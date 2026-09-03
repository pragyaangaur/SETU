"""The forecast network.

The model reads a window of solar wind history measured at the first Lagrange
point and predicts the distribution of the ground magnetic rate of change at an
Indian observatory, at several horizons ahead. It is a dilated causal convolution
stack, which is a good fit for this problem because the response of the
magnetosphere is spread over hours and a stack of dilated layers reaches back over
hours with very few parameters.

Two things make this a physics informed model rather than a plain regression.

The first is the input. The network is handed the coupling functions from
``setu.ml.features`` alongside the raw measurements, so the combinations that
magnetospheric physics says carry the energy are available directly instead of
having to be discovered from a small dataset.

The second is a constraint in the loss. The forecast is required to be
non decreasing in the energy coupling inputs, which is enforced by perturbing those
inputs upward and penalising any case where the prediction falls. A network that
satisfies this constraint cannot predict a calmer ground in response to a stronger
driver, which is a statement about physics and not about the training data.

The output quantiles are made non crossing by construction. The lowest quantile is
predicted directly and each higher one is reached by adding a strictly positive
increment, so the ninetieth percentile can never come out below the tenth.
"""

import numpy as np

from setu.config import FORECAST_HORIZONS_MIN, QUANTILES
from setu.ml.features import FEATURE_NAMES
from setu.ml.nn import (Adam, CausalConv1d, ChannelNorm, Dense, Dropout, ReLU,
                        ResidualBlock, pinball_loss)

# The target is stored as the natural logarithm of the rate of change plus a floor.
# The floor keeps quiet minutes finite and sets the resolution at the bottom end.
TARGET_FLOOR = 0.01


def to_log_target(dbdt_nt_per_s):
    """Compress the target, which spans several orders of magnitude."""
    return np.log(np.asarray(dbdt_nt_per_s, dtype=float) + TARGET_FLOOR)


def from_log_target(y):
    """Undo the compression. The logarithm is monotonic, so quantiles survive it."""
    return np.exp(np.asarray(y, dtype=float)) - TARGET_FLOOR


def softplus(x):
    return np.logaddexp(0.0, x)


def softplus_grad(x):
    return 1.0 / (1.0 + np.exp(-x))


class GICNet:
    """Dilated causal convolution network with monotone quantile outputs."""

    def __init__(self, n_features=None, channels=32, blocks=(1, 2, 4, 8),
                 window=96, horizons=FORECAST_HORIZONS_MIN, quantiles=QUANTILES,
                 dropout=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.n_features = n_features or len(FEATURE_NAMES)
        self.window = window
        self.horizons = tuple(horizons)
        self.quantiles = tuple(quantiles)
        self.channels = channels
        self.block_dilations = tuple(blocks)

        self.stem = CausalConv1d(self.n_features, channels, 3, 1, rng)
        self.stem_norm = ChannelNorm(channels)
        self.stem_act = ReLU()
        self.blocks = [ResidualBlock(channels, d, 3, dropout, rng) for d in blocks]

        # The head reads the last time step only, because that is the present
        # moment and a causal stack has already folded the history into it.
        self.head_drop = Dropout(dropout, rng)
        self.head1 = Dense(channels, 64, rng)
        self.head_act = ReLU()
        self.head2 = Dense(64, len(self.horizons) * len(self.quantiles), rng)
        # The last layer starts small so the network begins with a nearly flat
        # prediction near the middle of the target range. Starting from a large
        # random output makes the first epochs fight their own initialisation
        # instead of learning anything.
        self.head2.w *= 0.05
        self.head2.b[...] = -2.0

        self.layers = ([self.stem, self.stem_norm, self.stem_act] + self.blocks
                       + [self.head_drop, self.head1, self.head_act, self.head2])

    @property
    def receptive_field(self):
        """How many time steps of history reach the output, in samples."""
        return 1 + 2 + 4 * sum(self.block_dilations)

    def _named(self, which):
        out = {}
        simple = {"stem": self.stem, "stem_norm": self.stem_norm,
                  "head1": self.head1, "head2": self.head2}
        for name, layer in simple.items():
            source = layer.parameters() if which == "p" else layer.gradients()
            for k, v in source.items():
                out[f"{name}.{k}"] = v
        for i, block in enumerate(self.blocks):
            part = (block.named_parameters(f"block{i}") if which == "p"
                    else block.named_gradients(f"block{i}"))
            out.update(part)
        return out

    def parameters(self):
        return self._named("p")

    def gradients(self):
        return self._named("g")

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def forward(self, x, training=True):
        """Run the network.

        Args:
            x: Standardised inputs of shape (batch, features, window).

        Returns:
            Predictions of shape (batch, horizons, quantiles), in log target space,
            already sorted so that quantiles do not cross.
        """
        h = self.stem.forward(x, training)
        h = self.stem_norm.forward(h, training)
        h = self.stem_act.forward(h, training)
        for block in self.blocks:
            h = block.forward(h, training)
        last = h[:, :, -1]
        if training:
            self._time_shape = h.shape
        last = self.head_drop.forward(last, training)
        d = self.head1.forward(last, training)
        d = self.head_act.forward(d, training)
        raw = self.head2.forward(d, training)
        raw = raw.reshape(-1, len(self.horizons), len(self.quantiles))
        if training:
            self._raw = raw
        base = raw[:, :, :1]
        increments = softplus(raw[:, :, 1:])
        return np.concatenate([base, base + np.cumsum(increments, axis=2)], axis=2)

    def backward(self, grad_out):
        """Push the gradient of the loss back to the parameters.

        Args:
            grad_out: Gradient with respect to the sorted output, of shape
                (batch, horizons, quantiles).

        Returns:
            The gradient with respect to the input, which the monotonicity
            constraint does not need but which is useful for inspecting what the
            network is sensitive to.
        """
        raw = self._raw
        g = np.zeros_like(raw)
        # The base level feeds every quantile, because each one is the base plus a
        # running sum of increments.
        g[:, :, 0] = grad_out.sum(axis=2)
        # Increment k contributes to every quantile from k onward.
        reverse = np.cumsum(grad_out[:, :, ::-1], axis=2)[:, :, ::-1]
        g[:, :, 1:] = reverse[:, :, 1:] * softplus_grad(raw[:, :, 1:])

        h = self.head2.backward(g.reshape(raw.shape[0], -1))
        h = self.head_act.backward(h)
        h = self.head1.backward(h)
        h = self.head_drop.backward(h)

        full = np.zeros(self._time_shape)
        full[:, :, -1] = h
        for block in reversed(self.blocks):
            full = block.backward(full)
        full = self.stem_act.backward(full)
        full = self.stem_norm.backward(full)
        return self.stem.backward(full)

    def predict_quantiles(self, x):
        """Forecast in the physical unit of nanotesla per second."""
        return from_log_target(self.forward(x, training=False))

    def exceedance_probability(self, x, threshold_nt_per_s):
        """Probability that the rate of change goes above a threshold.

        The model outputs a set of quantiles, which is a description of the
        cumulative distribution sampled at fixed probabilities. Reading it the
        other way round, by interpolating in the value axis, gives the probability
        of passing any level an operator cares about. This is the number that
        drives the decision layer, because acting on a storm has a cost and a
        point forecast gives no way to weigh it.
        """
        q = self.predict_quantiles(x)
        levels = np.asarray(self.quantiles)
        out = np.zeros(q.shape[:2])
        for b in range(q.shape[0]):
            for h in range(q.shape[1]):
                values = q[b, h]
                if threshold_nt_per_s <= values[0]:
                    out[b, h] = 1.0
                elif threshold_nt_per_s >= values[-1]:
                    # Beyond the highest quantile the tail is extrapolated with an
                    # exponential, which is the usual shape for this quantity.
                    span = max(values[-1] - values[-2], 1e-6)
                    decay = np.exp(-(threshold_nt_per_s - values[-1]) / span)
                    out[b, h] = (1.0 - levels[-1]) * decay
                else:
                    out[b, h] = 1.0 - np.interp(threshold_nt_per_s, values, levels)
        return out

    def save(self, path):
        state = {k: v for k, v in self.parameters().items()}
        state["_config"] = np.array([self.n_features, self.channels, self.window,
                                     len(self.horizons), len(self.quantiles)])
        state["_dilations"] = np.array(self.block_dilations)
        state["_horizons"] = np.array(self.horizons)
        state["_quantiles"] = np.array(self.quantiles)
        np.savez_compressed(path, **state)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        cfg = data["_config"]
        model = cls(n_features=int(cfg[0]), channels=int(cfg[1]),
                    blocks=tuple(int(d) for d in data["_dilations"]),
                    window=int(cfg[2]), horizons=tuple(int(h) for h in data["_horizons"]),
                    quantiles=tuple(float(q) for q in data["_quantiles"]), dropout=0.0)
        for k, v in model.parameters().items():
            v[...] = data[k]
        return model


def monotonicity_penalty(model, x, feature_index, delta=0.25, weight=1.0):
    """Penalise a forecast that falls when the energy input rises.

    The constraint is applied by finite difference across a pair of perturbations.
    The chosen input channels are pushed up by a fixed amount and down by the same
    amount, and any case where the lower input predicts a rougher ground than the
    higher one is charged.

    Perturbing in both directions matters. An earlier version compared the
    perturbed prediction against the unperturbed one and left the unperturbed one
    out of the gradient. That version could satisfy the constraint by raising the
    whole output, which it then did without limit, because raising everything also
    raised the reference it was being compared against. Charging both sides means a
    uniform shift cancels exactly, so the only way to reduce the penalty is to
    change the slope, which is what the constraint is actually about.

    Args:
        model: The network.
        x: Standardised inputs of shape (batch, features, window).
        feature_index: Indices of the channels the constraint applies to.
        delta: Size of the perturbation in each direction, in standardised units.
        weight: Multiplier applied to the gradient before it is accumulated.

    Returns:
        The penalty value. The gradient is accumulated into the model directly,
        because the two sides of the difference need a backward pass each and the
        caches for them cannot both be held at once.
    """
    lowered = x.copy()
    lowered[:, feature_index, :] -= delta
    raised = x.copy()
    raised[:, feature_index, :] += delta

    y_low = model.forward(lowered, training=False)
    y_high = model.forward(raised, training=True)
    violation = np.maximum(y_low - y_high, 0.0)
    scale = 2.0 * weight / y_high.size

    # Push the prediction at the stronger driver upward.
    model.backward(-scale * violation)
    # Push the prediction at the weaker driver downward by the same amount.
    model.forward(lowered, training=True)
    model.backward(scale * violation)

    return float((violation ** 2).mean())
