"""A small neural network framework written directly on NumPy.

There is no deep learning library in the dependency list of this project. The
network is written out by hand instead, with an explicit forward pass and an
explicit backward pass for every layer. That keeps the install to four ordinary
scientific packages, which means the whole system runs on any machine that can run
Python, and it keeps the model small enough that every part of it can be explained.

Tensors are laid out as (batch, channel, time) throughout, which is the natural
order for one dimensional convolution over a time series.
"""

import numpy as np


class Layer:
    """Base class. A layer holds parameters, gradients, and a cache for backward."""

    def parameters(self):
        return {}

    def gradients(self):
        return {}

    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def zero_grad(self):
        for g in self.gradients().values():
            g.fill(0.0)


class CausalConv1d(Layer):
    """Dilated causal convolution over time.

    Causal means the output at one minute depends only on that minute and earlier
    ones, so the network can never see the future. Dilation spreads the taps of the
    kernel apart, which lets a stack of a few layers reach back hours without the
    parameter count of a wide kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, rng=None):
        rng = rng or np.random.default_rng(0)
        fan_in = in_channels * kernel_size
        scale = np.sqrt(2.0 / fan_in)  # He initialisation, matched to the ReLU below
        self.w = rng.normal(0.0, scale, (out_channels, in_channels, kernel_size))
        self.b = np.zeros(out_channels)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) * dilation
        self._taps = None

    def parameters(self):
        return {"w": self.w, "b": self.b}

    def gradients(self):
        return {"w": self.dw, "b": self.db}

    def forward(self, x, training=True):
        b, c, t = x.shape
        xp = np.zeros((b, c, t + self.pad))
        xp[:, :, self.pad:] = x
        taps = [xp[:, :, j * self.dilation: j * self.dilation + t]
                for j in range(self.kernel_size)]
        out = np.zeros((b, self.w.shape[0], t))
        for j, tap in enumerate(taps):
            out += np.einsum("bct,oc->bot", tap, self.w[:, :, j], optimize=True)
        if training:
            self._taps = taps
            self._shape = (b, c, t)
        return out + self.b[None, :, None]

    def backward(self, grad):
        b, c, t = self._shape
        self.db += grad.sum(axis=(0, 2))
        dxp = np.zeros((b, c, t + self.pad))
        for j, tap in enumerate(self._taps):
            self.dw[:, :, j] += np.einsum("bot,bct->oc", grad, tap, optimize=True)
            dxp[:, :, j * self.dilation: j * self.dilation + t] += np.einsum(
                "bot,oc->bct", grad, self.w[:, :, j], optimize=True)
        self._taps = None
        return dxp[:, :, self.pad:]


class ReLU(Layer):
    """Rectifier.

    The mask is kept only on a training pass. An evaluation pass must leave the
    cache alone, because the monotonicity constraint runs an evaluation forward in
    between a training forward and its backward, and overwriting the mask there
    would silently corrupt the gradient.
    """

    def __init__(self):
        self._mask = None

    def forward(self, x, training=True):
        if training:
            self._mask = x > 0
        return np.maximum(x, 0.0)

    def backward(self, grad):
        return grad * self._mask


class ChannelNorm(Layer):
    """Normalise across the channel axis at every time step, then rescale.

    This is layer normalisation applied per time step. It keeps the activations of
    a deep dilated stack in a usable range without needing batch statistics, which
    matters here because the batches are small.
    """

    def __init__(self, channels, eps=1e-5):
        self.g = np.ones(channels)
        self.b = np.zeros(channels)
        self.dg = np.zeros_like(self.g)
        self.db = np.zeros_like(self.b)
        self.eps = eps

    def parameters(self):
        return {"g": self.g, "b": self.b}

    def gradients(self):
        return {"g": self.dg, "b": self.db}

    def forward(self, x, training=True):
        mu = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mu) * inv
        if training:
            self._cache = (xhat, inv)
        return xhat * self.g[None, :, None] + self.b[None, :, None]

    def backward(self, grad):
        xhat, inv = self._cache
        c = xhat.shape[1]
        self.dg += (grad * xhat).sum(axis=(0, 2))
        self.db += grad.sum(axis=(0, 2))
        dxhat = grad * self.g[None, :, None]
        return inv / c * (c * dxhat
                          - dxhat.sum(axis=1, keepdims=True)
                          - xhat * (dxhat * xhat).sum(axis=1, keepdims=True))


class Dropout(Layer):
    def __init__(self, rate=0.1, rng=None):
        self.rate = rate
        self.rng = rng or np.random.default_rng(0)

    def forward(self, x, training=True):
        if not training or self.rate <= 0:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) >= self.rate) / (1.0 - self.rate)
        return x * self._mask

    def backward(self, grad):
        return grad if self._mask is None else grad * self._mask


class Dense(Layer):
    """Fully connected layer acting on a flat vector per sample."""

    def __init__(self, in_features, out_features, rng=None):
        rng = rng or np.random.default_rng(0)
        self.w = rng.normal(0.0, np.sqrt(2.0 / in_features), (in_features, out_features))
        self.b = np.zeros(out_features)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def parameters(self):
        return {"w": self.w, "b": self.b}

    def gradients(self):
        return {"w": self.dw, "b": self.db}

    def forward(self, x, training=True):
        if training:
            self._x = x
        return x @ self.w + self.b

    def backward(self, grad):
        self.dw += self._x.T @ grad
        self.db += grad.sum(axis=0)
        return grad @ self.w.T


class ResidualBlock(Layer):
    """Two dilated convolutions with a skip connection around them.

    The skip connection is what makes a deep stack trainable, because the gradient
    has a path back to the input that does not pass through every convolution. A
    one by one convolution is inserted on the skip path when the channel count
    changes.
    """

    def __init__(self, channels, dilation, kernel_size=3, dropout=0.1, rng=None):
        rng = rng or np.random.default_rng(0)
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation, rng)
        self.norm1 = ChannelNorm(channels)
        self.act1 = ReLU()
        self.drop = Dropout(dropout, rng)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation, rng)
        self.norm2 = ChannelNorm(channels)
        self.act2 = ReLU()
        self.sublayers = [self.conv1, self.norm1, self.act1, self.drop,
                          self.conv2, self.norm2, self.act2]

    def forward(self, x, training=True):
        h = x
        for layer in self.sublayers:
            h = layer.forward(h, training)
        return x + h

    def backward(self, grad):
        h = grad
        for layer in reversed(self.sublayers):
            h = layer.backward(h)
        return grad + h

    def zero_grad(self):
        for layer in self.sublayers:
            layer.zero_grad()

    def named_parameters(self, prefix):
        out = {}
        for i, layer in enumerate(self.sublayers):
            for k, v in layer.parameters().items():
                out[f"{prefix}.{i}.{k}"] = v
        return out

    def named_gradients(self, prefix):
        out = {}
        for i, layer in enumerate(self.sublayers):
            for k, v in layer.gradients().items():
                out[f"{prefix}.{i}.{k}"] = v
        return out


class Adam:
    """Adam with decoupled weight decay and global gradient norm clipping.

    Clipping matters here because the target is a heavy tailed quantity. One
    extreme minute in a batch can otherwise produce a step large enough to undo
    an epoch of progress.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, clip=1.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip = clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        if self.clip:
            total = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
            scale = min(1.0, self.clip / (total + 1e-12))
        else:
            scale = 1.0
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, p in self.params.items():
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            step = self.lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)
            if self.weight_decay:
                step = step + self.lr * self.weight_decay * p
            p -= step


def pinball_loss(pred, target, quantiles, weights=None):
    """Quantile loss, also called the pinball loss.

    For a quantile level q the loss charges q for underprediction and 1-q for
    overprediction. Minimising it drives the output to the qth quantile of the
    conditional distribution of the target, which is how this model produces a
    probability of exceeding a threshold rather than a single number.

    Args:
        pred: Array of shape (batch, horizons, quantiles).
        target: Array of shape (batch, horizons).
        quantiles: Sequence of quantile levels, matching the last axis of ``pred``.
        weights: Optional per sample and per horizon weights of shape
            (batch, horizons). They are used to pull the fit toward the disturbed
            minutes, which are rare and are the only ones anybody cares about.

    Returns:
        A pair of the mean loss and the gradient with respect to ``pred``.
    """
    q = np.asarray(quantiles).reshape(1, 1, -1)
    diff = target[:, :, None] - pred
    loss = np.maximum(q * diff, (q - 1.0) * diff)
    grad = np.where(diff > 0, -q, 1.0 - q)
    if weights is not None:
        w = np.asarray(weights, dtype=float)[:, :, None]
        loss = loss * w
        grad = grad * w
    return float(loss.mean()), grad / pred.size
