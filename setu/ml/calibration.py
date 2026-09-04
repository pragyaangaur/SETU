"""Make the predicted quantiles mean what they say.

A model that outputs a ninetieth percentile should be above the observation ninety
percent of the time. Two things in this project push it away from that. Weighting
the loss toward disturbed minutes pulls the upper quantiles up, and training on
storms smaller than the ones held out pushes them down. The first version of the
model under covered badly, and the correction for it overshot.

Rather than guess at the weighting until the coverage lands, the mapping is
measured and inverted. This is the recalibration of Kuleshov, Fenner, and Ermon
(2018), applied to quantile outputs.

The idea is short. For every validation sample the predicted distribution is asked
where the observation actually fell, which gives a number between zero and one. If
the model were calibrated those numbers would be spread evenly over the interval.
They are not, and their empirical distribution is exactly the distortion. Inverting
it gives, for any level the caller asks for, the level the raw model has to be read
at to deliver it.

The map is fitted on validation data only. Fitting it on the test storms would make
the calibration report meaningless, which is the whole reason the report exists.
"""

import numpy as np


class QuantileCalibrator:
    """A monotone map from requested quantile level to raw model level."""

    def __init__(self, levels=None, grid_size=201):
        self.levels = np.asarray(levels, dtype=float) if levels is not None else None
        self.grid_size = grid_size
        self.requested = None
        self.raw = None

    def _predicted_position(self, pred_quantiles, target):
        """Where each observation fell inside its own predicted distribution.

        Args:
            pred_quantiles: Array of shape (samples, horizons, levels).
            target: Array of shape (samples, horizons).

        Returns:
            A flat array of positions between zero and one, one per observation.
        """
        pred = np.asarray(pred_quantiles, dtype=float)
        obs = np.asarray(target, dtype=float)
        positions = np.empty(obs.shape, dtype=float)
        for h in range(pred.shape[1]):
            for i in range(pred.shape[0]):
                row = pred[i, h]
                value = obs[i, h]
                if value <= row[0]:
                    # Below the lowest predicted quantile. Placed proportionally
                    # inside the bottom band rather than pinned at zero, so that a
                    # long quiet stretch does not pile up on the boundary.
                    span = max(row[1] - row[0], 1e-9)
                    positions[i, h] = self.levels[0] * np.clip(
                        1.0 - (row[0] - value) / span, 0.0, 1.0)
                elif value >= row[-1]:
                    span = max(row[-1] - row[-2], 1e-9)
                    tail = 1.0 - np.exp(-(value - row[-1]) / span)
                    positions[i, h] = self.levels[-1] + (1.0 - self.levels[-1]) * tail
                else:
                    positions[i, h] = np.interp(value, row, self.levels)
        return positions.ravel()

    def fit(self, pred_quantiles, target, levels=None):
        """Learn the distortion from a held out set the model did not train on."""
        if levels is not None:
            self.levels = np.asarray(levels, dtype=float)
        if self.levels is None:
            raise ValueError("the quantile levels must be given before fitting")

        positions = self._predicted_position(pred_quantiles, target)
        positions = positions[np.isfinite(positions)]
        if positions.size < 50:
            raise ValueError("not enough valid samples to fit a calibration map")

        # The empirical distribution of those positions is the distortion. Reading
        # it backwards gives the raw level needed to deliver a requested level.
        grid = np.linspace(0.0, 1.0, self.grid_size)
        empirical = np.searchsorted(np.sort(positions), grid, side="right") / positions.size

        # Both axes must increase for the inverse to be well defined, so the
        # empirical curve is forced upward where sampling noise dips it.
        empirical = np.maximum.accumulate(empirical)
        self.requested = empirical
        self.raw = grid
        return self

    def raw_level(self, requested_level):
        """Which raw model level delivers the requested coverage."""
        if self.requested is None:
            raise RuntimeError("the calibrator has not been fitted")
        return float(np.clip(np.interp(requested_level, self.requested, self.raw),
                             1e-4, 1.0 - 1e-4))

    def calibrate(self, pred_quantiles, requested_levels=None):
        """Rewrite a set of predicted quantiles so the levels mean what they say.

        Each requested level is mapped to the raw level that delivers it, and the
        value at that raw level is read off the model's own quantiles by
        interpolation. The output keeps the same shape and the same ordering.
        """
        pred = np.asarray(pred_quantiles, dtype=float)
        requested = (np.asarray(requested_levels, dtype=float)
                     if requested_levels is not None else self.levels)
        raw_levels = np.array([self.raw_level(q) for q in requested])

        out = np.empty(pred.shape[:2] + (len(requested),))
        for h in range(pred.shape[1]):
            for i in range(pred.shape[0]):
                out[i, h] = np.interp(raw_levels, self.levels, pred[i, h])
        # Interpolation cannot cross, but floating point can tie, so the result is
        # made strictly non decreasing to keep every downstream reader safe.
        return np.maximum.accumulate(out, axis=2)

    def state(self) -> dict:
        return {"levels": self.levels, "requested": self.requested, "raw": self.raw}

    @classmethod
    def from_state(cls, state) -> "QuantileCalibrator":
        obj = cls(levels=np.asarray(state["levels"], dtype=float))
        obj.requested = np.asarray(state["requested"], dtype=float)
        obj.raw = np.asarray(state["raw"], dtype=float)
        return obj


def coverage_error(pred_quantiles, target, levels) -> float:
    """Mean absolute gap between nominal and observed coverage, over all levels.

    One number for how honest a set of predictions is. Zero is perfect.
    """
    t = np.asarray(target)[:, :, None]
    observed = (t <= np.asarray(pred_quantiles)).mean(axis=(0, 1))
    return float(np.mean(np.abs(observed - np.asarray(levels))))
