"""Assemble training samples from real solar wind and real ground magnetometer data.

One sample is a window of solar wind history ending at the present minute, paired
with what the ground magnetic field actually did some minutes later. The pairing is
what encodes the lead time, and the lead time is the whole point of the system.

Two choices here are worth stating plainly.

The input is resampled to five minute cadence. The magnetosphere responds to the
solar wind over tens of minutes, so five minute resolution loses very little, and
it lets one window reach back five hours at a quarter of the cost.

The target is the largest rate of change inside a short window centred on the
horizon, rather than the value at one exact minute. An operator cannot act on the
value at one exact minute, and the peak inside a window is both more useful and
more predictable than a single sample of a very spiky quantity.
"""

import logging

import numpy as np
import pandas as pd

from setu.config import CADENCE_S, FORECAST_HORIZONS_MIN
from setu.data.magnetometer import fetch_observatory, to_disturbance
from setu.data.omni import fetch_range, fill_gaps
from setu.data.storms import StormEvent
from setu.ml.features import FEATURE_NAMES, Standardiser, build_features
from setu.ml.model import to_log_target
from setu.physics.geoelectric import dbdt

log = logging.getLogger(__name__)

INPUT_CADENCE_MIN = 5
TARGET_WINDOW_MIN = 15  # peak is taken over this span, centred on the horizon


def event_frames(event: StormEvent, observatory: str = "ABG"):
    """Load one storm and return aligned solar wind features and ground target.

    Returns:
        A pair of frames on a common five minute index. The first holds the model
        inputs, the second holds the one minute peak rate of change resampled to
        the same grid.
    """
    days = (event.end - event.start).days + 1
    solar_wind = fill_gaps(fetch_range(event.start, event.end))
    ground = to_disturbance(fetch_observatory(observatory, event.start, days))

    rate = pd.Series(
        dbdt(ground["bx"].ffill().bfill().values,
             ground["by"].ffill().bfill().values, CADENCE_S),
        index=ground.index, name="dbdt",
    )
    # Any minute where the raw record was missing is dropped rather than filled,
    # so an interpolated value never becomes a training target.
    rate[ground["bx"].isna() | ground["by"].isna()] = np.nan

    features = build_features(solar_wind)
    grid = features.resample(f"{INPUT_CADENCE_MIN}min").mean()
    peak = rate.rolling(f"{TARGET_WINDOW_MIN}min", center=True, min_periods=8).max()
    peak = peak.resample(f"{INPUT_CADENCE_MIN}min").max()

    joined = grid.join(peak.rename("dbdt"), how="inner")
    return joined[FEATURE_NAMES], joined["dbdt"]


def windowed_samples(features: pd.DataFrame, target: pd.Series, window: int,
                     horizons=FORECAST_HORIZONS_MIN):
    """Cut one storm into overlapping input windows with their future targets.

    A sample is kept only when the input window is complete and every horizon has
    a valid target, so a gap in either record removes the samples that touch it
    instead of being papered over.
    """
    steps = [h // INPUT_CADENCE_MIN for h in horizons]
    x_all = features.values
    y_all = target.values
    n = len(features)
    xs, ys, stamps = [], [], []
    for t in range(window - 1, n - max(steps)):
        chunk = x_all[t - window + 1: t + 1]
        future = np.array([y_all[t + s] for s in steps])
        if np.isnan(chunk).any() or np.isnan(future).any():
            continue
        xs.append(chunk.T)  # stored as (features, time)
        ys.append(future)
        stamps.append(features.index[t])
    if not xs:
        return (np.zeros((0, len(FEATURE_NAMES), window)),
                np.zeros((0, len(horizons))), pd.DatetimeIndex([]))
    return np.asarray(xs), np.asarray(ys), pd.DatetimeIndex(stamps)


def build_dataset(events, window: int, observatories=("ABG", "HYB"),
                  horizons=FORECAST_HORIZONS_MIN):
    """Build one array set from a list of storms and a list of observatories.

    Using more than one observatory roughly doubles the number of samples and it
    also stops the model from memorising the quirks of a single instrument. Both
    stations sit in the same low geomagnetic latitude band, so the large scale
    storm time variation they see is the same.
    """
    xs, ys, stamps, tags = [], [], [], []
    for event in events:
        for obs in observatories:
            try:
                features, target = event_frames(event, obs)
            except Exception as exc:
                log.warning("skipping %s at %s: %s", event.key, obs, exc)
                continue
            x, y, idx = windowed_samples(features, target, window, horizons)
            if len(x) == 0:
                log.warning("no usable samples for %s at %s", event.key, obs)
                continue
            log.info("%s at %s: %d samples", event.key, obs, len(x))
            xs.append(x)
            ys.append(y)
            stamps.append(idx)
            tags.extend([f"{event.key}/{obs}"] * len(x))
    if not xs:
        raise RuntimeError("no samples could be built from the given events")
    return {
        "x": np.concatenate(xs),
        "y": to_log_target(np.concatenate(ys)),
        "y_raw": np.concatenate(ys),
        "time": np.concatenate([i.values for i in stamps]),
        "tag": np.asarray(tags),
    }


def standardise(train: dict, *others: dict):
    """Fit the scaler on training data only and apply it everywhere.

    Fitting on anything else would leak information about the held out storms into
    the model, and the whole value of holding a storm out is that it did not touch
    the model in any way.
    """
    scaler = Standardiser(FEATURE_NAMES)
    flat = train["x"].transpose(0, 2, 1).reshape(-1, train["x"].shape[1])
    scaler.fit(flat)

    def apply(block):
        shape = block["x"].shape
        flat_block = block["x"].transpose(0, 2, 1).reshape(-1, shape[1])
        scaled = scaler.transform(flat_block)
        block = dict(block)
        block["x"] = scaled.reshape(shape[0], shape[2], shape[1]).transpose(0, 2, 1)
        return block

    return scaler, apply(train), [apply(o) for o in others]
