"""Checks on the data layer, especially the clock the solar wind sits on.

Getting the time base wrong is the failure that cost the first version of this
model its ability to warn about a storm onset, and it is invisible in any loss
curve. These tests make it visible instead.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from setu.config import EARTH_RADIUS_KM, PHASE_FRONT_CORRECTION
from setu.data.omni import (MAX_TIMESHIFT_S, MIN_TIMESHIFT_S, advection_delay_s,
                            to_l1_time_base)
from setu.ml.features import FEATURE_NAMES, Standardiser, build_features


def synthetic_wind(minutes=600, delay_s=3000.0):
    """A record with a known shock at a known time and a constant delay."""
    index = pd.date_range("2024-01-01", periods=minutes, freq="1min")
    frame = pd.DataFrame(index=index)
    shock = minutes // 2
    frame["b_total"] = np.where(np.arange(minutes) < shock, 5.0, 30.0)
    frame["bx_gsm"] = 1.0
    frame["by_gsm"] = 2.0
    frame["bz_gsm"] = np.where(np.arange(minutes) < shock, 1.0, -20.0)
    frame["speed"] = np.where(np.arange(minutes) < shock, 400.0, 700.0)
    frame["density"] = 5.0
    frame["pressure"] = 2.0
    frame["timeshift"] = delay_s
    frame["vx_gse"] = -frame["speed"]
    frame["sc_x_re"] = 235.0
    frame["sc_y_re"] = 20.0
    frame["sc_z_re"] = 5.0
    return frame, index[shock]


def test_archive_time_base_moves_the_shock_earlier_by_the_recorded_shift():
    frame, shock_time = synthetic_wind(delay_s=3000.0)
    shifted = to_l1_time_base(frame, method="archive")
    moved = shifted["b_total"].diff().idxmax()
    assert abs((shock_time - moved).total_seconds() - 3000.0) < 120.0


def test_advection_time_base_uses_position_over_speed():
    """The delay has to fall when the solar wind speeds up."""
    frame, _ = synthetic_wind()
    delay = advection_delay_s(frame)
    slow = delay.iloc[0]
    fast = delay.iloc[-1]
    assert fast < slow
    expected_slow = 235.0 * EARTH_RADIUS_KM * PHASE_FRONT_CORRECTION / 400.0
    assert slow == pytest.approx(expected_slow, rel=1e-6)


def test_the_l1_base_buys_warning_time():
    """This is the whole point of the change, so it gets its own test."""
    frame, shock_time = synthetic_wind()
    shifted = to_l1_time_base(frame, method="advection")
    moved = shifted["b_total"].diff().idxmax()
    warning_minutes = (shock_time - moved).total_seconds() / 60.0
    assert warning_minutes > 20.0


def test_rows_with_an_impossible_delay_are_dropped():
    frame, _ = synthetic_wind()
    frame.loc[frame.index[:50], "timeshift"] = MIN_TIMESHIFT_S - 1.0
    frame.loc[frame.index[50:100], "timeshift"] = MAX_TIMESHIFT_S + 1.0
    shifted = to_l1_time_base(frame, method="archive")
    assert len(shifted) < len(frame)


def test_unknown_time_base_is_rejected():
    frame, _ = synthetic_wind()
    with pytest.raises(ValueError):
        to_l1_time_base(frame, method="whatever")


def test_features_include_the_propagation_delay_and_it_is_sensible():
    frame, _ = synthetic_wind()
    features = build_features(frame)
    assert "propagation_delay_min" in features
    delay = features["propagation_delay_min"]
    # 235 Earth radii at 400 km per second is a little over sixty minutes, and at
    # 700 km per second it is a little over thirty five.
    assert 55.0 < delay.iloc[0] < 70.0
    assert 30.0 < delay.iloc[-1] < 40.0


def test_feature_order_is_fixed():
    """The scaler and the saved model both index features by position."""
    frame, _ = synthetic_wind()
    assert list(build_features(frame).columns) == FEATURE_NAMES


def test_standardiser_fits_and_inverts_scale():
    frame, _ = synthetic_wind()
    values = build_features(frame).values
    scaler = Standardiser()
    scaled = scaler.fit_transform(values)
    assert np.nanmax(np.abs(np.nanmean(scaled, axis=0))) < 1e-8
    restored = Standardiser.from_state(scaler.state())
    assert np.allclose(restored.transform(values), scaled, equal_nan=True)


def test_the_phase_front_correction_shortens_the_delay():
    """A tilted front reaches the Earth sooner than a flat one, so the correction
    has to be below one and the delay has to come out shorter than the plain
    distance over speed value."""
    assert 0.8 < PHASE_FRONT_CORRECTION < 1.0
    frame, _ = synthetic_wind()
    plain = 235.0 * EARTH_RADIUS_KM / 400.0
    assert advection_delay_s(frame).iloc[0] < plain


def test_no_lookahead_in_the_advection_delay():
    """The delay at one minute must use only that minute's own telemetry."""
    frame, _ = synthetic_wind()
    full = advection_delay_s(frame)
    truncated = advection_delay_s(frame.iloc[:100])
    assert np.allclose(full.iloc[:100].values, truncated.values, equal_nan=True)


def test_windowed_samples_returns_the_current_value_with_the_future_ones():
    """The persistence baseline needs the value at the moment of the forecast, and
    a caller that unpacks the wrong number of returns should fail loudly here
    rather than in a replay hours later."""
    import pandas as pd

    from setu.ml.dataset import windowed_samples
    from setu.ml.features import FEATURE_NAMES

    rng = np.random.default_rng(5)
    index = pd.date_range("2024-01-01", periods=200, freq="5min")
    features = pd.DataFrame(
        rng.normal(size=(200, len(FEATURE_NAMES))), index=index, columns=FEATURE_NAMES)
    target = pd.Series(np.abs(rng.normal(size=200)), index=index)

    x, y, now, stamps = windowed_samples(features, target, window=20,
                                         horizons=(30, 60))
    assert x.shape[0] == y.shape[0] == now.shape[0] == len(stamps)
    assert y.shape[1] == 2
    assert now.ndim == 1
    # The current value must be the target at the window end, not at a horizon.
    position = list(features.index).index(stamps[0])
    assert now[0] == pytest.approx(target.iloc[position])
