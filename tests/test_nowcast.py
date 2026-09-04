"""Tests for the standing forecast service and its verification record.

The point of the ledger is that it cannot be adjusted after the fact, so the parts
worth testing are the ones that could quietly break that. A forecast has to be
scored against the right minute, a day that has been closed has to keep its counts,
and a station that stopped reporting must never be treated as ground truth.

Everything here runs offline on constructed records. Nothing reaches the network.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from setu import nowcast


def _entry(issued_at, persistence, observed, alarm, backfilled=False):
    """One ledger row, built the way the service builds it."""
    issued = pd.Timestamp(issued_at)
    horizons = {}
    for horizon in (30, 45, 60, 90):
        horizons[str(horizon)] = {
            "valid_at": str(issued + pd.Timedelta(minutes=horizon)),
            "quantiles": [0.01, 0.02, 0.03, 0.05, 0.09, 0.2],
            "median": 0.03,
            "probability": {"0.1": 0.9 if alarm else 0.01,
                            "0.3": 0.1, "1.0": 0.0},
            "probability_cut": 0.15,
            "alarm": alarm,
            "observed_dbdt": observed,
            "verified_at": None if observed is None else str(issued),
        }
    return {"run_at": str(issued), "issued_at": str(issued),
            "sources": ["ACE"], "observatory": "HYB",
            "persistence_dbdt": persistence, "backfilled": backfilled,
            "horizons": horizons}


def test_target_matches_the_training_definition():
    """The verifier has to compute the same quantity the model was trained on.

    That is the peak of the rate of change of the horizontal field over a window
    centred on the minute, not the value at the minute itself. A single spike
    inside the window has to show up at the centre of it.
    """
    index = pd.date_range("2026-01-01", periods=61, freq="1min")
    bx = np.zeros(61)
    bx[30] = 60.0  # a one minute spike of 60 nT
    frame = pd.DataFrame({"bx": bx, "by": np.zeros(61)}, index=index)

    series = nowcast.ground_series(frame)
    centre = pd.Timestamp("2026-01-01 00:30")

    # np.gradient over a one minute cadence turns a 60 nT spike into 0.5 nT/s.
    assert nowcast._peak_at(series, centre) == pytest.approx(0.5, rel=1e-6)
    # The same peak is seen from either edge of the centred window.
    assert nowcast._peak_at(series, centre - pd.Timedelta(minutes=7)) == pytest.approx(0.5)
    # Well outside the window it is not.
    assert nowcast._peak_at(series, centre + pd.Timedelta(minutes=20)) == pytest.approx(0.0)


def test_peak_is_none_when_the_record_does_not_reach():
    index = pd.date_range("2026-01-01", periods=30, freq="1min")
    frame = pd.DataFrame({"bx": np.zeros(30), "by": np.zeros(30)}, index=index)
    series = nowcast.ground_series(frame)
    assert nowcast._peak_at(series, pd.Timestamp("2026-01-02")) is None


def test_verify_attaches_the_outcome_and_leaves_settled_rows_alone(monkeypatch):
    """A forecast is scored once, against the minute it was made for."""
    now = dt.datetime(2026, 1, 1, 12, 0)
    monkeypatch.setattr(nowcast, "_now", lambda: now)

    index = pd.date_range("2026-01-01 08:00", periods=300, freq="1min")
    bx = np.zeros(300)
    bx[index.get_loc(pd.Timestamp("2026-01-01 10:30"))] = 60.0
    frame = pd.DataFrame({"bx": bx, "by": np.zeros(300)}, index=index)
    monkeypatch.setattr(nowcast, "first_reporting", lambda **kw: ("HYB", frame))

    # Issued at 10:00, so the thirty minute horizon is valid at 10:30, which is
    # exactly the minute of the spike.
    entries = [_entry("2026-01-01 10:00", 0.0, None, False)]
    scored = nowcast.verify(entries)

    assert scored == 4
    assert entries[0]["horizons"]["30"]["observed_dbdt"] == pytest.approx(0.5)
    assert entries[0]["horizons"]["90"]["observed_dbdt"] == pytest.approx(0.0)
    assert entries[0]["horizons"]["30"]["verified_against"] == "HYB"

    # A second pass changes nothing, because a settled row is never revisited.
    assert nowcast.verify(entries) == 0


def test_verify_will_not_score_a_forecast_that_has_not_come_due(monkeypatch):
    now = dt.datetime(2026, 1, 1, 10, 5)
    monkeypatch.setattr(nowcast, "_now", lambda: now)

    def fail(**kw):
        raise AssertionError("the ground record must not be fetched with nothing due")

    monkeypatch.setattr(nowcast, "first_reporting", fail)
    entries = [_entry("2026-01-01 10:00", 0.0, None, False)]
    assert nowcast.verify(entries) == 0


def test_a_closed_day_keeps_its_counts_when_more_rows_are_folded_in():
    """Folding is additive. A day already in the record is added to, never replaced.

    This is the property the whole record rests on. If a later run could rewrite an
    earlier day, a bad week could be made to disappear.
    """
    first = nowcast.fold([_entry("2026-01-01 10:00", 0.0, 0.5, True)], {"days": {}})
    both = nowcast.fold([_entry("2026-01-01 11:00", 0.0, 0.0, False)], first)

    counts = both["days"]["2026-01-01"]["horizons"]["30"]
    assert counts["n"] == 2
    assert counts["events"] == 1
    # The first row alarmed on an event, the second did not alarm and nothing
    # happened, so the model has one hit and one correct negative.
    assert counts["model"] == {"hits": 1, "false_alarms": 0, "misses": 0,
                               "correct_negatives": 1}
    # Persistence saw a quiet ground both times, so it missed the event.
    assert counts["persistence"]["misses"] == 1


def test_a_row_with_no_outcome_is_never_counted():
    folded = nowcast.fold([_entry("2026-01-01 10:00", 0.0, None, True)], {"days": {}})
    assert folded["days"]["2026-01-01"]["horizons"] == {}


def test_the_model_and_the_baseline_are_scored_on_the_same_minutes():
    """A row only enters the count when both had something to say.

    A row with no persistence value is dropped from both sides rather than counted
    for the model alone, which would let the model be scored on minutes its
    baseline was excused from.
    """
    entries = [_entry("2026-01-01 10:00", None, 0.5, True),
               _entry("2026-01-01 11:00", 0.0, 0.5, True)]
    board = nowcast.scoreboard({"days": {}}, entries)
    assert board["horizons"]["30"]["n"] == 1
    assert board["horizons"]["30"]["model"]["n"] == board["horizons"]["30"]["persistence"]["n"]


def test_heidke_skill_is_absent_rather_than_zero_when_nothing_happened():
    """A quiet stretch has no skill to report, and zero would read as a failure."""
    entries = [_entry(f"2026-01-01 {h:02d}:00", 0.0, 0.0, False) for h in range(10)]
    board = nowcast.scoreboard({"days": {}}, entries)
    block = board["horizons"]["30"]

    assert board["quiet"] is True
    assert block["events"] == 0
    assert block["model"]["hss"] is None
    assert block["model"]["correct_negatives"] == 10
    assert block["model"]["false_alarms"] == 0


def test_skill_is_computed_the_standard_way():
    """A worked contingency table, checked against the Heidke definition by hand."""
    skill = nowcast._skill({"hits": 8, "false_alarms": 2, "misses": 2,
                            "correct_negatives": 88})
    assert skill["pod"] == pytest.approx(0.8)
    assert skill["far"] == pytest.approx(0.2)
    expected = ((10 * 10) + (90 * 90)) / 100.0
    assert skill["hss"] == pytest.approx((96 - expected) / (100 - expected))


def test_live_and_backfilled_rows_are_counted_apart():
    entries = [_entry("2026-01-01 10:00", 0.0, 0.0, False, backfilled=True),
               _entry("2026-01-01 11:00", 0.0, 0.0, False)]
    board = nowcast.scoreboard({"days": {}}, entries)
    assert board["forecasts_backfilled"] == 1
    assert board["forecasts_issued_live"] == 1
    assert board["forecasts_issued"] == 2


def test_a_station_that_stopped_reporting_is_not_treated_as_ground_truth(monkeypatch):
    """Row count alone is not evidence that a station is live.

    A station that fell silent two days ago still returns a full day of good rows
    for the day before it stopped. An earlier version of this check accepted that
    and scored forecasts against a record which ended before they were issued, so
    the newest minute has to be recent as well.
    """
    from setu.data import magnetometer

    now = dt.datetime(2026, 1, 3, 12, 0)

    def fake_fetch(code, hours=6, timeout=60):
        end = (dt.datetime(2026, 1, 1, 23, 59) if code == "ABG"
               else dt.datetime(2026, 1, 3, 11, 55))
        index = pd.date_range(end=end, periods=600, freq="1min")
        return pd.DataFrame({"bx": np.zeros(600), "by": np.zeros(600)}, index=index)

    class FrozenDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now.replace(tzinfo=tz)

    monkeypatch.setattr(magnetometer, "fetch_recent", fake_fetch)
    monkeypatch.setattr(magnetometer.dt, "datetime", FrozenDateTime)

    code, frame = magnetometer.first_reporting(("ABG", "HYB"))
    assert code == "HYB"
    assert frame.index[-1] == pd.Timestamp("2026-01-03 11:55")

    # With only the stale station on offer, nothing is returned at all.
    assert magnetometer.first_reporting(("ABG",)) == (None, None)
