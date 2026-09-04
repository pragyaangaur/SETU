"""The standing forecast service, and the ledger that keeps it honest.

Everything else in this project is measured on storms that have already happened.
That is the right way to develop a model and it is the wrong way to prove one,
because the storms were chosen after the fact and the model was tuned while they
were on the table. A forecast only really counts when it is written down before
the outcome exists.

This module is that commitment. Each run does two things in order.

It issues a forecast from the solar wind as it stands at this minute, for four
horizons, and appends it to a ledger with the time it becomes checkable. The entry
is written before anything is known about the outcome.

It then goes back through the ledger, finds every forecast whose valid time has
passed, fetches what the ground magnetometer actually recorded at that minute, and
writes the answer beside the forecast. Nothing is ever removed and nothing is
edited once an outcome is attached, so the record can only accumulate.

The persistence baseline is issued at the same moment from the same information,
which is the ground disturbance as it stands when the forecast is made. That number
is what the model has to beat, and putting it in the same row means the comparison
cannot drift.

The result is a running public score, computed on data that did not exist when the
model was trained, on a station that reports within a few minutes. It is the only
number in the project that nobody can tune.
"""

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from setu.config import (CADENCE_S, DBDT_THRESHOLDS_NT_PER_S, DOCS_DATA_DIR,
                         FORECAST_HORIZONS_MIN, QUANTILES)
from setu.data.magnetometer import first_reporting
from setu.data.realtime import current_conditions, fetch_live
from setu.decision.policy import PolicyOptimiser
from setu.decision.scenarios import build_scenarios
from setu.grid.network import Network
from setu.grid.voltage import VoltageModel
from setu.ml.dataset import INPUT_CADENCE_MIN, TARGET_WINDOW_MIN
from setu.ml.features import FEATURE_NAMES, build_features
from setu.physics.geoelectric import dbdt
from setu.physics.gic import GICSolver
from setu.pipeline import exceedance

log = logging.getLogger(__name__)

LEDGER_PATH = DOCS_DATA_DIR / "ledger.json"

# The ledger is fetched whole by a browser, so it cannot grow without limit. Rows
# stay in it for two days, which is long enough to plot, and are then folded into a
# daily verification record that keeps the counts every score is computed from.
#
# The roll up never expires and it is never rewritten. Once a day has been closed
# its counts are final, which is the property that makes the record worth anything:
# a bad week cannot be quietly dropped later.
LEDGER_HOURS = 48
ROLLUP_PATH = DOCS_DATA_DIR / "verification.json"

# A forecast is checked once its valid minute plus half the target window has
# passed, with a few minutes on top for the observatory to publish.
PUBLICATION_MARGIN_MIN = 8

# The alerting level the running score is kept at. This is the level where an
# Indian low latitude observatory starts to show a disturbance worth acting on.
SCORE_THRESHOLD = 0.1


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)


def ground_series(frame: pd.DataFrame) -> pd.Series:
    """The forecast target computed from a raw magnetometer frame.

    This has to be the same quantity the model was trained against or the score
    means nothing, so it is built the same way: the magnitude of the rate of change
    of the horizontal field, then the peak over a window centred on each minute.
    """
    rate = pd.Series(dbdt(frame["bx"].values, frame["by"].values, CADENCE_S),
                     index=frame.index, name="dbdt")
    return rate.rolling(f"{TARGET_WINDOW_MIN}min", center=True, min_periods=8).max()


def _peak_at(series: pd.Series, when: pd.Timestamp):
    """The target value at one minute, or None if the record does not cover it."""
    half = pd.Timedelta(minutes=TARGET_WINDOW_MIN / 2)
    window = series.loc[when - half: when + half].dropna()
    if window.empty:
        return None
    return float(window.max())


def _day_of(entry: dict) -> str:
    return entry["issued_at"][:10]


def fold(entries: list, rollup: dict) -> dict:
    """Add a set of finished rows into the daily verification record.

    Only rows whose outcome is already attached are folded, and a row is counted
    once. A day already present in the record is added to rather than replaced, so
    a run that folds the last few rows of a day does not erase the rest of it.
    """
    days = dict(rollup.get("days", {}))
    for entry in entries:
        day = _day_of(entry)
        kind = "backfilled" if entry.get("backfilled") else "live"
        block = days.setdefault(day, {"live": 0, "backfilled": 0, "horizons": {}})
        block[kind] = block.get(kind, 0) + 1

        persistence = entry.get("persistence_dbdt")
        for horizon, record in entry["horizons"].items():
            observed = record.get("observed_dbdt")
            if observed is None or persistence is None:
                continue
            counts = block["horizons"].setdefault(horizon, {
                "n": 0, "events": 0,
                "model": {"hits": 0, "false_alarms": 0, "misses": 0,
                          "correct_negatives": 0},
                "persistence": {"hits": 0, "false_alarms": 0, "misses": 0,
                                "correct_negatives": 0},
                "absolute_error_sum": 0.0,
                "brier_sum": 0.0,
            })
            happened = observed >= SCORE_THRESHOLD
            counts["n"] += 1
            counts["events"] += int(happened)
            for name, alarmed in (("model", bool(record["alarm"])),
                                  ("persistence", persistence >= SCORE_THRESHOLD)):
                if alarmed and happened:
                    counts[name]["hits"] += 1
                elif alarmed:
                    counts[name]["false_alarms"] += 1
                elif happened:
                    counts[name]["misses"] += 1
                else:
                    counts[name]["correct_negatives"] += 1
            counts["absolute_error_sum"] += abs(record["median"] - observed)
            probability = record["probability"][str(SCORE_THRESHOLD)]
            counts["brier_sum"] += (probability - float(happened)) ** 2

    return {"threshold": SCORE_THRESHOLD,
            "what_this_is": (
                "One row a day, closed and never rewritten. Each day holds the "
                "counts every score on the console is computed from, for the model "
                "and for the persistence baseline it is measured against, on "
                "forecasts that were written down before their outcomes existed."),
            "days": dict(sorted(days.items()))}


def load_rollup(path=ROLLUP_PATH) -> dict:
    if not path.exists():
        return {"threshold": SCORE_THRESHOLD, "days": {}}
    return json.loads(path.read_text())


def _skill(counts: dict) -> dict:
    """Probability of detection, false alarm ratio, and Heidke skill from counts."""
    hits = counts["hits"]
    false_alarms = counts["false_alarms"]
    misses = counts["misses"]
    correct_negatives = counts["correct_negatives"]
    total = hits + false_alarms + misses + correct_negatives
    out = dict(counts, n=total, pod=None, far=None, hss=None)
    if hits + misses:
        out["pod"] = hits / (hits + misses)
    if hits + false_alarms:
        out["far"] = false_alarms / (hits + false_alarms)
    # Heidke skill is undefined when nothing ever happened. Reporting zero there
    # would read as a failure rather than as an absence of events.
    expected = (((hits + misses) * (hits + false_alarms)
                 + (correct_negatives + misses) * (correct_negatives + false_alarms))
                / total) if total else 0.0
    if total and total != expected:
        out["hss"] = (hits + correct_negatives - expected) / (total - expected)
    return out


def load_ledger(path=LEDGER_PATH) -> list:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("entries", payload if isinstance(payload, list) else [])


def issue(model, scaler, calibrator=None, n_scenarios: int = 60) -> dict:
    """Produce one forecast from the solar wind as it stands right now.

    The entry it returns carries everything needed to score it later without
    trusting anything computed after the fact: the minute the forecast was issued,
    the minute each horizon becomes checkable, the full predicted distribution, the
    alarm decision, and the persistence forecast made from the same information.
    """
    frame = fetch_live()
    features = build_features(frame).resample(f"{INPUT_CADENCE_MIN}min").mean()
    features = features.ffill().bfill()
    if len(features) < model.window:
        raise RuntimeError(
            f"the feed returned {len(features)} five minute steps and the model "
            f"needs {model.window}")

    issued_at = pd.Timestamp(features.index[-1])
    window = features.iloc[-model.window:][FEATURE_NAMES].values
    scaled = scaler.transform(window).T[None, :, :]
    quantiles = model.predict_quantiles(scaled)
    if calibrator is not None:
        quantiles = calibrator.calibrate(quantiles)
    quantiles = quantiles[0]

    # The ground as it stands at the moment of the forecast. This is the
    # persistence forecast an operator already has for free, and it is recorded in
    # the same row so the comparison is made on identical information.
    observatory, ground = first_reporting()
    persistence = None
    ground_observed_at = None
    if ground is not None and len(ground):
        series = ground_series(ground)
        persistence = _peak_at(series, pd.Timestamp(series.index[-1]))
        ground_observed_at = str(series.index[-1])

    horizons = {}
    for index, horizon in enumerate(FORECAST_HORIZONS_MIN):
        row = [float(v) for v in quantiles[index]]
        probability = {str(t): float(exceedance(row, t))
                       for t in DBDT_THRESHOLDS_NT_PER_S}
        cut = alarm_cut(horizon)
        horizons[str(horizon)] = {
            "valid_at": str(issued_at + pd.Timedelta(minutes=horizon)),
            "quantiles": row,
            "median": row[len(row) // 2],
            "probability": probability,
            "probability_cut": cut,
            "alarm": bool(probability[str(SCORE_THRESHOLD)] >= cut),
            "observed_dbdt": None,
            "verified_at": None,
        }

    # The consequence and the recommendation are attached to the shortest horizon,
    # because that is the only one guaranteed to sit inside the propagation delay
    # during a fast storm, and it is the one an operator would act on.
    network = Network()
    solver = GICSolver(network)
    voltage = VoltageModel(network)
    scenarios = build_scenarios(quantiles[0], list(QUANTILES),
                                n_samples=n_scenarios, seed=0)
    worst = max(scenarios, key=lambda s: s.peak_dbdt)
    result = solver.solve(worst.ex, worst.ey)
    assessment = voltage.assess(result.reactive_loss_mvar)
    plan = PolicyOptimiser(network).optimise(scenarios).summary()

    return {
        "run_at": str(_now()),
        "issued_at": str(issued_at),
        "sources": frame.attrs.get("sources", []),
        "observatory": observatory,
        "ground_observed_at": ground_observed_at,
        "persistence_dbdt": persistence,
        "horizons": horizons,
        "consequence": {
            "worst_site": result.codes[int(np.argmax(result.per_phase_per_unit))],
            "peak_amp": float(result.per_phase_per_unit.max()),
            "reactive_mvar": float(result.reactive_loss_mvar.sum()),
            "worst_voltage_pu": assessment["worst_deviation_pu"],
            "load_at_risk_mw": voltage.load_at_risk_mw(result.reactive_loss_mvar),
        },
        "plan": plan,
    }


def alarm_cut(horizon_minutes: int, threshold: str = "0.1",
              default: float = 0.15) -> float:
    """The probability cut chosen on validation during training, for one horizon.

    Reading it from the training report rather than picking a number here is what
    makes the live score describe the same system the held out score describes.
    """
    from setu.config import ARTIFACT_DIR

    for name in ("training_report_l1.json", "training_report.json"):
        path = ARTIFACT_DIR / name
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        entry = (report.get("thresholds", {}).get(threshold, {})
                 .get(f"{horizon_minutes}min"))
        if entry and "probability_cut" in entry:
            return float(entry["probability_cut"])
    return default


def verify(entries: list) -> int:
    """Attach the observed outcome to every forecast whose valid time has passed.

    An entry is only ever written once. Once an outcome is attached it is left
    alone, so a later run cannot quietly improve an earlier answer.

    Returns:
        How many horizon forecasts were newly scored.
    """
    pending = []
    deadline = _now() - dt.timedelta(minutes=PUBLICATION_MARGIN_MIN
                                     + TARGET_WINDOW_MIN / 2)
    for entry in entries:
        for horizon, record in entry["horizons"].items():
            if record.get("observed_dbdt") is not None:
                continue
            valid_at = pd.Timestamp(record["valid_at"])
            if valid_at <= deadline:
                pending.append((entry, record, valid_at))
    if not pending:
        return 0

    earliest = min(v for _, _, v in pending)
    hours = int(np.ceil((_now() - earliest.to_pydatetime()).total_seconds() / 3600)) + 1
    hours = max(2, min(hours, 72))

    observatory, frame = first_reporting(hours=hours)
    if frame is None:
        log.warning("no Indian observatory is reporting, nothing can be scored")
        return 0
    series = ground_series(frame)

    scored = 0
    for entry, record, valid_at in pending:
        observed = _peak_at(series, valid_at)
        if observed is None:
            continue
        record["observed_dbdt"] = observed
        record["verified_at"] = str(_now())
        record["verified_against"] = observatory
        scored += 1
    return scored


def scoreboard(rollup: dict, entries: list = None) -> dict:
    """The running score, over every forecast the record has ever closed.

    The daily record holds the counts and the live ledger holds the rows that have
    not been folded into it yet, so the two are added together here. Every number
    is over exactly the same set of minutes for the model and for the persistence
    baseline, because a row only enters the count when both had something to say.
    """
    entries = entries or []
    pending = [e for e in entries
               if any(r.get("observed_dbdt") is not None
                      for r in e["horizons"].values())]
    combined = fold(pending, rollup)["days"]

    per_horizon, issued_live, issued_backfilled = {}, 0, 0
    for day in combined.values():
        issued_live += day.get("live", 0)
        issued_backfilled += day.get("backfilled", 0)
        for horizon, counts in day["horizons"].items():
            block = per_horizon.setdefault(horizon, {
                "n": 0, "events": 0, "absolute_error_sum": 0.0, "brier_sum": 0.0,
                "model": {"hits": 0, "false_alarms": 0, "misses": 0,
                          "correct_negatives": 0},
                "persistence": {"hits": 0, "false_alarms": 0, "misses": 0,
                                "correct_negatives": 0}})
            block["n"] += counts["n"]
            block["events"] += counts["events"]
            block["absolute_error_sum"] += counts["absolute_error_sum"]
            block["brier_sum"] += counts["brier_sum"]
            for name in ("model", "persistence"):
                for key, value in counts[name].items():
                    block[name][key] += value

    horizons = {}
    for horizon in FORECAST_HORIZONS_MIN:
        block = per_horizon.get(str(horizon))
        if block is None:
            horizons[str(horizon)] = {"n": 0, "events": 0, "model": None,
                                      "persistence": None}
            continue
        n = block["n"]
        horizons[str(horizon)] = {
            "n": n,
            "events": block["events"],
            "model": _skill(block["model"]),
            "persistence": _skill(block["persistence"]),
            "mean_absolute_error": (block["absolute_error_sum"] / n) if n else None,
            "brier": (block["brier_sum"] / n) if n else None,
        }

    days = sorted(combined)
    verified = sum(h["n"] for h in horizons.values())
    # Rows that are in the ledger and whose valid minute has not arrived yet. These
    # are the live commitments still outstanding, so they are shown rather than
    # left out, but they cannot count towards any score.
    awaiting = sum(1 for e in entries
                   if all(r.get("observed_dbdt") is None
                          for r in e["horizons"].values()))
    return {
        "threshold": SCORE_THRESHOLD,
        # These two are counted apart on purpose. A row written before its outcome
        # existed is a commitment. A row replayed through hours the feed still held
        # is only an out of sample test, and reporting one total would let the
        # weaker claim borrow the strength of the other.
        "forecasts_issued_live": issued_live,
        "forecasts_backfilled": issued_backfilled,
        "forecasts_issued": issued_live + issued_backfilled,
        "horizon_forecasts_verified": verified,
        "awaiting_outcome": awaiting,
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "days_running": len(days),
        "quiet": all(h["events"] == 0 for h in horizons.values()),
        "horizons": horizons,
    }


def run(model, scaler, calibrator=None, path=LEDGER_PATH,
        rollup_path=ROLLUP_PATH, n_scenarios: int = 60) -> dict:
    """One pass of the service: issue, verify, fold, score, and write both files."""
    entries = load_ledger(path)
    rollup = load_rollup(rollup_path)

    entry = issue(model, scaler, calibrator, n_scenarios=n_scenarios)
    # A run that repeats a minute already in the ledger replaces that row rather
    # than adding a second forecast for the same instant, which would let one
    # minute count twice. Any outcome already attached to that minute is carried
    # across, because an outcome is never thrown away once it is known.
    previous = {e["issued_at"]: e for e in entries}
    if entry["issued_at"] in previous:
        for horizon, record in previous[entry["issued_at"]]["horizons"].items():
            if record.get("observed_dbdt") is not None and horizon in entry["horizons"]:
                entry["horizons"][horizon].update(
                    observed_dbdt=record["observed_dbdt"],
                    verified_at=record.get("verified_at"),
                    verified_against=record.get("verified_against"))
    entries = [e for e in entries if e["issued_at"] != entry["issued_at"]]
    entries.append(entry)
    entries.sort(key=lambda e: e["issued_at"])

    newly_scored = verify(entries)

    # Rows older than the ledger window move into the daily record. A row is only
    # folded once every horizon it will ever have has been settled, so a forecast
    # is never counted before its longest horizon has come due.
    horizon_span = dt.timedelta(minutes=max(FORECAST_HORIZONS_MIN)
                                + TARGET_WINDOW_MIN + PUBLICATION_MARGIN_MIN)
    settled_before = pd.Timestamp(_now() - dt.timedelta(hours=LEDGER_HOURS)
                                  - horizon_span)
    ageing = [e for e in entries if pd.Timestamp(e["issued_at"]) < settled_before]
    if ageing:
        rollup = fold(ageing, rollup)
        rollup_path.write_text(json.dumps(rollup, indent=1, default=float))
        entries = [e for e in entries if pd.Timestamp(e["issued_at"]) >= settled_before]

    board = scoreboard(rollup, entries)
    path.write_text(json.dumps({
        "written_at": str(_now()),
        "what_this_is": (
            "Every row here was written before its outcome existed, except the "
            "rows marked backfilled, which were replayed through the hours the "
            "real time feed still held. The forecast, the alarm decision, and the "
            "persistence baseline are recorded together at the moment of issue, "
            "and the observed value is attached afterwards from the ground "
            "magnetometer. Rows older than two days move into verification.json, "
            "which is a daily record that is never rewritten."),
        "scoreboard": board,
        "latest": entry,
        "entries": entries,
    }, indent=1, default=float))
    return {"entry": entry, "scoreboard": board, "newly_scored": newly_scored,
            "folded": len(ageing), "path": path, "entries": len(entries)}


def backfill(model, scaler, calibrator=None, path=LEDGER_PATH,
             stride_steps: int = 3) -> dict:
    """Open the ledger with the forecasts the feed still holds enough history for.

    The real time solar wind feed carries about a day of history, which is more
    than the model needs for one forecast, so the hours already in the feed can be
    walked through as if the service had been running through them. Every forecast
    made this way uses only feed data from before its own issue minute, and it is
    checked against the ground record in the same way a live one is.

    These rows are marked, and they are counted separately from the ones issued
    live, because a forecast made from a record that already exists is not the same
    claim as one written down before the outcome did. Both are honest and only one
    of them is a commitment.

    Args:
        stride_steps: How many five minute steps to advance between forecasts.
    """
    frame = fetch_live()
    features = build_features(frame).resample(f"{INPUT_CADENCE_MIN}min").mean()
    features = features.ffill().bfill()
    if len(features) < model.window + stride_steps:
        raise RuntimeError("the feed does not hold enough history to backfill")

    observatory, ground = first_reporting(hours=48)
    if ground is None:
        raise RuntimeError("no Indian observatory is reporting, so nothing can be scored")
    series = ground_series(ground)

    existing = load_ledger(path)
    seen = {e["issued_at"] for e in existing}
    cut = {h: alarm_cut(h) for h in FORECAST_HORIZONS_MIN}

    added = 0
    for end in range(model.window, len(features) + 1, stride_steps):
        issued_at = pd.Timestamp(features.index[end - 1])
        if str(issued_at) in seen:
            continue
        window = features.iloc[end - model.window: end][FEATURE_NAMES].values
        quantiles = model.predict_quantiles(scaler.transform(window).T[None, :, :])
        if calibrator is not None:
            quantiles = calibrator.calibrate(quantiles)
        quantiles = quantiles[0]

        persistence = _peak_at(series, issued_at)
        if persistence is None:
            continue

        horizons = {}
        for index, horizon in enumerate(FORECAST_HORIZONS_MIN):
            row = [float(v) for v in quantiles[index]]
            probability = {str(t): float(exceedance(row, t))
                           for t in DBDT_THRESHOLDS_NT_PER_S}
            valid_at = issued_at + pd.Timedelta(minutes=horizon)
            observed = _peak_at(series, valid_at)
            horizons[str(horizon)] = {
                "valid_at": str(valid_at),
                "quantiles": row,
                "median": row[len(row) // 2],
                "probability": probability,
                "probability_cut": cut[horizon],
                "alarm": bool(probability[str(SCORE_THRESHOLD)] >= cut[horizon]),
                "observed_dbdt": observed,
                "verified_at": None if observed is None else str(_now()),
                "verified_against": None if observed is None else observatory,
            }

        existing.append({
            "run_at": str(_now()),
            "issued_at": str(issued_at),
            "sources": frame.attrs.get("sources", []),
            "observatory": observatory,
            "ground_observed_at": str(series.index[-1]),
            "persistence_dbdt": persistence,
            "backfilled": True,
            "horizons": horizons,
            "consequence": None,
            "plan": None,
        })
        added += 1

    existing.sort(key=lambda e: e["issued_at"])

    # The rollup is deliberately not written here. A forecast lives in exactly one
    # of the two files, and it moves from the ledger into the daily record only
    # when it is dropped from the ledger, which ``run`` does once every horizon has
    # settled. Writing it in both places counted every backfilled row twice.
    board = scoreboard(load_rollup(), existing)
    path.write_text(json.dumps({
        "written_at": str(_now()),
        "what_this_is": (
            "Every row here was written before its outcome existed, except the "
            "rows marked backfilled, which were replayed through the hours the "
            "real time feed still held. The observed value beside each forecast "
            "comes from the ground magnetometer afterwards."),
        "scoreboard": board,
        "latest": existing[-1],
        "entries": existing,
    }, indent=1, default=float))
    return {"added": added, "scoreboard": board, "path": path,
            "entries": len(existing)}
