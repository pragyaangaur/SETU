"""Run the whole chain end to end on one storm, minute by minute.

This is the part that makes the project a system rather than a set of modules. It
takes a real storm, walks forward through it in the order the data would have
arrived, and at every step produces what an operator would have seen: a
probabilistic forecast of the ground disturbance, the current that follows at each
transformer, the voltage consequence, and the action the decision layer recommends.

Nothing here looks ahead. The forecast at a given minute uses only solar wind that
had already arrived at the first Lagrange point by that minute.
"""

import logging

import numpy as np

from setu.config import (DBDT_THRESHOLDS_NT_PER_S, DEFAULT_TIME_BASE,
                         FORECAST_HORIZONS_MIN, QUANTILES)
from setu.data.storms import get_event
from setu.decision.policy import PolicyOptimiser
from setu.decision.scenarios import build_scenarios
from setu.grid.network import Network
from setu.grid.voltage import VoltageModel
from setu.ml.dataset import INPUT_CADENCE_MIN, event_frames, windowed_samples
from setu.ml.features import Standardiser
from setu.ml.model import GICNet, from_log_target
from setu.physics.gic import GICSolver

log = logging.getLogger(__name__)


def replay(event_key: str, model: GICNet, scaler: Standardiser,
           observatory: str = "ABG", stride: int = 3, n_scenarios: int = 60,
           plan_threshold: float = 0.25, horizon_index: int = 1,
           time_base: str = DEFAULT_TIME_BASE, calibrator=None) -> dict:
    """Walk through one storm and record what the system would have said.

    Args:
        event_key: Key from the storm catalogue.
        model: A trained forecast network.
        scaler: The standardiser fitted during training.
        observatory: Which ground station to verify against.
        stride: How many five minute steps to advance between decision points. A
            control room does not re run a plan every five minutes, and a larger
            stride keeps the replay quick.
        n_scenarios: Scenarios sampled per decision point.
        plan_threshold: Probability of passing the lowest alert level above which
            the decision layer is run at all. Below it the recommendation is to do
            nothing, which is the right answer almost all of the time.
        horizon_index: Which forecast horizon the plan is built for.
        calibrator: Optional quantile calibration map fitted during training. When
            it is given, the forecast an operator sees is the calibrated one, which
            is the whole point of having fitted it.

    Returns:
        A dictionary of arrays and records, ready to be written out for the
        dashboard or plotted.
    """
    event = get_event(event_key)
    features, target = event_frames(event, observatory, time_base)
    x, y_actual, y_now, stamps = windowed_samples(features, target, model.window)
    if len(x) == 0:
        raise RuntimeError(f"no usable samples for {event_key} at {observatory}")

    scaled = scaler.transform(
        x.transpose(0, 2, 1).reshape(-1, x.shape[1])
    ).reshape(x.shape[0], x.shape[2], x.shape[1]).transpose(0, 2, 1)

    quantiles = np.concatenate([
        from_log_target(model.forward(scaled[i: i + 256], training=False))
        for i in range(0, len(scaled), 256)
    ])
    if calibrator is not None:
        quantiles = calibrator.calibrate(quantiles)

    network = Network()
    solver = GICSolver(network)
    voltage = VoltageModel(network)
    optimiser = PolicyOptimiser(network)

    horizon = FORECAST_HORIZONS_MIN[horizon_index]
    steps = []
    for i in range(0, len(scaled), stride):
        row = quantiles[i, horizon_index]
        probability = {
            f"{t}": float(exceedance(row, t)) for t in DBDT_THRESHOLDS_NT_PER_S
        }
        scenarios = build_scenarios(row, QUANTILES, n_samples=n_scenarios,
                                    seed=int(i))
        median = build_scenarios(row, QUANTILES, n_samples=1, seed=int(i))[0]
        result = solver.solve(median.ex, median.ey)
        assessment = voltage.assess(result.reactive_loss_mvar)

        record = {
            "time": str(stamps[i]),
            "observed_dbdt": float(y_actual[i, horizon_index]),
            # What the ground was doing when the forecast was issued. This is the
            # persistence forecast, carried through so the console can show what
            # the model is being compared against.
            "current_dbdt": float(y_now[i]),
            "forecast_quantiles": [float(v) for v in row],
            "probability": probability,
            "peak_per_phase_amp": float(result.per_phase_per_unit.max()),
            "worst_site": result.codes[int(np.argmax(result.per_phase_per_unit))],
            "reactive_mvar": float(result.reactive_loss_mvar.sum()),
            "worst_voltage_pu": assessment["worst_deviation_pu"],
            "reserve_exhausted": assessment["reserve_exhausted"],
            "load_at_risk_mw": voltage.load_at_risk_mw(result.reactive_loss_mvar),
            "per_site_amp": {c: float(v) for c, v in
                             zip(result.codes, result.per_phase_per_unit)},
            "plan": None,
        }
        if probability[str(DBDT_THRESHOLDS_NT_PER_S[0])] >= plan_threshold:
            record["plan"] = optimiser.optimise(scenarios).summary()
        steps.append(record)

    return {
        "time_base": time_base,
        "time_meaning": ("every timestamp is when the spacecraft took the "
                         "measurement, and the observed value beside it is what the "
                         f"ground did {horizon} minutes later"),
        "event": {"key": event.key, "name": event.name, "note": event.note,
                  "min_sym_h": event.min_sym_h, "role": event.role},
        "observatory": observatory,
        "horizon_minutes": horizon,
        "steps": steps,
        "quantile_levels": list(QUANTILES),
    }


def exceedance(quantile_row, threshold):
    """Probability of passing a threshold, read from one row of quantiles."""
    values = np.asarray(quantile_row, dtype=float)
    levels = np.asarray(QUANTILES)
    if threshold <= values[0]:
        return 1.0
    if threshold >= values[-1]:
        span = max(values[-1] - values[-2], 1e-6)
        return float((1.0 - levels[-1]) * np.exp(-(threshold - values[-1]) / span))
    return float(1.0 - np.interp(threshold, values, levels))


def episodes(exceeded, gap_steps=24):
    """Group a boolean series into separate disturbed periods.

    A storm is not one continuous event. It has a sudden commencement, a main
    phase, and often several substorm injections hours apart. Treating the whole
    record as a single episode would score only the first onset and ignore
    everything after it, which is most of the storm.
    """
    indices = np.where(np.asarray(exceeded, dtype=bool))[0]
    if indices.size == 0:
        return []
    groups, start, previous = [], indices[0], indices[0]
    for i in indices[1:]:
        if i - previous > gap_steps:
            groups.append((int(start), int(previous)))
            start = i
        previous = i
    groups.append((int(start), int(previous)))
    return groups


def lead_time_summary(replayed: dict, threshold: float = 0.1,
                      probability_cut: float = None, gap_steps: int = 24) -> dict:
    """How much warning the system gave before each disturbed period.

    The first number an operator asks for is not a skill score. It is how many
    minutes of warning they get before the ground starts moving, so it is computed
    here directly from the replay.

    Warning is measured per episode rather than once for the whole storm. For each
    disturbed period the last alarm that was already standing before it began is
    found, and the warning is the time from that alarm to the start of the period,
    plus the forecast horizon itself. An episode with no standing alarm is recorded
    as a miss, and misses are reported alongside the warnings rather than dropped.

    The probability cut has to be the one chosen on validation and carried through
    unchanged. An earlier version of this function used a hard coded 0.4 while
    validation had chosen 0.14, so it reported that nothing was ever warned when in
    fact four disturbed minutes in five were being caught.
    """
    if probability_cut is None:
        probability_cut = replayed.get("probability_cut", 0.15)
    steps = replayed["steps"]
    if len(steps) < 2:
        return {"threshold": threshold, "note": "the replay is too short to score"}

    observed = np.array([s["observed_dbdt"] for s in steps])
    probability = np.array([s["probability"][str(threshold)] for s in steps])
    horizon = replayed["horizon_minutes"]
    minutes_per_step = int(round(
        (np.datetime64(steps[1]["time"]) - np.datetime64(steps[0]["time"]))
        / np.timedelta64(1, "m")))

    alarmed = probability >= probability_cut
    found = episodes(observed >= threshold, gap_steps)
    if not found:
        return {"threshold": threshold, "episodes": 0,
                "note": "the ground never passed this level during the event"}

    warnings, misses = [], []
    for start, end in found:
        prior = np.where(alarmed[:start])[0]
        # An alarm counts only if it was still standing when the episode began,
        # which means no quiet gap longer than one episode gap between the two.
        if prior.size and start - prior[-1] <= gap_steps:
            onset = prior[-1]
            # Walk back to the beginning of that unbroken run of alarms.
            while onset > 0 and alarmed[onset - 1]:
                onset -= 1
            warnings.append({
                "episode_start": steps[start]["time"],
                "peak_dbdt": float(observed[start:end + 1].max()),
                "lead_minutes": int((start - onset) * minutes_per_step + horizon),
            })
        else:
            misses.append({"episode_start": steps[start]["time"],
                           "peak_dbdt": float(observed[start:end + 1].max())})

    leads = [w["lead_minutes"] for w in warnings]

    # Two numbers are needed to describe this fairly, and reporting only the first
    # is misleading in one direction while reporting only the second is misleading
    # in the other.
    #
    # The episode count asks whether an alarm was already standing when a disturbed
    # period began. It is the strictest possible reading and it is what an operator
    # would want. It also charges the system for shocks that were physically
    # impossible to forecast, because a forecast issued at a given horizon can only
    # use solar wind that had already reached the spacecraft by then, and a fast
    # shock reaches the spacecraft barely half an hour before it reaches the Earth.
    #
    # The coverage number asks what fraction of all disturbed minutes carried an
    # alarm. That is what the skill scores measure, and it says how the system
    # behaves through the body of a storm rather than at its very first instant.
    disturbed = observed >= threshold
    coverage_of_disturbed = (float(alarmed[disturbed].mean()) if disturbed.any()
                             else None)

    return {
        "threshold": threshold,
        "probability_cut": probability_cut,
        "episodes": len(found),
        "warned": len(warnings),
        "missed": len(misses),
        "median_lead_minutes": int(np.median(leads)) if leads else None,
        "max_lead_minutes": int(max(leads)) if leads else None,
        "disturbed_steps": int(disturbed.sum()),
        "disturbed_steps_alarmed": coverage_of_disturbed,
        "alarm_steps": int(alarmed.sum()),
        "warnings": warnings,
        "misses": misses,
    }
