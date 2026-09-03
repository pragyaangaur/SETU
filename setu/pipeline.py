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

from setu.config import DBDT_THRESHOLDS_NT_PER_S, FORECAST_HORIZONS_MIN, QUANTILES
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
           plan_threshold: float = 0.25, horizon_index: int = 1) -> dict:
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
        plan_threshold: Probability of passing the middle alert level above which
            the decision layer is run at all. Below it the recommendation is to do
            nothing, which is the right answer almost all of the time.
        horizon_index: Which forecast horizon the plan is built for.

    Returns:
        A dictionary of arrays and records, ready to be written out for the
        dashboard or plotted.
    """
    event = get_event(event_key)
    features, target = event_frames(event, observatory)
    x, y_actual, stamps = windowed_samples(features, target, model.window)
    if len(x) == 0:
        raise RuntimeError(f"no usable samples for {event_key} at {observatory}")

    scaled = scaler.transform(
        x.transpose(0, 2, 1).reshape(-1, x.shape[1])
    ).reshape(x.shape[0], x.shape[2], x.shape[1]).transpose(0, 2, 1)

    quantiles = np.concatenate([
        from_log_target(model.forward(scaled[i: i + 256], training=False))
        for i in range(0, len(scaled), 256)
    ])

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
        if probability[str(DBDT_THRESHOLDS_NT_PER_S[1])] >= plan_threshold:
            record["plan"] = optimiser.optimise(scenarios).summary()
        steps.append(record)

    return {
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


def lead_time_summary(replayed: dict, threshold: float = 0.3,
                      probability_cut: float = 0.4) -> dict:
    """How much warning the system would have given, and whether it was right.

    The number an operator asks for first is not a skill score. It is how many
    minutes of warning they get before the ground starts moving, so that is
    computed here directly from the replay.
    """
    steps = replayed["steps"]
    observed = np.array([s["observed_dbdt"] for s in steps])
    probability = np.array([s["probability"][str(threshold)] for s in steps])
    horizon = replayed["horizon_minutes"]

    alarmed = probability >= probability_cut
    exceeded = observed >= threshold
    if not exceeded.any():
        return {"threshold": threshold, "exceeded": False,
                "note": "the ground never passed this level during the event"}

    first_exceed = int(np.argmax(exceeded))
    first_alarm = int(np.argmax(alarmed)) if alarmed.any() else None
    if first_alarm is None or first_alarm > first_exceed:
        return {"threshold": threshold, "exceeded": True, "warned": False,
                "note": "the system did not raise this level before the ground did"}

    stride_minutes = INPUT_CADENCE_MIN * max(
        1, (len(steps) > 1) * int(round(
            (np.datetime64(steps[1]["time"]) - np.datetime64(steps[0]["time"]))
            / np.timedelta64(1, "m") / INPUT_CADENCE_MIN)))
    return {
        "threshold": threshold,
        "exceeded": True,
        "warned": True,
        "lead_minutes": int((first_exceed - first_alarm) * stride_minutes + horizon),
        "first_alarm": steps[first_alarm]["time"],
        "first_exceedance": steps[first_exceed]["time"],
        "peak_observed_dbdt": float(observed.max()),
    }
