"""Command line entry point for the whole system.

    python -m setu.cli train
    python -m setu.cli replay --event 2024-05-10
    python -m setu.cli placement --budget 8
    python -m setu.cli live
    python -m setu.cli nowcast
    python -m setu.cli benchmark

The last two are the difference between a study and a service. ``live`` prints what
the system says at this minute. ``nowcast`` writes that down before the outcome
exists and comes back later to mark it right or wrong, which is what the unattended
job runs every quarter of an hour.
"""

import argparse
import json
import logging

import numpy as np

from setu.config import ARTIFACT_DIR, DOCS_DATA_DIR, FORECAST_HORIZONS_MIN
from setu.decision.blockers import greedy_placement, marginal_value
from setu.decision.scenarios import build_scenarios
from setu.decision.policy import PolicyOptimiser
from setu.grid.network import Network
from setu.grid.voltage import VoltageModel
from setu.ml.calibration import QuantileCalibrator
from setu.ml.features import FEATURE_NAMES, Standardiser
from setu.ml.model import GICNet
from setu.physics.earth import (BENGAL_TRIPURA_BASIN, BRAHMAPUTRA_VALLEY,
                                EARTH_MODELS, SHILLONG_PLATEAU, contrast_amplification)
from setu.physics.gic import GICSolver, uniform_field_case
from setu.pipeline import lead_time_summary, replay

log = logging.getLogger(__name__)

TIME_BASE = "l1"

# The model trained on the spacecraft clock is the one the system runs on. The
# untagged names are the fallback, so an older run still loads.
MODEL_CANDIDATES = (ARTIFACT_DIR / f"gicnet_{TIME_BASE}.npz", ARTIFACT_DIR / "gicnet.npz")
SCALER_CANDIDATES = (ARTIFACT_DIR / f"scaler_{TIME_BASE}.npz", ARTIFACT_DIR / "scaler.npz")
CALIBRATOR_CANDIDATES = (ARTIFACT_DIR / f"calibrator_{TIME_BASE}.npz",
                         ARTIFACT_DIR / "calibrator.npz")

# The extreme scenario set. These quantiles correspond to a ground disturbance well
# beyond anything in the modern Indian record, at the level historical accounts put
# the 1859 event. They are used for planning studies and they are labelled as
# hypothetical everywhere they appear.
EXTREME_QUANTILES = [1.0, 1.8, 3.2, 5.6, 10.0, 18.0]
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.98]


def load_model():
    """Load the trained network and the scaler that goes with it.

    The two have to come from the same run, because the scaler holds the column
    order and the statistics the network was fitted against. Mixing a model from
    one run with a scaler from another produces plausible looking nonsense, so the
    pair is chosen together and the feature count is checked.
    """
    for model_path, scaler_path in zip(MODEL_CANDIDATES, SCALER_CANDIDATES):
        if model_path.exists() and scaler_path.exists():
            model = GICNet.load(model_path)
            scaler = Standardiser.from_state(np.load(scaler_path, allow_pickle=False))
            if model.n_features != len(scaler.feature_names):
                raise SystemExit(
                    f"{model_path.name} expects {model.n_features} features and "
                    f"{scaler_path.name} has {len(scaler.feature_names)}. These are "
                    "from different runs, so retrain with 'python -m setu.cli train'.")
            if list(scaler.feature_names) != FEATURE_NAMES:
                raise SystemExit(
                    f"{scaler_path.name} was fitted on a different feature set than "
                    "the code now builds, so retrain with 'python -m setu.cli train'.")
            calibrator = None
            for path in CALIBRATOR_CANDIDATES:
                if path.exists():
                    calibrator = QuantileCalibrator.from_state(
                        np.load(path, allow_pickle=False))
                    break
            return model, scaler, calibrator
    raise SystemExit("no trained model found, run 'python -m setu.cli train' first")


def cmd_train(args):
    from setu.ml.train import main as train_main
    report = train_main(tag=args.tag, epochs=args.epochs, channels=args.channels,
                        window=args.window, batch_size=args.batch_size,
                        physics_weight=args.physics_weight,
                        physics_every=args.physics_every,
                        tail_weight=args.tail_weight, time_base=args.time_base,
                        dropout=args.dropout, weight_decay=args.weight_decay,
                        input_noise=args.input_noise)
    summary = {k: v for k, v in report.items()
               if k not in ("history", "thresholds")}
    print(json.dumps(summary, indent=2, default=float))
    for threshold, horizons in report["thresholds"].items():
        for horizon, scores in horizons.items():
            print(f"  {threshold} nT/s at {horizon:>6s}: POD {scores['pod']:.3f}  "
                  f"FAR {scores['far']:.3f}  HSS {scores['hss']:.3f}  "
                  f"PSS {scores['pss']:.3f}")


def probability_cut_for(horizon_minutes, threshold="0.1", default=0.15):
    """The alarm cut chosen on validation during training, for one horizon.

    Carrying this through rather than picking a fresh number is what makes the
    warning statistics describe the same system the skill scores describe.
    """
    for tag in (TIME_BASE, None):
        path = ARTIFACT_DIR / (f"training_report_{tag}.json" if tag
                               else "training_report.json")
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        entry = report.get("thresholds", {}).get(threshold, {}).get(f"{horizon_minutes}min")
        if entry and "probability_cut" in entry:
            return float(entry["probability_cut"])
    return default


def cmd_replay(args):
    model, scaler, calibrator = load_model()
    result = replay(args.event, model, scaler, observatory=args.observatory,
                    stride=args.stride, time_base=TIME_BASE, calibrator=calibrator,
                    horizon_index=args.horizon_index)
    # The alarm threshold has to be the one training chose on validation, not a
    # number picked here, or the warning statistics describe a different system
    # from the one that was scored.
    result["probability_cut"] = probability_cut_for(result["horizon_minutes"])
    result["lead_time"] = {
        str(t): lead_time_summary(result, t) for t in (0.1, 0.3)
    }
    result["planning_study"] = extreme_planning_study()
    out = DOCS_DATA_DIR / f"replay_{args.event}.json"
    out.write_text(json.dumps(result, indent=1, default=float))
    peak = max(s["observed_dbdt"] for s in result["steps"])
    print(f"replayed {result['event']['name']} at {args.observatory}")
    print(f"  {len(result['steps'])} decision points, peak observed {peak:.3f} nT/s")
    for level, summary in result["lead_time"].items():
        warned = summary.get("warned")
        if warned is None:
            print(f"  {level} nT/s: {summary.get('note', 'nothing to report')}")
        else:
            share = summary.get("disturbed_steps_alarmed")
            share_text = ("not applicable" if share is None
                          else f"{share * 100:.0f} percent")
            print(f"  {level} nT/s: {share_text} of disturbed steps carried an "
                  f"alarm; warned before {warned} of {summary['episodes']} "
                  f"disturbed periods began")
    print(f"  written to {out}")


def extreme_planning_study():
    """Run the decision layer against the hypothetical extreme scenario set.

    Observed Indian storms do not push this network anywhere near its limits, which
    is the honest finding and is reported as such. A planning study still needs to
    show what the decision layer does when the storm is large enough to matter, so
    the extreme set is run once and exported alongside the replay, clearly labelled
    as hypothetical.
    """
    scenarios = build_scenarios(EXTREME_QUANTILES, QUANTILE_LEVELS,
                                n_samples=80, seed=7)
    plan = PolicyOptimiser().optimise(scenarios)
    out = plan.summary()
    out["scenario_note"] = ("hypothetical extreme event, well beyond anything in the "
                            "modern Indian record, used for planning only")
    return out


def cmd_placement(args):
    scenarios = build_scenarios(EXTREME_QUANTILES, QUANTILE_LEVELS,
                                n_samples=args.scenarios, seed=3)
    greedy = greedy_placement(scenarios, budget=args.budget, metric=args.metric)
    single = marginal_value(scenarios, metric=args.metric)
    payload = {"greedy": greedy, "single_site_ranking": single,
               "scenario_note": "hypothetical extreme event, not an observed storm"}
    out = DOCS_DATA_DIR / "placement.json"
    out.write_text(json.dumps(payload, indent=1, default=float))

    print(f"unprotected baseline: {greedy['baseline']} ({args.metric})")
    for step in greedy["steps"]:
        print(f"  {step['devices']:2d} devices  {step['name']:22s} "
              f"{step['reduction_percent']:5.1f}% reduction")
    print("\nranking if each site were scored on its own:")
    for row in single[:5]:
        print(f"  {row['name']:22s} {row['reduction_percent']:5.2f}%")
    print(f"\nwritten to {out}")


def cmd_live(args):
    """Run the whole chain on the solar wind as it is right now.

    This is the demonstration that the system is deployment technology rather than
    a study. It reads the live feed, builds the same features the model was trained
    on, produces a probabilistic forecast, pushes it through the induction and the
    network physics, and prints what an operator would be looking at. Nothing in
    the path is different from the historical replay except where the data came
    from.
    """
    import pandas as pd

    from setu.data.realtime import compare_delay_estimate, current_conditions, fetch_live
    from setu.ml.dataset import INPUT_CADENCE_MIN
    from setu.ml.features import FEATURE_NAMES, build_features
    from setu.pipeline import exceedance

    conditions = current_conditions()
    print("solar wind now")
    for key, value in conditions.items():
        print(f"  {key:24s} {value}")

    check = compare_delay_estimate()
    print("\ndelay estimate checked against the NOAA propagated product")
    print(f"  ours {check['our_delay_min']} min, theirs {check['noaa_delay_min']} min, "
          f"median difference {check['median_difference_min']} min")

    model, scaler, calibrator = load_model()
    frame = fetch_live()
    features = build_features(frame).resample(f"{INPUT_CADENCE_MIN}min").mean()
    features = features.ffill().bfill()
    if len(features) < model.window:
        raise SystemExit(
            f"the feed returned {len(features)} steps and the model needs "
            f"{model.window}, try again when more history is available")

    window = features.iloc[-model.window:][FEATURE_NAMES].values
    scaled = scaler.transform(window).T[None, :, :]
    quantiles = model.predict_quantiles(scaled)
    if calibrator is not None:
        quantiles = calibrator.calibrate(quantiles)
    quantiles = quantiles[0]

    print("\nforecast of the ground rate of change at an Indian low latitude station")
    for h_index, horizon in enumerate(FORECAST_HORIZONS_MIN):
        row = quantiles[h_index]
        median = row[len(row) // 2]
        probabilities = " ".join(
            f"P(>{t})={exceedance(row, t):.2f}" for t in (0.1, 0.3, 1.0))
        print(f"  {horizon:3d} min ahead: median {median:.3f} nT/s   {probabilities}")

    network = Network()
    solver = GICSolver(network)
    voltage = VoltageModel(network)
    scenarios = build_scenarios(quantiles[1], list(model.quantiles),
                                n_samples=args.scenarios, seed=0)
    worst = max(scenarios, key=lambda s: s.peak_dbdt)
    result = solver.solve(worst.ex, worst.ey)
    assessment = voltage.assess(result.reactive_loss_mvar)
    worst_site = result.codes[int(np.argmax(result.per_phase_per_unit))]

    print("\nnetwork consequence in the worst of "
          f"{args.scenarios} sampled scenarios")
    print(f"  peak current            {result.per_phase_per_unit.max():.3f} A per phase at {worst_site}")
    print(f"  regional reactive load  {result.reactive_loss_mvar.sum():.1f} MVAr")
    print(f"  worst voltage deviation {assessment['worst_deviation_pu'] * 100:.2f} percent")
    print(f"  load exposed            {voltage.load_at_risk_mw(result.reactive_loss_mvar):.1f} MW")

    plan = PolicyOptimiser(network).optimise(scenarios).summary()
    print("\nrecommended action")
    if plan["actions"]:
        for action in plan["actions"]:
            print(f"  {action['label']} ({action['cost_lakh']} lakh, "
                  f"{action['lead_time_min']} min)")
    else:
        print("  none. The forecast risk does not justify the cost of acting.")

    payload = {"generated_at": str(pd.Timestamp.now("UTC")),
               "conditions": conditions, "delay_check": check,
               "forecast": {str(h): [float(v) for v in quantiles[i]]
                            for i, h in enumerate(FORECAST_HORIZONS_MIN)},
               "quantile_levels": list(model.quantiles),
               "consequence": {
                   "peak_amp": float(result.per_phase_per_unit.max()),
                   "worst_site": worst_site,
                   "reactive_mvar": float(result.reactive_loss_mvar.sum()),
                   "load_at_risk_mw": voltage.load_at_risk_mw(result.reactive_loss_mvar),
               },
               "plan": plan}
    out = DOCS_DATA_DIR / "live.json"
    out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwritten to {out}")


def cmd_nowcast(args):
    """Issue one forecast into the ledger and score every forecast that has come due.

    This is the command the unattended service runs. Where ``live`` prints what the
    system says at this minute and throws it away, this one writes it down before
    the outcome exists and comes back later to mark it right or wrong. The score it
    keeps is the only one in the project computed on data that did not exist when
    the model was trained.
    """
    from setu import nowcast

    model, scaler, calibrator = load_model()
    outcome = nowcast.run(model, scaler, calibrator, n_scenarios=args.scenarios)
    entry = outcome["entry"]
    board = outcome["scoreboard"]

    print(f"forecast issued at {entry['issued_at']} from {', '.join(entry['sources'])}")
    ground = entry["persistence_dbdt"]
    if ground is None:
        print("  no Indian observatory is reporting, so no baseline this run")
    else:
        print(f"  ground now at {entry['observatory']}: {ground:.3f} nT/s")
    for horizon in FORECAST_HORIZONS_MIN:
        row = entry["horizons"][str(horizon)]
        state = "ALARM" if row["alarm"] else "quiet"
        print(f"  {horizon:3d} min: median {row['median']:.3f} nT/s   "
              f"P(>0.1)={row['probability']['0.1']:.2f}   {state}")

    action = entry["plan"]["actions"]
    print(f"  recommendation: {action[0]['label'] if action else 'no action needed'}")

    print(f"\nrecord: {board['forecasts_issued']} forecasts over "
          f"{board['days_running']} days since {board['first_day']}, "
          f"{board['horizon_forecasts_verified']} of them scored against the "
          f"ground ({outcome['newly_scored']} new this run)")
    print(f"  {board['forecasts_issued_live']} were written before the outcome "
          f"existed, {board['forecasts_backfilled']} were replayed through the feed, "
          f"and {board['awaiting_outcome']} are still waiting on theirs")
    if board["quiet"]:
        print("  the ground has not passed 0.1 nT/s since the ledger opened, so "
              "there is nothing to score skill on yet")
    else:
        for horizon in FORECAST_HORIZONS_MIN:
            block = board["horizons"][str(horizon)]
            model_hss = block["model"]["hss"]
            base_hss = block["persistence"]["hss"]
            if model_hss is None:
                continue
            base = "n/a" if base_hss is None else f"{base_hss:.2f}"
            print(f"  {horizon:3d} min: HSS {model_hss:.2f} against "
                  f"persistence {base} over {block['model']['n']} scored minutes")
    print(f"\nwritten to {outcome['path']}")


def cmd_benchmark(args):
    """Report the standard benchmark cases, which is what a reviewer checks first."""
    network = Network()
    solver = GICSolver(network)
    print(network.summary())

    print("\nconductivity contrast at a five minute period")
    for other, label in ((BRAHMAPUTRA_VALLEY, "Brahmaputra valley"),
                         (BENGAL_TRIPURA_BASIN, "Bengal and Tripura basin")):
        ratio = contrast_amplification(SHILLONG_PLATEAU, other)
        print(f"  Shillong Plateau over {label}: {ratio:.2f} times")

    print("\none volt per kilometre benchmark, worst five substations")
    for direction, ex, ey in (("northward", 1.0, 0.0), ("eastward", 0.0, 1.0)):
        result = uniform_field_case(solver, ex, ey)
        order = np.argsort(-np.abs(result.neutral_current))[:5]
        sites = ", ".join(f"{result.codes[i]} {result.neutral_current[i]:.0f} A"
                          for i in order)
        print(f"  {direction} field: {sites}")

    print("\nearth current sum (Kirchhoff check): "
          f"{uniform_field_case(solver, 0.7, 1.3).neutral_current.sum():.2e} A")

    payload = {
        "network": network.summary(),
        "contrast": {name: contrast_amplification(SHILLONG_PLATEAU, model)
                     for name, model in EARTH_MODELS.items()},
        "benchmark_1v_per_km": {
            direction: {code: float(v) for code, v in
                        zip(solver.codes,
                            uniform_field_case(solver, ex, ey).neutral_current)}
            for direction, ex, ey in (("north", 1.0, 0.0), ("east", 0.0, 1.0))
        },
        "substations": [
            {"code": s.code, "name": s.name, "lat": s.lat, "lon": s.lon,
             "kv": s.kv, "state": s.state, "earth_model": s.earth_model,
             "transformer_type": s.transformer_type, "criticality": s.criticality}
            for s in network.substations
        ],
        "lines": [{"from": l.frm, "to": l.to, "kv": l.kv,
                   "length_km": round(network.line_length_km(l), 1), "note": l.note}
                  for l in network.lines],
    }
    out = DOCS_DATA_DIR / "network.json"
    out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwritten to {out}")


def build_parser():
    parser = argparse.ArgumentParser(prog="setu", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="fit the forecast network on real storm data")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--channels", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--input-noise", type=float, default=0.05)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--physics-weight", type=float, default=0.5)
    p.add_argument("--physics-every", type=int, default=4)
    p.add_argument("--tail-weight", type=float, default=3.0)
    p.add_argument("--time-base", default="l1", choices=("l1", "bowshock"),
                   help="l1 puts the solar wind on the spacecraft clock, which is "
                        "the only setting that leaves any warning time")
    p.add_argument("--tag", default=None,
                   help="suffix for the saved files, so two runs can be compared")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("replay", help="walk through one storm end to end")
    p.add_argument("--event", default="2024-05-10")
    p.add_argument("--observatory", default="ABG")
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--horizon-index", type=int, default=0,
                   help="which forecast horizon to drive the replay with, "
                        "defaulting to the shortest because it is the only one "
                        "inside the propagation delay during a fast storm")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("placement", help="site neutral blocking devices")
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--scenarios", type=int, default=40)
    p.add_argument("--metric", default="reactive",
                   choices=("reactive", "peak_current"))
    p.set_defaults(func=cmd_placement)

    p = sub.add_parser("live", help="run the whole chain on the solar wind right now")
    p.add_argument("--scenarios", type=int, default=60)
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("nowcast", help="issue a forecast into the ledger and score "
                                       "the ones that have come due")
    p.add_argument("--scenarios", type=int, default=60)
    p.set_defaults(func=cmd_nowcast)

    p = sub.add_parser("benchmark", help="print the standard checks and export the network")
    p.set_defaults(func=cmd_benchmark)
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
