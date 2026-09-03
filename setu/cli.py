"""Command line entry point for the whole system.

Four commands cover everything the project does.

    python -m setu.cli train
    python -m setu.cli replay --event 2024-05-10
    python -m setu.cli placement --budget 8
    python -m setu.cli benchmark
"""

import argparse
import json
import logging

import numpy as np

from setu.config import ARTIFACT_DIR, DOCS_DATA_DIR
from setu.decision.blockers import greedy_placement, marginal_value
from setu.decision.scenarios import build_scenarios
from setu.grid.network import Network
from setu.ml.features import Standardiser
from setu.ml.model import GICNet
from setu.physics.earth import (BENGAL_TRIPURA_BASIN, BRAHMAPUTRA_VALLEY,
                                EARTH_MODELS, SHILLONG_PLATEAU, contrast_amplification)
from setu.physics.gic import GICSolver, uniform_field_case
from setu.pipeline import lead_time_summary, replay

log = logging.getLogger(__name__)

MODEL_PATH = ARTIFACT_DIR / "gicnet.npz"
SCALER_PATH = ARTIFACT_DIR / "scaler.npz"

# The extreme scenario set. These quantiles correspond to a ground disturbance well
# beyond anything in the modern Indian record, at the level historical accounts put
# the 1859 event. They are used for planning studies and they are labelled as
# hypothetical everywhere they appear.
EXTREME_QUANTILES = [1.0, 1.8, 3.2, 5.6, 10.0, 18.0]
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.98]


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit("no trained model found, run 'python -m setu.cli train' first")
    data = np.load(SCALER_PATH, allow_pickle=False)
    return GICNet.load(MODEL_PATH), Standardiser.from_state(data)


def cmd_train(args):
    from setu.ml.train import main as train_main
    report = train_main(epochs=args.epochs, channels=args.channels,
                        window=args.window, physics_weight=args.physics_weight)
    print(json.dumps({k: v for k, v in report.items() if k != "history"},
                     indent=2, default=float))


def cmd_replay(args):
    model, scaler = load_model()
    result = replay(args.event, model, scaler, observatory=args.observatory,
                    stride=args.stride)
    result["lead_time"] = {
        str(t): lead_time_summary(result, t) for t in (0.1, 0.3)
    }
    out = DOCS_DATA_DIR / f"replay_{args.event}.json"
    out.write_text(json.dumps(result, indent=1, default=float))
    peak = max(s["observed_dbdt"] for s in result["steps"])
    print(f"replayed {result['event']['name']} at {args.observatory}")
    print(f"  {len(result['steps'])} decision points, peak observed {peak:.3f} nT/s")
    for level, summary in result["lead_time"].items():
        print(f"  lead time at {level} nT/s: {summary}")
    print(f"  written to {out}")


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
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--physics-weight", type=float, default=0.5)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("replay", help="walk through one storm end to end")
    p.add_argument("--event", default="2024-05-10")
    p.add_argument("--observatory", default="ABG")
    p.add_argument("--stride", type=int, default=3)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("placement", help="site neutral blocking devices")
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--scenarios", type=int, default=40)
    p.add_argument("--metric", default="reactive",
                   choices=("reactive", "peak_current"))
    p.set_defaults(func=cmd_placement)

    p = sub.add_parser("benchmark", help="print the standard checks and export the network")
    p.set_defaults(func=cmd_benchmark)
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
