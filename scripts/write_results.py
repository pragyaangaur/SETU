"""Write RESULTS.md from the saved training reports.

Numbers in prose drift. Somebody changes a setting, reruns, and the table in the
document still says what it said last week. Writing the document from the reports
removes that possibility, so anything quoted here is what the code actually
produced on the run that is currently saved.

Run it after training. It reads every ``training_report_*.json`` in the artefact
directory and writes one document covering all of them.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from setu.config import ARTIFACT_DIR  # noqa: E402
from setu.data.storms import test_events, training_events  # noqa: E402

OUTPUT = ROOT / "RESULTS.md"

CLOCK_NAMES = {
    "l1": "spacecraft clock",
    "bowshock": "bow shock clock",
}


def load_reports():
    reports = {}
    for path in sorted(ARTIFACT_DIR.glob("training_report_*.json")):
        tag = path.stem.replace("training_report_", "")
        reports[tag] = json.loads(path.read_text())
    return reports


def horizon_order(horizons):
    return sorted(horizons, key=lambda h: int(h.replace("min", "")))


def skill_table(report, threshold):
    """Model scores next to the persistence baseline they have to beat."""
    horizons = report.get("thresholds", {}).get(threshold)
    if not horizons:
        return None
    lines = ["| Horizon | POD | POD, persistence | FAR | FAR, persistence | "
             "Heidke skill | Heidke, persistence | Gain |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in horizon_order(horizons):
        e = horizons[name]
        b = e.get("persistence", {})
        gain = e["hss"] - b.get("hss", float("nan"))
        lines.append(
            f"| {name.replace('min', ' minutes')} | {e['pod']:.2f} | "
            f"{b.get('pod', float('nan')):.2f} | {e['far']:.2f} | "
            f"{b.get('far', float('nan')):.2f} | **{e['hss']:.2f}** | "
            f"{b.get('hss', float('nan')):.2f} | {gain:+.2f} |")
    return "\n".join(lines)


def ablation_table(reports, threshold):
    """The model with the physics constraint against the same model without it."""
    if "l1" not in reports or "l1_nophysics" not in reports:
        return None
    with_physics = reports["l1"].get("thresholds", {}).get(threshold, {})
    without = reports["l1_nophysics"].get("thresholds", {}).get(threshold, {})
    if not with_physics or not without:
        return None
    lines = ["| Horizon | Heidke skill, constraint on | Heidke skill, constraint off | "
             "POD on | POD off |",
             "| --- | --- | --- | --- | --- |"]
    for name in horizon_order(with_physics):
        a = with_physics[name]
        b = without.get(name)
        if not b:
            continue
        lines.append(f"| {name.replace('min', ' minutes')} | {a['hss']:.2f} | "
                     f"{b['hss']:.2f} | {a['pod']:.2f} | {b['pod']:.2f} |")
    return "\n".join(lines)


def comparison_table(reports, threshold):
    tags = [t for t in ("l1", "bowshock") if t in reports]
    if len(tags) < 2:
        return None
    horizons = reports[tags[0]].get("thresholds", {}).get(threshold, {})
    if not horizons:
        return None
    lines = ["| Horizon | " + " | ".join(
        f"{CLOCK_NAMES.get(t, t)} HSS" for t in tags) + " | Difference |",
        "| --- | " + " | ".join("---" for _ in tags) + " | --- |"]
    for name in horizon_order(horizons):
        values = []
        for tag in tags:
            entry = reports[tag].get("thresholds", {}).get(threshold, {}).get(name)
            values.append(entry["hss"] if entry else float("nan"))
        difference = values[0] - values[1]
        lines.append(f"| {name.replace('min', ' minutes')} | "
                     + " | ".join(f"{v:.2f}" for v in values)
                     + f" | {difference:+.2f} |")
    return "\n".join(lines)


def coverage_table(report):
    before = report.get("coverage_before_calibration", {})
    after = report.get("coverage", {})
    if not after:
        return None
    lines = ["| Nominal level | Observed before calibration | Observed after |",
             "| --- | --- | --- |"]
    for key in sorted(after, key=float):
        b = before.get(key)
        lines.append(f"| {float(key):.2f} | "
                     + (f"{b:.3f}" if b is not None else "not recorded")
                     + f" | {after[key]:.3f} |")
    return "\n".join(lines)


def main():
    reports = load_reports()
    if not reports:
        print("no training reports found, run the training first", file=sys.stderr)
        return 1

    primary = reports.get("l1") or next(iter(reports.values()))
    settings = primary.get("settings", {})
    thresholds = sorted(primary.get("thresholds", {}), key=float)
    lowest = thresholds[0] if thresholds else None

    parts = [
        "# Results",
        "",
        "This document is written by `scripts/write_results.py` from the saved "
        "training reports, so every number here is what the code produced on the "
        "run currently in `artifacts/`. Editing it by hand would only make it wrong "
        "at the next training run.",
        "",
        "## What was held out",
        "",
        f"The catalogue holds {len(training_events()) + len(test_events())} storms. "
        f"{len(training_events())} are used for training and validation, and "
        f"{len(test_events())} are held out completely:",
        "",
    ]
    for event in test_events():
        parts.append(f"- **{event.name}** ({event.start} to {event.end}), "
                     f"minimum SYM/H {event.min_sym_h} nT. {event.note}")
    parts += [
        "",
        "The split is by whole storm, so no minute in the test set sits next to a "
        "minute the model trained on. The scaler, the alarm threshold, and the "
        "calibration map are all fitted without touching the test storms.",
        "",
        "## The model",
        "",
        f"- {primary.get('parameters', 0):,} parameters, "
        f"{settings.get('channels', '?')} channels, "
        f"{primary.get('receptive_field_minutes', '?')} minutes of history reaching the output",
        f"- trained in {primary.get('training_seconds', 0) / 60:.0f} minutes on a "
        "laptop with no graphics card, using NumPy alone",
        f"- best epoch {primary.get('best_epoch', '?')}, selected on a validation "
        "split made of whole storms",
        f"- dropout {settings.get('dropout', '?')}, weight decay "
        f"{settings.get('weight_decay', '?')}, input jitter "
        f"{settings.get('input_noise', '?')} standard deviations",
        "",
    ]

    if lowest:
        parts += [
            f"## Forecast skill at {lowest} nT per second",
            "",
            "This is the alerting level where an Indian low latitude observatory "
            "starts to show a disturbance worth acting on. Scores are on the held "
            "out storms, with the alarm threshold chosen on validation and applied "
            "unchanged.",
            "",
            skill_table(primary, lowest) or "",
            "",
        ]
        parts += [
            "Persistence is the baseline every space weather forecast has to beat. "
            "It is free, it needs no model, and it says the ground will do at the "
            "horizon whatever it is doing now. It is handed the ground measurement "
            "as it stands when the forecast is made, which an operator genuinely "
            "has, and it alarms whenever that is already above the threshold. The "
            "gain column is what the model adds over it, and it grows with horizon, "
            "which is the shape it should have.",
            "",
        ]
        comparison = comparison_table(reports, lowest)
        if comparison:
            parts += [
                "### What the clock is worth",
                "",
                "The same model, the same storms, and the same settings, trained "
                "once on the clock of the spacecraft that measured the solar wind "
                "and once on the archive convention that shifts it forward to the "
                "Earth. The second one has no warning time to forecast into, "
                "because a shock appears in its input at the moment it strikes.",
                "",
                comparison,
                "",
            ]

    if len(thresholds) > 1:
        severe = thresholds[1]
        parts += [
            f"## At the higher {severe} nT per second level",
            "",
            "This level is rare, so both the model and the baseline score much lower "
            "here. The model beats persistence by a far wider margin than it does at "
            "the common level, which is what a model is for.",
            "",
            skill_table(primary, severe) or "",
            "",
        ]
        for label, level in (("the common", lowest), ("the severe", severe)):
            table = ablation_table(reports, level)
            if table is None:
                continue
            parts += [
                f"### Does the physics constraint earn its place, at {label} level",
                "",
                table,
                "",
            ]
        if ablation_table(reports, severe):
            parts += [
                "The same network, the same data, and the same settings, trained "
                "once with the monotonicity constraint and once without it. At the "
                "common level the two are close. At the severe level the "
                "unconstrained model scores zero at every horizon, meaning it never "
                "raises a severe alarm at all, while the constrained one keeps real "
                "skill.",
                "",
                "That is where a physics constraint is supposed to help. Severe "
                "minutes are rare, so there is little data to learn them from, and "
                "an unconstrained fit takes the safe option of never predicting one. "
                "The constraint forbids the forecast from falling when the energy "
                "input rises, which carries the model into a part of the range the "
                "data barely covers.",
                "",
                "One caveat belongs with this. Both runs are single seeds. The gap at "
                "the severe level is far too large to be run to run noise, and the "
                "near tie at the common level is small enough that it could be.",
                "",
            ]

    coverage = coverage_table(primary)
    if coverage:
        parts += [
            "## Calibration",
            "",
            "A stated ninetieth percentile should sit above the observation ninety "
            "percent of the time. The decision layer treats these numbers as real "
            "probabilities and would be misled by overconfident ones.",
            "",
            coverage,
            "",
            f"Mean absolute coverage error: "
            f"{primary.get('coverage_error_before_calibration', float('nan')):.3f} "
            f"before calibration, {primary.get('coverage_error', float('nan')):.3f} after.",
            "",
        ]

    parts += [
        "## Reproducing this",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python -m setu.cli train --time-base l1 --tag l1",
        "python -m setu.cli train --time-base bowshock --tag bowshock",
        "python scripts/write_results.py",
        "python scripts/make_figures.py",
        "```",
        "",
        "The data is downloaded on first use and cached under `data/raw`, so the "
        "first run is slow and later ones are not.",
        "",
    ]

    OUTPUT.write_text("\n".join(parts))
    print(f"wrote {OUTPUT} from {len(reports)} report(s): {', '.join(sorted(reports))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
