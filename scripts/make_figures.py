"""Draw the figures for the report and the presentation.

Everything here is generated from the code and the real data, so a figure can never
drift away from the result it is supposed to show. Each one is saved twice, as a
PNG for slides and as an SVG for anything that needs to scale.

Figures that need a trained model are skipped with a message when no model is
found, so the script still produces most of its output on a fresh clone.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from setu.config import ARTIFACT_DIR, QUANTILES  # noqa: E402
from setu.data.magnetometer import fetch_observatory, to_disturbance  # noqa: E402
from setu.data.omni import fetch_range, fill_gaps, to_l1_time_base  # noqa: E402
from setu.data.storms import get_event  # noqa: E402
from setu.decision.blockers import greedy_placement, marginal_value  # noqa: E402
from setu.decision.scenarios import build_scenarios  # noqa: E402
from setu.grid.network import Network  # noqa: E402
from setu.physics.earth import EARTH_MODELS  # noqa: E402
from setu.physics.geoelectric import dbdt, field_magnitude, geoelectric_field  # noqa: E402
from setu.physics.gic import GICSolver, uniform_field_case  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# One palette used everywhere, so the figures read as a set.
INK = "#1f2328"
DIM = "#6b7480"
ACCENT = "#1f6feb"
WARN = "#bf8700"
BAD = "#cf222e"
GOOD = "#1a7f37"
BAND = "#1f6feb"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "font.size": 9,
    "axes.edgecolor": DIM,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "text.color": INK,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})


def save(fig, name):
    for extension in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{extension}")
    plt.close(fig)
    print(f"  wrote {name}.png and {name}.svg")


def figure_ground_response():
    """How much electric field each ground model gives for the same disturbance."""
    periods = np.logspace(1.5, 4.2, 200)
    frequencies = 1.0 / periods

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    order = ["shillong_plateau", "arunachal_himalaya", "brahmaputra_valley",
             "bengal_tripura_basin"]
    colours = [BAD, WARN, ACCENT, GOOD]

    for name, colour in zip(order, colours):
        model = EARTH_MODELS[name]
        label = name.replace("_", " ").title()
        axes[0].loglog(periods, np.abs(model.surface_impedance(frequencies)) * 1e3,
                       color=colour, lw=1.8, label=label)
        axes[1].loglog(periods, model.apparent_resistivity(frequencies),
                       color=colour, lw=1.8)

    axes[0].set_xlabel("period, second")
    axes[0].set_ylabel("surface impedance, milliohm")
    axes[0].set_title("Response of the ground to a magnetic disturbance")
    axes[0].axvspan(100, 1800, color=DIM, alpha=0.08)
    axes[0].annotate("band that carries\nmost storm energy", xy=(420, 0.4),
                     color=DIM, fontsize=7.5, ha="center")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")

    axes[1].set_xlabel("period, second")
    axes[1].set_ylabel("apparent resistivity, ohm metre")
    axes[1].set_title("Resistivity each profile presents")
    fig.suptitle("The Shillong Plateau is the worst ground in the region",
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.82, wspace=0.26)
    save(fig, "01_ground_response")


def figure_direction_sensitivity():
    """Exposure of the network against the direction of the electric field."""
    network = Network()
    solver = GICSolver(network)
    angles = np.arange(0, 360, 5)
    peak, total = [], []
    for angle in angles:
        radians = np.radians(angle)
        result = uniform_field_case(solver, np.cos(radians), np.sin(radians))
        peak.append(np.abs(result.neutral_current).max())
        total.append(result.reactive_loss_mvar.sum())

    fig = plt.figure(figsize=(10.0, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    theta = np.radians(angles)
    ax.plot(theta, peak, color=BAD, lw=2)
    ax.fill(theta, peak, color=BAD, alpha=0.15)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Largest neutral current, ampere,\nby electric field direction",
                 pad=12, fontsize=9)
    ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(1, 2, 2)
    subs = {s.code: s for s in network.substations}
    for line in network.lines:
        a, b = subs[line.frm], subs[line.to]
        ax2.plot([a.lon, b.lon], [a.lat, b.lat], color=DIM,
                 lw=2.0 if line.kv >= 400 else 0.9, alpha=0.6, zorder=1)
    east = uniform_field_case(solver, 0.0, 1.0)
    sizes = 12 + 240 * (np.abs(east.neutral_current) / np.abs(east.neutral_current).max())
    ax2.scatter([subs[c].lon for c in east.codes], [subs[c].lat for c in east.codes],
                s=sizes, c=np.abs(east.neutral_current), cmap="YlOrRd",
                edgecolor=INK, linewidth=0.4, zorder=3)
    for code in ("BWNC", "ALPD", "BNGR", "MRNI", "BYRN"):
        s = subs[code]
        ax2.annotate(s.name, (s.lon, s.lat), fontsize=7, color=INK,
                     xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("longitude, degree east")
    ax2.set_ylabel("latitude, degree north")
    ax2.set_title("Exposure to an eastward field of one volt per kilometre",
                  fontsize=9)
    lons = [subs[c].lon for c in east.codes]
    ax2.set_xlim(min(lons) - 0.5, max(lons) + 1.0)
    fig.suptitle("The grid is stretched east to west, and so is the driver",
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.80, wspace=0.25)
    save(fig, "02_direction_and_map")


def figure_storm_chain(event_key="2024-05-10", observatory="ABG"):
    """The whole physics chain through one real storm, one panel per stage."""
    event = get_event(event_key)
    days = (event.end - event.start).days + 1
    wind = fill_gaps(to_l1_time_base(fetch_range(event.start, event.end)))
    ground = to_disturbance(fetch_observatory(observatory, event.start, days))

    rate = pd.Series(dbdt(ground["bx"].ffill().bfill().values,
                          ground["by"].ffill().bfill().values), index=ground.index)
    plateau = EARTH_MODELS["shillong_plateau"]
    basin = EARTH_MODELS["bengal_tripura_basin"]
    ex_p, ey_p = geoelectric_field(ground["bx"].ffill().bfill().values,
                                   ground["by"].ffill().bfill().values, plateau)
    ex_b, ey_b = geoelectric_field(ground["bx"].ffill().bfill().values,
                                   ground["by"].ffill().bfill().values, basin)

    network = Network()
    solver = GICSolver(network)
    field_plateau = field_magnitude(ex_p, ey_p)
    worst_index = int(np.argmax(field_plateau))
    peak_result = solver.solve(
        {m: ex_p[worst_index] if "plateau" in m or "himalaya" in m else ex_b[worst_index]
         for m in EARTH_MODELS},
        {m: ey_p[worst_index] if "plateau" in m or "himalaya" in m else ey_b[worst_index]
         for m in EARTH_MODELS})

    fig, axes = plt.subplots(4, 1, figsize=(9.5, 8.2), sharex=True)

    axes[0].plot(wind.index, wind["b_total"], color=ACCENT, lw=0.9, label="field magnitude")
    axes[0].plot(wind.index, wind["bz_gsm"], color=BAD, lw=0.9, label="northward component")
    axes[0].axhline(0, color=DIM, lw=0.5)
    axes[0].set_ylabel("nT")
    axes[0].set_title("1. Solar wind, on the clock of the spacecraft that measured it")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")

    axes[1].plot(ground.index, ground["bx"], color=INK, lw=0.7)
    axes[1].set_ylabel("nT")
    axes[1].set_title(f"2. Ground magnetic disturbance at {observatory}")

    axes[2].plot(rate.index, rate.values, color=WARN, lw=0.7)
    axes[2].set_ylabel("nT per second")
    axes[2].set_title("3. Rate of change, which is what induction responds to")

    axes[3].plot(ground.index, field_magnitude(ex_p, ey_p), color=BAD, lw=0.8,
                 label="Shillong Plateau")
    axes[3].plot(ground.index, field_magnitude(ex_b, ey_b), color=GOOD, lw=0.8,
                 label="Bengal and Tripura basin")
    axes[3].set_ylabel("volt per km")
    axes[3].set_title("4. Induced electric field, and how much the ground matters")
    axes[3].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[3].set_xlabel("time, UTC")

    peak_amp = peak_result.per_phase_per_unit.max()
    worst_site = peak_result.codes[int(np.argmax(peak_result.per_phase_per_unit))]
    fig.suptitle(
        f"{event.name}: from the Sun to a transformer neutral\n"
        f"peak modelled current {peak_amp:.2f} A per phase at {worst_site}, "
        f"minimum SYM/H {event.min_sym_h} nT",
        fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.90, hspace=0.42)
    save(fig, "03_storm_chain")


def figure_placement():
    """Greedy device placement against the ranking intuition would give."""
    scenarios = build_scenarios([1.0, 1.8, 3.2, 5.6, 10.0, 18.0],
                                [0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
                                n_samples=40, seed=3)
    greedy = greedy_placement(scenarios, budget=8)
    single = marginal_value(scenarios)[:8]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    devices = [s["devices"] for s in greedy["steps"]]
    reduction = [s["reduction_percent"] for s in greedy["steps"]]
    axes[0].plot(devices, reduction, "o-", color=ACCENT, lw=2, ms=5)
    axes[0].fill_between(devices, reduction, color=ACCENT, alpha=0.12)
    for step in greedy["steps"]:
        # Alternate the label side so the last one does not run off the axis.
        last = step["devices"] == len(greedy["steps"])
        axes[0].annotate(step["name"], (step["devices"], step["reduction_percent"]),
                         fontsize=6.8, color=DIM,
                         xytext=(-6, -12) if last else (5, -9),
                         ha="right" if last else "left",
                         textcoords="offset points")
    axes[0].set_xlim(0.6, len(greedy["steps"]) + 0.4)
    axes[0].set_xlabel("blocking devices installed")
    axes[0].set_ylabel("reduction in regional reactive absorption, percent")
    axes[0].set_title("Each device, placed by search")

    names = [r["name"] for r in single]
    values = [r["reduction_percent"] for r in single]
    top_three = [s["name"] for s in greedy["steps"][:3]]
    colours = [ACCENT if n in top_three else BAD for n in names]
    axes[1].barh(range(len(names)), values, color=colours, height=0.62)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("reduction from one device alone, percent")
    axes[1].set_title("What scoring each site on its own would say", fontsize=9)

    missing = [n for n in top_three if n not in names]
    if missing:
        axes[1].annotate(
            f"{missing[0]} is picked third by the search\nand does not appear on this "
            f"list at all,\nbecause it only matters once the two\nabove it are blocked",
            xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=7.2, color=BAD,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=BAD, lw=0.8))
    fig.suptitle("Blocking one substation pushes its current into the neighbours",
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.80, wspace=0.34)
    save(fig, "04_placement")


def figure_skill(report_path=None):
    """Held out skill scores, and how they compare between the two clocks."""
    reports = {}
    for tag in ("l1", "bowshock"):
        path = ARTIFACT_DIR / f"training_report_{tag}.json"
        if path.exists():
            reports[tag] = json.loads(path.read_text())
    if not reports:
        print("  skipping skill figure, no training reports found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    labels = {"l1": "spacecraft clock", "bowshock": "bow shock clock"}
    colours = {"l1": ACCENT, "bowshock": DIM}

    for tag, report in reports.items():
        thresholds = report.get("thresholds", {})
        if not thresholds:
            continue
        level = sorted(thresholds)[0]
        horizons = thresholds[level]
        names = sorted(horizons, key=lambda h: int(h.replace("min", "")))
        hss = [horizons[h]["hss"] for h in names]
        pod = [horizons[h]["pod"] for h in names]
        x = np.arange(len(names))
        offset = -0.18 if tag == "l1" else 0.18
        axes[0].bar(x + offset, hss, width=0.34, color=colours[tag],
                    label=labels[tag])
        axes[1].bar(x + offset, pod, width=0.34, color=colours[tag])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names)

    axes[0].set_ylabel("Heidke skill score")
    axes[0].set_title("Skill against chance, on held out storms")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("probability of detection")
    axes[1].set_title("Fraction of disturbed minutes caught")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=DIM, alpha=0.15)
    fig.suptitle("What the clock is worth", fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.80, wspace=0.28)
    save(fig, "05_skill")


def figure_reliability():
    """Are the stated probabilities honest."""
    path = ARTIFACT_DIR / "training_report_l1.json"
    if not path.exists():
        print("  skipping reliability figure, no L1 training report found")
        return
    report = json.loads(path.read_text())
    thresholds = report.get("thresholds", {})
    if not thresholds:
        print("  skipping reliability figure, the report has no threshold scores")
        return
    level = sorted(thresholds)[0]
    horizons = thresholds[level]
    name = sorted(horizons, key=lambda h: int(h.replace("min", "")))[len(horizons) // 2]
    rows = horizons[name].get("reliability", [])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].plot([0, 1], [0, 1], color=DIM, ls="--", lw=1, label="perfect")
    axes[0].plot([r["forecast"] for r in rows], [r["observed"] for r in rows],
                 "o-", color=ACCENT, lw=1.8, ms=5, label="model")
    axes[0].set_xlabel("forecast probability")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_title(f"Reliability at {level} nT per second, {name} ahead")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    cover = report.get("coverage", {})
    levels = sorted(float(k) for k in cover)
    observed = [cover[str(k)] if str(k) in cover else cover[k] for k in levels]
    axes[1].plot([0, 1], [0, 1], color=DIM, ls="--", lw=1)
    axes[1].plot(levels, observed, "o-", color=BAD, lw=1.8, ms=5)
    axes[1].set_xlabel("nominal quantile")
    axes[1].set_ylabel("fraction of observations below it")
    axes[1].set_title("Calibration of the whole distribution")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    fig.suptitle("A probability is only useful if it means what it says",
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.80, wspace=0.28)
    save(fig, "06_reliability")


def main():
    print(f"writing figures to {FIG_DIR}")
    figure_ground_response()
    figure_direction_sensitivity()
    figure_placement()
    figure_storm_chain()
    figure_skill()
    figure_reliability()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
