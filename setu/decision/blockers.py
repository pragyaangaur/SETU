"""Where to install a limited number of neutral blocking devices.

A neutral blocking device is a capacitor placed in series with the earthed neutral
of a transformer. It passes the alternating fault current the protection needs and
blocks the direct current that causes saturation. The devices work, and they are
expensive enough that no utility installs them everywhere.

That makes siting them a combinatorial problem on the network graph, and it is the
kind of problem where intuition does badly. Blocking one substation does not remove
its current from the system. It pushes that current into the neighbouring
substations, and a device placed in the wrong order can make a neighbour worse. The
solver in ``setu.physics.gic`` already couples the whole network, so the search
below simply asks it what actually happens rather than assuming anything.

The search is a forward greedy selection. Greedy selection is the standard approach
for problems of this shape, and it is used here with the honest caveat that this
objective is not proven submodular, so the result is a strong solution rather than
a proven optimal one. For the small numbers of devices that a real budget allows,
the greedy answer is checked against an exhaustive search in the tests.
"""

from itertools import combinations

import numpy as np

from setu.grid.network import Network
from setu.grid.voltage import VoltageModel
from setu.physics.gic import GICSolver


def worst_case_metric(network, blocked, scenarios, metric="reactive"):
    """Score one placement across a scenario set.

    Args:
        metric: ``reactive`` scores the total reactive power absorbed across the
            region, which is what drives voltage collapse. ``peak_current`` scores
            the largest per phase current at any single transformer, which is what
            drives thermal damage. The two give different answers, and which one to
            use is a policy choice rather than a technical one.
    """
    solver = GICSolver(network, set(blocked))
    values = []
    for sc in scenarios:
        result = solver.solve(sc.ex, sc.ey)
        if metric == "peak_current":
            values.append(float(result.per_phase_per_unit.max()))
        else:
            values.append(float(result.reactive_loss_mvar.sum()))
    values = np.asarray(values)
    # The upper tail is scored rather than the mean, because a device that helps
    # only during small storms is not worth installing.
    return float(np.percentile(values, 90))


def greedy_placement(scenarios, budget=8, network=None, metric="reactive",
                     candidates=None):
    """Place devices one at a time, each time picking the best remaining site.

    Returns:
        A list of steps. Each step names the site chosen, the score after choosing
        it, and the reduction against the unprotected network as a percentage.
    """
    network = network or Network()
    pool = list(candidates or [s.code for s in network.substations])
    baseline = worst_case_metric(network, [], scenarios, metric)

    chosen, steps = [], []
    current = baseline
    for _ in range(min(budget, len(pool))):
        best_site, best_score = None, current
        for code in pool:
            if code in chosen:
                continue
            score = worst_case_metric(network, chosen + [code], scenarios, metric)
            if score < best_score:
                best_site, best_score = code, score
        if best_site is None:
            break
        chosen.append(best_site)
        current = best_score
        steps.append({
            "site": best_site,
            "name": next(s.name for s in network.substations if s.code == best_site),
            "score": round(current, 2),
            "reduction_percent": round(100.0 * (baseline - current) / baseline, 1),
            "devices": len(chosen),
        })
    return {"baseline": round(baseline, 2), "metric": metric, "steps": steps,
            "chosen": chosen}


def exhaustive_placement(scenarios, budget, network=None, metric="reactive",
                         candidates=None):
    """Try every combination of a given size. Only usable for a small budget.

    This exists so that the greedy result can be checked rather than trusted. The
    number of combinations grows very fast, so the caller is responsible for
    keeping the budget and the candidate list small.
    """
    network = network or Network()
    pool = list(candidates or [s.code for s in network.substations])
    best, best_score = None, np.inf
    for combo in combinations(pool, budget):
        score = worst_case_metric(network, list(combo), scenarios, metric)
        if score < best_score:
            best, best_score = list(combo), score
    return {"chosen": best, "score": round(float(best_score), 2)}


def marginal_value(scenarios, network=None, metric="reactive"):
    """Value of a single device at every site, with nothing else installed.

    This is the ranking that intuition would produce, and comparing it against the
    greedy result shows how much the interaction between sites matters. If the two
    lists agree the problem was easy, and if they disagree the greedy search has
    earned its place.
    """
    network = network or Network()
    baseline = worst_case_metric(network, [], scenarios, metric)
    rows = []
    for sub in network.substations:
        score = worst_case_metric(network, [sub.code], scenarios, metric)
        rows.append({"site": sub.code, "name": sub.name,
                     "reduction_percent": round(100.0 * (baseline - score) / baseline, 2)})
    rows.sort(key=lambda r: -r["reduction_percent"])
    return rows
