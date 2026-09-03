"""Choose what to do, given a forecast that is a distribution and not a number.

The problem is a decision under uncertainty with an asymmetric cost. Acting when
the storm turns out to be small wastes money. Not acting when it turns out to be
large costs load, and in the worst case costs a transformer. Minimising the average
cost gets this wrong, because the average is dominated by the many small storms and
it will quietly accept a small chance of a very bad outcome.

The objective used here is the conditional value at risk. It is the average cost
over the worst tail of the scenario set, so it is driven by the cases that actually
matter. The final objective is a blend of the average and the tail, which lets an
operator dial how cautious the system is with one number.

The search is a forward greedy selection. Actions interact, because switching one
neutral out pushes its current into the neighbouring sites, so each candidate is
re-evaluated against the full physics with the already chosen actions in place
rather than being scored once at the start.
"""

from dataclasses import dataclass

import numpy as np

from setu.decision.actions import Action, default_actions
from setu.grid.network import Network
from setu.grid.voltage import DEFAULT_RESERVE_MVAR, FALLBACK_RESERVE_MVAR, VoltageModel
from setu.physics.gic import GICSolver

# Value of unserved energy, in lakh rupees per megawatt of load lost for the
# duration of one storm. The Central Electricity Regulatory Commission has used
# figures of this order in reliability studies, and the number is exposed here so
# it can be changed rather than buried.
VALUE_OF_LOST_LOAD_LAKH_PER_MW = 2.6

# Cost of losing one transformer, in lakh rupees. A large unit takes many months to
# replace and there is no spare pool in the region, so this dominates everything
# else once the thermal limit is passed.
TRANSFORMER_LOSS_LAKH = 4200.0

# Winding hot spot rise at which a transformer is taken as damaged. The IEEE
# guidance puts short term emergency limits near this level for a modern unit.
HOTSPOT_LIMIT_K = 60.0


@dataclass
class Plan:
    """The chosen actions and what they are expected to achieve."""

    actions: list
    baseline: dict
    outcome: dict
    steps: list

    @property
    def total_cost_lakh(self) -> float:
        return float(sum(a.cost_lakh for a in self.actions))

    def summary(self) -> dict:
        saved = self.baseline["cvar"] - self.outcome["cvar"]
        return {
            "actions": [{"key": a.key, "label": a.label,
                         "cost_lakh": a.cost_lakh, "lead_time_min": a.lead_time_min}
                        for a in self.actions],
            "action_cost_lakh": round(self.total_cost_lakh, 1),
            "baseline_expected_lakh": round(self.baseline["expected"], 1),
            "baseline_tail_lakh": round(self.baseline["cvar"], 1),
            "planned_expected_lakh": round(self.outcome["expected"], 1),
            "planned_tail_lakh": round(self.outcome["cvar"], 1),
            "tail_risk_avoided_lakh": round(saved, 1),
            "baseline_load_at_risk_mw": round(self.baseline["load_at_risk_p95"], 1),
            "planned_load_at_risk_mw": round(self.outcome["load_at_risk_p95"], 1),
            "benefit_to_cost": (round(saved / self.total_cost_lakh, 2)
                                if self.total_cost_lakh > 0 else None),
        }


class PolicyOptimiser:
    """Search the action set against a sampled scenario set."""

    def __init__(self, network: Network = None, actions=None,
                 tail_fraction: float = 0.10, risk_aversion: float = 0.75,
                 lead_time_min: int = 45):
        self.net = network or Network()
        self.actions = [a for a in (actions or default_actions(self.net))
                        if a.lead_time_min <= lead_time_min]
        self.tail_fraction = tail_fraction
        self.risk_aversion = risk_aversion
        self.lead_time_min = lead_time_min
        self._solvers = {}

    def _solver(self, blocked_key) -> GICSolver:
        """Cache one solver per set of blocked neutrals, since building it is the
        expensive part and the same set is scored many times."""
        if blocked_key not in self._solvers:
            self._solvers[blocked_key] = GICSolver(self.net, set(blocked_key))
        return self._solvers[blocked_key]

    def _voltage_model(self, added_reserve) -> VoltageModel:
        reserves = {s.code: DEFAULT_RESERVE_MVAR.get(s.code, FALLBACK_RESERVE_MVAR)
                    for s in self.net.substations}
        for code, extra in added_reserve.items():
            reserves[code] = reserves.get(code, FALLBACK_RESERVE_MVAR) + extra
        return VoltageModel(self.net, reserves)

    def evaluate(self, scenarios, chosen) -> dict:
        """Cost of one action set across every scenario.

        Returns the expected cost, the conditional value at risk over the worst
        tail, and the load at risk at the ninety fifth percentile, all in the units
        named in this module.
        """
        blocked = tuple(sorted({code for a in chosen for code in a.block_neutrals}))
        added = {}
        for a in chosen:
            for code, mvar in a.added_reserve.items():
                added[code] = added.get(code, 0.0) + mvar

        solver = self._solver(blocked)
        voltage = self._voltage_model(added)
        action_cost = float(sum(a.cost_lakh for a in chosen))

        costs = np.zeros(len(scenarios))
        load = np.zeros(len(scenarios))
        for i, sc in enumerate(scenarios):
            result = solver.solve(sc.ex, sc.ey)
            at_risk = voltage.load_at_risk_mw(result.reactive_loss_mvar)
            damaged = int(np.sum(result.hotspot_rise_k >= HOTSPOT_LIMIT_K))
            load[i] = at_risk
            costs[i] = (at_risk * VALUE_OF_LOST_LOAD_LAKH_PER_MW
                        + damaged * TRANSFORMER_LOSS_LAKH)

        weights = np.array([sc.weight for sc in scenarios])
        weights = weights / weights.sum()
        expected = float(np.sum(weights * costs))

        # Conditional value at risk: the mean of the worst tail of the scenarios.
        order = np.argsort(-costs)
        keep = max(1, int(round(self.tail_fraction * len(costs))))
        cvar = float(costs[order[:keep]].mean())

        objective = ((1.0 - self.risk_aversion) * expected
                     + self.risk_aversion * cvar + action_cost)
        return {
            "expected": expected + action_cost,
            "cvar": cvar + action_cost,
            "objective": objective,
            "action_cost": action_cost,
            "load_at_risk_mean": float(np.sum(weights * load)),
            "load_at_risk_p95": float(np.percentile(load, 95)),
            "blocked": list(blocked),
        }

    def optimise(self, scenarios, max_actions: int = 6,
                 min_improvement_lakh: float = 5.0) -> Plan:
        """Greedily add actions while each one still pays for itself.

        An action is only accepted if it reduces the risk objective by more than a
        small margin, so the optimiser stops on its own rather than filling the
        budget. That margin is what keeps the recommendation short enough for an
        operator to act on inside the lead time.
        """
        baseline = self.evaluate(scenarios, [])
        chosen, steps = [], []
        current = baseline

        for _ in range(max_actions):
            remaining = [a for a in self.actions if a not in chosen]
            if not remaining:
                break
            scored = [(self.evaluate(scenarios, chosen + [a]), a) for a in remaining]
            best_result, best_action = min(scored, key=lambda p: p[0]["objective"])
            gain = current["objective"] - best_result["objective"]
            if gain <= min_improvement_lakh:
                break
            chosen.append(best_action)
            steps.append({
                "action": best_action.key,
                "label": best_action.label,
                "objective_before": round(current["objective"], 1),
                "objective_after": round(best_result["objective"], 1),
                "gain_lakh": round(gain, 1),
            })
            current = best_result

        return Plan(actions=chosen, baseline=baseline, outcome=current, steps=steps)
