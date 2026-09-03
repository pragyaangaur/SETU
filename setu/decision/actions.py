"""The actions an operator can actually take before a storm arrives.

Every action in this file is something a North Eastern Regional control room could
order with the equipment it already has, inside the lead time the forecast
provides. Nothing here needs new hardware, and nothing here needs a change of
operating code. That constraint is deliberate, because an optimiser that
recommends an impossible action is worse than no optimiser at all.

Costs are in lakh rupees for the duration of one storm event, and they are order of
magnitude figures meant for ranking one action against another rather than for
settlement.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    """One thing an operator can do, with what it costs and what it changes.

    Attributes:
        key: Identifier used in reports and on the dashboard.
        label: Short description for a human reader.
        cost_lakh: Direct cost of taking the action for one event.
        lead_time_min: Minutes needed to carry the action out. An action that takes
            longer than the forecast horizon cannot be used.
        block_neutrals: Substations whose neutral path is opened by this action.
        added_reserve: Extra reactive reserve in megavar, by substation.
        reversible: Whether the action can be undone quickly if the storm misses.
    """

    key: str
    label: str
    cost_lakh: float
    lead_time_min: int
    block_neutrals: tuple = ()
    added_reserve: dict = field(default_factory=dict)
    reversible: bool = True
    note: str = ""


def default_actions(network) -> list:
    """Build the action set for the North East Region model.

    Three families of action are offered.

    Switching a transformer neutral out of service removes that site as a path for
    induced current, at the price of losing the transformer. It is applied only at
    the sites where the transformer family is one of the vulnerable ones, because
    switching out a three phase three limb unit gives up capacity for very little
    benefit.

    Moving the direct current link off earth return removes the two converter
    stations as current paths at once. This is the single most effective action in
    the set and also one of the most expensive, because it derates the link.

    Committing extra reactive reserve does not reduce the current at all. It buys
    room to absorb the reactive power the current causes, which is the other half
    of the problem and is much cheaper.
    """
    actions = []
    vulnerable = {"single_phase_bank", "hvdc_converter", "autotransformer"}

    for sub in network.substations:
        if sub.transformer_type not in vulnerable:
            continue
        # A more critical site costs more to switch out, because more transfer
        # depends on it.
        cost = 12.0 + 90.0 * sub.criticality
        actions.append(Action(
            key=f"switch_out:{sub.code}",
            label=f"Switch out one transformer neutral at {sub.name}",
            cost_lakh=round(cost, 1),
            lead_time_min=20,
            block_neutrals=(sub.code,),
            note=f"{sub.transformer_type.replace('_', ' ')} at {sub.kv} kV",
        ))

    actions.append(Action(
        key="hvdc_metallic_return",
        label="Move the direct current link from earth return to metallic return",
        cost_lakh=180.0,
        lead_time_min=30,
        block_neutrals=("BWNC", "ALPD"),
        note="Removes both converter stations as current paths and derates the link.",
    ))

    reserve_sites = {"AZRA": 180.0, "MISA": 150.0, "BWNC": 200.0,
                     "SLCR": 140.0, "SRJM": 130.0, "BNGR": 220.0}
    for code, mvar in reserve_sites.items():
        name = next(s.name for s in network.substations if s.code == code)
        actions.append(Action(
            key=f"reserve:{code}",
            label=f"Hold {mvar:.0f} MVAr of extra reactive reserve at {name}",
            cost_lakh=round(0.09 * mvar, 1),
            lead_time_min=15,
            added_reserve={code: mvar},
            note="Costs generation efficiency and nothing else. Fully reversible.",
        ))

    actions.append(Action(
        key="regional_reserve_dispatch",
        label="Bring the regional reactive reserve to its storm posture",
        cost_lakh=95.0,
        lead_time_min=45,
        added_reserve={code: 90.0 for code in
                       ("BNGN", "AZRA", "MISA", "BWNC", "MRNI", "SLCR", "SRJM", "IMPL")},
        note="Raises reserve everywhere by a modest amount rather than a lot in one place.",
    ))
    return actions
