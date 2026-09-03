"""A representative model of the North East Region transmission grid.

The topology, the substation locations, and the plant capacities follow the public
picture of the 400 kV and 220 kV network of the North Eastern Regional grid, which
includes the Biswanath Chariali terminal of the high voltage direct current link to
the national grid and the Siliguri corridor connection to the Eastern Region.

Electrical parameters are representative values for Indian transmission practice
rather than measured data, because real network parameters are not public. The
model is built to be topologically and electrically honest, so a conclusion drawn
from it about which parts of the network are exposed will hold, while an absolute
current in ampere should be read as an estimate.

Every substation is tagged with the layered Earth profile it stands on, which is
what lets the model show the conductivity contrast across the region.
"""

from dataclasses import dataclass, field
from math import cos, radians

import numpy as np

# Transformer families, ordered by how badly each one behaves under a direct
# current bias in its neutral.
#
# ``k_var`` is the reactive power absorbed per ampere of geomagnetically induced
# current per phase, in megavar per ampere. ``hotspot_k`` is the rise of the
# winding hot spot temperature per ampere per phase, in kelvin per ampere, at the
# quasi steady state. Both are representative values taken from the ranges
# reported in the international literature, most of which is summarised in the
# NERC geomagnetic disturbance task force reports and in Price (2002).
TRANSFORMER_TYPES = {
    "three_phase_three_limb": {
        "k_var": 0.28,
        "hotspot_k": 0.55,
        "note": "Flux returns through the tank, so the core saturates late. Most immune type.",
    },
    "three_phase_five_limb": {
        "k_var": 0.62,
        "hotspot_k": 1.10,
        "note": "Outer limbs give the zero sequence flux a path, so saturation starts earlier.",
    },
    "autotransformer": {
        "k_var": 0.85,
        "hotspot_k": 1.45,
        "note": "Common at 400 over 220 kV in India. Series winding carries the bias directly.",
    },
    "single_phase_bank": {
        "k_var": 1.15,
        "hotspot_k": 2.10,
        "note": "Each unit has an independent flux path, so it saturates at low bias. Worst case.",
    },
    "hvdc_converter": {
        "k_var": 1.30,
        "hotspot_k": 2.40,
        "note": "Single phase construction plus an existing direct current duty. Least margin.",
    },
}


@dataclass
class Substation:
    """One node of the direct current network.

    Attributes:
        code: Short identifier used everywhere else in the project.
        name: Human readable name for reports and for the dashboard.
        lat: Latitude in degree north.
        lon: Longitude in degree east.
        kv: Highest voltage level present at the site.
        earth_model: Name of the layered Earth profile under the site.
        ground_resistance: Substation grounding grid resistance to remote earth,
            in ohm. A low value means the site is an easy path for current into
            the ground, so it draws more of it.
        transformer_type: Key into ``TRANSFORMER_TYPES``.
        transformer_count: Number of parallel units at the site, which sets how
            the total current divides between them.
        winding_resistance: Direct current resistance of one transformer winding
            path to the neutral, in ohm per phase.
        criticality: Weight between zero and one for how much of the regional
            load or generation depends on this node. Used by the decision layer.
        state: Indian state, for reporting only.
    """

    code: str
    name: str
    lat: float
    lon: float
    kv: int
    earth_model: str
    ground_resistance: float
    transformer_type: str
    transformer_count: int
    winding_resistance: float
    criticality: float
    state: str

    def __post_init__(self):
        if self.transformer_type not in TRANSFORMER_TYPES:
            raise ValueError(f"unknown transformer type {self.transformer_type!r}")
        if self.ground_resistance <= 0:
            raise ValueError("ground resistance must be positive")


@dataclass
class Line:
    """One transmission corridor between two substations.

    ``resistance`` is the direct current resistance of a single phase conductor
    over the whole route, in ohm. The solver divides it by three because the three
    phases sit in parallel for a direct current that enters through the neutral.
    """

    frm: str
    to: str
    kv: int
    resistance: float
    circuits: int = 1
    note: str = ""


# Coordinates are the published locations of the substations and power stations.
SUBSTATIONS = [
    Substation("BNGN", "Bongaigaon", 26.48, 90.55, 400, "brahmaputra_valley",
               0.35, "autotransformer", 3, 0.55, 0.85, "Assam"),
    Substation("SLKT", "Salakati", 26.42, 90.62, 400, "brahmaputra_valley",
               0.42, "three_phase_three_limb", 2, 0.62, 0.55, "Assam"),
    Substation("AGIA", "Agia", 26.15, 90.70, 400, "brahmaputra_valley",
               0.48, "autotransformer", 2, 0.58, 0.50, "Assam"),
    Substation("BYRN", "Byrnihat", 25.90, 91.87, 400, "shillong_plateau",
               0.95, "single_phase_bank", 3, 0.50, 0.90, "Meghalaya"),
    Substation("AZRA", "Azara", 26.10, 91.58, 400, "brahmaputra_valley",
               0.40, "autotransformer", 3, 0.55, 0.95, "Assam"),
    Substation("SHLG", "Shillong", 25.57, 91.88, 220, "shillong_plateau",
               1.20, "three_phase_three_limb", 2, 0.75, 0.45, "Meghalaya"),
    Substation("KPLI", "Kopili", 25.65, 92.80, 220, "shillong_plateau",
               1.05, "three_phase_five_limb", 2, 0.70, 0.40, "Assam"),
    Substation("MISA", "Misa", 26.42, 92.83, 400, "brahmaputra_valley",
               0.38, "autotransformer", 3, 0.55, 0.80, "Assam"),
    Substation("SMGR", "Samaguri", 26.35, 92.92, 400, "brahmaputra_valley",
               0.45, "three_phase_three_limb", 2, 0.60, 0.50, "Assam"),
    Substation("BWNC", "Biswanath Chariali", 26.72, 93.15, 400, "brahmaputra_valley",
               0.22, "hvdc_converter", 4, 0.35, 1.00, "Assam"),
    Substation("BLPR", "Balipara", 26.83, 92.78, 400, "brahmaputra_valley",
               0.44, "autotransformer", 2, 0.58, 0.60, "Assam"),
    Substation("TZPR", "Tezpur", 26.63, 92.80, 220, "brahmaputra_valley",
               0.60, "three_phase_three_limb", 2, 0.72, 0.35, "Assam"),
    Substation("KMNG", "Kameng", 27.20, 92.55, 400, "arunachal_himalaya",
               0.85, "three_phase_five_limb", 2, 0.60, 0.65, "Arunachal Pradesh"),
    Substation("RNGN", "Ranganadi", 27.35, 93.90, 220, "arunachal_himalaya",
               0.90, "three_phase_five_limb", 2, 0.68, 0.50, "Arunachal Pradesh"),
    Substation("ITNG", "Itanagar", 27.10, 93.62, 220, "arunachal_himalaya",
               1.00, "three_phase_three_limb", 2, 0.75, 0.35, "Arunachal Pradesh"),
    Substation("MRNI", "Mariani", 26.65, 94.30, 400, "brahmaputra_valley",
               0.46, "autotransformer", 2, 0.58, 0.55, "Assam"),
    Substation("DMPR", "Dimapur", 25.90, 93.72, 220, "brahmaputra_valley",
               0.70, "three_phase_three_limb", 2, 0.72, 0.45, "Nagaland"),
    Substation("IMPL", "Imphal", 24.80, 93.94, 400, "bengal_tripura_basin",
               0.75, "autotransformer", 2, 0.58, 0.60, "Manipur"),
    Substation("LKTK", "Loktak", 24.50, 93.80, 132, "bengal_tripura_basin",
               1.10, "three_phase_three_limb", 2, 0.85, 0.30, "Manipur"),
    Substation("SLCR", "Silchar", 24.82, 92.80, 400, "bengal_tripura_basin",
               0.55, "single_phase_bank", 2, 0.52, 0.70, "Assam"),
    Substation("AZWL", "Aizawl", 23.73, 92.72, 220, "bengal_tripura_basin",
               0.95, "three_phase_three_limb", 2, 0.78, 0.35, "Mizoram"),
    Substation("KMGT", "Kumarghat", 24.20, 92.02, 400, "bengal_tripura_basin",
               0.65, "autotransformer", 2, 0.60, 0.45, "Tripura"),
    Substation("SRJM", "Surajmaninagar", 23.75, 91.28, 400, "bengal_tripura_basin",
               0.58, "autotransformer", 2, 0.58, 0.60, "Tripura"),
    Substation("PLTN", "Palatana", 23.55, 91.40, 400, "bengal_tripura_basin",
               0.62, "three_phase_five_limb", 2, 0.62, 0.65, "Tripura"),
    Substation("ALPD", "Alipurduar", 26.48, 89.53, 400, "brahmaputra_valley",
               0.30, "hvdc_converter", 2, 0.38, 0.75, "West Bengal"),
    Substation("BNGR", "Binaguri", 26.62, 88.72, 400, "brahmaputra_valley",
               0.33, "autotransformer", 3, 0.55, 0.90, "West Bengal"),
]

# Line resistances are set from route length at roughly 0.028 ohm per kilometre
# for a twin bundle 400 kV conductor and 0.075 ohm per kilometre at 220 kV, which
# are ordinary values for Indian transmission conductor.
LINES = [
    Line("BNGR", "ALPD", 400, 2.4, 2, "Siliguri corridor, the only link to the Eastern Region"),
    Line("ALPD", "BNGN", 400, 2.9, 2, "corridor continuation into Assam"),
    Line("ALPD", "BWNC", 400, 8.4, 1, "high voltage direct current route, earth return path"),
    Line("BNGN", "SLKT", 400, 0.3, 2, ""),
    Line("BNGN", "AGIA", 400, 1.2, 2, ""),
    Line("AGIA", "AZRA", 400, 2.6, 2, ""),
    Line("AZRA", "BYRN", 400, 1.4, 2, ""),
    Line("BYRN", "SHLG", 220, 3.6, 2, "climbs onto the resistive plateau"),
    Line("BYRN", "KPLI", 220, 7.2, 1, ""),
    Line("AZRA", "MISA", 400, 3.6, 2, ""),
    Line("MISA", "SMGR", 400, 0.4, 2, ""),
    Line("MISA", "KPLI", 220, 5.5, 1, ""),
    Line("SMGR", "BWNC", 400, 2.7, 2, ""),
    Line("BWNC", "BLPR", 400, 1.3, 2, ""),
    Line("BLPR", "KMNG", 400, 2.1, 2, ""),
    Line("BLPR", "TZPR", 220, 1.5, 2, ""),
    Line("BWNC", "RNGN", 220, 6.8, 1, ""),
    Line("RNGN", "ITNG", 220, 2.4, 1, ""),
    Line("SMGR", "MRNI", 400, 3.9, 2, ""),
    Line("MRNI", "DMPR", 220, 4.6, 1, ""),
    Line("DMPR", "IMPL", 400, 5.1, 1, ""),
    Line("IMPL", "LKTK", 132, 3.2, 1, ""),
    Line("IMPL", "SLCR", 400, 3.4, 1, ""),
    Line("KPLI", "SLCR", 220, 8.9, 1, "long radial link across the Barail range"),
    Line("SLCR", "AZWL", 220, 6.4, 1, ""),
    Line("SLCR", "KMGT", 400, 2.2, 2, ""),
    Line("KMGT", "SRJM", 400, 2.6, 2, ""),
    Line("SRJM", "PLTN", 400, 0.7, 2, ""),
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great circle distance in kilometre between two points on the Earth."""
    r = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + cos(p1) * cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


class Network:
    """The assembled direct current network, ready for the current solver."""

    def __init__(self, substations=None, lines=None):
        self.substations = list(substations if substations is not None else SUBSTATIONS)
        self.lines = list(lines if lines is not None else LINES)
        self.index = {s.code: i for i, s in enumerate(self.substations)}
        codes = set(self.index)
        for ln in self.lines:
            if ln.frm not in codes or ln.to not in codes:
                raise ValueError(f"line {ln.frm}-{ln.to} refers to an unknown substation")

    @property
    def n(self) -> int:
        return len(self.substations)

    def line_vector_km(self, line: Line):
        """Northward and eastward extent of a line in kilometre.

        The sign matters, because the induced voltage is a dot product with the
        electric field and a line running the other way sees the opposite sign.
        """
        a = self.substations[self.index[line.frm]]
        b = self.substations[self.index[line.to]]
        dx = haversine_km(a.lat, a.lon, b.lat, a.lon) * (1 if b.lat >= a.lat else -1)
        dy = haversine_km(a.lat, a.lon, a.lat, b.lon) * (1 if b.lon >= a.lon else -1)
        return dx, dy

    def line_length_km(self, line: Line) -> float:
        a = self.substations[self.index[line.frm]]
        b = self.substations[self.index[line.to]]
        return haversine_km(a.lat, a.lon, b.lat, b.lon)

    def summary(self) -> str:
        total_km = sum(self.line_length_km(l) for l in self.lines)
        return (
            f"North East Region model: {self.n} substations, {len(self.lines)} corridors, "
            f"{total_km:,.0f} km of route"
        )
