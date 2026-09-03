"""One dimensional Earth conductivity models for the North East Indian region.

The geoelectric field that drives geomagnetically induced current depends far more
on ground conductivity than on anything in the power network. This module holds the
layered resistivity profiles used in the rest of the project and the plane wave
surface impedance that follows from them.

The profiles below are representative models assembled from the published
magnetotelluric and deep resistivity literature for the region. They are not
inverted from raw field data, and they are meant to bracket the real behaviour
rather than to reproduce any single survey. Sources are named in each docstring so
that a reader can check the assumption.
"""

from dataclasses import dataclass, field

import numpy as np

from setu.config import MU0


@dataclass(frozen=True)
class EarthModel:
    """A stack of uniform layers over a half space.

    Attributes:
        name: Short identifier used in outputs and plots.
        resistivities: Layer resistivity in ohm metre, from the surface downward.
            The last entry is the resistivity of the terminating half space.
        thicknesses: Layer thickness in metre, from the surface downward. This list
            is one shorter than ``resistivities`` because the half space has no
            thickness.
        description: Where the profile comes from and what it represents.
    """

    name: str
    resistivities: tuple
    thicknesses: tuple
    description: str = ""

    def __post_init__(self):
        if len(self.thicknesses) != len(self.resistivities) - 1:
            raise ValueError(
                "a model with n resistivities needs n-1 thicknesses, "
                f"got {len(self.resistivities)} and {len(self.thicknesses)}"
            )
        if any(r <= 0 for r in self.resistivities):
            raise ValueError("resistivity must be positive")
        if any(t <= 0 for t in self.thicknesses):
            raise ValueError("thickness must be positive")

    @property
    def conductivities(self) -> np.ndarray:
        return 1.0 / np.asarray(self.resistivities, dtype=float)

    def surface_impedance(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """Plane wave surface impedance Z(omega) in ohm, one value per frequency.

        This is the standard recursion for a layered half space under a vertically
        incident plane wave, evaluated from the bottom layer upward. The zero
        frequency term is returned as zero, which is correct because a static
        magnetic field induces no electric field.

        Reference: Wait (1954) for the recursion, and Boteler and Pirjola (1998)
        for its use in geomagnetically induced current work.
        """
        f = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
        omega = 2.0 * np.pi * f
        rho = np.asarray(self.resistivities, dtype=float)
        thick = np.asarray(self.thicknesses, dtype=float)

        # Work on the non-zero frequencies and fill the DC term in afterwards.
        nonzero = omega != 0.0
        w = omega[nonzero]

        # Bottom half space. The propagation constant uses the quasi-static
        # approximation, where displacement current is neglected.
        k_bottom = np.sqrt(1j * w * MU0 / rho[-1])
        z = 1j * w * MU0 / k_bottom

        # Walk upward through the finite thickness layers.
        for i in range(len(thick) - 1, -1, -1):
            k = np.sqrt(1j * w * MU0 / rho[i])
            z_intrinsic = 1j * w * MU0 / k
            t = np.tanh(k * thick[i])
            z = z_intrinsic * (z + z_intrinsic * t) / (z_intrinsic + z * t)

        out = np.zeros(f.shape, dtype=complex)
        out[nonzero] = z
        return out

    def apparent_resistivity(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """Apparent resistivity curve, useful for sanity checking a profile."""
        f = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
        z = self.surface_impedance(f)
        omega = 2.0 * np.pi * f
        out = np.full(f.shape, np.nan)
        m = omega > 0
        out[m] = np.abs(z[m]) ** 2 / (omega[m] * MU0)
        return out

    def skin_depth(self, frequency_hz: float, layer: int = 0) -> float:
        """Electromagnetic skin depth in metre for one layer at one frequency."""
        rho = self.resistivities[layer]
        return float(np.sqrt(2.0 * rho / (2.0 * np.pi * frequency_hz * MU0)))


# Four profiles that together cover the ground the North East Region grid stands on.
# They are ordered from most resistive to least resistive, which is also the order
# from largest to smallest induced electric field for a given magnetic disturbance.

SHILLONG_PLATEAU = EarthModel(
    name="shillong_plateau",
    resistivities=(3000.0, 8000.0, 2000.0, 100.0, 20.0),
    thicknesses=(2.0e3, 18.0e3, 20.0e3, 60.0e3),
    description=(
        "Precambrian gneissic block of the Shillong Plateau. Thin weathered cover "
        "over a highly resistive crystalline crust, with conductivity rising in the "
        "lower crust and upper mantle. This is the worst case ground in the region "
        "and it hosts Byrnihat and the Meghalaya substations."
    ),
)

BRAHMAPUTRA_VALLEY = EarthModel(
    name="brahmaputra_valley",
    resistivities=(15.0, 60.0, 1500.0, 300.0, 20.0),
    thicknesses=(1.5e3, 4.0e3, 25.0e3, 60.0e3),
    description=(
        "Alluvial fill of the Brahmaputra valley over Precambrian basement. The "
        "conductive sediment shields the surface field, so the induced electric "
        "field here is much smaller than on the plateau a short distance away. "
        "Most of the Assam 400 kV backbone sits on this ground."
    ),
)

BENGAL_TRIPURA_BASIN = EarthModel(
    name="bengal_tripura_basin",
    resistivities=(8.0, 3.0, 25.0, 400.0, 20.0),
    thicknesses=(2.0e3, 8.0e3, 12.0e3, 70.0e3),
    description=(
        "Very thick and very conductive sedimentary pile of the Bengal basin and "
        "the Tripura fold belt. This is the most conductive ground in the study "
        "area and it produces the smallest surface electric field."
    ),
)

ARUNACHAL_HIMALAYA = EarthModel(
    name="arunachal_himalaya",
    resistivities=(500.0, 30.0, 2500.0, 200.0, 20.0),
    thicknesses=(3.0e3, 5.0e3, 25.0e3, 60.0e3),
    description=(
        "Himalayan thrust belt of Arunachal Pradesh. Resistive cover over a "
        "conductive mid crustal thrust zone, which is a common signature in "
        "magnetotelluric profiles across the range. The hydro plants at Kameng "
        "and Ranganadi feed into this ground."
    ),
)

EARTH_MODELS = {
    m.name: m
    for m in (
        SHILLONG_PLATEAU,
        BRAHMAPUTRA_VALLEY,
        BENGAL_TRIPURA_BASIN,
        ARUNACHAL_HIMALAYA,
    )
}


def get_model(name: str) -> EarthModel:
    """Look a model up by name and fail loudly if it is not there."""
    try:
        return EARTH_MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown earth model {name!r}, available: {sorted(EARTH_MODELS)}"
        ) from None


def contrast_amplification(resistive: EarthModel, conductive: EarthModel,
                           frequency_hz: float = 1.0 / 300.0) -> float:
    """Ratio of surface impedance between two profiles at one frequency.

    This single number is the argument for studying the North East. It says how
    much larger the induced electric field is on the resistive side of a
    conductivity boundary than on the conductive side, for the same magnetic
    disturbance. The default period of five minutes sits inside the band that
    carries most of the storm energy that reaches a transformer neutral. Longer
    periods sample deeper and more conductive rock, so the contrast falls away
    above roughly half an hour.

    The real amplification at a sharp lateral boundary is larger than this, because
    charge accumulates on the interface itself. A one dimensional model cannot
    represent that effect, so this ratio is a lower bound.
    """
    zr = abs(resistive.surface_impedance(np.array([frequency_hz]))[0])
    zc = abs(conductive.surface_impedance(np.array([frequency_hz]))[0])
    return float(zr / zc)
