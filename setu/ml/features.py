"""Turn raw solar wind measurements into inputs a network can learn from.

Feeding raw field and velocity components to a network works, and it works better
if the network is also handed the combinations that decades of magnetospheric
physics say are the ones that matter. Those combinations are called coupling
functions. They describe how much energy the solar wind actually delivers into the
magnetosphere, which is the physical cause of the ground disturbance that this
project is trying to predict.

Putting the coupling functions in the input is the first of the two physics
constraints in this model. The second is a penalty in the loss function, which
lives in ``setu.ml.model``.

References for the coupling functions:
  Newell and others (2007), for the rate of magnetic flux opening at the
  magnetopause, which is the single best predictor of most magnetospheric indices.
  Perreault and Akasofu (1978), for the epsilon parameter.
  Kan and Lee (1979), for the merging electric field.
"""

import numpy as np
import pandas as pd

RAW_INPUTS = ["b_total", "bx_gsm", "by_gsm", "bz_gsm", "speed", "density", "pressure"]

DERIVED_INPUTS = [
    "clock_sin", "clock_cos", "bt", "bs", "newell", "epsilon",
    "kan_lee", "dyn_pressure", "d_pressure", "d_bz", "alfven_mach",
    "propagation_delay_min",
]

# Distance from the spacecraft to the Earth when the position is not in the record.
# The first Lagrange point sits about 1.5 million kilometres sunward, which is
# roughly 235 Earth radii.
DEFAULT_SC_DISTANCE_RE = 235.0
EARTH_RADIUS_KM = 6371.0

FEATURE_NAMES = RAW_INPUTS + DERIVED_INPUTS


def clock_angle(by: np.ndarray, bz: np.ndarray) -> np.ndarray:
    """Angle of the interplanetary field in the plane facing the magnetosphere.

    Zero points north and pi points south. Reconnection at the front of the
    magnetosphere is strongest when this angle is near pi, which is why almost
    every coupling function is written as a function of it.
    """
    return np.arctan2(by, bz)


def newell_coupling(by, bz, speed):
    """Rate at which magnetic flux is opened at the magnetopause.

    The units here are arbitrary, because the constant in front of the expression
    is absorbed by the normalisation step. What matters is the shape, which is a
    four thirds power of speed and a sine of the clock angle raised to eight
    thirds.
    """
    bt = np.hypot(by, bz)
    theta = clock_angle(by, bz)
    return np.abs(speed) ** (4.0 / 3.0) * bt ** (2.0 / 3.0) * np.abs(np.sin(theta / 2.0)) ** (8.0 / 3.0)


def akasofu_epsilon(b_total, by, bz, speed):
    """Energy input rate into the magnetosphere, in arbitrary units."""
    theta = clock_angle(by, bz)
    return np.abs(speed) * b_total ** 2 * np.sin(theta / 2.0) ** 4


def kan_lee_field(by, bz, speed):
    """Dawn to dusk merging electric field across the magnetosphere."""
    bt = np.hypot(by, bz)
    theta = clock_angle(by, bz)
    return np.abs(speed) * bt * np.sin(theta / 2.0) ** 2


def southward_field(bz: np.ndarray) -> np.ndarray:
    """Southward part of the interplanetary field, zero when the field points north."""
    return np.where(bz < 0, -bz, 0.0)


def build_features(solar_wind: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full input frame from a solar wind record.

    Args:
        solar_wind: Frame with the columns listed in ``RAW_INPUTS``, at one minute
            cadence and indexed by time.

    Returns:
        A frame with the raw columns and the derived ones, in the fixed order given
        by ``FEATURE_NAMES``.
    """
    missing = [c for c in RAW_INPUTS if c not in solar_wind]
    if missing:
        raise ValueError(f"solar wind frame is missing columns: {missing}")

    f = solar_wind[RAW_INPUTS].copy()
    by, bz = f["by_gsm"].values, f["bz_gsm"].values
    v, n, b = f["speed"].values, f["density"].values, f["b_total"].values
    theta = clock_angle(by, bz)

    f["clock_sin"] = np.sin(theta)
    f["clock_cos"] = np.cos(theta)
    f["bt"] = np.hypot(by, bz)
    f["bs"] = southward_field(bz)
    f["newell"] = newell_coupling(by, bz, v)
    f["epsilon"] = akasofu_epsilon(b, by, bz, v)
    f["kan_lee"] = kan_lee_field(by, bz, v)

    # Dynamic pressure in nanopascal, from proton density and speed.
    f["dyn_pressure"] = 1.6726e-6 * n * v ** 2

    # Rates of change over five minutes. A sudden commencement is a step in
    # pressure, and it is the fastest ground disturbance in any storm, so the model
    # needs to see the step and not only the level.
    f["d_pressure"] = f["dyn_pressure"].diff(5).fillna(0.0)
    f["d_bz"] = pd.Series(bz, index=f.index).diff(5).fillna(0.0)

    # Alfven Mach number. A low value means the solar wind is unusually magnetised,
    # which changes how the magnetosphere responds and was a notable feature of the
    # May 2024 event.
    with np.errstate(divide="ignore", invalid="ignore"):
        v_alfven = 21.8 * b / np.sqrt(np.maximum(n, 1e-6))
        f["alfven_mach"] = np.where(v_alfven > 0, np.abs(v) / v_alfven, np.nan)

    # How long this parcel of solar wind will take to reach the Earth, in minute.
    # When the record is on the spacecraft clock this is the single most important
    # input in the whole set, because it tells the network how far ahead of the
    # arrival it currently is. Without it the network would have to infer the delay
    # from the speed on its own, and the delay is what separates a horizon that has
    # to stay quiet from one that has to raise the alarm.
    distance_re = (solar_wind["sc_x_re"].abs()
                   if "sc_x_re" in solar_wind else DEFAULT_SC_DISTANCE_RE)
    travel_speed = f["speed"].where(f["speed"] > 100.0)
    f["propagation_delay_min"] = (distance_re * EARTH_RADIUS_KM / travel_speed) / 60.0

    return f[FEATURE_NAMES]


class Standardiser:
    """Centre and scale each feature, with the statistics fitted once on training data.

    Several inputs are strongly skewed, so a signed logarithm is applied first to
    the ones that are always positive and span orders of magnitude. Without it a
    single extreme minute would dominate the scale for the whole column.
    """

    LOG_COLUMNS = ("newell", "epsilon", "kan_lee", "dyn_pressure", "density",
                   "pressure", "alfven_mach")

    def __init__(self, feature_names=None):
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.log_mask = np.array([c in self.LOG_COLUMNS for c in self.feature_names])
        self.mean = None
        self.std = None

    def _pre(self, x: np.ndarray) -> np.ndarray:
        out = np.array(x, dtype=float, copy=True)
        out[:, self.log_mask] = np.log1p(np.clip(out[:, self.log_mask], 0.0, None))
        return out

    def fit(self, x: np.ndarray) -> "Standardiser":
        pre = self._pre(x)
        self.mean = np.nanmean(pre, axis=0)
        self.std = np.nanstd(pre, axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("the standardiser must be fitted before it is used")
        return (self._pre(x) - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std,
                "feature_names": np.array(self.feature_names)}

    @classmethod
    def from_state(cls, state) -> "Standardiser":
        obj = cls([str(v) for v in state["feature_names"]])
        obj.mean = np.asarray(state["mean"], dtype=float)
        obj.std = np.asarray(state["std"], dtype=float)
        return obj
