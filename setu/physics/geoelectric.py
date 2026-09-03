"""Convert a ground magnetic field time series into a surface geoelectric field.

The method is the plane wave method. A magnetic variation at the surface is treated
as a vertically incident uniform plane wave, and the surface impedance of the local
layered Earth relates the horizontal electric field to the horizontal magnetic
field at every frequency. In the frequency domain this is a single multiplication,
so the whole calculation is a forward transform, a filter, and an inverse
transform.

The rotation between the two horizontal components follows the usual convention for
an isotropic layered Earth, where the northward electric field is driven by the
eastward magnetic field and the eastward electric field is driven by the northward
magnetic field with the opposite sign.

Reference: Boteler and Pirjola (1998), and Pirjola (2002) for the plane wave
assumption and its limits.
"""

import numpy as np

from setu.config import CADENCE_S, MU0, NT_TO_T
from setu.physics.earth import EarthModel


def _taper(n: int, fraction: float = 0.05) -> np.ndarray:
    """Split cosine taper that suppresses the wrap around artefact of the FFT.

    Only the ends of the record are touched, so the interior of a storm is left
    alone. A fraction of five percent is enough for a record of several hours.
    """
    w = np.ones(n)
    edge = max(1, int(n * fraction))
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
    w[:edge] = ramp
    w[-edge:] = ramp[::-1]
    return w


def _prepare(series: np.ndarray, taper_fraction: float):
    """Remove the mean and the linear trend, then taper.

    A static offset carries no information for induction and a residual trend
    leaks into the lowest frequency bins, so both are removed before the
    transform. The removed trend is not added back, because the electric field
    that corresponds to a constant magnetic field is zero.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    t = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    detrended = x - (slope * t + intercept)
    return detrended * _taper(n, taper_fraction)


def geoelectric_field(bx_nt: np.ndarray, by_nt: np.ndarray, model: EarthModel,
                      cadence_s: float = CADENCE_S,
                      taper_fraction: float = 0.05):
    """Compute the horizontal geoelectric field from a horizontal magnetic field.

    Args:
        bx_nt: Northward magnetic field in nanotesla, one minute cadence.
        by_nt: Eastward magnetic field in nanotesla, same length as ``bx_nt``.
        model: Layered Earth profile for the site.
        cadence_s: Sample spacing in second.
        taper_fraction: Fraction of the record tapered at each end.

    Returns:
        A pair ``(ex, ey)`` of arrays in volt per kilometre, which is the unit
        the power systems literature uses.
    """
    bx = np.asarray(bx_nt, dtype=float)
    by = np.asarray(by_nt, dtype=float)
    if bx.shape != by.shape:
        raise ValueError("the two magnetic components must have the same length")
    n = bx.size
    if n < 16:
        raise ValueError("a record shorter than 16 samples cannot be transformed")

    freqs = np.fft.rfftfreq(n, d=cadence_s)
    z = model.surface_impedance(freqs)

    fx = np.fft.rfft(_prepare(bx, taper_fraction) * NT_TO_T)
    fy = np.fft.rfft(_prepare(by, taper_fraction) * NT_TO_T)

    # Electric field in volt per metre, then converted to volt per kilometre.
    ex = np.fft.irfft(z * fy / MU0, n=n) * 1.0e3
    ey = np.fft.irfft(-z * fx / MU0, n=n) * 1.0e3
    return ex, ey


def dbdt(bx_nt: np.ndarray, by_nt: np.ndarray,
         cadence_s: float = CADENCE_S) -> np.ndarray:
    """Magnitude of the rate of change of the horizontal magnetic field.

    This is the quantity the forecast model predicts, in nanotesla per second. It
    is used instead of the field itself because induction responds to the rate of
    change, so a large but steady depression of the field is harmless while a fast
    small excursion is not.
    """
    bx = np.asarray(bx_nt, dtype=float)
    by = np.asarray(by_nt, dtype=float)
    dx = np.gradient(bx, cadence_s)
    dy = np.gradient(by, cadence_s)
    return np.hypot(dx, dy)


def field_magnitude(ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
    """Magnitude of a horizontal vector field, in the unit of its components."""
    return np.hypot(np.asarray(ex, dtype=float), np.asarray(ey, dtype=float))


def line_voltage(ex: np.ndarray, ey: np.ndarray,
                 dx_km: float, dy_km: float) -> np.ndarray:
    """Induced voltage along a straight transmission line, in volt.

    The plane wave field is uniform over the scale of one line, so the induced
    electromotive force is the dot product of the field with the line vector. This
    is the only place where the geometry of the network meets the geophysics, and
    it is the reason a long line is worse than a short one.

    Args:
        ex: Northward electric field in volt per kilometre.
        ey: Eastward electric field in volt per kilometre.
        dx_km: Northward extent of the line in kilometre, from end to end.
        dy_km: Eastward extent of the line in kilometre.
    """
    return np.asarray(ex, dtype=float) * dx_km + np.asarray(ey, dtype=float) * dy_km
