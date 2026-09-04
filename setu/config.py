"""Shared constants and paths.

Physical constants are SI unless a name says otherwise. Magnetic field values are
handled in nanotesla because that is the unit every magnetometer archive uses, and
they are converted to tesla at the point where they enter a physics calculation.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
DOCS_DATA_DIR = ROOT / "docs" / "data"

for _d in (RAW_DIR, ARTIFACT_DIR, DOCS_DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MU0 = 4.0e-7 * 3.141592653589793  # vacuum permeability, henry per metre
NT_TO_T = 1.0e-9
CADENCE_S = 60.0  # every time series in this project is one minute cadence

# Operational alerting thresholds on the horizontal field rate of change.
# The 0.3 nT/s level is where low latitude studies start to report measurable
# neutral currents, and 1.0 nT/s is a severe storm level for this latitude band.
DBDT_THRESHOLDS_NT_PER_S = (0.1, 0.3, 1.0)

# Forecast horizons in minutes, measured from the moment the spacecraft at the
# first Lagrange point took the measurement.
#
# The set has to straddle the propagation delay, and the reason is a hard physical
# limit rather than a modelling choice. A disturbance can only be forecast from a
# horizon shorter than the time it takes to arrive, because at any longer horizon
# the disturbance had not yet reached the spacecraft when the forecast was issued.
# During the fast solar wind of 10 May 2024 the delay was about thirty minutes, so
# no horizon of forty five minutes or more could have caught that shock, and an
# earlier version of this project used forty five as its shortest and missed it.
#
# The delay runs from about twenty five minutes to eighty depending on speed, so
# the useful warning is short exactly when the storm is fast. Thirty minutes is
# included to cover the fast cases, ninety to cover the slow ones, and the
# propagation delay is given to the network as an input so it can tell which
# situation it is in.
FORECAST_HORIZONS_MIN = (30, 45, 60, 90)

# Geometry of the solar wind monitor at the first Lagrange point.
#
# EARTH_RADIUS_KM is the usual value. L1_DISTANCE_RE is the median sunward
# distance of the monitoring spacecraft, measured across 1.05 million minutes of
# the OMNI archive after excluding the minutes when the reporting spacecraft was
# not near the first Lagrange point.
#
# PHASE_FRONT_CORRECTION is the part worth explaining. Treating the solar wind as
# travelling straight down the Sun to Earth line overestimates the travel time,
# because the real front of a disturbance is tilted and reaches the Earth sooner
# than a flat front would. The archive computes the true delay from the
# orientation of that front, and comparing the two over the same 1.03 million
# minutes gives a ratio of 0.953. Applying it turns a plain distance over speed
# estimate into one that matches the archive in the median while still using
# nothing a real time system would not already have.
#
# Recompute both with scripts/calibrate_delay.py if the archive is extended.
EARTH_RADIUS_KM = 6371.0
L1_DISTANCE_RE = 225.7
PHASE_FRONT_CORRECTION = 0.953

# Which clock the solar wind is placed on. The bow shock base is what the OMNI
# archive publishes and it leaves no warning time at all for a shock arrival. The
# L1 base is the clock of the spacecraft and it is what a warning system has.
DEFAULT_TIME_BASE = "l1"

# Quantile levels used by the forecast model. The median is included so the model
# can also be read as a point forecast when an operator wants one number.
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.98)

# Reference magnetic observatory for the region. Shillong is a real Indian
# Institute of Geomagnetism observatory and sits on the resistive plateau.
REFERENCE_OBSERVATORY = {
    "code": "SHL",
    "name": "Shillong",
    "latitude": 25.57,
    "longitude": 91.88,
    "geomagnetic_latitude": 16.1,
}
