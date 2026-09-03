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

# Forecast horizons in minutes. The short horizon matches the travel time of the
# solar wind from the L1 point to the magnetopause during a fast stream.
FORECAST_HORIZONS_MIN = (30, 45, 60)

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
