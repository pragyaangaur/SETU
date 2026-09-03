"""Read one minute solar wind data from the NASA OMNI archive.

OMNI is the standard merged solar wind dataset. It takes measurements made by the
spacecraft at the first Lagrange point, mainly ACE and Wind and DSCOVR, and shifts
them forward in time to the nose of the bow shock. That shift is exactly the lead
time a grid operator gets, so OMNI is the right training source for a system whose
whole purpose is to use that lead time.

Files are plain text, one month per file, hosted by the Space Physics Data
Facility. The column layout is fixed and documented by the archive, so the reader
below indexes columns by position.

Data source: NASA Goddard Space Flight Center, Space Physics Data Facility,
OMNIWeb high resolution OMNI dataset. Please cite the archive if you use this.
"""

import datetime as dt
import io
import logging

import numpy as np
import pandas as pd
import requests

from setu.config import RAW_DIR

log = logging.getLogger(__name__)

BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min"

# Position of each field in the fixed width record, and the value the archive uses
# to mark a gap. Anything at or above the fill value is turned into a missing value.
COLUMNS = {
    "b_total": (13, 9999.99),
    "bx_gsm": (14, 9999.99),
    "by_gsm": (17, 9999.99),
    "bz_gsm": (18, 9999.99),
    "speed": (21, 99999.9),
    "density": (25, 999.99),
    "pressure": (27, 99.99),
    "ae_index": (37, 99999.0),
    "sym_h": (41, 99999.0),
}


def _month_url(year: int, month: int) -> str:
    return f"{BASE_URL}/omni_min{year}{month:02d}.asc"


def fetch_month(year: int, month: int, timeout: int = 60,
                use_cache: bool = True) -> pd.DataFrame:
    """Download and parse one month of one minute OMNI data.

    Files are cached under ``data/raw`` so a repeated run does not hit the archive
    again. The archive is a public service, so it is worth being polite to it.
    """
    cache = RAW_DIR / f"omni_min{year}{month:02d}.asc"
    if use_cache and cache.exists():
        text = cache.read_text()
    else:
        url = _month_url(year, month)
        log.info("downloading %s", url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        cache.write_text(text)

    raw = np.loadtxt(io.StringIO(text))
    if raw.ndim == 1:
        raw = raw[None, :]

    stamps = [
        dt.datetime(int(r[0]), 1, 1)
        + dt.timedelta(days=int(r[1]) - 1, hours=int(r[2]), minutes=int(r[3]))
        for r in raw
    ]
    frame = pd.DataFrame(index=pd.DatetimeIndex(stamps, name="time"))
    for name, (col, fill) in COLUMNS.items():
        values = raw[:, col].astype(float)
        values[values >= fill] = np.nan
        frame[name] = values
    return frame


def fetch_range(start: dt.date, end: dt.date, **kwargs) -> pd.DataFrame:
    """Download every month that overlaps a date range and join the result."""
    frames = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        frames.append(fetch_month(year, month, **kwargs))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    joined = pd.concat(frames).sort_index()
    mask = (joined.index >= pd.Timestamp(start)) & (
        joined.index < pd.Timestamp(end) + pd.Timedelta(days=1)
    )
    return joined.loc[mask]


def fill_gaps(frame: pd.DataFrame, limit_minutes: int = 30) -> pd.DataFrame:
    """Interpolate short gaps and leave long ones missing.

    A gap of a few minutes in the solar wind record is safe to bridge, because the
    quantities vary smoothly on that scale. A gap of hours is a real loss of
    information and pretending otherwise would put invented data into training.
    """
    out = frame.interpolate(method="time", limit=limit_minutes, limit_area="inside")
    return out
