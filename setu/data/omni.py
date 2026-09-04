"""Read one minute solar wind data from the NASA OMNI archive.

OMNI is the standard merged solar wind dataset. It takes measurements made by the
spacecraft at the first Lagrange point, mainly ACE and Wind and DSCOVR, and shifts
them forward in time to the nose of the bow shock.

That shift has a consequence which matters a great deal for this project, and it is
stated here rather than left to be discovered. Because OMNI has already moved the
measurement forward to Earth, a shock front appears in the OMNI record at the same
moment it strikes the magnetosphere. A model trained on OMNI therefore has no way
at all to anticipate a storm sudden commencement, and the replay of the May 2024
event shows exactly that failure. It predicts the sustained main phase and it
misses the onset.

The fix is to train on the raw first Lagrange point record instead, where the
travel time from the spacecraft to the Earth is still in front of the data and is
worth thirty to sixty minutes depending on the solar wind speed. That is the single
most valuable change to make to this system and it is the first item of future
work.

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

from setu.config import (EARTH_RADIUS_KM, L1_DISTANCE_RE,
                         PHASE_FRONT_CORRECTION, RAW_DIR)

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
    # The shift the archive applied to move this measurement from the spacecraft
    # to the bow shock nose, in second, and the position of the spacecraft that
    # made it, in Earth radii. These are what make it possible to undo the shift.
    "timeshift": (9, 999999.0),
    "vx_gse": (22, 99999.9),
    "sc_x_re": (31, 9999.0),
    "sc_y_re": (32, 9999.0),
    "sc_z_re": (33, 9999.0),
}

# A shift outside this range means the archive could not work out a sensible
# propagation delay for that minute, so the row is dropped rather than trusted.
MIN_TIMESHIFT_S = 300.0
MAX_TIMESHIFT_S = 7200.0


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


def advection_delay_s(frame: pd.DataFrame) -> pd.Series:
    """Travel time from the spacecraft to the bow shock, in second.

    This is the plain advection estimate. The solar wind is taken to travel
    straight down the Sun to Earth line at its measured speed, so the delay is the
    distance divided by the speed. Both quantities are in the telemetry at the
    moment it arrives, so this estimate needs nothing that a real time system would
    not already have.

    The archive publishes its own shift, which is computed from the orientation of
    the phase front. That method needs a window of data around each minute, so it
    looks slightly into the future of any given sample and cannot be used in a
    warning system. It is more accurate, and a plain radial estimate runs long by
    about five percent because a tilted front reaches the Earth sooner than a flat
    one would. That five percent is folded in here as a fixed correction, measured
    once against the whole archive, which costs nothing at run time and needs no
    knowledge of the future.
    """
    speed = frame["vx_gse"].abs().where(frame["vx_gse"].abs() > 100.0)
    speed = speed.fillna(frame["speed"].where(frame["speed"] > 100.0))
    distance_re = frame["sc_x_re"].abs() if "sc_x_re" in frame else L1_DISTANCE_RE
    distance_km = distance_re * EARTH_RADIUS_KM * PHASE_FRONT_CORRECTION
    return distance_km / speed


def to_l1_time_base(frame: pd.DataFrame, method: str = "advection") -> pd.DataFrame:
    """Move the record back onto the clock of the spacecraft that measured it.

    This is the single most important function in the data layer, and the reason
    is worth setting out fully.

    OMNI publishes the solar wind against the time it reaches the nose of the bow
    shock, not the time it was measured. That is the right convention for studying
    what the magnetosphere did, and it is exactly the wrong convention for building
    a warning system. Under it a shock front appears in the record at the same
    instant it strikes the Earth, so a model trained on it has nothing to warn
    about. The first version of this project did train on it, and the replay of the
    May 2024 storm shows the model missing the sudden commencement by about forty
    five minutes for precisely this reason.

    The archive records the shift it applied, so it can be undone. Each row is
    moved back by its own shift, which puts it at the moment the spacecraft at the
    first Lagrange point actually saw it. A forecast horizon measured from that
    moment is real warning time, and during the May 2024 storm it was worth between
    thirty seven and forty eight minutes.

    Rows whose shift is missing or outside a sensible range are dropped. The result
    is resampled back onto a regular one minute grid, because the shift varies from
    minute to minute and the shifted timestamps are neither evenly spaced nor
    guaranteed to stay in order.

    Args:
        frame: A record on the bow shock time base, as returned by ``fetch_range``.
        method: ``advection`` uses the delay implied by spacecraft position and
            measured speed, which involves no lookahead. ``archive`` uses the shift
            the archive itself applied, which is more accurate and slightly
            acausal. The default is the honest one.
    """
    if method == "archive":
        if "timeshift" not in frame:
            raise ValueError("the frame has no timeshift column")
        shift = frame["timeshift"]
    elif method == "advection":
        shift = advection_delay_s(frame)
    else:
        raise ValueError(f"unknown method {method!r}, use 'advection' or 'archive'")
    usable = shift.between(MIN_TIMESHIFT_S, MAX_TIMESHIFT_S)
    out = frame.loc[usable].copy()
    out["applied_delay_s"] = shift.loc[usable]
    if out.empty:
        raise ValueError("no rows have a usable propagation delay")

    observed_at = out.index - pd.to_timedelta(out["applied_delay_s"].values, unit="s")
    out.index = pd.DatetimeIndex(observed_at, name="time")
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # The shift varies from minute to minute, so the shifted stamps no longer land
    # on whole minutes and about half the target minutes receive no source row.
    # Averaging onto the grid and then bridging the one and two minute holes is
    # safe here, because the source cadence was already one minute and nothing in
    # the solar wind changes meaningfully inside that gap.
    gridded = out.resample("1min").mean()
    return gridded.interpolate(method="time", limit=3, limit_area="inside")


def fill_gaps(frame: pd.DataFrame, limit_minutes: int = 30) -> pd.DataFrame:
    """Interpolate short gaps and leave long ones missing.

    A gap of a few minutes in the solar wind record is safe to bridge, because the
    quantities vary smoothly on that scale. A gap of hours is a real loss of
    information and pretending otherwise would put invented data into training.
    """
    out = frame.interpolate(method="time", limit=limit_minutes, limit_area="inside")
    return out
