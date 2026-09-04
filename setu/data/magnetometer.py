"""Read one minute ground magnetometer data from the INTERMAGNET data service.

The forecast target of this project is the rate of change of the horizontal
magnetic field at the ground, so a real ground record is needed to train against.

Two Indian observatories are available through the public INTERMAGNET service,
Alibag and Hyderabad, both run by the Indian Institute of Geomagnetism. Shillong
and Tirunelveli are also Indian Institute of Geomagnetism observatories but they
are not published through this service, so a formal data request to the institute
is needed to add them. Until that data arrives the model is trained on the two
available Indian stations, which sit in the same low geomagnetic latitude band and
see the same large scale storm time variation.

Data source: INTERMAGNET, served by the British Geological Survey Geomagnetism
Information Node. INTERMAGNET data carry their own conditions of use and should be
acknowledged in any publication.
"""

import datetime as dt
import logging

import numpy as np
import pandas as pd
import requests

from setu.config import RAW_DIR

log = logging.getLogger(__name__)

SERVICE_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"

# Observatories used by this project, with their geographic and geomagnetic
# latitude. The geomagnetic latitude is what controls how strongly a station feels
# the auroral and the equatorial current systems.
OBSERVATORIES = {
    "ABG": {"name": "Alibag", "lat": 18.62, "lon": 72.87, "cgm_lat": 10.2},
    "HYB": {"name": "Hyderabad", "lat": 17.42, "lon": 78.55, "cgm_lat": 8.9},
}

# Stations that would improve the model and are not published openly. These are
# listed in code so that the gap is visible rather than quietly ignored.
REQUESTED_OBSERVATORIES = {
    "SHL": {"name": "Shillong", "lat": 25.57, "lon": 91.88, "cgm_lat": 16.1,
            "status": "held by the Indian Institute of Geomagnetism, data request pending"},
    "TIR": {"name": "Tirunelveli", "lat": 8.70, "lon": 77.80, "cgm_lat": 0.3,
            "status": "held by the Indian Institute of Geomagnetism, data request pending"},
}

FILL_VALUES = (99999.0, 88888.0, 99999.00)


def _parse_iaga2002(text: str) -> pd.DataFrame:
    """Turn an IAGA 2002 text response into a frame of X, Y, Z and F.

    Header lines are marked by a trailing pipe character and are skipped. The four
    element columns follow the date, the time, and the day of year.
    """
    stamps, values = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith("|"):
            continue
        parts = stripped.split()
        if len(parts) < 7:
            continue
        try:
            stamp = dt.datetime.strptime(parts[0] + " " + parts[1][:12],
                                         "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            continue
        stamps.append(stamp)
        values.append([float(v) for v in parts[3:7]])
    if not stamps:
        raise ValueError("no data rows found in the service response")
    arr = np.asarray(values, dtype=float)
    for fill in FILL_VALUES:
        arr[np.isclose(arr, fill)] = np.nan
    return pd.DataFrame(arr, columns=["bx", "by", "bz", "bf"],
                        index=pd.DatetimeIndex(stamps, name="time"))


def fetch_observatory(code: str, start: dt.date, days: int,
                      timeout: int = 90, use_cache: bool = True) -> pd.DataFrame:
    """Download a block of one minute data for one observatory.

    Args:
        code: IAGA three letter code, for example ``ABG``.
        start: First day to download.
        days: Number of whole days.

    Returns:
        A frame indexed by time with the northward, eastward, and vertical field
        in nanotesla and the total field.
    """
    cache = RAW_DIR / f"mag_{code}_{start:%Y%m%d}_{days}d.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    params = {
        "Request": "GetData",
        "format": "iaga2002",
        "testObsys": "0",
        "observatoryIagaCode": code,
        "samplesPerDay": "1440",
        "publicationState": "Best available",
        "dataStartDate": start.isoformat(),
        "dataDuration": str(days),
    }
    log.info("requesting %s for %s days from %s", code, days, start)
    resp = requests.get(SERVICE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    frame = _parse_iaga2002(resp.text)
    frame.to_csv(cache)
    return frame


def fetch_long_record(code: str, start: dt.date, end: dt.date,
                      chunk_days: int = 30, **kwargs) -> pd.DataFrame:
    """Download a long record in chunks, because the service limits one request."""
    frames = []
    cursor = start
    while cursor < end:
        span = min(chunk_days, (end - cursor).days)
        if span <= 0:
            break
        try:
            frames.append(fetch_observatory(code, cursor, span, **kwargs))
        except Exception as exc:
            log.warning("chunk starting %s failed: %s", cursor, exc)
        cursor += dt.timedelta(days=span)
    if not frames:
        raise RuntimeError(f"no data could be retrieved for {code}")
    return pd.concat(frames).sort_index()


def to_disturbance(frame: pd.DataFrame, window: str = "6h") -> pd.DataFrame:
    """Remove the quiet day baseline so only the disturbance field is left.

    The absolute field at a station is dominated by the main field of the Earth and
    by the regular daily variation, neither of which is a storm. A long running
    median is subtracted, which removes both without removing the fast excursions
    that matter for induction.
    """
    out = frame.copy()
    for col in ("bx", "by", "bz"):
        if col in out:
            baseline = out[col].rolling(window, center=True, min_periods=1).median()
            out[col] = out[col] - baseline
    return out


def fetch_recent(code: str, hours: int = 6, timeout: int = 60) -> pd.DataFrame:
    """Download the last few hours of one minute data for one observatory.

    This is the near real time counterpart of :func:`fetch_observatory`. The
    service publishes reported data within a few minutes of the minute it covers,
    which is what makes it possible to check a forecast against the ground while
    the forecast is still recent.

    Nothing is cached, because a cache would freeze the record at the moment it was
    first written and this function exists to see what has happened since.

    Args:
        code: IAGA three letter code.
        hours: How far back to ask for. The service works in whole days, so two
            days are requested when the window crosses midnight.

    Returns:
        A frame of the last ``hours`` of one minute field values, with the fill
        values removed and the empty tail of the current day dropped. Minutes the
        observatory has not reported yet are simply absent.
    """
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    start_day = (now - dt.timedelta(hours=hours)).date()
    days = (now.date() - start_day).days + 1

    params = {
        "Request": "GetData",
        "format": "iaga2002",
        "testObsys": "0",
        "observatoryIagaCode": code,
        "samplesPerDay": "1440",
        "publicationState": "Best available",
        "dataStartDate": start_day.isoformat(),
        "dataDuration": str(days),
    }
    resp = requests.get(SERVICE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    frame = _parse_iaga2002(resp.text)
    frame = frame.dropna(subset=["bx", "by"])
    return frame[frame.index >= now - dt.timedelta(hours=hours)]


def first_reporting(codes=("ABG", "HYB"), hours: int = 6, min_rows: int = 60,
                    max_age_min: int = 120):
    """Return whichever Indian observatory is actually reporting right now.

    Which station is live is not fixed. Alibag and Hyderabad both publish through
    the same service and either one can fall silent for days, so the caller is
    given whichever is currently up rather than one chosen in advance.

    Row count alone is not enough to decide that. A station that stopped reporting
    two days ago still returns a full day of good rows for the day before it
    stopped, and an earlier version of this function accepted exactly that and
    scored forecasts against a record that ended before they were issued. The last
    minute in the record has to be recent as well, and of the stations that pass,
    the freshest one wins.

    Returns:
        A pair of the code and its frame, or ``(None, None)`` if none report.
    """
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    best = (None, None)
    best_age = dt.timedelta(minutes=max_age_min)
    for code in codes:
        try:
            frame = fetch_recent(code, hours=hours)
        except Exception as exc:
            log.warning("%s is not reachable: %s", code, exc)
            continue
        if len(frame) < min_rows:
            log.warning("%s returned only %d usable minutes", code, len(frame))
            continue
        age = now - frame.index[-1].to_pydatetime()
        if age > best_age:
            log.warning("%s last reported %s, which is %.0f minutes ago",
                        code, frame.index[-1], age.total_seconds() / 60.0)
            continue
        best, best_age = (code, frame), age
    return best
