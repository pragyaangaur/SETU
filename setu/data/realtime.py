"""Live solar wind from the operational monitors at the first Lagrange point.

This module is the operational counterpart of ``setu.data.omni``. OMNI is a
carefully reprocessed archive that appears months after the fact, and it is what
the model is trained on. It is not something a control room can read. The feed here
is the real thing, updated every minute, free, and needing no account.

Both sources are kept behind the same shape of interface and produce the same
column names, so the same feature builder and the same trained model run on either
one. The archive is for learning and the feed is for operating.

Which spacecraft the feed is reporting is not fixed. NOAA publishes whichever
monitor is currently primary and marks the operational record with an active flag,
so this module filters on that flag and reports the source it found rather than
assuming one. At the time of writing the feed carries records from ACE, IMAP, and
SOLAR1.

The feed also publishes a propagated product that carries both the time a
measurement was taken and the time that parcel of solar wind is expected to reach
the Earth. That pair is the operational form of exactly what
``setu.data.omni.to_l1_time_base`` reconstructs from the archive, and
``compare_delay_estimate`` uses it to check the delay this project computes against
the one NOAA computes.

Data source: NOAA Space Weather Prediction Center real time solar wind products.
These are public and carry no access restriction.
"""

import logging

import numpy as np
import pandas as pd
import requests

from setu.config import EARTH_RADIUS_KM, L1_DISTANCE_RE, PHASE_FRONT_CORRECTION

log = logging.getLogger(__name__)

MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
PROPAGATED_URL = ("https://services.swpc.noaa.gov/products/geospace/"
                  "propagated-solar-wind-1-hour.json")

# Rows carrying a worse quality code than this are dropped. Zero is clean.
MAX_QUALITY_CODE = 0


def _records_to_frame(records, columns, only_active=True) -> pd.DataFrame:
    """Turn a list of feed records into a tidy frame at one minute cadence.

    The feed interleaves several spacecraft and is not sorted by time, so the
    records are filtered to the operational one, sorted, and de-duplicated before
    anything else happens.
    """
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("the feed returned no records")
    if only_active and "active" in frame:
        frame = frame[frame["active"].fillna(False).astype(bool)]
    if "overall_quality" in frame:
        quality = pd.to_numeric(frame["overall_quality"], errors="coerce")
        frame = frame[quality.fillna(0) <= MAX_QUALITY_CODE]
    if frame.empty:
        raise ValueError("no active good quality records in the feed")

    frame["time_tag"] = pd.to_datetime(frame["time_tag"], errors="coerce")
    frame = frame.dropna(subset=["time_tag"]).sort_values("time_tag")
    frame = frame.drop_duplicates(subset="time_tag", keep="last")
    frame = frame.set_index("time_tag")
    frame.index.name = "time"

    keep = {new: old for new, old in columns.items() if old in frame}
    out = pd.DataFrame(index=frame.index)
    for new, old in keep.items():
        out[new] = pd.to_numeric(frame[old], errors="coerce")
    out.attrs["sources"] = sorted(set(frame["source"].dropna())) if "source" in frame else []
    return out


def fetch_live(timeout: int = 30) -> pd.DataFrame:
    """Fetch the current solar wind in the layout the rest of the project expects.

    Returns:
        A frame at one minute cadence indexed by time, already on the clock of the
        spacecraft, because the feed has never applied any shift to it. That is the
        same clock the model is trained on, so the frame can go straight into
        ``setu.ml.features.build_features``.
    """
    magnetic = _records_to_frame(
        requests.get(MAG_URL, timeout=timeout).json(),
        {"b_total": "bt", "bx_gsm": "bx_gsm", "by_gsm": "by_gsm", "bz_gsm": "bz_gsm"},
    )
    plasma = _records_to_frame(
        requests.get(WIND_URL, timeout=timeout).json(),
        {"speed": "proton_speed", "density": "proton_density",
         "vx_gse": "proton_vx_gse"},
    )
    sources = sorted(set(magnetic.attrs.get("sources", []))
                     | set(plasma.attrs.get("sources", [])))

    joined = magnetic.join(plasma, how="outer").resample("1min").mean()
    joined = joined.interpolate(method="time", limit=5, limit_area="inside")

    # Dynamic pressure in nanopascal. The archive publishes this directly and the
    # feed does not, so it is computed here with the same expression.
    joined["pressure"] = 1.6726e-6 * joined["density"] * joined["speed"] ** 2

    # The feed carries no spacecraft position, so the nominal distance to the first
    # Lagrange point is used. That point wanders by a few percent over a year,
    # which moves the delay by about two minutes and is small next to the spread
    # the solar wind speed itself causes.
    joined["sc_x_re"] = L1_DISTANCE_RE
    if "vx_gse" not in joined or joined["vx_gse"].isna().all():
        joined["vx_gse"] = -joined["speed"]
    joined["timeshift"] = (L1_DISTANCE_RE * EARTH_RADIUS_KM * PHASE_FRONT_CORRECTION
                           / joined["speed"].where(joined["speed"] > 100.0))
    joined.attrs["sources"] = sources
    return joined


def current_conditions(timeout: int = 30) -> dict:
    """A one glance summary of what the solar wind is doing right now.

    This is what a duty engineer reads before anything else, so it is kept separate
    from the model and is readable on its own.
    """
    frame = fetch_live(timeout).ffill()
    frame = frame.dropna(subset=["b_total", "speed"])
    if frame.empty:
        raise RuntimeError("the feed returned nothing usable")

    latest = frame.iloc[-1]
    delay_min = float(latest["timeshift"]) / 60.0
    # Southward field is what drives reconnection at the front of the
    # magnetosphere, so it is called out separately from the field magnitude.
    southward = max(0.0, -float(latest["bz_gsm"]))
    return {
        "observed_at": str(frame.index[-1]),
        "sources": frame.attrs.get("sources", []),
        "b_total_nt": round(float(latest["b_total"]), 2),
        "bz_gsm_nt": round(float(latest["bz_gsm"]), 2),
        "southward_field_nt": round(southward, 2),
        "speed_km_s": round(float(latest["speed"]), 1),
        "density_per_cm3": round(float(latest["density"]), 2),
        "propagation_delay_min": round(delay_min, 1),
        "arrives_at": str(frame.index[-1] + pd.Timedelta(minutes=delay_min)),
    }


def compare_delay_estimate(timeout: int = 30) -> dict:
    """Check this project's propagation delay against the one NOAA publishes.

    NOAA's propagated product carries the time each measurement was taken and the
    time that parcel is expected to reach the Earth, so the difference between them
    is their own delay estimate. This project computes its delay from spacecraft
    distance and measured speed alone. Comparing the two is a free and continuous
    check on an assumption that the whole warning time rests on.
    """
    payload = requests.get(PROPAGATED_URL, timeout=timeout).json()
    header, rows = payload[0], payload[1:]
    frame = pd.DataFrame(rows, columns=header)
    observed = pd.to_datetime(frame["time_tag"], utc=True)
    arrives = pd.to_datetime(frame["propagated_time_tag"], utc=True)
    theirs = (arrives - observed).dt.total_seconds() / 60.0

    speed = pd.to_numeric(frame["speed"], errors="coerce")
    ours = (L1_DISTANCE_RE * EARTH_RADIUS_KM * PHASE_FRONT_CORRECTION / speed) / 60.0

    difference = (ours - theirs).dropna()
    return {
        "samples": int(len(difference)),
        "noaa_delay_min": round(float(theirs.median()), 1),
        "our_delay_min": round(float(ours.median()), 1),
        "median_difference_min": round(float(difference.median()), 1),
        "worst_difference_min": round(float(difference.abs().max()), 1),
    }
