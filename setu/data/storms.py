"""Catalogue of the geomagnetic storms used for training and for evaluation.

Storms are rare, so a model trained on a random slice of time would see almost
nothing worth learning. The catalogue below names specific events, splits them into
training and held out sets, and records why each one is interesting. Keeping the
split explicit and by event, rather than by random sample, is what stops the model
from being scored on minutes that sit next to minutes it was trained on.

Minimum SYM/H values are from the OMNI archive and are quoted to give a sense of
scale. A storm below minus 250 nT is severe by any definition.
"""

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class StormEvent:
    key: str
    name: str
    start: dt.date
    end: dt.date
    min_sym_h: int
    role: str
    note: str


EVENTS = [
    StormEvent("2024-05-10", "Gannon storm", dt.date(2024, 5, 8), dt.date(2024, 5, 14),
               -518, "test",
               "The largest storm in more than twenty years. Held out entirely, so "
               "every number reported for it is a genuine forecast."),
    StormEvent("2003-10-29", "Halloween storm", dt.date(2003, 10, 28), dt.date(2003, 11, 3),
               -432, "test",
               "The reference extreme event. Held out as a second independent test."),
    StormEvent("2015-03-17", "St Patrick's Day storm", dt.date(2015, 3, 16), dt.date(2015, 3, 21),
               -234, "train",
               "The best studied storm at Indian observatories in the modern record."),
    StormEvent("2017-09-07", "September 2017 storm", dt.date(2017, 9, 6), dt.date(2017, 9, 12),
               -146, "train", "Two active region flares followed by a fast stream."),
    StormEvent("2001-03-31", "March 2001 storm", dt.date(2001, 3, 29), dt.date(2001, 4, 4),
               -387, "train", "Solar cycle 23 maximum, very fast solar wind."),
    StormEvent("2004-11-07", "November 2004 storm", dt.date(2004, 11, 6), dt.date(2004, 11, 12),
               -373, "train", "Repeated coronal mass ejection arrivals over several days."),
    StormEvent("2005-05-15", "May 2005 storm", dt.date(2005, 5, 14), dt.date(2005, 5, 18),
               -305, "train", "Sharp sudden commencement with a strong southward field."),
    StormEvent("2000-07-15", "Bastille Day storm", dt.date(2000, 7, 14), dt.date(2000, 7, 18),
               -301, "train", "Classic fast coronal mass ejection from an X class flare."),
    StormEvent("2023-04-23", "April 2023 storm", dt.date(2023, 4, 22), dt.date(2023, 4, 26),
               -213, "train", "Rise of solar cycle 25, modern instrumentation."),
    StormEvent("2023-11-05", "November 2023 storm", dt.date(2023, 11, 4), dt.date(2023, 11, 8),
               -175, "train", "Moderate event, useful for calibrating the lower threshold."),
    StormEvent("2021-11-04", "November 2021 storm", dt.date(2021, 11, 3), dt.date(2021, 11, 7),
               -118, "train", "Weak event, included so the model learns where nothing happens."),
    StormEvent("2022-04-14", "April 2022 storm", dt.date(2022, 4, 13), dt.date(2022, 4, 17),
               -100, "train", "Weak event, same purpose as the one above."),
    StormEvent("2018-08-25", "August 2018 storm", dt.date(2018, 8, 24), dt.date(2018, 8, 29),
               -174, "train", "Slow coronal mass ejection with a long southward field."),
    StormEvent("2016-10-12", "October 2016 storm", dt.date(2016, 10, 11), dt.date(2016, 10, 16),
               -104, "train", "Coronal hole stream rather than an ejection."),
    StormEvent("2012-03-09", "March 2012 storm", dt.date(2012, 3, 7), dt.date(2012, 3, 13),
               -148, "train", "Large flare with a fast but poorly connected ejection."),
    StormEvent("2006-12-14", "December 2006 storm", dt.date(2006, 12, 13), dt.date(2006, 12, 17),
               -211, "train", "Late cycle 23 event with a very clean sudden commencement."),
]

BY_KEY = {e.key: e for e in EVENTS}


def training_events():
    return [e for e in EVENTS if e.role == "train"]


def test_events():
    return [e for e in EVENTS if e.role == "test"]


def get_event(key: str) -> StormEvent:
    try:
        return BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown storm {key!r}, available: {sorted(BY_KEY)}") from None
