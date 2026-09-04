"""Check every storm in the catalogue against the archive and correct it.

The catalogue carries a minimum SYM/H for each event to give a sense of scale.
Quoting those from memory is how wrong numbers get into a presentation, so this
script measures each one from the OMNI record and rewrites the file where it
disagrees. Run it after adding an event.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from setu.data.omni import fetch_range  # noqa: E402
from setu.data.storms import EVENTS  # noqa: E402

SOURCE = ROOT / "setu" / "data" / "storms.py"


def main(apply_changes=True):
    text = SOURCE.read_text()
    changed = 0
    for event in EVENTS:
        try:
            frame = fetch_range(event.start, event.end)
        except Exception as exc:
            print(f"{event.key}: could not fetch, {exc}")
            continue
        if "sym_h" not in frame or frame["sym_h"].isna().all():
            print(f"{event.key}: no SYM/H in the record")
            continue
        measured = int(round(float(frame["sym_h"].min())))
        flag = "same" if measured == event.min_sym_h else f"was {event.min_sym_h}"
        print(f"{event.key:12s} {event.name:28s} measured {measured:6d} nT  ({flag})")
        if measured != event.min_sym_h and apply_changes:
            pattern = re.compile(
                r'(StormEvent\("' + re.escape(event.key) + r'".*?\n\s*)'
                + str(event.min_sym_h) + r'(, ")', re.DOTALL)
            new_text, count = pattern.subn(r"\g<1>" + str(measured) + r"\g<2>", text, count=1)
            if count:
                text = new_text
                changed += 1
            else:
                print(f"  could not rewrite {event.key} automatically")
    if apply_changes and changed:
        SOURCE.write_text(text)
        print(f"\nupdated {changed} entries in {SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
