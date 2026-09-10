"""Rebuild the daily verification record from the history of the ledger.

The service folds rows older than two days out of the ledger and into
docs/data/verification.json, but the workflow only committed the ledger. Each run
starts from a fresh checkout, so every fold was written into a file that was
thrown away when the run ended, and the closed days were kept nowhere except in
the history of the ledger itself.

Every row the service ever wrote is still in that history, because every run
committed the ledger. This reads each row back out, merges the outcomes attached
to it across commits, and folds every row that is no longer in the current ledger
through the same fold() the service uses. A row still in the ledger is left
alone, so no forecast is counted in both files.

    python scripts/rebuild_rollup.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from setu.nowcast import LEDGER_PATH, ROLLUP_PATH, fold, load_ledger  # noqa: E402

LEDGER = "docs/data/ledger.json"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def main():
    rows = {}
    for commit in git("log", "HEAD", "--format=%h", "--reverse", "--", LEDGER).split():
        payload = json.loads(git("show", f"{commit}:{LEDGER}"))
        for entry in payload.get("entries", []):
            known = rows.setdefault(entry["issued_at"], entry)
            for horizon, record in entry["horizons"].items():
                if record.get("observed_dbdt") is not None:
                    known["horizons"][horizon] = record

    current = {e["issued_at"] for e in load_ledger(LEDGER_PATH)}
    closed = [rows[k] for k in sorted(rows) if k not in current]
    rollup = fold(closed, {"days": {}})
    ROLLUP_PATH.write_text(json.dumps(rollup, indent=1, default=float))
    live = sum(day["live"] for day in rollup["days"].values())
    backfilled = sum(day["backfilled"] for day in rollup["days"].values())
    print(f"folded {len(closed)} rows ({live} live, {backfilled} backfilled) "
          f"into {ROLLUP_PATH.relative_to(ROOT)}, leaving {len(current)} in the ledger")


if __name__ == "__main__":
    main()
