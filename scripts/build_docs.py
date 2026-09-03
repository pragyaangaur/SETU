"""Bundle the exported results into one JavaScript file for the dashboard.

The dashboard is a single static page with no build step and no dependencies, so
that it can be served by GitHub Pages and also opened straight from the file system
during a demonstration where the network cannot be trusted. A browser refuses to
read a local JSON file over the file protocol, so the data is written as a script
that assigns to a global instead.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "data"
OUTPUT = ROOT / "docs" / "data.js"


def main():
    payload = {}
    for path in sorted(DATA_DIR.glob("*.json")):
        payload[path.stem] = json.loads(path.read_text())
    if not payload:
        print("no exported json found, run the cli commands first", file=sys.stderr)
        return 1
    OUTPUT.write_text("window.SETU_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"bundled {len(payload)} files into {OUTPUT} ({size_kb:.0f} kB)")
    for key in payload:
        print(f"  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
