"""Measure the two geometry constants the warning time depends on.

The forecast horizon is only real warning time if the propagation delay from the
spacecraft to the Earth is right. This script measures the two numbers that go into
it, straight from the OMNI archive, and prints values to paste into
``setu.config``. Run it if the archive is extended or if the monitoring spacecraft
changes.
"""

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Column positions in the OMNI one minute record, zero based.
TIMESHIFT, SPEED, SC_X = 9, 21, 31
EARTH_RADIUS_KM = 6371.0

# A spacecraft outside this band was not near the first Lagrange point when it
# reported, so its geometry says nothing about the monitor this project assumes.
L1_BAND_RE = (150.0, 280.0)


def main():
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "omni_min*.asc")))
    if not files:
        print("no OMNI files cached, run a training or a fetch first")
        return 1

    effective, physical = [], []
    for path in files:
        raw = np.loadtxt(path)
        shift, speed, x = raw[:, TIMESHIFT], raw[:, SPEED], raw[:, SC_X]
        good = ((shift > 300) & (shift < 7200) & (speed > 100) & (speed < 3000)
                & (x > L1_BAND_RE[0]) & (x < L1_BAND_RE[1]))
        effective.append(shift[good] * speed[good] / EARTH_RADIUS_KM)
        physical.append(x[good])

    effective = np.concatenate(effective)
    physical = np.concatenate(physical)
    ratio = float(np.median(effective) / np.median(physical))

    print(f"months read: {len(files)}")
    print(f"minutes used: {len(effective):,}")
    print(f"L1_DISTANCE_RE = {np.median(physical):.1f}")
    print(f"PHASE_FRONT_CORRECTION = {ratio:.3f}")
    print()
    print("The correction is below one because a tilted disturbance front reaches")
    print("the Earth sooner than a flat front travelling the same distance would.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
