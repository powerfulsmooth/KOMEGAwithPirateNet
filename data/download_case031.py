#!/usr/bin/env python3
"""
Download the full ERCOFTAC Classic Collection Case031 dataset
(Backward-Facing Step, DNS by Le, Moin & Kim 1997, Re_h = 5100).

Source : http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case031
License: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike)

Usage:
    python download_case031.py
Downloads every file into ./raw/case031/ . Re-run is idempotent (skips
files already present unless --force is given).

NOTE: The 6 mean-velocity / Reynolds-stress profile files (x-*.dat) plus
stat-inf.dat are the only files the PINN actually needs. The 30 budget
files (rs11/rs22/rs33/rs12/rskk) are optional (Reynolds-stress and k
budgets, useful if you want dissipation epsilon for an exact omega BC).
"""
import os
import sys
import urllib.request

BASE = ("http://cfd.mace.manchester.ac.uk/ercoftac/lib/exe/"
        "fetch.php?media=cdata:case031:")

# Streamwise grid index  ->  ERCOFTAC x/h  (Pioch x/h = ERCOFTAC x/h + 3)
STATIONS = {
    181: -2.99,   # Pioch x/h = 0.00  (inlet)
    360:  4.00,   # Pioch x/h = 7.00
    411:  6.00,   # Pioch x/h = 9.00
    513:  9.98,   # Pioch x/h = 12.98
    641: 14.98,   # Pioch x/h = 17.98
    744: 19.00,   # Pioch x/h = 22.00 (outlet)
}

# Essential = mean velocity + Reynolds stresses + integral stats
ESSENTIAL = [f"x-{i}.dat" for i in STATIONS] + ["stat-inf.dat", "readme.txt"]
# Optional = full budgets (epsilon etc.)
BUDGETS = [f"{p}-{i}.dat" for i in STATIONS
           for p in ("rs11", "rs22", "rs33", "rs12", "rskk")]


def fetch(fname, outdir, force=False):
    dst = os.path.join(outdir, fname)
    if os.path.exists(dst) and not force:
        print(f"  skip  {fname}")
        return
    url = BASE + fname
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        with open(dst, "wb") as f:
            f.write(data)
        print(f"  ok    {fname}  ({len(data)} B)")
    except Exception as e:  # noqa
        print(f"  FAIL  {fname}: {e}", file=sys.stderr)


def main():
    force = "--force" in sys.argv
    full = "--full" in sys.argv  # also pull the 30 budget files
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "raw", "case031")
    os.makedirs(outdir, exist_ok=True)
    files = list(ESSENTIAL) + (BUDGETS if full else [])
    print(f"Downloading {len(files)} files -> {outdir}")
    for fname in files:
        fetch(fname, outdir, force=force)
    print("Done. (use --full for the 30 budget files, --force to overwrite)")


if __name__ == "__main__":
    main()
