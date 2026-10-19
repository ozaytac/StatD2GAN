#!/usr/bin/env python3
"""make data: obtain the ERA5 input series.

Contains modified Copernicus Climate Change Service information [2026]. Neither the European
Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or
data it contains. Source: Hersbach, H. et al. (2023): ERA5 hourly data on single levels from 1940
to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS),
DOI: 10.24381/cds.adbb2d47. Use is governed by the licence shown on the CDS dataset page
(CC BY 4.0 at the time of writing; check the current terms before redistributing).

1. If data/raw/*.csv are present they are checked against data/raw.sha256 (frozen); done.
2. Otherwise the frozen scripts/download_era5.py re-downloads the five point series from the CDS
   into data/raw_redownload/ (needs a CDS account and ~/.cdsapirc; queue time can be hours).
   A re-download is not byte-identical to the pinned files (CSV formatting, possible ERA5 revisions),
   so the frozen pipeline's sha check (Makefile: data/.raw_ok) will refuse it. scripts/verify_raw.py
   --redownload compares numerically when the pinned files are available. With --install the
   re-downloaded files are copied to data/raw/ so that `make smoke` and `make distance` can run;
   results obtained that way are approximate and are not the frozen inputs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, ensure_dirs, run  # noqa: E402

PY = sys.executable


def main(install: bool) -> int:
    ensure_dirs(ROOT)
    raws = sorted((ROOT / "data" / "raw").glob("*.csv"))
    if raws:
        r = subprocess.run([PY, "scripts/verify_raw.py"], cwd=ROOT)
        print("data: pinned ERA5 files", "verified" if r.returncode == 0 else "DO NOT MATCH data/raw.sha256")
        return r.returncode
    if not (Path.home() / ".cdsapirc").exists():
        sys.exit("data/raw/*.csv absent and no ~/.cdsapirc: register at https://cds.climate.copernicus.eu, "
                 "accept the ERA5 licence, create ~/.cdsapirc, then re-run `make data`")
    run([PY, "scripts/download_era5.py"], ROOT)
    if install:
        (ROOT / "data" / "raw").mkdir(exist_ok=True)
        for f in sorted((ROOT / "data" / "raw_redownload").glob("*.csv")):
            shutil.copy2(f, ROOT / "data" / "raw" / f.name)
        print("data: re-downloaded files installed in data/raw/ (NOT the pinned frozen inputs; `make verify` reports them)")
    else:
        print("data: re-downloaded into data/raw_redownload/; add INSTALL=1 to copy them into data/raw/")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    sys.exit(main(ap.parse_args().install))
