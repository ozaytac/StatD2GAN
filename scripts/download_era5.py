"""Optional: re-download ERA5 hourly single-level point series into data/raw_redownload/{loc}.csv.
Requires a CDS account (~/.cdsapirc). Then `python scripts/verify_raw.py` compares against the
pinned raw files. NOT part of `make all` (needs credentials + hours of queue time)."""
import sys, zipfile, tempfile
from pathlib import Path
import cdsapi, pandas as pd, xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g.config import P, ROOT

D = P["data"]; out = ROOT / "data/raw_redownload"; out.mkdir(exist_ok=True)
c = cdsapi.Client()
for loc, (lat, lon) in D["locations"].items():
    if (out / f"{loc}.csv").exists():
        continue
    frames = []
    for y in range(D["years"][0], D["years"][1] + 1):        # one request per year (CDS size limits)
        tgt = Path(tempfile.mkdtemp()) / f"{loc}_{y}.zip"
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "variable": D["era5_vars"], "year": str(y),
            "month": [f"{m:02d}" for m in range(1, 13)], "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)], "area": [lat, lon, lat, lon],
            "data_format": "netcdf", "download_format": "zip"}, str(tgt))
        with zipfile.ZipFile(tgt) as z:
            z.extractall(tgt.parent)
        ds = xr.merge([xr.open_dataset(f) for f in sorted(tgt.parent.glob("*.nc"))])
        frames.append(ds.isel(latitude=0, longitude=0).to_dataframe().reset_index())
    df = pd.concat(frames).rename(columns={"time": "valid_time"}) if "time" in frames[0] else pd.concat(frames)
    df = df[["valid_time", "u10", "v10", "d2m", "t2m", "msl", "sp", "latitude", "longitude"]].sort_values("valid_time")
    df.to_csv(out / f"{loc}.csv", index=False)
