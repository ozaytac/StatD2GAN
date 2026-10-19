## Data licence note

`raw/*.csv` are hourly ERA5 single-level reanalysis point series (2010-2024) at the grid points listed in
`protocol.yaml`, extracted from the Copernicus Climate Data Store. They are **not** covered by the MIT licence of the code;
cite Copernicus/ECMWF and follow the Copernicus Licence. Integrity: `raw.sha256`. `scripts/download_era5.py`
re-creates them from CDS; `scripts/verify_raw.py --redownload` compares numerically.
