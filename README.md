# StatD2GAN v2.0.0: reproducibility package

Code, frozen results and post-freeze reporting layer for *Marginal Calibration Masks
Generator Differences in Held-Out Evaluation of Multivariate Time Series, with an Application to
ERA5 Weather Sequences* (Mustafa Özaytaç and Özge Karadağ Ataş), preprint arXiv:2609.33761.

```
git clone https://github.com/ozaytac/StatD2GAN && cd StatD2GAN
make env          # optional: .venv with pinned packages (WITH_TORCH=1 adds torch)
make all          # verify + tables + paper; CPU, about a minute; no weights, no training
```

The v3.0.0 campaign (275 runs: 9 arms x 5 ERA5 locations x 5 seeds, plus RCGAN and TimeGAN
re-implementations x 25) was frozen before any result was seen (protocol sha256
`2d9a3b9d210df5b2...`, `MANIFEST.json`). This release (v2.0.0 of the repository) changes none of it: it adds a
post-freeze reporting layer, a one-command build and the generator weights. The labels v3.0.0 (frozen
campaign) and v3.1.0 (post-freeze layer) are internal development versions and appear in file headers and
manifests; the public release number is v2.0.0. The v3.0.0 README is kept unchanged as
`frozen_meta/README.md`; it describes the frozen campaign pipeline (`make frozen-<target>`).

## First commands after `git clone`

```
make env WITH_TORCH=1     # .venv with requirements-postfreeze.txt and torch 2.11.0
make verify               # integrity; passes with "weights not present" until the weights are fetched
make tables               # rebuild and check every post-freeze number, table and figure
make smoke LOC=ankara     # preprocessing, torch unit tests and a 2-epoch end-to-end run
make fetch-weights        # the 275 generator weight files (about 5 GB), for make distance
```

## Targets

| Target | Produces / checks | Needs | Hardware, time |
|---|---|---|---|
| `make verify` | sha256 of every frozen file (MANIFEST.json), every post-freeze file (MANIFEST_post_freeze.json) and any weights present. Passes with a warning when only the weights are absent | Python 3 | CPU, seconds |
| `make tables` | regenerates `generated/numbers_extra.tex`, `tables_pf.tex`, `table_dist_pf.tex` and the `figures/*_pf.*` in `build/tables/`, requires the `.tex` files to be byte-identical to the committed ones, runs `tools/macro_check.py` and `tools/lint_paper.py` | `requirements-postfreeze.txt` | CPU, under a minute |
| `make paper` | `build/paper/main.pdf` from `paper/main.tex` | latexmk, pdfLaTeX, `elsarticle`, `lm`; skipped with a message otherwise | CPU, under a minute |
| `make smoke [LOC=ankara]` | frozen preprocessing of LOC (the `.npz` is checked), frozen torch unit tests (incl. a 2-epoch run on synthetic data), then one 2-epoch end-to-end run (`full_model`, seed 42: training, calibration, projection, full evaluation) in `build/smoke/` | torch | CPU or GPU, minutes |
| `make data` | checks `data/raw/*.csv` against the frozen `data/raw.sha256` (or re-downloads from the CDS if absent) | CDS account only for a download | seconds |
| `make fetch-weights` | the 275 generator weight files `runs/<loc>/*.pt` (five archives, 5.03 GB), each checked against `weights_manifest.json` and MANIFEST.json | network | about 5 GB |
| `make distance [LOC=..]` | recomputes `results_extra/distance_cal_train.csv` (paper, Appendix B) from the weights in `build/distance/` and compares it with the registered CSV | weights, torch | GPU recommended, a few minutes per location |
| `make train LOC=.. CONFIRM=yes` | re-runs the frozen campaign in `build/train/`; without `CONFIRM=yes` it only prepares `build/train/` (raw-data check, preprocessing) and stops | torch, GPU | **GPU-hours** (the frozen 275 runs took 34 GPU-hours of training) |
| `make all` | `verify tables paper` | | CPU, about a minute |
| `make all-heavy` | `all fetch-weights distance` (no training) | see above | GPU recommended |
| `make frozen-<target>` | a target of the frozen `Makefile` (e.g. `make frozen-test-fast`) | as that target | |

GNU make reads `GNUmakefile` before the frozen `Makefile`, so `make all` never starts training.
Nothing writes to a frozen file; every target works in `build/` or fills files that are absent.
`.pytest_cache/` is listed in the frozen MANIFEST.json but is pytest's own cache; it is not in git and
`make verify` treats it as optional.

## Layers

```
frozen v3.0.0 (MANIFEST.json, never modified)            post-freeze (MANIFEST_post_freeze.json)
  protocol.yaml, protocol.lock, Makefile, src/, scripts/   scripts/{report,fig1,figs,table_seq,table_pilot,distance}_extra.py, post_freeze.py
  results/*.csv, results/decisions.json                    results_extra/distance_cal_train.csv
  generated/numbers.tex, generated/tables.tex              generated/numbers_extra.tex, tables_pf.tex, table_dist_pf.tex
  figures/fig*.{pdf,png}                                   figures/*_pf.*, figures/figs_extra_record.json
  runs/**/*.json, runs/_env/, logs/                        paper/main.tex, tools/, GNUmakefile, weights_manifest.json
  runs/**/*.pt (Zenodo archives, make fetch-weights)       requirements-*.txt, .github/workflows/ci.yml
  data/raw/*.csv, data/raw.sha256                          README.md, CITATION.cff, .zenodo.json (superseded;
                                                           v3.0.0 originals in frozen_meta/)
```

Every number in the paper comes from `generated/numbers.tex`, `generated/tables.tex` (frozen) or
`generated/numbers_extra.tex` (post-freeze). The post-freeze scripts read only frozen outputs, with
two exceptions: `scripts/distance_extra.py` re-runs inference from the stored weights with the frozen
seeds and frozen code (no training), and `scripts/report_extra.py` reads its output.

## Devices and versions

* Training (v3.0.0 campaign): torch 2.11.0+cu128, CUDA 12.8, Python 3.13.15, NVIDIA RTX PRO 6000
  Blackwell (run environment record `runs/_env/584162a1e69e.json`; `requirements-frozen-env.txt`).
* Distance recomputation (`make distance`, Appendix B): torch 2.11.0+cu130, CUDA 13.0, same GPU
  model. Every one of the 250 calibrated runs reproduced the stored raw and calibrated KS with a
  maximum difference of 0 (gate tolerance 1e-4).
* On other devices the noise draws can differ, so agreement is approximate at best; if any run
  fails the gate, the script stops and produces no numbers. Bitwise agreement on other devices is
  not claimed.
* Post-freeze reporting (`make tables`): `requirements-postfreeze.txt` (Python 3.13).

## ERA5 data

`data/raw/*.csv` are hourly ERA5 single-level point series, 2010 to 2024, at the five grid points in
`protocol.yaml`, extracted from the Copernicus Climate Data Store and unchanged since the freeze
(`data/raw.sha256`). They are redistributed under the dataset's licence and are not covered by the
code licence.

* Licence: CC-BY, as listed on the CDS catalogue entry
  (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels); Creative Commons
  Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/.
* Citation: Hersbach, H., Bell, B., Berrisford, P., et al. (2023): ERA5 hourly data on single levels
  from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS),
  DOI: 10.24381/cds.adbb2d47.
* Acknowledgement, in the form required by ECMWF
  (https://confluence.ecmwf.int/x/srnICw, "How to acknowledge and cite a Climate Data Store (CDS)
  catalogue entry and the data published as part of it"): Contains modified Copernicus Climate
  Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any
  use that may be made of the Copernicus information or data it contains.
* Modifications: extraction of the five grid-point series and conversion to CSV; the derived
  features and windows are computed by `src/s2g/data.py`.

## Weights

`weights_manifest.json` lists the 275 `.pt` files (generator `state_dict`, fp32; 5,028,671,257 bytes)
with size and sha256 (identical to MANIFEST.json) and the five per-location archives
`weights_<loc>.tar` stored in the companion Zenodo record
[10.5281/zenodo.23480462](https://doi.org/10.5281/zenodo.23480462) (Zenodo allows 100 files per record;
`make fetch-weights` downloads and verifies them). They are not in git and not in the code archive.

## Citation and licence

Cite the article (`CITATION.cff`, preferred citation) and this software release. Code: MIT
(`LICENSE`). ERA5 data: see above.
