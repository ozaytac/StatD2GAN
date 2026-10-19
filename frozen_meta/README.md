# StatD2GAN: reproducibility package (v3)

Raw ERA5 → every table, figure and number in the paper, one command: `make all`.

```
data/raw/*.csv (pinned, sha256)  ──►  preprocess ──►  train (GPU) ──► collect ──► analyze (CSV only) ──► figures / tables ──► archive
                                       data/processed   runs/*/*.json+.pt   results/*.csv         figures/ generated/*.tex     dist/*.tar.gz
```

| Layer | Runs on | Reads | Writes |
|---|---|---|---|
| train | GPU | `data/processed`, `protocol.yaml` | `runs/<loc>/<arm>_<seed>.{json,pt}` (once) |
| collect | CPU | `runs/*/*.json` | `results/{abl_perseed,baselines,projection,seq_level}.csv`, one script each |
| analyze | CPU | `results/*.csv` only | `results/{ks_ratio,summary,pct_change,wilcoxon,bootstrap,baseline_tests,run_noise}.csv`, `decisions.json` |
| figures / tables | CPU | `results/*.csv` | `figures/*.{pdf,png}`, `generated/{tables,numbers}.tex` |

Rules enforced by code: `protocol.yaml` is frozen (`make freeze`, sha256 lock; scripts refuse to run if it changes);
every intermediate is written once (`write_csv`, run JSONs, `decisions.json`); every derived CSV starts with a
`# produced by scripts/…` header; the paper only `\input`s `generated/*.tex` (`make lint-paper` rejects typed statistics).

## Order of work
1. Review `protocol.yaml` (DRAFT). Set `budget.gpu_hours`, `status: FROZEN`, commit, `make freeze`.
2. `make smoke` (probe of all 9 arms + one real run per baseline → `runs/budget.json`, ladder applied from timings only).
3. `make pilot` → gate is computed by code (`runs/pilot_gate.json`) → `make train` (resumable) → `make all`.
Without a GPU: unpack the archive's `runs/` and run `make analyze figures tables`.

## Checks
`make test-fast` (CPU; includes an end-to-end pipeline test on synthetic run records), `make test` (adds torch and a tiny training run).

## Reproducibility statement
Analysis layers are bit-reproducible from `runs/*.json`. GAN training on GPU is **not** bitwise reproducible
(cuDNN/AMP nondeterminism); the saved generator weights (`runs/*/*.pt`) reproduce each run's evaluation draws exactly
(dedicated RNG streams), and a re-training reproduces results statistically, not digit for digit.
The KS reference ("noise floor") is the real train vs held-out KS sup-distance: a practical reference, not a mathematical lower bound.

Data: ERA5 (Copernicus, Copernicus licence; see `data/README`). Code: MIT.

## Running one location per session (Colab)

`StatD2GAN_v3_lokasyon_run.ipynb` runs one (location, seed) per session, e.g. `ankara, 42` (11 runs); raw CSVs are read from Drive `data/<loc>.csv` and writes to `<loc>_sonuc/`
(`kosumlar/<loc>_<arm>_seed<seed>.json|.pt`, `ozet/<loc>_seed<seed>_ozet.csv`, `ozet/<loc>_tum_kosumlar.jsonl`, `loglar/`, `env/`).
The frozen protocol and `budget.json` live in `ortak/`; later sessions verify the same sha256.
After all five locations: `LOCATION = 'birlestir'` runs collect, analyse, figures, tables, archive.
Direct CLI: `python scripts/run_campaign.py --stage {pilot,full,baselines} --loc <loc> --seed <seed>`.
