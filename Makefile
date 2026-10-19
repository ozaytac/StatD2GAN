# One command: `make all` (GPU box) builds every table, figure and number from the pinned raw ERA5 files.
# Layers: data -> train (GPU) -> collect/analyze (CPU, CSV only) -> figures/tables (CPU) -> archive.
# Every intermediate is written once; a step is redone only after its explicit clean-* target.
SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
PY ?= python
LOG = mkdir -p logs && $(PY)
tee = 2>&1 | tee logs/$(1).log

.PHONY: all test test-fast train analyze figures tables archive lint-paper smoke pilot verify-era5 \
        clean-collect clean-analysis clean-figures
all: archive

# ---- layer 0: protocol, environment, data ---------------------------------------------------
protocol.lock: protocol.yaml
	$(PY) scripts/freeze.py
runs/_env/.stamp: protocol.lock
	$(PY) scripts/env_snapshot.py && touch $@
data/.raw_ok: data/raw.sha256
	$(PY) scripts/verify_raw.py && touch $@
data/processed/.done: data/.raw_ok protocol.lock
	$(LOG) scripts/preprocess.py $(call tee,preprocess) && touch $@
results/ks_floor.csv: data/processed/.done
	$(LOG) scripts/make_floor.py $(call tee,make_floor)

# ---- layer 1: train (GPU) -------------------------------------------------------------------
runs/budget.json: data/processed/.done runs/_env/.stamp
	$(LOG) scripts/smoke.py $(call tee,smoke)
smoke: runs/budget.json
runs/.pilot_done: runs/budget.json
	$(LOG) scripts/run_campaign.py --stage pilot $(call tee,pilot)
pilot: runs/.pilot_done
runs/pilot_gate.json: runs/.pilot_done
	$(LOG) scripts/pilot_gate.py $(call tee,pilot_gate)
runs/.full_done: runs/pilot_gate.json
	$(LOG) scripts/run_campaign.py --stage full $(call tee,campaign_full)
runs/.baselines_done: runs/budget.json
	$(LOG) scripts/run_campaign.py --stage baselines $(call tee,campaign_baselines)
train: runs/.full_done runs/.baselines_done

# ---- layer 2: collect run JSONs into CSVs, one named script per CSV; analyze (CSV only) -----
results/abl_perseed.csv: runs/.full_done
	$(LOG) scripts/make_abl_perseed.py $(call tee,make_abl_perseed)
results/baselines.csv: runs/.baselines_done
	$(LOG) scripts/make_baselines.py $(call tee,make_baselines)
results/projection.csv: runs/.full_done
	$(LOG) scripts/make_projection.py $(call tee,make_projection)
results/seq_level.csv: runs/.full_done runs/.baselines_done data/processed/.done
	$(LOG) scripts/make_seq_level.py $(call tee,make_seq_level)
results/decisions.json: results/abl_perseed.csv results/baselines.csv results/projection.csv results/seq_level.csv results/ks_floor.csv
	$(LOG) scripts/analyze.py $(call tee,analyze)
analyze: results/decisions.json

# ---- layer 3: figures and tables (CPU) ------------------------------------------------------
figures/fig4_variance_decomposition.pdf: results/decisions.json
	$(LOG) scripts/figures.py $(call tee,figures)
figures: figures/fig4_variance_decomposition.pdf
generated/tables.tex: results/decisions.json runs/pilot_gate.json
	$(LOG) scripts/report.py $(call tee,report)
tables: generated/tables.tex

# ---- checks, archive -----------------------------------------------------------------------
test:
	$(PY) -m pytest -q
test-fast:
	$(PY) -m pytest -q -m "not slow"
lint-paper:
	$(PY) scripts/lint_paper.py paper/main.tex
archive: figures tables
	$(PY) scripts/archive.py
verify-era5:
	$(PY) scripts/download_era5.py && $(PY) scripts/verify_raw.py --redownload

# ---- explicit redo (results are never silently overwritten) --------------------------------
clean-collect:
	rm -f results/abl_perseed.csv results/baselines.csv results/projection.csv results/seq_level.csv
clean-analysis:
	rm -f results/{ks_ratio,summary,pct_change,wilcoxon,bootstrap,baseline_tests,run_noise}.csv results/decisions.json generated/*.tex
clean-figures:
	rm -f figures/*
