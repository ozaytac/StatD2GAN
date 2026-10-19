# StatD2GAN v3 release: post-freeze entry point.
#
# GNU make reads this file before ./Makefile. ./Makefile is the frozen v3 campaign pipeline (listed in
# MANIFEST.json, kept byte-identical); its targets stay available as `make -f Makefile <target>`.
# Nothing here writes to a frozen file: every target works in build/ or fills files that are absent.
#
#   make all        verify + tables + paper              CPU, about a minute; no weights, no training
#   make all-heavy  all + fetch-weights + distance       about 5 GB download; GPU recommended
#   make help       the full list
SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
.DEFAULT_GOAL := all
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
LOC ?=
SEED ?=
CONFIRM ?= no
WITH_TORCH ?= 0
INSTALL ?= 0

.PHONY: help env dirs verify smoke tables distance paper train fetch-weights data all all-heavy ci clean-build

help:
	@echo "make env            create .venv (requirements-postfreeze.txt; WITH_TORCH=1 adds torch)"
	@echo "make verify         sha256 of the frozen layer, the post-freeze layer and any weights   CPU, seconds"
	@echo "make tables         regenerate post-freeze numbers, tables, figures; macro_check; lint   CPU, ~1 min"
	@echo "make paper          latexmk paper/main.tex (skipped with a message if TeX/elsarticle absent)"
	@echo "make smoke [LOC=]   preprocess + torch unit tests + 2-epoch end-to-end run (one location) CPU/GPU, minutes"
	@echo "make data [INSTALL=1]  ERA5: verify pinned files or re-download from the CDS"
	@echo "make fetch-weights  download and verify the 275 .pt files (about 5 GB, Zenodo)"
	@echo "make distance [LOC=]  recompute distance_cal_train.csv from the weights               GPU recommended"
	@echo "make train LOC=.. CONFIRM=yes  re-run the campaign in build/train                       GPU-HOURS"
	@echo "make all = verify tables paper;  make all-heavy = all fetch-weights distance (no train)"
	@echo "frozen pipeline: make frozen-<target> (= make -f Makefile <target>, after make dirs)"

env:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-postfreeze.txt
	@if [ "$(WITH_TORCH)" = "1" ]; then .venv/bin/pip install "torch==2.11.0"; \
	  else echo "torch not installed (make env WITH_TORCH=1 for smoke/distance/train)"; fi

# output directories of the frozen code (git does not keep empty directories; tools/_common.py)
dirs:
	$(PY) -c "import sys; sys.path.insert(0, 'tools'); from _common import ensure_dirs; ensure_dirs()"

frozen-%: dirs
	$(MAKE) -f Makefile $*

verify:
	$(PY) tools/verify.py

tables:
	$(PY) tools/repro_tables.py

paper:
	$(PY) tools/build_paper.py

smoke: dirs
	$(PY) tools/smoke.py $(if $(LOC),--loc $(LOC)) $(if $(SEED),--seed $(SEED))

data: dirs
	$(PY) tools/data.py $(if $(filter 1,$(INSTALL)),--install)

fetch-weights:
	$(PY) tools/fetch_weights.py

distance: dirs
	$(PY) tools/repro_distance.py $(if $(LOC),--loc $(LOC))

train: dirs
	$(PY) tools/train.py --confirm $(CONFIRM) $(if $(LOC),--loc $(LOC)) $(if $(SEED),--seed $(SEED))

all: verify tables paper

all-heavy: all fetch-weights distance

# continuous integration (no torch, no weights, no TeX): see .github/workflows/ci.yml
ci: verify tables
	$(PY) tests/test_distance_extra.py

clean-build:
	rm -rf build
