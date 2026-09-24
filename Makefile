PY ?= python

.PHONY: install data train figures report all test lint clean

install:            ## install the package + dev tools
	$(PY) -m pip install -e ".[dev]"

data:               ## download -> ingest -> resolve entities -> fact table -> DuckDB
	$(PY) -m drugsyn data

train:              ## cross-validated models under four split strategies
	$(PY) -m drugsyn train

figures:            ## portfolio figures in reports/figures
	$(PY) -m drugsyn figures

report:             ## reports/REPORT.md
	$(PY) -m drugsyn report

all:                ## everything, end to end
	$(PY) -m drugsyn all

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests

clean:              ## remove derived data (keeps downloaded raw files)
	rm -rf data/interim data/processed
