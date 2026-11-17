ifeq ($(OS),Windows_NT)
PYTHON = python
VENV_PY = .venv/Scripts/python.exe
else
PYTHON = python3
VENV_PY = .venv/bin/python
endif

CONFIG ?= configs/xor.json
EPOCHS ?=
OUTPUT ?= runs/latest

.PHONY: help venv install dev test train xor digits benchmark gradcheck clean distclean

help:
	@echo Targets:
	@echo   venv       create the local virtual environment in .venv
	@echo   install    install the package and runtime dependencies
	@echo   dev        install the package with development dependencies
	@echo   test       run the test suite
	@echo   gradcheck  verify analytic gradients against finite differences
	@echo   xor        train the XOR configuration
	@echo   digits     train the synthetic digits configuration
	@echo   train      train with CONFIG=path/to/config.json OUTPUT=runs/name
	@echo   benchmark  run every configuration in configs/ and write a report
	@echo   clean      remove build, cache and run artifacts
	@echo   distclean  clean and remove the virtual environment

venv:
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip

install: venv
	$(VENV_PY) -m pip install -r requirements.txt
	$(VENV_PY) -m pip install -e .

dev: venv
	$(VENV_PY) -m pip install -r requirements-dev.txt
	$(VENV_PY) -m pip install -e .

test:
	$(VENV_PY) -m pytest -q

gradcheck:
	$(VENV_PY) -m nnlab gradcheck

xor:
	$(VENV_PY) -m nnlab train --config configs/xor.json --output runs/xor

digits:
	$(VENV_PY) -m nnlab train --config configs/digits.json --output runs/digits

train:
	$(VENV_PY) -m nnlab train --config $(CONFIG) --output $(OUTPUT)

benchmark:
	$(VENV_PY) -m nnlab benchmark --config-dir configs --output runs/benchmark

clean:
	$(PYTHON) scripts/clean.py

distclean:
	$(PYTHON) scripts/clean.py --all
