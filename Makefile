PYTHON ?= python

.PHONY: install test experiment conjecture figures readme all clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

experiment:
	$(PYTHON) scripts/run_experiment.py

conjecture:
	$(PYTHON) scripts/test_conjecture.py

figures:
	$(PYTHON) scripts/make_figures.py

readme:
	$(PYTHON) scripts/update_readme.py

all: install test experiment conjecture figures readme

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__
