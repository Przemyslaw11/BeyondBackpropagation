PYTHON ?= python3
CONFIG ?= configs/mf/mnist_mlp_2x1000.yaml

.PHONY: install format format-check lint typecheck test test-fast test-cov check validate-config smoke clean help

install:
	$(PYTHON) -m pip install -e .

format:
	ruff format src/beyond_backprop tests

format-check:
	ruff format --check src/beyond_backprop tests

lint:
	ruff check src/beyond_backprop tests

typecheck:
	mypy src/beyond_backprop

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow and not gpu"

test-cov:
	$(PYTHON) -m pytest --cov=beyond_backprop --cov-report=term-missing

check: format-check lint typecheck test-fast

validate-config:
	PYTHONPATH=src:. $(PYTHON) -m beyond_backprop.cli.main validate-config --config $(CONFIG)

smoke:
	$(PYTHON) -m pytest -m smoke

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache

help:
	@printf '%s\n' \
		'install         Install the package in editable mode' \
		'format          Format canonical package and tests' \
		'format-check    Check formatting' \
		'lint            Run Ruff lint checks' \
		'typecheck       Run mypy on canonical package' \
		'test            Run the complete test suite' \
		'test-fast       Run CPU/non-slow tests' \
		'test-cov        Run tests with coverage' \
		'check           Run formatting, lint, type, and fast tests' \
		'validate-config Validate CONFIG without loading data' \
		'smoke           Run the tiny offline smoke test' \
		'clean           Remove generated build and cache files'
