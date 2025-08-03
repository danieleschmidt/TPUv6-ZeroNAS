.PHONY: install test lint format clean example help

help:	## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:	## Install package and dependencies
	pip install -e .
	pip install -e ".[dev]"

test:	## Run tests
	pytest tests/ -v

test-coverage:	## Run tests with coverage
	pytest tests/ --cov=tpuv6_zeronas --cov-report=html --cov-report=term

lint:	## Run linting
	flake8 tpuv6_zeronas tests examples
	mypy tpuv6_zeronas

format:	## Format code
	black tpuv6_zeronas tests examples scripts
	
format-check:	## Check code formatting
	black --check tpuv6_zeronas tests examples scripts

example:	## Run basic example
	python scripts/run_example.py

clean:	## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

build:	## Build package
	python setup.py sdist bdist_wheel

search:	## Run architecture search
	python -m tpuv6_zeronas.cli search --max-iterations 50 --population-size 20 --optimize-for-tpuv6

benchmark:	## Benchmark example architecture (requires architecture file)
	@echo "Usage: make benchmark ARCH_FILE=path/to/architecture.json"
	@if [ -n "$(ARCH_FILE)" ]; then \
		python -m tpuv6_zeronas.cli benchmark --architecture $(ARCH_FILE); \
	fi

all: clean format lint test	## Run full development pipeline