.PHONY: help venv install install-dev fmt lint test test-cov bench clean

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

install: ## Install package in production mode
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"

fmt: ## Format code with black and isort
	black mempack tests examples benchmarks
	isort mempack tests examples benchmarks

lint: ## Run linting with ruff and mypy
	ruff check mempack tests examples benchmarks
	mypy mempack

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=mempack --cov-report=html --cov-report=term-missing

bench: ## Run benchmarks
	python3 benchmarks/bench_search.py

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
