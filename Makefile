# LLMs with Hugging Face - Course Examples
# Makefile for linting and validation

.PHONY: all setup lint format test test-fast coverage check clean help install sync

# Variables - use uv run for all Python operations
PYTHON := uv run python
RUFF := uv run ruff
PYTEST := uv run pytest

# Default target
all: check

# Help
help:
	@printf '%s\n' 'LLMs with Hugging Face'
	@printf '%s\n' '======================'
	@printf '\n'
	@printf '%s\n' 'Targets:'
	@printf '%s\n' '  make sync       Install all dependencies with uv'
	@printf '%s\n' '  make setup      Setup development environment'
	@printf '%s\n' '  make install    Alias for sync'
	@printf '%s\n' '  make lint       Run ruff linter'
	@printf '%s\n' '  make format     Auto-format code with ruff'
	@printf '%s\n' '  make test       Run pytest test suite'
	@printf '%s\n' '  make test-fast  Quick syntax validation'
	@printf '%s\n' '  make coverage   Run tests with coverage'
	@printf '%s\n' '  make check      Full quality check'
	@printf '%s\n' '  make clean      Remove generated files'

# Sync dependencies with uv
sync:
	@printf '%s\n' '=== Syncing dependencies ==='
	uv sync --all-extras
	@printf '%s\n' '=== Sync complete ==='

# Setup development environment
setup: sync
	@printf '%s\n' '=== Setting up environment ==='
	uv run pre-commit install 2>/dev/null || true
	@printf '%s\n' '=== Setup complete ==='

# Install dependencies (alias for sync)
install: sync

# Lint Python files
# @perf: uses ruff for fast linting
# Ignore E501 (line too long) and E722 (bare except) for example code
lint:
	@printf '%s\n' '=== Linting ==='
	$(RUFF) check src/ tests/ examples/ --ignore E501,E722
	$(RUFF) format --check src/ tests/ examples/
	@printf '%s\n' '=== Lint complete ==='

# Auto-format code
format:
	@printf '%s\n' '=== Formatting ==='
	$(RUFF) check --fix src/ tests/ examples/ --ignore E501,E722
	$(RUFF) format src/ tests/ examples/
	@printf '%s\n' '=== Format complete ==='

# Test - run pytest suite
test:
	@printf '%s\n' '=== Running tests ==='
	$(PYTEST) tests/ -v
	@printf '%s\n' '=== Tests complete ==='

# Fast test - syntax validation only
# @perf: optimized for CI feedback loop
test-fast:
	@printf '%s\n' '=== Validating Python syntax ==='
	@for f in $$(find examples src tests -name "*.py" 2>/dev/null); do \
		$(PYTHON) -m py_compile "$$f" && printf '  ✓ %s\n' "$$f"; \
	done
	@printf '%s\n' '=== Validation complete ==='

# Coverage - run tests with coverage report
coverage:
	@printf '%s\n' '=== Running tests with coverage ==='
	$(PYTEST) tests/ --cov=src --cov-report=term-missing
	@printf '%s\n' '=== Coverage complete ==='

# Full quality check
# @perf: parallel lint and test execution
check: lint test
	@printf '%s\n' '=== All checks passed ==='

# Clean generated files
clean:
	@printf '%s\n' '=== Cleaning ==='
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	@printf '%s\n' '=== Clean complete ==='
