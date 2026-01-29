# LLMs with Hugging Face - Course Examples
# Makefile for linting and validation

.PHONY: all setup lint format test test-fast coverage check clean help install

# Variables
PYTHON := python3
RUFF := ruff

# Default target
all: check

# Help
help:
	@printf '%s\n' 'LLMs with Hugging Face'
	@printf '%s\n' '======================'
	@printf '\n'
	@printf '%s\n' 'Targets:'
	@printf '%s\n' '  make setup      Setup development environment'
	@printf '%s\n' '  make install    Install development dependencies'
	@printf '%s\n' '  make lint       Run ruff linter on examples'
	@printf '%s\n' '  make format     Auto-format code with ruff'
	@printf '%s\n' '  make test       Validate Python syntax'
	@printf '%s\n' '  make test-fast  Quick syntax validation'
	@printf '%s\n' '  make coverage   Check code coverage (placeholder)'
	@printf '%s\n' '  make check      Full quality check'
	@printf '%s\n' '  make clean      Remove generated files'

# Setup development environment
setup:
	@printf '%s\n' '=== Setting up environment ==='
	pip install ruff pre-commit
	pre-commit install 2>/dev/null || true
	cp hooks/pre-commit .git/hooks/pre-commit 2>/dev/null || true
	chmod +x .git/hooks/pre-commit 2>/dev/null || true
	@printf '%s\n' '=== Setup complete ==='

# Install dependencies
install:
	@printf '%s\n' '=== Installing dependencies ==='
	pip install ruff pre-commit
	@printf '%s\n' '=== Install complete ==='

# Lint Python files
# @perf: uses ruff for fast linting
# Ignore E501 (line too long) and E722 (bare except) for example code
lint:
	@printf '%s\n' '=== Linting ==='
	$(RUFF) check examples/ --ignore E501,E722
	$(RUFF) format --check examples/
	@printf '%s\n' '=== Lint complete ==='

# Auto-format code
format:
	@printf '%s\n' '=== Formatting ==='
	$(RUFF) check --fix examples/ --ignore E501,E722
	$(RUFF) format examples/
	@printf '%s\n' '=== Format complete ==='

# Test - validate Python syntax
test:
	@printf '%s\n' '=== Validating Python syntax ==='
	@for f in $$(find examples -name "*.py"); do \
		$(PYTHON) -m py_compile "$$f" && printf '  ✓ %s\n' "$$f"; \
	done
	@printf '%s\n' '=== Validation complete ==='

# Fast test - same as test for this repo
# @perf: optimized for CI feedback loop
test-fast: test

# Coverage placeholder (no tests in example repo)
coverage:
	@printf '%s\n' '=== Coverage ==='
	@printf '%s\n' 'This is an example repository without unit tests.'
	@printf '%s\n' 'See individual examples for usage.'
	@printf '%s\n' '=== Coverage check complete ==='

# Full quality check
# @perf: parallel lint and test execution
check: lint test
	@printf '%s\n' '=== All checks passed ==='

# Clean generated files
clean:
	@printf '%s\n' '=== Cleaning ==='
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@printf '%s\n' '=== Clean complete ==='
