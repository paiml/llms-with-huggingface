# Contributing to LLMs with Hugging Face

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues

1. Check if the issue already exists
2. Create a new issue with a clear description
3. Include steps to reproduce if applicable

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run linting: `make lint`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add comments for complex logic
- Keep functions focused and small

### Documentation

- Update README.md if adding new features
- Add docstrings to functions
- Update lab files if changing examples

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llms-with-huggingface.git
cd llms-with-huggingface

# Install all dependencies with uv
uv sync --all-extras

# Run linting
uv run ruff check .

# Run tests
uv run pytest

# Or use make targets
make lint
make test
```

## Questions?

Open an issue or reach out to the maintainers.

Thank you for contributing!
