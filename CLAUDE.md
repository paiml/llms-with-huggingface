# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational course repository for building LLM applications with Hugging Face and OpenAI-compatible APIs. Contains hands-on labs and Python examples demonstrating chat interfaces, RAG, API development, small language models, and agentic workflows.

## Packaging

**ONLY use `uv` for all Python packaging operations.** Never use `pip` directly.

```bash
# Install all dependencies (creates venv automatically)
uv sync --all-extras

# Run a command in the venv
uv run <command>

# Run Python scripts
uv run python script.py

# Run pytest
uv run pytest

# Add a package
uv add <package>
```

## Commands

```bash
# Full quality check (lint + syntax validation)
make check

# Lint Python files with ruff
make lint

# Auto-format code
make format

# Validate Python syntax
make test

# Install dev dependencies
make install
```

### Running Examples

Examples require environment variables and a local LLM server (Ollama):

```bash
# Start Ollama and pull a model
ollama pull qwen2.5-coder:7b-instruct

# Set environment and run
export MODEL_NAME="qwen2.5-coder:7b-instruct"
export OPENAI_API_BASE="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"

uv run python examples/1-simple/chat.py
```

### Running the FastAPI Server (example 3)

```bash
cd examples/3-api
uv run uvicorn solution:app --reload
```

## Architecture

### Examples Structure

- `examples/1-simple/` - Basic chat interface, async operations, structured output with Pydantic
- `examples/2-rag/` - RAG pipeline using Sentence Transformers and Qdrant vector database
- `examples/3-api/` - FastAPI endpoints integrating RAG for wine recommendations
- `examples/4-small-lm/` - Local model configuration and optimization
- `examples/5-agentic/` - LangChain agents with tool/function calling
- `examples/shared/` - Reusable `ChatConfig` and utilities in `chat_utils.py`

### Key Patterns

All examples use OpenAI-compatible API (`openai` package) pointing to local Ollama or remote endpoints. Configuration follows this pattern:

```python
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
)
```

RAG examples use:
- `sentence-transformers` with `all-MiniLM-L6-v2` for embeddings
- `qdrant-client` for vector storage (in-memory for demos)

Agentic examples use:
- `langchain` with `@tool` decorator for function calling
- `ChatOpenAI` connected to local LLM

## Dependencies

Each example has its own `requirements.txt`. Core packages:
- `openai` - API client (all examples)
- `sentence-transformers`, `qdrant-client` - RAG (example 2-3)
- `fastapi`, `uvicorn`, `pandas` - API server (example 3)
- `langchain`, `langchain-openai` - Agents (example 5)

## Linting

Uses `ruff` with E501 (line length) and E722 (bare except) ignored for example code readability.
