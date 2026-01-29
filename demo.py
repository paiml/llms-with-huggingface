#!/usr/bin/env python3
"""Interactive demo showcasing LLM capabilities with rich terminal output.

This demo provides a visual, interactive experience of the Research Assistant.
Run with: uv run python demo.py
"""

import sys
import time

# Rich terminal output for visual stability
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Create global console for rich output
console = Console()
RICH_AVAILABLE = True


def print_banner() -> None:
    """Print the demo banner with ASCII art."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██╗     ██╗     ███╗   ███╗███████╗                        ║
║   ██║     ██║     ████╗ ████║██╔════╝                        ║
║   ██║     ██║     ██╔████╔██║███████╗                        ║
║   ██║     ██║     ██║╚██╔╝██║╚════██║                        ║
║   ███████╗███████╗██║ ╚═╝ ██║███████║                        ║
║   ╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝                        ║
║                                                               ║
║   with Hugging Face - Research Assistant Demo                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    if RICH_AVAILABLE:
        console = Console()
        console.print(banner, style="bold cyan")
    else:
        print(banner)


def show_features() -> None:
    """Display available features in a table."""
    if not RICH_AVAILABLE:
        print("\nFeatures: Chat, RAG, API, Agents")
        return

    console = Console()
    table = Table(title="Available Features", show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Status", justify="center")

    table.add_row("Chat Interface", "OpenAI-compatible chat with local LLMs", "✅")
    table.add_row("RAG Pipeline", "Semantic search with Qdrant + embeddings", "✅")
    table.add_row("REST API", "FastAPI endpoints at /chat, /research", "✅")
    table.add_row("Agents", "LangChain agents with tool calling", "✅")
    table.add_row("59 Tests", "Comprehensive test coverage", "✅")

    console.print(table)


def simulate_rag_search() -> None:
    """Simulate a RAG search with visual feedback."""
    if not RICH_AVAILABLE:
        print("\n[Simulating RAG search...]")
        print("Found: 3 relevant documents")
        return

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Encoding query...", total=100)
        for _ in range(100):
            time.sleep(0.01)
            progress.update(task, advance=1)

        progress.update(task, description="[cyan]Searching vector store...")
        for _ in range(100):
            time.sleep(0.01)

        progress.update(task, description="[green]Found 3 relevant documents!")

    # Show results
    results = Table(title="Search Results", show_header=True)
    results.add_column("Score", style="cyan", justify="right")
    results.add_column("Title", style="green")
    results.add_column("Preview", style="dim")

    results.add_row("0.95", "Machine Learning Basics", "ML is a subset of AI...")
    results.add_row("0.87", "Deep Learning Guide", "Neural networks are...")
    results.add_row("0.82", "Python for Data Science", "Python is widely used...")

    console.print(results)


def show_api_endpoints() -> None:
    """Display API endpoint documentation."""
    if not RICH_AVAILABLE:
        print("\nAPI Endpoints:")
        print("  GET  /health   - Health check")
        print("  POST /chat     - Chat with LLM")
        print("  POST /research - RAG-powered research")
        return

    console = Console()
    md = Markdown("""
## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Chat with LLM |
| POST | `/research` | RAG-powered research |
| POST | `/documents` | Add to knowledge base |

**Try it:**
```bash
curl http://localhost:8000/health
```
""")
    console.print(Panel(md, title="REST API", border_style="green"))


def main() -> int:
    """Run the interactive demo."""
    print_banner()

    if RICH_AVAILABLE:
        console = Console()
        console.print("\n[bold]Welcome to the LLMs with Hugging Face Demo![/bold]\n")
    else:
        print("\nWelcome to the LLMs with Hugging Face Demo!\n")

    show_features()

    if RICH_AVAILABLE:
        console = Console()
        console.print("\n[bold cyan]Simulating RAG Search...[/bold cyan]\n")

    simulate_rag_search()
    show_api_endpoints()

    if RICH_AVAILABLE:
        console = Console()
        console.print(
            Panel(
                "[green]Demo complete![/green]\n\n"
                "Next steps:\n"
                "1. Start Ollama: [cyan]ollama serve[/cyan]\n"
                "2. Run the API: [cyan]uv run uvicorn src.api.main:app[/cyan]\n"
                "3. Open docs: [cyan]http://localhost:8000/docs[/cyan]",
                title="What's Next?",
                border_style="green",
            )
        )
    else:
        print("\nDemo complete! Run 'uv run uvicorn src.api.main:app' to start the API.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
