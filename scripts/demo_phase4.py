#!/usr/bin/env python3
"""Phase 4 Demo Script - Test chatbot Phase 4 commands without backend server.

This script simulates Phase 4 functionality with mock data to demonstrate:
- /suggest - RAG-powered smart suggestions
- /iterate - Iterative improvement loops
- /compare - Version comparison with diffs
- /rollback - Version rollback
- /versions - Version history display

Usage:
    python scripts/demo_phase4.py
"""

from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
import time

console = Console()


# ========== Mock Data ==========

MOCK_PAPER = {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "AI-Powered Autonomous Research: A Novel Framework",
    "current_version": "1.2.0",
    "quality_score": 7.8,
}

MOCK_VERSIONS = [
    {
        "version": "1.2.0",
        "type": "MINOR",
        "summary": "Iteration 2: score=7.8",
        "quality": 7.8,
        "created": "2025-10-10 14:30:00",
    },
    {
        "version": "1.1.0",
        "type": "MINOR",
        "summary": "Iteration 1: score=7.2",
        "quality": 7.2,
        "created": "2025-10-10 14:15:00",
    },
    {
        "version": "1.0.0",
        "type": "MAJOR",
        "summary": "Start iterative improvement session (target: 8.5)",
        "quality": 6.5,
        "created": "2025-10-10 14:00:00",
    },
]

MOCK_SUGGESTIONS = [
    {
        "section": "Abstract",
        "expected_gain": 0.8,
        "patterns_used": 5,
        "exemplars": 2,
        "changes": ["Enhanced clarity", "Added quantitative results", "Improved structure"],
    },
    {
        "section": "Introduction",
        "expected_gain": 0.6,
        "patterns_used": 3,
        "exemplars": 1,
        "changes": ["Strengthened motivation", "Added recent citations"],
    },
    {
        "section": "Methodology",
        "expected_gain": 0.5,
        "patterns_used": 4,
        "exemplars": 2,
        "changes": ["Clarified experimental setup", "Added reproducibility details"],
    },
]

MOCK_DIFF = """--- Abstract (1.0.0)
+++ Abstract (1.2.0)
@@ -1,5 +1,7 @@
-This paper presents a framework for autonomous research.
+This paper presents a novel AI-powered framework for autonomous scientific research.

-We propose a multi-agent system.
+We propose a multi-agent system that achieves 95% automation in hypothesis generation
+and experimental design, representing a 3x improvement over existing approaches.

-Results show promising performance.
+Experimental results demonstrate state-of-the-art performance across 5 benchmark datasets,
+with statistical significance (p < 0.001)."""


# ========== Command Simulations ==========


def demo_versions():
    """Simulate /versions command."""
    console.print("\n[cyan]📚 Fetching version history...[/cyan]\n")
    time.sleep(0.5)

    table = Table(
        title=f"📜 Version History: {MOCK_PAPER['title']}", box=box.ROUNDED, show_lines=True
    )
    table.add_column("Version", style="cyan", width=10)
    table.add_column("Type", style="yellow", width=10)
    table.add_column("Quality", style="magenta", width=10)
    table.add_column("Summary", style="white", width=40)
    table.add_column("Created", style="dim", width=20)

    for v in MOCK_VERSIONS:
        is_current = "⭐ " if v["version"] == MOCK_PAPER["current_version"] else ""
        table.add_row(
            f"{is_current}{v['version']}",
            v["type"],
            f"{v['quality']:.1f}/10",
            v["summary"],
            v["created"],
        )

    console.print(table)
    console.print(f"\n[dim]Current version: {MOCK_PAPER['current_version']} | Total versions: {len(MOCK_VERSIONS)}[/dim]\n")


def demo_suggest():
    """Simulate /suggest command."""
    console.print("\n[cyan]🧠 Generating RAG-powered smart suggestions...[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing paper sections...", total=None)
        time.sleep(1)
        progress.update(task, description="Querying ChromaDB for similar patterns...")
        time.sleep(1)
        progress.update(task, description="Finding exemplar papers...")
        time.sleep(1)
        progress.update(task, description="Generating suggestions...")
        time.sleep(0.5)

    table = Table(title="💡 Smart Suggestions (RAG-Enhanced)", box=box.ROUNDED)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Section", style="green", width=15)
    table.add_column("Expected Gain", style="magenta", width=15)
    table.add_column("Patterns", style="yellow", width=10)
    table.add_column("Exemplars", style="blue", width=10)
    table.add_column("Changes", style="white", width=50)

    for i, sug in enumerate(MOCK_SUGGESTIONS, 1):
        table.add_row(
            str(i),
            sug["section"],
            f"+{sug['expected_gain']:.1f}",
            str(sug["patterns_used"]),
            str(sug["exemplars"]),
            "\n".join(f"• {c}" for c in sug["changes"]),
        )

    console.print(table)
    console.print(f"\n[dim]✅ RAG-Enhanced | Total: {len(MOCK_SUGGESTIONS)} suggestions[/dim]\n")
    console.print("[yellow]💡 Use '/apply <section>' to apply a suggestion![/yellow]\n")


def demo_iterate():
    """Simulate /iterate command."""
    target_score = 8.5
    console.print(f"\n[cyan]🔄 Starting iterative improvement (target: {target_score})...[/cyan]\n")

    iterations = [
        {"round": 1, "score": 6.5, "improvements": 3, "time": "2.3s"},
        {"round": 2, "score": 7.2, "improvements": 3, "time": "2.1s"},
        {"round": 3, "score": 7.8, "improvements": 2, "time": "1.8s"},
    ]

    for iteration in iterations:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Iteration {iteration['round']}/{len(iterations)}...", total=None
            )
            time.sleep(0.8)
            progress.update(task, description=f"Analyzing quality...")
            time.sleep(0.5)
            progress.update(task, description=f"Generating suggestions...")
            time.sleep(0.5)
            progress.update(task, description=f"Applying improvements...")
            time.sleep(0.7)

        console.print(
            f"[green]✓[/green] Iteration {iteration['round']} complete: "
            f"score={iteration['score']:.1f} "
            f"(+{iteration['score'] - (iterations[iteration['round']-2]['score'] if iteration['round'] > 1 else 6.5):+.1f}) | "
            f"{iteration['improvements']} improvements | {iteration['time']}\n"
        )

    # Summary panel
    summary = Panel(
        f"[bold green]🎯 Iterative Improvement Complete![/bold green]\n\n"
        f"[cyan]Session ID:[/cyan] session-abc123\n"
        f"[cyan]Iterations:[/cyan] {len(iterations)}\n"
        f"[cyan]Improvements Applied:[/cyan] {sum(i['improvements'] for i in iterations)}\n"
        f"[cyan]Initial Score:[/cyan] 6.5/10\n"
        f"[cyan]Final Score:[/cyan] 7.8/10\n"
        f"[cyan]Improvement:[/cyan] [green]+1.3[/green]\n"
        f"[cyan]Target Reached:[/cyan] [yellow]No[/yellow] (target: {target_score})\n"
        f"[cyan]Version:[/cyan] 1.0.0 → 1.2.0",
        border_style="green",
        box=box.DOUBLE,
    )
    console.print(summary)
    console.print()


def demo_compare():
    """Simulate /compare command."""
    version_a = "1.0.0"
    version_b = "1.2.0"

    console.print(f"\n[cyan]🔍 Comparing versions {version_a} and {version_b}...[/cyan]\n")
    time.sleep(1)

    # Version info table
    info_table = Table(title="📊 Version Comparison", box=box.ROUNDED)
    info_table.add_column("Metric", style="cyan", width=20)
    info_table.add_column(f"v{version_a}", style="yellow", width=15)
    info_table.add_column(f"v{version_b}", style="green", width=15)
    info_table.add_column("Change", style="magenta", width=15)

    info_table.add_row("Quality Score", "6.5/10", "7.8/10", "[green]+1.3[/green]")
    info_table.add_row(
        "Summary",
        "Start iterative...",
        "Iteration 2: score=7.8",
        "2 iterations",
    )
    info_table.add_row("Sections Changed", "-", "3", "+3")

    console.print(info_table)
    console.print()

    # Diff visualization
    diff_panel = Panel(
        Syntax(MOCK_DIFF, "diff", theme="monokai", line_numbers=False),
        title="📝 Abstract Section Diff",
        border_style="blue",
        box=box.ROUNDED,
    )
    console.print(diff_panel)
    console.print(f"\n[dim]💡 Quality improved by +1.3 points across 3 sections[/dim]\n")


def demo_rollback():
    """Simulate /rollback command."""
    target_version = "1.1.0"

    console.print(f"\n[yellow]⏪ Rolling back to version {target_version}...[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating backup...", total=None)
        time.sleep(0.8)
        progress.update(task, description="Restoring content...")
        time.sleep(1)
        progress.update(task, description="Restoring sections...")
        time.sleep(0.7)
        progress.update(task, description="Creating rollback version...")
        time.sleep(0.5)

    result_panel = Panel(
        f"[bold green]✅ Rollback Successful![/bold green]\n\n"
        f"[cyan]Rolled back from:[/cyan] 1.2.0\n"
        f"[cyan]Rolled back to:[/cyan] {target_version}\n"
        f"[cyan]New version:[/cyan] 2.0.0 (rollback snapshot)\n"
        f"[cyan]Backup created:[/cyan] [green]Yes[/green]\n"
        f"[cyan]Quality score:[/cyan] 7.2/10 (was 7.8/10)",
        border_style="green",
        box=box.DOUBLE,
    )
    console.print(result_panel)
    console.print()


# ========== Interactive Demo ==========


def show_welcome():
    """Show welcome banner."""
    welcome = Panel(
        "[bold cyan]Phase 4 Demo: Intelligent Paper Improvement System[/bold cyan]\n\n"
        "This demo simulates Phase 4 chatbot commands with mock data.\n"
        "No backend server required!\n\n"
        "[yellow]Available Commands:[/yellow]\n"
        "  1. /versions  - Show version history\n"
        "  2. /suggest   - Get RAG-powered suggestions\n"
        "  3. /iterate   - Run iterative improvement loop\n"
        "  4. /compare   - Compare two versions\n"
        "  5. /rollback  - Rollback to previous version\n"
        "  6. all        - Run all demos\n"
        "  7. quit       - Exit demo",
        border_style="bright_blue",
        box=box.DOUBLE,
    )
    console.print(welcome)
    console.print()


def run_demo():
    """Run interactive demo."""
    show_welcome()

    # Display paper info
    paper_info = Panel(
        f"[bold]Mock Paper Loaded[/bold]\n\n"
        f"[cyan]Title:[/cyan] {MOCK_PAPER['title']}\n"
        f"[cyan]ID:[/cyan] {MOCK_PAPER['id']}\n"
        f"[cyan]Current Version:[/cyan] {MOCK_PAPER['current_version']}\n"
        f"[cyan]Quality Score:[/cyan] {MOCK_PAPER['quality_score']}/10",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(paper_info)
    console.print()

    commands = {
        "1": ("versions", demo_versions),
        "/versions": ("versions", demo_versions),
        "2": ("suggest", demo_suggest),
        "/suggest": ("suggest", demo_suggest),
        "3": ("iterate", demo_iterate),
        "/iterate": ("iterate", demo_iterate),
        "4": ("compare", demo_compare),
        "/compare": ("compare", demo_compare),
        "5": ("rollback", demo_rollback),
        "/rollback": ("rollback", demo_rollback),
        "all": ("all", None),
    }

    while True:
        try:
            choice = console.input("[bold cyan]💬 Enter command (or 'quit' to exit):[/bold cyan] ").strip().lower()

            if choice in ["quit", "exit", "q"]:
                console.print("\n👋 [green]Demo complete! Thanks for testing Phase 4![/green]\n")
                break

            if choice == "all":
                console.print("\n[bold magenta]🎬 Running All Demos[/bold magenta]\n")
                demo_versions()
                console.print("[dim]Press Ctrl+C to stop, or wait for next demo...[/dim]")
                time.sleep(2)
                demo_suggest()
                time.sleep(2)
                demo_iterate()
                time.sleep(2)
                demo_compare()
                time.sleep(2)
                demo_rollback()
                console.print("\n[bold green]✅ All demos complete![/bold green]\n")
                continue

            if choice in commands:
                name, func = commands[choice]
                func()
            else:
                console.print(f"[red]❌ Unknown command: {choice}[/red]\n")

        except KeyboardInterrupt:
            console.print("\n\n👋 [yellow]Demo interrupted. Goodbye![/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]\n")


if __name__ == "__main__":
    run_demo()
