#!/usr/bin/env python3
"""Phase 4 Auto Demo - Non-interactive demonstration of all Phase 4 features.

This script automatically runs through all Phase 4 commands with mock data.

Usage:
    python scripts/demo_phase4_auto.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.demo_phase4 import (
    console,
    show_welcome,
    demo_versions,
    demo_suggest,
    demo_iterate,
    demo_compare,
    demo_rollback,
    MOCK_PAPER,
)
from rich.panel import Panel
from rich import box
import time


def run_auto_demo():
    """Run all demos automatically."""
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

    console.print("[bold magenta]🎬 Running All Phase 4 Demos Automatically[/bold magenta]\n")
    time.sleep(1)

    # Demo 1: Versions
    console.print("[bold cyan]═══ Demo 1/5: Version History ═══[/bold cyan]\n")
    demo_versions()
    time.sleep(2)

    # Demo 2: Suggestions
    console.print("[bold cyan]═══ Demo 2/5: Smart Suggestions ═══[/bold cyan]\n")
    demo_suggest()
    time.sleep(2)

    # Demo 3: Iterative Improvement
    console.print("[bold cyan]═══ Demo 3/5: Iterative Improvement ═══[/bold cyan]\n")
    demo_iterate()
    time.sleep(2)

    # Demo 4: Version Comparison
    console.print("[bold cyan]═══ Demo 4/5: Version Comparison ═══[/bold cyan]\n")
    demo_compare()
    time.sleep(2)

    # Demo 5: Rollback
    console.print("[bold cyan]═══ Demo 5/5: Version Rollback ═══[/bold cyan]\n")
    demo_rollback()
    time.sleep(1)

    # Summary
    summary = Panel(
        "[bold green]✅ All Phase 4 Demos Complete![/bold green]\n\n"
        "[cyan]Demonstrated Features:[/cyan]\n"
        "  ✓ Version tracking with semantic versioning\n"
        "  ✓ RAG-powered smart suggestions from ChromaDB\n"
        "  ✓ Iterative improvement loops with quality tracking\n"
        "  ✓ Version comparison with diff visualization\n"
        "  ✓ Version rollback with automatic backups\n\n"
        "[yellow]To test interactively:[/yellow]\n"
        "  python scripts/demo_phase4.py\n\n"
        "[yellow]To test with real backend:[/yellow]\n"
        "  1. Start API: uvicorn src.main:app --reload\n"
        "  2. Start ChromaDB: chroma run --path ./chroma_data\n"
        "  3. Run chatbot: python scripts/chat_reviewer_enhanced.py",
        border_style="bright_blue",
        box=box.DOUBLE,
    )
    console.print(summary)
    console.print()


if __name__ == "__main__":
    try:
        run_auto_demo()
    except KeyboardInterrupt:
        console.print("\n\n👋 [yellow]Demo interrupted. Goodbye![/yellow]\n")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]\n")
        raise
