#!/usr/bin/env python3
"""
Simple RAG Enhancement Workflow Test Script
Tests the workflow configuration and basic functionality
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_configuration():
    """Test workflow configuration loading"""
    console.print("[bold blue]Testing workflow configuration...[/bold blue]")

    config_path = Path("workflow_config.yaml")
    if not config_path.exists():
        console.print("[red]❌ workflow_config.yaml not found![/red]")
        return False

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check basic structure
        required_keys = ["project", "phases", "metrics"]
        for key in required_keys:
            if key not in config:
                console.print(f"[red]❌ Missing required key: {key}[/red]")
                return False

        console.print("[green]✅ Configuration structure valid[/green]")

        # Display project info
        project = config["project"]
        console.print(f"[cyan]Project:[/cyan] {project['name']}")
        console.print(f"[cyan]Version:[/cyan] {project['version']}")
        console.print(f"[cyan]Start Date:[/cyan] {project['start_date']}")

        return True

    except Exception as e:
        console.print(f"[red]❌ Configuration error: {e}[/red]")
        return False

def test_task_structure():
    """Test task structure from configuration"""
    console.print("\n[bold blue]Testing task structure...[/bold blue]")

    try:
        with open("workflow_config.yaml") as f:
            config = yaml.safe_load(f)

        tasks_found = 0
        phases = config.get("phases", {})

        table = Table(title="Workflow Tasks Overview")
        table.add_column("Phase", style="cyan")
        table.add_column("Sprint", style="yellow")
        table.add_column("Task", style="white")
        table.add_column("Priority", style="red")
        table.add_column("Hours", style="green")

        for phase_id, phase_data in phases.items():
            for sprint_id, sprint_data in phase_data.get("sprints", {}).items():
                for task_id, task_data in sprint_data.get("tasks", {}).items():
                    tasks_found += 1
                    table.add_row(
                        phase_id,
                        sprint_id,
                        task_data.get("name", task_id)[:40] + "...",
                        task_data.get("priority", "P2"),
                        str(task_data.get("estimated_hours", "N/A"))
                    )

        console.print(table)
        console.print(f"\n[green]✅ Found {tasks_found} tasks across {len(phases)} phases[/green]")
        return True

    except Exception as e:
        console.print(f"[red]❌ Task structure error: {e}[/red]")
        return False

def test_file_creation():
    """Test file and directory creation"""
    console.print("\n[bold blue]Testing file structure creation...[/bold blue]")

    # Test directories
    test_dirs = [
        "src/services/rag/unified",
        "src/services/rag/evaluation",
        "tests/rag/unit",
        "data/evaluation",
        "src/monitoring"
    ]

    created_dirs = 0
    for dir_path in test_dirs:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]✅ Directory created/exists: {dir_path}[/green]")
            created_dirs += 1
        except Exception as e:
            console.print(f"[red]❌ Failed to create directory {dir_path}: {e}[/red]")

    # Test file creation
    test_file = Path("src/services/rag/rag_evaluator.py")
    if not test_file.exists():
        try:
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text('''"""
RAG Evaluator Test Implementation
Created for workflow testing
"""

class RAGEvaluator:
    def __init__(self):
        self.metrics = {}

    async def evaluate(self, query, context, answer):
        """Placeholder evaluation method"""
        return {"faithfulness": 0.8, "relevancy": 0.9}

# Test implementation
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    print("RAG Evaluator test implementation ready")
''')
            console.print(f"[green]✅ Test file created: {test_file}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to create test file: {e}[/red]")
    else:
        console.print(f"[yellow]ℹ️ Test file already exists: {test_file}[/yellow]")

    return created_dirs == len(test_dirs)

def test_progress_tracking():
    """Test progress tracking functionality"""
    console.print("\n[bold blue]Testing progress tracking...[/bold blue]")

    # Create mock progress data
    progress_data = {
        "last_updated": datetime.now().isoformat(),
        "current_phase": "phase1",
        "current_sprint": "sprint1_1",
        "tasks": {
            "ragas_integration": {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "notes": "Testing workflow system"
            },
            "benchmark_dataset": {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "notes": None
            }
        }
    }

    try:
        with open("test_workflow_progress.json", "w") as f:
            json.dump(progress_data, f, indent=2)

        # Read it back
        with open("test_workflow_progress.json") as f:
            loaded_data = json.load(f)

        console.print("[green]✅ Progress tracking data structure works[/green]")

        # Display sample progress
        table = Table(title="Sample Progress Tracking")
        table.add_column("Task", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Started", style="white")

        for task_id, task_data in loaded_data["tasks"].items():
            started = task_data["started_at"]
            if started:
                started = started.split("T")[0]  # Just the date
            else:
                started = "Not started"

            table.add_row(
                task_id,
                task_data["status"],
                started
            )

        console.print(table)
        return True

    except Exception as e:
        console.print(f"[red]❌ Progress tracking error: {e}[/red]")
        return False

def test_workflow_documentation():
    """Test workflow documentation files"""
    console.print("\n[bold blue]Testing workflow documentation...[/bold blue]")

    docs = [
        "RAG_ENHANCEMENT_WORKFLOW.md",
        "workflow_config.yaml",
        "WORKFLOW_QUICK_START.md"
    ]

    doc_status = []
    for doc in docs:
        path = Path(doc)
        if path.exists():
            size = path.stat().st_size
            console.print(f"[green]✅ {doc} exists ({size} bytes)[/green]")
            doc_status.append(True)
        else:
            console.print(f"[red]❌ {doc} missing[/red]")
            doc_status.append(False)

    return all(doc_status)

def run_full_test():
    """Run complete workflow test suite"""
    console.print(Panel.fit(
        "[bold white]🚀 RAG Enhancement Workflow Test Suite[/bold white]",
        border_style="blue"
    ))

    tests = [
        ("Configuration Loading", test_configuration),
        ("Task Structure", test_task_structure),
        ("File Creation", test_file_creation),
        ("Progress Tracking", test_progress_tracking),
        ("Documentation", test_workflow_documentation)
    ]

    results = []
    for test_name, test_func in tests:
        console.print(f"\n[bold]Running {test_name}...[/bold]")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"[red]❌ {test_name} failed with error: {e}[/red]")
            results.append((test_name, False))

    # Summary
    console.print("\n" + "="*50)
    console.print("[bold]TEST RESULTS SUMMARY[/bold]")
    console.print("="*50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        color = "green" if result else "red"
        console.print(f"[{color}]{status}[/{color}] {test_name}")
        if result:
            passed += 1

    console.print(f"\n[bold]Overall: {passed}/{len(results)} tests passed[/bold]")

    if passed == len(results):
        console.print("\n[bold green]🎉 All tests passed! Workflow system is ready.[/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("1. Run: [yellow]./start_workflow.sh[/yellow]")
        console.print("2. Start first task: [yellow]python3 test_workflow.py start ragas_integration[/yellow]")
    else:
        console.print(f"\n[bold red]❌ {len(results)-passed} tests failed. Please fix issues before proceeding.[/bold red]")

def simulate_task_execution():
    """Simulate starting and completing a task"""
    console.print("\n[bold blue]Simulating task execution...[/bold blue]")

    # Simulate starting a task
    task_data = {
        "id": "ragas_integration",
        "name": "Extend RAG evaluator with RAGAS metrics",
        "status": "in_progress",
        "started_at": datetime.now().isoformat(),
        "file_path": "src/services/rag/rag_evaluator.py"
    }

    console.print(Panel(
        f"[green]Started task: {task_data['name']}[/green]\n"
        f"File: {task_data['file_path']}\n"
        f"Status: {task_data['status']}",
        title="Task Simulation"
    ))

    # Check if implementation file exists
    impl_file = Path(task_data['file_path'])
    if impl_file.exists():
        console.print(f"[green]✅ Implementation file exists: {impl_file}[/green]")
    else:
        console.print(f"[yellow]ℹ️ Implementation file would be created: {impl_file}[/yellow]")

    console.print("[cyan]Task simulation complete![/cyan]")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        simulate_task_execution()
    else:
        run_full_test()