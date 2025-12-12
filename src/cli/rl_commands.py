"""
CLI Commands for RL System Management

Provides comprehensive command-line interface for managing the RL-enhanced
agent selection system, including:

- System status and health checks
- Performance monitoring and metrics
- Traffic migration and rollback
- Model management and training
- Configuration management
- Troubleshooting utilities

This module integrates with existing CLI infrastructure and provides
both interactive and scripted management capabilities.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import click
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from src.core.rl_system import (
        get_rl_system_manager,
        is_rl_enabled,
        get_rl_health_status,
        get_rl_metrics,
        migrate_to_rl,
        rollback_from_rl,
        validate_rl_config
    )
    from src.agents.rl.deployment import create_rl_system_manager
    RL_SYSTEM_AVAILABLE = True
except ImportError:
    RL_SYSTEM_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def print_status(message: str, style: str = "info"):
    """Print status message with styling"""
    if RICH_AVAILABLE and console:
        if style == "info":
            console.print(f"ℹ️  {message}", style="blue")
        elif style == "success":
            console.print(f"✅ {message}", style="green")
        elif style == "warning":
            console.print(f"⚠️  {message}", style="yellow")
        elif style == "error":
            console.print(f"❌ {message}", style="red")
        else:
            console.print(message)
    else:
        print(f"[{style.upper()}] {message}")


def print_json(data: Dict[str, Any]):
    """Print JSON data with formatting"""
    if RICH_AVAILABLE and console:
        console.print_json(json.dumps(data, default=str))
    else:
        print(json.dumps(data, indent=2, default=str))


def create_status_table(data: Dict[str, Any]) -> Table:
    """Create a status table for display"""
    if not RICH_AVAILABLE:
        return None

    table = Table(title="RL System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="white")

    # Overall status
    overall_status = data.get("status", "unknown")
    status_color = "green" if overall_status == "healthy" else "red"
    table.add_row("Overall System", overall_status, f"Running: {data.get('running', False)}")

    # Component status
    components = data.get("components", {})
    for name, info in components.items():
        component_status = info.get("status", "unknown")
        details = []
        for key, value in info.items():
            if key != "status":
                details.append(f"{key}: {value}")
        table.add_row(name.replace("_", " ").title(), component_status, ", ".join(details))

    return table


@click.group()
def rl():
    """RL system management commands."""
    if not RL_SYSTEM_AVAILABLE:
        print_status("RL system not available. Please install RL components.", "error")
        sys.exit(1)


@rl.command()
def status():
    """Show RL system status and health."""
    print_status("Checking RL system status...")

    try:
        # Get status information
        enabled = is_rl_enabled()
        health = get_rl_health_status()

        if RICH_AVAILABLE and console:
            # Create status panel
            status_text = "🟢 Enabled" if enabled else "🔴 Disabled"
            panel = Panel(
                f"RL System: {status_text}\nLast Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                title="RL System Overview",
                expand=False
            )
            console.print(panel)

            # Show detailed status table
            if health and "status" in health:
                table = create_status_table(health)
                if table:
                    console.print(table)
        else:
            print(f"RL System Enabled: {enabled}")
            print_json(health)

    except Exception as e:
        print_status(f"Failed to get status: {e}", "error")
        sys.exit(1)


@rl.command()
def metrics():
    """Show RL system metrics and performance data."""
    print_status("Collecting RL system metrics...")

    try:
        metrics_data = get_rl_metrics()

        if RICH_AVAILABLE and console:
            # Performance metrics table
            performance = metrics_data.get("performance", {})
            if performance:
                table = Table(title="Performance Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")

                table.add_row("Strategies Active", str(performance.get("strategies_active", 0)))
                table.add_row("Total Selections (1h)", str(performance.get("total_selections_1h", 0)))
                table.add_row("Error Rate (1h)", f"{performance.get('error_rate_1h', 0):.2%}")

                console.print(table)

            # Traffic distribution
            traffic = metrics_data.get("traffic", {})
            if traffic:
                table = Table(title="Traffic Distribution")
                table.add_column("Setting", style="cyan")
                table.add_column("Value", style="magenta")

                table.add_row("RL Enabled", str(traffic.get("rl_enabled", False)))
                table.add_row("Current RL Traffic", f"{traffic.get('current_rl_traffic', 0):.1%}")
                table.add_row("A/B Testing", str(traffic.get("ab_testing", False)))

                console.print(table)

            # System health
            health = metrics_data.get("system_health", {})
            if health:
                resources = health.get("resources", {})
                if resources:
                    table = Table(title="System Resources")
                    table.add_column("Resource", style="cyan")
                    table.add_column("Usage", style="magenta")

                    table.add_row("CPU", f"{resources.get('cpu_percent', 0):.1f}%")
                    table.add_row("Memory", f"{resources.get('memory_percent', 0):.1f}%")
                    table.add_row("Disk", f"{resources.get('disk_usage_percent', 0):.1f}%")

                    console.print(table)
        else:
            print_json(metrics_data)

    except Exception as e:
        print_status(f"Failed to get metrics: {e}", "error")
        sys.exit(1)


@rl.command()
@click.option("--percentage", "-p", type=float, default=100.0,
              help="Target percentage of traffic for RL (0-100)")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def migrate(percentage: float, confirm: bool):
    """Start traffic migration to RL system."""
    if not 0 <= percentage <= 100:
        print_status("Percentage must be between 0 and 100", "error")
        sys.exit(1)

    if not confirm:
        if RICH_AVAILABLE and console:
            response = console.input(
                f"Start migration to {percentage}% RL traffic? [y/N]: "
            )
        else:
            response = input(f"Start migration to {percentage}% RL traffic? [y/N]: ")

        if response.lower() not in ('y', 'yes'):
            print_status("Migration cancelled", "warning")
            return

    print_status(f"Starting migration to {percentage}% RL traffic...")

    async def run_migration():
        try:
            result = await migrate_to_rl(percentage)

            if result["status"] == "success":
                print_status(f"Migration started: {result['migration_id']}", "success")
                print_status("Monitor progress with: rl-cli rl status", "info")
            else:
                print_status(f"Migration failed: {result.get('message', 'Unknown error')}", "error")
                sys.exit(1)

        except Exception as e:
            print_status(f"Migration failed: {e}", "error")
            sys.exit(1)

    asyncio.run(run_migration())


@rl.command()
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def rollback(confirm: bool):
    """Rollback from RL to traditional agent selection."""
    if not confirm:
        if RICH_AVAILABLE and console:
            response = console.input("Rollback to traditional agent selection? [y/N]: ")
        else:
            response = input("Rollback to traditional agent selection? [y/N]: ")

        if response.lower() not in ('y', 'yes'):
            print_status("Rollback cancelled", "warning")
            return

    print_status("Rolling back to traditional agent selection...")

    async def run_rollback():
        try:
            result = await rollback_from_rl()

            if result["status"] == "success":
                print_status("Rollback completed successfully", "success")
            else:
                print_status(f"Rollback failed: {result.get('message', 'Unknown error')}", "error")
                sys.exit(1)

        except Exception as e:
            print_status(f"Rollback failed: {e}", "error")
            sys.exit(1)

    asyncio.run(run_rollback())


@rl.command()
def validate():
    """Validate RL system configuration."""
    print_status("Validating RL system configuration...")

    try:
        validation = validate_rl_config()

        if RICH_AVAILABLE and console:
            # Status panel
            status_color = "green" if validation["valid"] else "red"
            status_text = "✅ Valid" if validation["valid"] else "❌ Invalid"
            panel = Panel(
                status_text,
                title="Configuration Validation",
                style=status_color
            )
            console.print(panel)

            # Configuration table
            config = validation.get("config", {})
            if config:
                table = Table(title="Configuration Values")
                table.add_column("Setting", style="cyan")
                table.add_column("Value", style="magenta")
                table.add_column("Status", style="white")

                for key, value in config.items():
                    table.add_row(key, str(value), "✅")

                console.print(table)

            # Errors and warnings
            errors = validation.get("errors", [])
            if errors:
                console.print("\n[red]Errors:[/red]")
                for error in errors:
                    console.print(f"  ❌ {error}")

            warnings = validation.get("warnings", [])
            if warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  ⚠️  {warning}")

        else:
            print(f"Valid: {validation['valid']}")
            if validation['errors']:
                print("Errors:")
                for error in validation['errors']:
                    print(f"  - {error}")
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")

        if not validation["valid"]:
            sys.exit(1)

    except Exception as e:
        print_status(f"Validation failed: {e}", "error")
        sys.exit(1)


@rl.command()
@click.argument("output_file", type=click.Path(), required=False)
def export_config(output_file: Optional[str]):
    """Export RL system configuration."""
    try:
        validation = validate_rl_config()
        config_data = {
            "timestamp": datetime.now().isoformat(),
            "validation": validation,
            "system_info": {
                "enabled": is_rl_enabled(),
                "health": get_rl_health_status()
            }
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            print_status(f"Configuration exported to: {output_file}", "success")
        else:
            print_json(config_data)

    except Exception as e:
        print_status(f"Export failed: {e}", "error")
        sys.exit(1)


@rl.command()
@click.argument("log_lines", type=int, default=50)
def logs(log_lines: int):
    """Show recent RL system logs."""
    print_status(f"Showing last {log_lines} log lines...")

    try:
        # Try to read logs from common locations
        log_paths = [
            "/app/logs/rl_system/rl_system.log",
            "logs/rl_system/rl_system.log",
            "./logs/rl_system.log"
        ]

        log_file = None
        for path in log_paths:
            if Path(path).exists():
                log_file = Path(path)
                break

        if not log_file:
            print_status("No log file found", "warning")
            print_status("Expected locations:", "info")
            for path in log_paths:
                print(f"  - {path}")
            return

        # Read last N lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-log_lines:]

        if RICH_AVAILABLE and console:
            console.print(f"\n[blue]Recent logs from {log_file}:[/blue]")
            for line in recent_lines:
                console.print(line.rstrip())
        else:
            print(f"\nRecent logs from {log_file}:")
            for line in recent_lines:
                print(line.rstrip())

    except Exception as e:
        print_status(f"Failed to read logs: {e}", "error")


@rl.command()
def dashboard():
    """Show monitoring dashboard URL."""
    print_status("RL System Monitoring Dashboard")

    try:
        manager = get_rl_system_manager()
        if manager and hasattr(manager, 'config'):
            port = manager.config.get("dashboard_port", 8001)
        else:
            port = 8001

        dashboard_url = f"http://localhost:{port}"

        if RICH_AVAILABLE and console:
            panel = Panel(
                f"Dashboard URL: [blue]{dashboard_url}[/blue]\n\n"
                f"Available endpoints:\n"
                f"• Status: [blue]{dashboard_url}/[/blue]\n"
                f"• Health: [blue]http://localhost:8000/health/rl[/blue]\n"
                f"• Metrics: [blue]http://localhost:8000/metrics/rl[/blue]",
                title="Monitoring Dashboard",
                expand=False
            )
            console.print(panel)
        else:
            print(f"Dashboard URL: {dashboard_url}")
            print("Health endpoint: http://localhost:8000/health/rl")
            print("Metrics endpoint: http://localhost:8000/metrics/rl")

    except Exception as e:
        print_status(f"Failed to get dashboard info: {e}", "error")


@rl.command()
@click.option("--watch", "-w", is_flag=True, help="Watch for changes")
@click.option("--interval", "-i", type=int, default=5, help="Update interval in seconds")
def monitor(watch: bool, interval: int):
    """Monitor RL system in real-time."""
    if not watch:
        # Single check
        status()
        return

    print_status(f"Monitoring RL system (updating every {interval}s, press Ctrl+C to stop)...")

    try:
        while True:
            if RICH_AVAILABLE and console:
                console.clear()
                console.print(f"[bold]RL System Monitor[/bold] - {datetime.now().strftime('%H:%M:%S')}")
                console.print("=" * 60)

            # Get current status
            health = get_rl_health_status()
            metrics_data = get_rl_metrics()

            # Display status
            if RICH_AVAILABLE and console:
                table = create_status_table(health)
                if table:
                    console.print(table)

                # Show key metrics
                performance = metrics_data.get("performance", {})
                if performance:
                    console.print(f"\nKey Metrics:")
                    console.print(f"  Total Selections (1h): {performance.get('total_selections_1h', 0)}")
                    console.print(f"  Error Rate (1h): {performance.get('error_rate_1h', 0):.2%}")

            else:
                print(f"Status: {health.get('status', 'unknown')}")
                print(f"Running: {health.get('running', False)}")

            import time
            time.sleep(interval)

    except KeyboardInterrupt:
        print_status("\nMonitoring stopped", "info")
    except Exception as e:
        print_status(f"Monitoring failed: {e}", "error")
        sys.exit(1)


# Entry point for CLI integration
def add_rl_commands(cli_app):
    """Add RL commands to existing CLI application"""
    if hasattr(cli_app, 'add_command'):
        cli_app.add_command(rl)
    else:
        # For other CLI frameworks, expose individual commands
        return {
            'rl-status': status,
            'rl-metrics': metrics,
            'rl-migrate': migrate,
            'rl-rollback': rollback,
            'rl-validate': validate,
            'rl-logs': logs,
            'rl-dashboard': dashboard,
            'rl-monitor': monitor
        }


if __name__ == "__main__":
    # Standalone CLI execution
    rl()