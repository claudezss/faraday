"""
CLI interface for faraday to fix power grid violations.
"""

import argparse
import sys
from pathlib import Path

import pandapower as pp
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from faraday.tools.pandapower import (
    get_violations,
)
from faraday.agents.workflow.graph import get_workflow
from faraday.agents.workflow.state import State, Violation
from faraday.tools.pandapower import (
    read_network,
    get_voltage_thresholds,
    set_voltage_thresholds,
)


console = Console()


def validate_network_file(network_path: Path) -> bool:
    """Validate that the network file exists and is a valid pandapower network."""
    if not network_path.exists():
        console.print(f"❌ Network file not found: {network_path}", style="red")
        return False

    try:
        net = pp.from_json(str(network_path))
        pp.runpp(net)
        return True
    except Exception as e:
        console.print(f"❌ Invalid network file: {e}", style="red")
        return False


def print_violations(violation: Violation, title: str = "Violations") -> None:
    """Print violations in a formatted table."""
    voltage_violations = violation.voltage
    thermal_violations = violation.thermal

    if not voltage_violations and not thermal_violations:
        console.print(f"✅ {title}: No violations found", style="green")
        return

    console.print(f"⚠️  {title}:", style="yellow")

    if voltage_violations:
        table = Table(title="Voltage Violations")
        table.add_column("Bus", justify="right")
        table.add_column("Voltage (pu)", justify="right")
        table.add_column("Severity", justify="center")

        for v in voltage_violations:
            thresholds = get_voltage_thresholds()
            severity = (
                "🔴 Critical"
                if v.v_mag_pu > thresholds.high_violation_upper
                or v.v_mag_pu < thresholds.high_violation_lower
                else "🟡 Medium"
            )
            table.add_row(str(v.bus_idx), f"{v.v_mag_pu:.3f}", severity)

        console.print(table)

    if thermal_violations:
        table = Table(title="Thermal Violations")
        table.add_column("Element", justify="left")
        table.add_column("Loading (%)", justify="right")
        table.add_column("Severity", justify="center")

        for v in thermal_violations:
            severity = "🔴 Critical" if v.loading_percent > 120 else "🟡 Medium"
            table.add_row(v.name, f"{v.loading_percent:.1f}", severity)

        console.print(table)


def print_actions(actions: list, title: str = "Executed Actions") -> None:
    """Print executed actions in a formatted table."""
    if not actions:
        console.print(f"ℹ️  {title}: No actions executed", style="blue")
        return

    table = Table(title=title)
    table.add_column("Action", justify="left")
    table.add_column("Target", justify="left")
    table.add_column("Parameters", justify="left")

    for action in actions:
        action_name = action.get("name", "Unknown")
        args = action.get("args", {})

        target = (
            args.get("switch_name")
            or args.get("load_name")
            or f"Bus {args.get('bus_index', 'Unknown')}"
        )

        params = []
        if "closed" in args:
            params.append(f"closed={args['closed']}")
        if "curtail_percent" in args:
            params.append(f"curtail={args['curtail_percent']}%")
        if "max_energy_kw" in args:
            params.append(f"capacity={args['max_energy_kw']}kW")

        table.add_row(action_name, target, ", ".join(params))

    console.print(table)


def run_workflow(
    network_path: Path, max_iterations: int = 5, verbose: bool = False
) -> bool:
    """Run the complete workflow to fix violations."""

    # Initialize state
    state = State(network_file_path=str(network_path), max_iterations=max_iterations)

    # Get workflow
    workflow = get_workflow()
    graph = workflow.compile()

    console.print(
        f"🔧 Starting violation fix workflow for: {network_path.name}", style="cyan"
    )

    # Initial network analysis
    net = read_network(str(network_path))
    pp.runpp(net)

    initial_violations = get_violations(net)

    print_violations(initial_violations, "Initial Violations")

    if not initial_violations.voltage and not initial_violations.thermal:
        console.print(
            "✅ No violations found. Network is already compliant.", style="green"
        )
        return True

    # Run workflow
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Processing network violations...", total=None)

            # Execute the workflow
            result = State(**graph.invoke(state))

            progress.update(task, description="Workflow completed")

        # Display results
        console.print("\n📊 Workflow Results:", style="cyan")

        # Show executed actions
        executed_actions = result.all_executed_actions
        print_actions(executed_actions)

        # Show final violations
        final_violations = result.iteration_results[-1].viola_after
        print_violations(final_violations, "Final Violations")

        # Show summary
        summary = result.summary
        if summary:
            console.print(f"\n📝 Summary: {summary}", style="blue")

        # Show explanation
        explanation = result.explanation
        if explanation and verbose:
            console.print(f"\n💡 Explanation: {explanation}", style="blue")

        # Final status
        total_final_violations = len(final_violations.voltage) + len(
            final_violations.thermal
        )

        total_initial_violations = len(initial_violations.voltage) + len(
            initial_violations.thermal
        )

        if total_final_violations == 0:
            console.print("✅ All violations successfully resolved!", style="green")
            return True
        elif total_final_violations < total_initial_violations:
            console.print(
                f"⚠️  Partially resolved: {total_initial_violations - total_final_violations} violations fixed, {total_final_violations} remaining",
                style="yellow",
            )
            return True
        else:
            console.print(
                f"❌ Unable to resolve violations after {result.get('iter', 0)} iterations",
                style="red",
            )
            return False

    except Exception as e:
        console.print(f"❌ Error during workflow execution: {e}", style="red")
        if verbose:
            console.print_exception()
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fix power grid violations using faraday",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  faraday fix network.json
  faraday fix network.json --verbose
  faraday fix network.json --max-iterations 10
        """,
    )

    parser.add_argument(
        "network_file", type=Path, help="Path to pandapower network JSON file"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of iterations to attempt (default: 5)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output including explanations",
    )

    parser.add_argument(
        "--output", "-o", type=Path, help="Output directory for results (optional)"
    )

    parser.add_argument(
        "--v-max",
        type=float,
        default=1.05,
        help="Maximum allowed voltage in per unit (default: 1.05)",
    )

    parser.add_argument(
        "--v-min",
        type=float,
        default=0.95,
        help="Minimum allowed voltage in per unit (default: 0.95)",
    )

    args = parser.parse_args()

    # Set voltage thresholds from command line arguments
    set_voltage_thresholds(v_max=args.v_max, v_min=args.v_min)

    # Display current voltage thresholds
    console.print(
        f"⚙️  Using voltage thresholds: v_max={args.v_max}, v_min={args.v_min}",
        style="blue",
    )

    # Validate network file
    if not validate_network_file(args.network_file):
        sys.exit(1)

    # Run workflow
    success = run_workflow(
        args.network_file, max_iterations=args.max_iterations, verbose=args.verbose
    )

    if not success:
        sys.exit(1)

    console.print("\n🎉 Workflow completed successfully!", style="green")


if __name__ == "__main__":
    main()
