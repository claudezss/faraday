"""
CLI interface for EnergiQ-Agent to fix power grid violations.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import pandapower as pp
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from energiq_agent.agents.graph import get_workflow
from energiq_agent.schemas import State
from energiq_agent.tools.pandapower import (
    get_network_status,
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


def print_violations(violations: Dict[str, Any], title: str = "Violations") -> None:
    """Print violations in a formatted table."""
    voltage_violations = violations.get("voltage", [])
    thermal_violations = violations.get("thermal", [])

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
                if v["v_mag_pu"] > thresholds.high_violation_upper
                or v["v_mag_pu"] < thresholds.high_violation_lower
                else "🟡 Medium"
            )
            table.add_row(str(v["bus_idx"]), f"{v['v_mag_pu']:.3f}", severity)

        console.print(table)

    if thermal_violations:
        table = Table(title="Thermal Violations")
        table.add_column("Element", justify="left")
        table.add_column("Loading (%)", justify="right")
        table.add_column("Severity", justify="center")

        for v in thermal_violations:
            severity = "🔴 Critical" if v["loading"] > 120 else "🟡 Medium"
            table.add_row(v["line_name"], f"{v['loading']:.1f}", severity)

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
    state = State(
        network_file_path=str(network_path),
        editing_network_file_path=None,
        work_dir=None,
        network=None,
        violation_before_action=None,
        violation_after_action=None,
        messages=[],
        action_plan=None,
        executed_actions=None,
        summary=None,
        explanation=None,
        iter=0,
    )

    # Get workflow
    workflow = get_workflow()
    graph = workflow.compile()

    console.print(
        f"🔧 Starting violation fix workflow for: {network_path.name}", style="cyan"
    )

    # Initial network analysis
    net = read_network(str(network_path))
    pp.runpp(net)
    initial_status = get_network_status(net)

    # Extract initial violations
    thresholds = get_voltage_thresholds()
    initial_violations = {
        "voltage": [
            {"bus_idx": bus["index"], "v_mag_pu": bus["v_mag_pu"]}
            for bus in initial_status.get("bus_status", [])
            if bus["v_mag_pu"] > thresholds.v_max or bus["v_mag_pu"] < thresholds.v_min
        ],
        "thermal": [
            {"line_name": line["name"], "loading": line["loading_percent"]}
            for line in initial_status.get("line_status", [])
            if line["loading_percent"] > 100
        ],
    }

    print_violations(initial_violations, "Initial Violations")

    if not initial_violations["voltage"] and not initial_violations["thermal"]:
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
            result = graph.invoke(state)

            progress.update(task, description="Workflow completed")

        # Display results
        console.print("\n📊 Workflow Results:", style="cyan")

        # Show executed actions
        executed_actions = result.get("executed_actions", [])
        print_actions(executed_actions)

        # Show final violations
        final_violations = result.get(
            "violation_after_action", {"voltage": [], "thermal": []}
        )
        print_violations(final_violations, "Final Violations")

        # Show summary
        summary = result.get("summary", "")
        if summary:
            console.print(f"\n📝 Summary: {summary}", style="blue")

        # Show explanation
        explanation = result.get("explanation", "")
        if explanation and verbose:
            console.print(f"\n💡 Explanation: {explanation}", style="blue")

        # Final status
        total_final_violations = len(final_violations["voltage"]) + len(
            final_violations["thermal"]
        )
        total_initial_violations = len(initial_violations["voltage"]) + len(
            initial_violations["thermal"]
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
        description="Fix power grid violations using EnergiQ-Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  energiq-agent fix network.json
  energiq-agent fix network.json --verbose
  energiq-agent fix network.json --max-iterations 10
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
