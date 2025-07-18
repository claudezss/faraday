"""
Execution node for executing action plans and validating results.
"""

import json
from pathlib import Path

import pandapower as pp
from langgraph.types import Command

from faraday.agents.executor import Executor
from faraday.schemas import State
from faraday.tools.pandapower import read_network, get_network_status


def executor(state: State):
    """Executes the structured plan on a copy of the current network and validates results."""
    plan = state["action_plan"]
    work_dir = Path(state["work_dir"])

    # Create a temporary copy of the current network for execution
    execution_network = state["current_network"].deepcopy()

    # Save temporary network to file for executor
    temp_network_path = work_dir / "temp_execution_network.json"
    pp.to_json(execution_network, str(temp_network_path))

    # Use optimized executor for small networks or plans with optimization metadata
    total_buses = len(execution_network.bus)
    use_optimized_executor = total_buses <= 100 or any(
        action.get("optimization_type") == "coordinated" for action in plan
    )

    if use_optimized_executor:
        executed = Executor.execute_optimized(str(temp_network_path), plan)
    else:
        executed = Executor.execute(str(temp_network_path), plan)

    # Load the modified network and validate
    modified_network = read_network(str(temp_network_path))

    try:
        pp.runpp(modified_network)
        execution_successful = True
    except Exception as e:
        execution_successful = False
        print(f"Power flow failed after execution: {e}")

    if execution_successful:
        status = get_network_status(modified_network)

        violations = {
            "voltage": [
                {"bus_idx": bus["index"], "v_mag_pu": bus["v_mag_pu"]}
                for bus in status["bus_status"]
                if bus["v_mag_pu"] > 1.05 or bus["v_mag_pu"] < 0.95
            ],
            "thermal": [
                {"line_name": line["name"], "loading": line["loading_percent"]}
                for line in status["line_status"]
                if line["loading_percent"] > 100
            ],
        }

        # Update current network state only if execution was successful
        current_network = modified_network
        # Save to editing network file
        pp.to_json(current_network, state["editing_network_file_path"])

        # Track this as a successful iteration
        iteration_result = {
            "iter": state.get("iter", 0) + 1,
            "executed_actions": executed,
            "violations_before": state.get("violation_before_action", {}),
            "violations_after": violations,
            "successful": True,
        }

        successful_changes = state.get("successful_changes", []) + executed

    else:
        # Execution failed - keep current state, report failure
        violations = state.get("violation_after_action", {})
        current_network = state["current_network"]
        status = {"error": "Power flow failed after execution"}

        iteration_result = {
            "iter": state.get("iter", 0) + 1,
            "executed_actions": executed,
            "violations_before": state.get("violation_before_action", {}),
            "violations_after": violations,
            "successful": False,
            "error": "Power flow failed",
        }

        successful_changes = state.get("successful_changes", [])

    # Append the newly executed actions to the existing list
    all_executed_actions = state.get("executed_actions", []) + executed
    all_iteration_results = state.get("iteration_results", []) + [iteration_result]

    with open(work_dir / "status_after_action.json", "w") as f:
        json.dump(status, f)

    return Command(
        update={
            "current_network": current_network,
            "network": current_network,  # Keep for backward compatibility
            "violation_after_action": violations,
            "executed_actions": all_executed_actions,
            "iteration_results": all_iteration_results,
            "successful_changes": successful_changes,
            "iter": state.get("iter", 0) + 1,
        }
    )
