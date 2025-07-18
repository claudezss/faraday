"""
Planning node for generating action plans to fix violations.
"""

import json
from pathlib import Path

from langgraph.types import Command

from faraday.agents.planner import Planner
from faraday.agents.workflow.state import State
from faraday.tools.pandapower import (
    get_network_status,
    get_advanced_network_status,
    get_optimized_network_status,
)
from ..config import llm


def planner(state: State):
    """Generates a structured plan (a list of tool calls) to fix violations."""
    work_dir = Path(state["work_dir"])

    # Create planning network by applying successful changes to original network
    planning_network = state["original_network"].deepcopy()

    # Apply previously successful changes to planning network
    if state.get("successful_changes"):
        # This would need to be implemented based on how changes are tracked
        # For now, use current_network which represents the last successful state
        planning_network = state["current_network"].deepcopy()

    # Use planning network for status analysis
    planning_network_for_analysis = planning_network

    # Phase 3: Intelligent context-aware planning mode selection
    total_buses = len(planning_network_for_analysis.bus)

    # Determine optimal context mode based on network characteristics
    if total_buses > 2000:
        # Very large networks: Use ultra-compact overview
        context_mode = "adaptive"
        max_tokens = 500
    elif total_buses > 500:
        # Large networks: Use graph-based representation for better action understanding
        context_mode = "graph"
        max_tokens = 1500
    elif total_buses > 100:
        # Medium networks: Use hierarchical representation
        context_mode = "hierarchical"
        max_tokens = 3000
    else:
        # Small networks
        context_mode = "full"
        max_tokens = None

    # Get optimized network status using advanced features on planning network
    if context_mode == "full":
        status = get_network_status(planning_network_for_analysis, hierarchical=True)
    elif context_mode == "hierarchical":
        status = get_network_status(planning_network_for_analysis, hierarchical=True)
    elif total_buses <= 100:
        # For small networks, use optimization-focused representation
        status = get_optimized_network_status(
            planning_network_for_analysis, context_mode="optimization"
        )
    else:
        # Use advanced Phase 3 representation for large networks
        status = get_advanced_network_status(
            planning_network_for_analysis,
            max_tokens=max_tokens,
            context_mode=context_mode,
        )

    with open(work_dir / "status_before_action.json", "w") as f:
        json.dump(status, f)

    model_with_tools = llm.bind_tools(Planner.get_tools())

    messages = [
        {"role": "system", "content": Planner.prompt()},
        {"role": "user", "content": f"Network status: {status}`"},
    ]

    if state.get("action_plan") and state.get("violation_after_action"):
        messages.append(
            {
                "role": "user",
                "content": f"You previously tried this action plan: {state['action_plan']}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"However, it resulted in these violations: {state['violation_after_action']}. Please generate a new plan to fix them.",
            }
        )

    ai_message = model_with_tools.invoke(messages)
    plan = ai_message.tool_calls

    # Extract violations based on status format (Enhanced format detection)
    violations = _extract_violations_from_status(status)

    return Command(
        update={
            "messages": state.get("messages", []) + messages,
            "action_plan": plan,
            "violation_before_action": violations,
        }
    )


def _extract_violations_from_status(status):
    """Extract violations from different status formats."""
    violations = {"voltage": [], "thermal": []}

    if "optimization_context" in status:
        # Optimization context format
        opt_context = status["optimization_context"]
        violations["voltage"] = [
            {"bus_idx": v["bus"], "v_mag_pu": v["v_mag_pu"]}
            for v in opt_context.get("voltage_violations", [])
        ]
        violations["thermal"] = [
            {"line_name": v["name"], "loading": v["loading_percent"]}
            for v in opt_context.get("thermal_violations", [])
        ]

    elif "graph_representation" in status:
        # Graph-based representation
        for violation in status["graph_representation"].get("violations", []):
            if violation["type"] == "voltage_violation":
                violations["voltage"].append(
                    {"bus_idx": violation["bus"], "v_mag_pu": violation["value"]}
                )
            elif violation["type"] == "thermal_violation":
                violations["thermal"].append(
                    {
                        "line_name": violation.get(
                            "name", f"element_{violation.get('from_bus')}"
                        ),
                        "loading": violation["value"],
                    }
                )

    elif "violation_details" in status:
        # Hierarchical status format
        for zone_name, zone_data in status.get("violation_details", {}).items():
            violations["voltage"].extend(
                [
                    {"bus_idx": v["bus"], "v_mag_pu": v["v_mag_pu"]}
                    for v in zone_data.get("voltage_violations", [])
                ]
            )
            violations["thermal"].extend(
                [
                    {"line_name": v["name"], "loading": v["loading_percent"]}
                    for v in zone_data.get("thermal_violations", [])
                ]
            )

    elif "level" in status and status["level"] == "overview":
        # Overview level representation
        for i, v_mag in enumerate(status.get("critical_buses", [])):
            violations["voltage"].append({"bus_idx": i, "v_mag_pu": v_mag})
        # Thermal violations count only available in overview

    else:
        # Standard status format
        violations = {
            "voltage": [
                {"bus_idx": bus["index"], "v_mag_pu": bus["v_mag_pu"]}
                for bus in status.get("bus_status", [])
                if bus["v_mag_pu"] > 1.05 or bus["v_mag_pu"] < 0.95
            ],
            "thermal": [
                {"line_name": line["name"], "loading": line["loading_percent"]}
                for line in status.get("line_status", [])
                if line["loading_percent"] > 100
            ],
        }

    return violations
