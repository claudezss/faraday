from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pandapower as pp
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.types import Command

from energiq_agent import WORKSPACE_NETWORKS, WORKSPACE
from energiq_agent.agents.executor import Executor
from energiq_agent.agents.planner import Planner
from energiq_agent.schemas import State
from energiq_agent.tools.pandapower import (
    get_network_status,
    read_network,
    get_advanced_network_status,
    get_optimized_network_status,
)

# Initialize the language model
llm = ChatOpenAI(
    base_url=os.environ.get("OPENAI_API_BASE") or "http://localhost:11434/v1/",
    api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
    model=os.environ.get("OPENAI_MODEL") or "qwen3:32b",
)


def cache_network(state: State):
    """Copies the initial network to a temporary editing directory."""
    short_uuid = str(uuid4())[:6]
    dst = WORKSPACE_NETWORKS / "editing" / short_uuid / "network.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(state["network_file_path"], str(dst.absolute()))
    return Command(
        update={
            "editing_network_file_path": str(dst.absolute()),
            "work_dir": str(dst.parent.absolute()),
            "network": read_network(str(dst.absolute())),
            "executed_actions": [],  # Initialize the list of executed actions
        }
    )


def planner(state: State):
    """Generates a structured plan (a list of tool calls) to fix violations."""
    work_dir = Path(state["work_dir"])
    state["network"] = read_network(state["editing_network_file_path"])

    # Phase 3: Intelligent context-aware planning mode selection
    total_buses = len(state["network"].bus)

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
        # Small networks: Use violations_only for efficiency
        context_mode = "violations_only"
        max_tokens = None

    # Get optimized network status using advanced features
    if context_mode == "violations_only":
        status = get_network_status(state["network"], violations_only=True)
        # If no violations, get full status
        if not status.get("bus_status", []) and not status.get("line_status", []):
            status = get_network_status(state["network"])
    elif context_mode == "hierarchical":
        status = get_network_status(state["network"], hierarchical=True)
    elif total_buses <= 100:
        # For small networks, use optimization-focused representation
        status = get_optimized_network_status(
            state["network"], context_mode="optimization"
        )
    else:
        # Use advanced Phase 3 representation for large networks
        status = get_advanced_network_status(
            state["network"], max_tokens=max_tokens, context_mode=context_mode
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

    return Command(
        update={
            "messages": state.get("messages", []) + messages,
            "action_plan": plan,
            "violation_before_action": violations,
        }
    )


def executor(state: State):
    """Executes the structured plan and updates the network state."""
    plan = state["action_plan"]
    work_dir = Path(state["work_dir"])

    # Use optimized executor for small networks or plans with optimization metadata
    total_buses = len(state["network"].bus)
    use_optimized_executor = total_buses <= 100 or any(
        action.get("optimization_type") == "coordinated" for action in plan
    )

    if use_optimized_executor:
        executed = Executor.execute_optimized(state["editing_network_file_path"], plan)
    else:
        executed = Executor.execute(state["editing_network_file_path"], plan)

    # Append the newly executed actions to the existing list
    all_executed_actions = state.get("executed_actions", []) + executed

    network = read_network(state["editing_network_file_path"])
    pp.runpp(network)
    status = get_network_status(network)

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

    pp.to_json(network, work_dir / state["editing_network_file_path"])

    with open(work_dir / "status_after_action.json", "w") as f:
        json.dump(status, f)

    return Command(
        update={
            "network": network,
            "violation_after_action": violations,
            "executed_actions": all_executed_actions,
            "iter": state.get("iter", 0) + 1,
        }
    )


def summarizer(state: State):
    """Generates a summary of the executed actions."""
    executed_actions = state.get("executed_actions", [])
    if not executed_actions:
        summary = "No actions were executed."
    else:
        action_report = "\n".join(
            [f"- {action['name']}({action['args']})" for action in executed_actions]
        )
        summary_prompt = f"Please provide a short summary of the following actions that were taken to resolve power grid violations:\n\n{action_report}"
        summary = llm.invoke(summary_prompt).content

    return Command(update={"summary": summary})


def explainer(state: State):
    """Generates an explanation of the actions and saves the conversation data."""
    executed_actions = state.get("executed_actions", [])
    if not executed_actions:
        explanation = "No actions were executed, so no explanation is needed."
    else:
        action_report = "\n".join(
            [f"- {action['name']}({action['args']})" for action in executed_actions]
        )
        explanation_prompt = f"""Please provide a brief explanation of why the following actions helped to resolve the power grid violations.

Initial Violations:
{state["violation_before_action"]}

Executed Actions:
{action_report}

Final Violations:
{state["violation_after_action"]}
"""
        explanation = llm.invoke(explanation_prompt).content

    # --- Enhanced Data Collection for Fine-tuning ---
    try:
        from energiq_agent.training.data_collector import EnhancedTrainingDataCollector

        # Initialize enhanced collector
        training_dir = WORKSPACE / "training_data_enhanced"
        collector = EnhancedTrainingDataCollector(training_dir)

        # Collect comprehensive training sample
        training_sample = collector.collect_training_sample(
            state=state,
            executed_actions=executed_actions,
            explanation=explanation,
            system_prompt=Planner.prompt(),
        )

        # Save in multiple formats
        collector.save_training_sample(training_sample)

    except Exception as e:
        # Fallback to legacy method if enhanced collection fails
        print(f"Enhanced training data collection failed: {e}")
        training_data_path = WORKSPACE / "training_data.json"

        net = read_network(state["network_file_path"])
        status = get_network_status(net)
        conversation_data = {
            "system_prompt": Planner.prompt(),
            "user_prompt": f"""Network Status: 
{status}
""",
            "assistant_response": {
                "actions": executed_actions,
                "explanation": explanation,
            },
        }

        # Append the new data to the JSON file
        if training_data_path.exists():
            with open(training_data_path, "r+") as f:
                data = json.load(f)
                data.append(conversation_data)
                f.seek(0)
                json.dump(data, f, indent=2)
        else:
            with open(training_data_path, "w") as f:
                json.dump([conversation_data], f, indent=2)

    return Command(update={"explanation": explanation})


def should_continue(state: State):
    """Determines the next step in the workflow."""
    if state["iter"] >= 5:
        return "summarizer"  # Go to summarizer if max iterations are reached
    if (
        len(state["violation_after_action"]["voltage"]) > 0
        or len(state["violation_after_action"]["thermal"]) > 0
    ):
        return "planner"
    return "summarizer"  # Go to summarizer if violations are resolved


def get_workflow():
    """Builds the LangGraph workflow."""
    workflow = StateGraph(State)
    workflow.add_node("cache_network", cache_network)
    workflow.add_node("planner", RunnableLambda(planner))
    workflow.add_node("executor", executor)
    workflow.add_node("summarizer", RunnableLambda(summarizer))
    workflow.add_node("explainer", RunnableLambda(explainer))

    workflow.set_entry_point("cache_network")
    workflow.add_edge("cache_network", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {"planner": "planner", "summarizer": "summarizer"},
    )
    workflow.add_edge("summarizer", "explainer")
    workflow.add_edge("explainer", "__end__")

    return workflow


if __name__ == "__main__":
    workflow = get_workflow()
    graph = workflow.compile()
