from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pandapower as pp
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.types import Command

from energiq_agent import DATA_DIR
from energiq_agent.agents.executor import Executor
from energiq_agent.agents.planner import Planner
from energiq_agent.schemas import State
from energiq_agent.tools.pandapower import get_network_status, read_network

# Initialize the language model
llm = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key="AIzaSyBkNL-ERU4mjt0AKRiD7EjvP0tqfZtvthk",
    model="gemini-2.5-flash"
)


def cache_network(state: State):
    """Copies the initial network to a temporary editing directory."""
    short_uuid = str(uuid4())[:6]
    dst = DATA_DIR / "networks" / "editing" / short_uuid / "network.json"
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
    status = get_network_status(state["network"])

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

    return Command(
        update={
            "action_plan": plan,
            "violation_before_action": violations,
        }
    )


def executor(state: State):
    """Executes the structured plan and updates the network state."""
    plan = state["action_plan"]
    work_dir = Path(state["work_dir"])

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

    workflow.set_entry_point("cache_network")
    workflow.add_edge("cache_network", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {"planner": "planner", "summarizer": "summarizer"},
    )
    workflow.add_edge("summarizer", "__end__")

    return workflow


if __name__ == "__main__":
    workflow = get_workflow()
    graph = workflow.compile()
