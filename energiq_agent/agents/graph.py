"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from pathlib import Path

from langchain_openai import ChatOpenAI
from typing import Annotated, Optional

from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict
from langgraph.types import Command
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
import shutil
import json
from energiq_agent import DATA_DIR
from energiq_agent.agents.critic import Critic
from energiq_agent.agents.executor import (
    Executor,
)
from energiq_agent.agents.planner import Planner
import pandapower as pp

from uuid import uuid4

from energiq_agent.tools.pandapower import (
    read_network,
    get_network_status,
    update_switch_status,
    curtail_load,
    add_battery,
)


class State(TypedDict):
    network_file_path: str
    reason_the_violation_occurred: Optional[str]
    editing_network_file_path: Optional[str]
    messages: Annotated[list, add_messages]
    network: Optional[pp.pandapowerNet]
    work_dir: Optional[str]
    action_plan: Optional[str]
    violation_before_action: Optional[dict]
    violation_after_action: Optional[dict]
    feedback: Optional[str]
    log: Optional[list[str]]
    iter: int


llm = ChatOpenAI(
    base_url="http://localhost:11434/v1/", api_key="EMPTY", model="qwen3:32b"
)

tools = [update_switch_status, curtail_load, add_battery]

_executor = Executor.create(llm)
_critic = Critic.create(llm)


def cache_network(state: State):
    short_uuid = str(uuid4())[:6]
    dst = DATA_DIR / "networks" / "editing" / short_uuid / "network.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(state["network_file_path"], str(dst.absolute()))
    return Command(
        update={
            "messages": state["messages"]
            + [
                {"role": "system", "content": f"Copied network to editing folder {dst}"}
            ],
            "editing_network_file_path": str(dst.absolute()),
            "work_dir": str(dst.parent.absolute()),
            "network": read_network(str(dst.absolute())),
        },
    )


def planner(state: State):
    work_dir = Path(state["work_dir"])
    state["network"] = read_network(state["editing_network_file_path"])
    status = get_network_status(state["network"])

    with open(work_dir / "status_before_action.json", "w") as f:
        json.dump(status, f)

    messages = [
        {"role": "system", "content": Planner.prompt()},
        {"role": "user", "content": f"network status: {status}"},
    ]

    if state.get("reason_the_violation_occurred", None) is not None:
        messages.append(
            {
                "role": "user",
                "content": f"This is the reason the violation occurred: {state['reason_the_violation_occurred']}",
            }
        )

    if state.get("feedback", None) is not None:
        messages.append(
            {
                "role": "user",
                "content": f"This is your previous proposed actions: {state['log']}",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"This is the action feedback from critic: {state['feedback']}. \n Please consider this feedback and give new complete action plan.",
            }
        )

    plan = llm.invoke(messages).content
    log = state.get("log", [])
    log_entry = f"\n--- Planner Round {len(log) + 1} ---\n{plan}"

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

    iter = state.get("iter", 0)

    return Command(
        update={
            "action_plan": plan,
            "network": state["network"],
            "log": log + [log_entry],
            "violation_before_action": violations,
            "iter": iter + 1,
        },
    )


def executor(state: State):
    action = state["action_plan"]

    work_dir = Path(state["work_dir"])

    action_result = _executor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
            Network File Path: {state["editing_network_file_path"]}"
            
            Action Plan: {action}""",
                },
            ]
        }
    )
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

    pp.to_json(network, work_dir / "editing_network_file_path")

    with open(work_dir / "status_after_action.json", "w") as f:
        json.dump(status, f)

    return Command(
        update={
            "action_result": action_result,
            "network": network,
            "violation_after_action": violations,
        },
    )


def critic(state: State):
    action = state["action_plan"]

    before_action_status = json.loads(
        open(Path(state["work_dir"]) / "status_before_action.json").read()
    )
    after_action_status = json.loads(
        open(Path(state["work_dir"]) / "status_after_action.json").read()
    )

    feedback = llm.invoke(
        [
            {"role": "system", "content": Critic.prompt()},
            {
                "role": "user",
                "content": f"""
                
                Before action status: {before_action_status}"
                
                After action status: {after_action_status}"

                Action Plan: {action}""",
            },
        ]
    ).content

    return Command(
        update={
            "feedback": feedback,
        },
    )


def condition_fn(state: State):
    if state["iter"] >= 3:
        return "end"
    if (
        len(state["violation_after_action"]["voltage"]) > 0
        or len(state["violation_after_action"]["thermal"]) > 0
    ):
        return "planner"
    else:
        return "end"


workflow = StateGraph(State)
workflow.add_node("cache_network", cache_network)
workflow.add_node("planner", RunnableLambda(planner))
workflow.add_node("executor", executor)
workflow.add_node("critic", RunnableLambda(critic))
workflow.set_entry_point("cache_network")
workflow.add_edge("cache_network", "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "critic")
workflow.add_conditional_edges("critic", condition_fn)
graph = workflow.compile()
