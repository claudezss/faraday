import pandapower as pp

from energiq_agent.agents.critic import Critic
from energiq_agent.agents.executor import Executor
from energiq_agent.agents.planner import Planner
from energiq_agent.simulator.api import (
    get_load_status,
    get_sync_generator_status,
    get_switch_status,
    get_bus_status,
    get_line_status,
)

import json


def get_network_status(net) -> dict:
    pp.runpp(net)
    load_status = get_load_status(net)
    sync_generator_status = get_sync_generator_status(net)
    switch_status = get_switch_status(net)
    bus_status = get_bus_status(net)
    line_status = get_line_status(net)

    return {
        "load_status": load_status,
        "sync_generator_status": sync_generator_status,
        "switch_status": switch_status,
        "bus_status": bus_status,
        "line_status": line_status,
    }


planner = Planner(reasoning_effort="high", model="qwen/qwen3-32b")
executor = Executor()
critic = Critic(model="qwen/qwen3-32b")

net = pp.from_json("D:\\Dev\\repo\\EnergiQ-Agent\\data\\networks\\cigre_mv\\net.json")

state = get_network_status(net)

plan = planner.run(state)

print(plan)

for i in range(10):
    execution_result = executor.run(plan)

    print(execution_result)

    new_state = json.loads(open(execution_result).read())

    feedback = critic.run(
        f"""
        initial_network_state_dict: {state}
        
        executed_action_dict: {plan}
        
        final_network_state_dict: {new_state}
        """
    )
    print(feedback)

    plan = planner.run(
        f"""
    Network State after executing action: {new_state}
    
    Critic Feedback:
    {feedback}
    """
    )

    print(plan)

    state = new_state

    print(f"Iter {i}")
