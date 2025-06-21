from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from energiq_agent.tools.pandapower import (
    curtail_load,
    add_battery,
    update_switch_status,
    run_powerflow,
    get_network_status,
)

PROMPT = """
/no_think You will receive the action plan from the power grid operator.
You need to follow the action plan to resolve the violations in the network.
You should use the network in editing to resolve the violations.
You should use existing tools and you can ignore the actions that hasn't support in tools.
Do not include any other text or explanation.

**Never** Repeat the same tool call!
"""


class Executor:
    @classmethod
    def prompt(cls):
        return PROMPT

    @classmethod
    def create(
        cls, model, additional_tools: list = None, *args, **kwargs
    ) -> CompiledGraph:
        if additional_tools is None:
            additional_tools = []
        return create_react_agent(
            model=model,
            tools=[
                curtail_load,
                add_battery,
                update_switch_status,
                run_powerflow,
                get_network_status,
            ]
            + additional_tools,
            name="executor",
            prompt=PROMPT,
            *args,
            **kwargs,
        )
