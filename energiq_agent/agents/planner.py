from energiq_agent.agents.prompts import PLANNER_PROMPT
from energiq_agent.tools.pandapower import (
    update_switch_status,
    curtail_load,
    add_battery,
)


class Planner:
    @classmethod
    def prompt(cls):
        return PLANNER_PROMPT

    @classmethod
    def get_tools(cls):
        return [
            update_switch_status,
            add_battery,
            curtail_load,
        ]
