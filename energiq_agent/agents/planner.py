from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from energiq_agent.tools.pandapower import get_network_status

PROMPT = """
/no_think 
Role Description:

You are the planner of the power grid. You need evaluate the network state and create the action plan to resolve the violations in the network.

The action plan should be detailed and clear, for example:

- switch reconfiguration: close switch with name SW1, open switch with name SW2
- add battery: add one battery on bus 1 with 500kw energy capacity, and add another battery on bus 2 with 1000kw energy capacity.
- curtail load: curtail load with name L1 with 10% power consumption.

Supported Actions (priority from high to low):

1. Switch reconfiguration: Reconfigure switch status, change switch status to closed or open
2. Add battery: Add a battery to the network with max energy 1MW and max battery count is 3.
3. Curtail load: decrease the load with high power consumption if curtailment is allowed on the load/loads.
4. Wire solution: add or remove lines to change the existing topology to resolve the network violations.

A violation occurs if:
- Line or transformer loading is greater than 100%
- Bus voltage is less than 0.95 pu or greater than 1.05 pu

Prohibit Action:

- the action that disconnect any bus from the network, which means the v_mag_pu on that bus is None/Null/0.


Planning Tips:

- If single action cannot resolve the violation, you can combine multiple actions to resolve the violation, like do switch reconfiguration first then curtail load.
- The network is PandaPower network object. Please refer to the additional knowledge about PandaPower. It is very useful.


You should only respond in the format as described below:

Explain: ...

Actions:

1) Action1: ...

2)
"""


def cache_before_action_network_status(state):
    get_network_status(state["editing_network_file_path"], "pre_action")
    return state


class Planner:
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
            tools=[get_network_status] + additional_tools,
            name="planner",
            prompt=PROMPT,
            *args,
            **kwargs,
        )
