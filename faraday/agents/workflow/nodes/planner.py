"""
Planning node for generating action plans to fix violations.
"""

from faraday.agents.prompts import PLANNER_PROMPT
from faraday.agents.workflow.state import State, IterationResult
from faraday.tools.pandapower import (
    get_violations,
    get_json_network_status,
)
from faraday.agents.workflow.config import llm
from faraday.tools.pandapower import read_network
from faraday.tools.pandapower import (
    update_switch_status,
    curtail_load,
    add_battery,
)
import logging
import json


logger = logging.getLogger(__name__)

TOOL_MAPPING = {
    "curtail_load": curtail_load,
    "add_battery": add_battery,
    "update_switch_status": update_switch_status,
}


def planner(state: State) -> State:
    """Generates a structured plan (a list of tool calls) to fix violations."""

    logger.info(f"Iteration {state.iter_num}: Planning")

    planning_network = read_network(state.editing_network_file_path)

    network_status = get_json_network_status(planning_network)

    if not state.messages:
        state.messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {
                "role": "user",
                "content": f"""
**Network Status:**
```json
{json.dumps(network_status, indent=2)}
```
""",
            },
        ]

    agent = llm.bind_tools(list(TOOL_MAPPING.values()))

    ai_message = agent.invoke(state.messages)

    plan = ai_message.tool_calls

    executed_actions = []

    logger.info(f"Iteration {state.iter_num}: Executing Plan")
    for tool_call in plan:
        tool_name = tool_call.get("name")
        arguments = tool_call.get("args")

        if tool_name in TOOL_MAPPING:
            tool_function = TOOL_MAPPING[tool_name]
            # Add the network_path to the arguments for each tool call
            arguments["network_path"] = state.editing_network_file_path
            try:
                tool_function(**arguments)
                executed_actions.append(tool_call)
            except Exception as e:
                print(f"Error executing tool {tool_name}: {e}")
        else:
            # Handle the case where the tool is not found
            print(f"Warning: Tool '{tool_name}' not found.")

    violas = get_violations(read_network(state.editing_network_file_path))

    iter_results = IterationResult(
        iter=state.iter_num + 1,
        executed_actions=executed_actions,
        viola_before=state.iteration_results[-1].viola_after
        if state.iter_num > 0
        else get_violations(read_network(state.org_network_copy_file_path)),
        viola_after=violas,
    )

    state.iteration_results.append(iter_results)

    if len(violas.thermal) == 0 and len(violas.voltage) == 0:
        state.messages += [
            {
                "role": "user",
                "content": "Congratulations! You have successfully resolved all violations.",
            },
        ]

    else:
        state.messages += [
            {
                "role": "user",
                "content": f"You previously plan are: {state.all_executed_actions}",
            },
            {
                "role": "user",
                "content": f"However, those violations still remain: {violas.model_dump()}. "
                f"Please refine and generate a new plan to resolve violations.",
            },
        ]

    return state
