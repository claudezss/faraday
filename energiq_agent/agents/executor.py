from typing import List, Dict, Any

from energiq_agent.tools.pandapower import (
    curtail_load,
    add_battery,
    update_switch_status,
)

# A mapping from tool names to the actual functions
TOOL_MAPPING = {
    "curtail_load": curtail_load,
    "add_battery": add_battery,
    "update_switch_status": update_switch_status,
}


class Executor:
    @staticmethod
    def execute(
        network_path: str, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Executes a list of tool calls and returns the list of executed actions.

        Args:
            network_path: The path to the network file to be modified.
            tool_calls: A list of tool calls from the planner.

        Returns:
            A list of the tool calls that were successfully executed.
        """
        if not tool_calls:
            return []

        executed_actions = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("args")

            if tool_name in TOOL_MAPPING:
                tool_function = TOOL_MAPPING[tool_name]
                # Add the network_path to the arguments for each tool call
                arguments["network_path"] = network_path
                try:
                    tool_function(**arguments)
                    executed_actions.append(tool_call)
                except Exception as e:
                    print(f"Error executing tool {tool_name}: {e}")
            else:
                # Handle the case where the tool is not found
                print(f"Warning: Tool '{tool_name}' not found.")
        return executed_actions
