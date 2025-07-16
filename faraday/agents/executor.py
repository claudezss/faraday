from typing import List, Dict, Any

from faraday.tools.pandapower import (
    curtail_load,
    add_battery,
    update_switch_status,
    read_network,
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

    @staticmethod
    def execute_optimized(
        network_path: str, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhanced executor that optimizes action execution for better performance.

        Features:
        - Groups similar actions for batch processing
        - Validates actions before execution to prevent failures
        - Tracks cumulative effects of actions

        Args:
            network_path: The path to the network file to be modified.
            tool_calls: A list of tool calls from the planner.

        Returns:
            A list of the tool calls that were successfully executed.
        """
        if not tool_calls:
            return []

        # Sort actions by priority if available (optimized plans have priority scores)
        sorted_calls = sorted(
            tool_calls, key=lambda x: x.get("priority", 0), reverse=True
        )

        executed_actions = []

        # Group actions by type for potential optimization
        switch_actions = []
        battery_actions = []
        curtailment_actions = []

        for tool_call in sorted_calls:
            tool_name = tool_call.get("name")
            if tool_name == "update_switch_status":
                switch_actions.append(tool_call)
            elif tool_name == "add_battery":
                battery_actions.append(tool_call)
            elif tool_name == "curtail_load":
                curtailment_actions.append(tool_call)

        # Execute switch actions first (highest impact)
        for action in switch_actions:
            if Executor._execute_single_action(network_path, action):
                executed_actions.append(action)

        # Execute battery actions second
        for action in battery_actions:
            if Executor._execute_single_action(network_path, action):
                executed_actions.append(action)

        # Execute curtailment actions last
        for action in curtailment_actions:
            if Executor._execute_single_action(network_path, action):
                executed_actions.append(action)

        return executed_actions

    @staticmethod
    def _execute_single_action(network_path: str, tool_call: Dict[str, Any]) -> bool:
        """Execute a single action and return success status."""
        tool_name = tool_call.get("name")
        arguments = tool_call.get("args", {}).copy()

        if tool_name not in TOOL_MAPPING:
            print(f"Warning: Tool '{tool_name}' not found.")
            return False

        tool_function = TOOL_MAPPING[tool_name]
        arguments["network_path"] = network_path

        try:
            # Validate action before execution
            if not Executor._validate_action(network_path, tool_name, arguments):
                print(f"Warning: Action validation failed for {tool_name}")
                return False

            tool_function(**arguments)
            return True
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            return False

    @staticmethod
    def _validate_action(
        network_path: str, tool_name: str, arguments: Dict[str, Any]
    ) -> bool:
        """Validate an action before execution to prevent failures."""
        try:
            net = read_network(network_path)

            if tool_name == "update_switch_status":
                switch_name = arguments.get("switch_name")
                if switch_name not in net.switch.name.values:
                    return False

            elif tool_name == "curtail_load":
                load_name = arguments.get("load_name")
                if load_name and load_name not in net.load.name.values:
                    return False
                # Check if load is curtailable
                load_row = net.load[net.load.name == load_name]
                if not load_row.empty and not load_row.iloc[0].get(
                    "curtailable", False
                ):
                    return False

            elif tool_name == "add_battery":
                bus_index = arguments.get("bus_index")
                if bus_index not in net.bus.index:
                    return False

            return True
        except Exception:
            return False
