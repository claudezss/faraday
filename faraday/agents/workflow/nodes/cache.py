"""
Network caching node for workflow initialization.
"""

import shutil
from uuid import uuid4

from langgraph.types import Command

from faraday import WORKSPACE_NETWORKS
from faraday.agents.workflow.state import State
from faraday.tools.pandapower import read_network


def cache_network(state: State):
    """Copies the initial network to a temporary editing directory and preserves original state."""
    short_uuid = str(uuid4())[:6]
    dst = WORKSPACE_NETWORKS / "editing" / short_uuid / "network.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(state["network_file_path"], str(dst.absolute()))

    # Load and preserve original network
    original_network = read_network(state["network_file_path"])
    current_network = read_network(str(dst.absolute()))

    return Command(
        update={
            "editing_network_file_path": str(dst.absolute()),
            "work_dir": str(dst.parent.absolute()),
            "original_network": original_network,
            "original_network_file_path": state["network_file_path"],
            "current_network": current_network,
            "network": current_network,  # Keep for backward compatibility
            "executed_actions": [],
            "iteration_results": [],
            "successful_changes": [],
        }
    )
