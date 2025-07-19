"""
Network caching node for workflow initialization.
"""

from uuid import uuid4

from faraday import WORKSPACE_NETWORKS
from faraday.agents.workflow.state import State
from faraday.tools.pandapower import read_network
import pandapower as pp

from faraday.utils import fill_missing_names


def cache_network(state: State) -> State:
    """Copies the initial network to a temporary editing directory."""
    short_uuid = str(uuid4())[:6]
    dst = WORKSPACE_NETWORKS / "editing" / short_uuid / "network.json"
    copy_net_dst = WORKSPACE_NETWORKS / "editing" / short_uuid / "org_network.json"

    dst.parent.mkdir(parents=True, exist_ok=True)

    net = fill_missing_names(read_network(state.network_file_path))

    pp.to_json(net, str(dst.absolute()))
    pp.to_json(net, str(copy_net_dst.absolute()))

    state.org_network_copy_file_path = str(copy_net_dst.absolute())
    state.editing_network_file_path = str(dst.absolute())
    state.work_dir = str(dst.parent.absolute())

    return state
