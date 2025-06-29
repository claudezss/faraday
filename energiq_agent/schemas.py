from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph.message import add_messages
import pandapower as pp


class State(TypedDict):
    # File paths
    network_file_path: str
    editing_network_file_path: Optional[str]
    work_dir: Optional[str]

    # Network state
    network: Optional[pp.pandapowerNet]
    violation_before_action: Optional[dict]
    violation_after_action: Optional[dict]

    # Agent state
    messages: Annotated[list, add_messages]
    action_plan: Optional[List[Dict[str, Any]]]
    executed_actions: Optional[List[Dict[str, Any]]]
    summary: Optional[str]

    # Control flow
    iter: int
