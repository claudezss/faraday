from typing import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated, Optional
import pandapower as pp


class State(TypedDict):
    network_file_path: str

    human_guidance: Optional[str]

    editing_network_file_path: Optional[str]

    messages: Annotated[list, add_messages]

    network: Optional[pp.pandapowerNet]

    work_dir: Optional[str]

    action_plan: Optional[str]

    violation_before_action: Optional[dict]

    violation_after_action: Optional[dict]

    feedback: Optional[str]

    log: Optional[list[str]]

    iter: int
