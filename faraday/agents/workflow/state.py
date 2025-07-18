from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph.message import add_messages
import pandapower as pp


class State(TypedDict):
    # File paths
    network_file_path: str
    editing_network_file_path: Optional[str]
    work_dir: Optional[str]

    # Immutable network state
    original_network: Optional[pp.pandapowerNet]
    original_network_file_path: Optional[str]

    # Working network state
    current_network: Optional[pp.pandapowerNet]
    network: Optional[pp.pandapowerNet]  # Keep for backward compatibility

    # Iteration tracking
    iteration_results: Optional[List[Dict[str, Any]]]
    successful_changes: Optional[List[Dict[str, Any]]]

    # Violation tracking
    violation_before_action: Optional[dict]
    violation_after_action: Optional[dict]

    # Agent state
    messages: Annotated[list, add_messages]
    action_plan: Optional[List[Dict[str, Any]]]
    executed_actions: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    explanation: Optional[str]

    # Control flow
    iter: int

    # Validation state
    validation_result: Optional[str]
    violations_improvement: Optional[int]
    rollback_required: Optional[bool]
