"""
Backward compatibility module for the workflow.

This module maintains backward compatibility while delegating to the new
organized workflow structure.
"""

# Import from the new workflow structure
from .workflow import get_workflow
from .workflow.nodes import (
    cache_network,
    planner,
    executor,
    validator,
    summarizer,
    explainer,
)

# For backward compatibility, expose all functions
__all__ = [
    "get_workflow",
    "cache_network",
    "planner",
    "executor",
    "validator",
    "summarizer",
    "explainer",
]

# Support direct execution
if __name__ == "__main__":
    workflow = get_workflow()
    graph = workflow.compile()
