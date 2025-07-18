"""
Control flow logic for the workflow.
"""

from faraday.agents.workflow.state import State


def should_continue(state: State):
    """Determines the next step in the workflow."""
    if state["iter"] >= 5:
        return "summarizer"  # Go to summarizer if max iterations are reached

    # Check validation result
    validation_result = state.get("validation_result")
    if validation_result == "failed_execution":
        return "summarizer"  # Stop if execution failed

    if (
        len(state["violation_after_action"]["voltage"]) > 0
        or len(state["violation_after_action"]["thermal"]) > 0
    ):
        return "planner"
    return "summarizer"  # Go to summarizer if violations are resolved
