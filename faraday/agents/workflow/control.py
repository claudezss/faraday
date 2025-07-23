"""
Control flow logic for the workflow.
"""

from faraday.agents.workflow.state import State


def should_continue(state: State) -> str:
    """Determines the next step in the workflow."""
    if state.iter_num >= state.max_iterations:
        return "summarizer"  # Go to summarizer if max iterations are reached

    viola = state.iteration_results[-1].viola_after

    if (
        len(viola.voltage) > 0
        or len(viola.thermal) > 0
        or len(viola.disconnected_buses) > 0
    ):
        return "planner"

    return "summarizer"  # Go to summarizer if violations are resolved
