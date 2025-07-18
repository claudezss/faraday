"""
Validation node for validating network changes and determining rollback requirements.
"""

from langgraph.types import Command

from faraday.agents.workflow.state import State


def validator(state: State):
    """Validates the current network state and determines if changes should be committed."""
    iteration_results = state.get("iteration_results", [])

    if not iteration_results:
        return Command(update={"validation_result": "no_iterations"})

    last_result = iteration_results[-1]

    # Check if last iteration was successful
    if not last_result.get("successful", False):
        return Command(
            update={"validation_result": "failed_execution", "rollback_required": True}
        )

    # Check for improvement in violations
    violations_before = last_result.get("violations_before", {})
    violations_after = last_result.get("violations_after", {})

    total_violations_before = len(violations_before.get("voltage", [])) + len(
        violations_before.get("thermal", [])
    )
    total_violations_after = len(violations_after.get("voltage", [])) + len(
        violations_after.get("thermal", [])
    )

    improvement = total_violations_before - total_violations_after

    return Command(
        update={
            "validation_result": "success" if improvement >= 0 else "degraded",
            "violations_improvement": improvement,
            "rollback_required": False,
        }
    )
