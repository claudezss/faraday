"""
Summarizer node for generating summaries of executed actions.
"""

from langgraph.types import Command

from faraday.agents.prompts import SUMMARIZER_PROMPT
from faraday.agents.workflow.state import State
from ..config import llm


def summarizer(state: State):
    """Generates a summary of the executed actions."""
    executed_actions = state.get("executed_actions", [])
    if not executed_actions:
        summary = "No actions were executed."
    else:
        action_report = "\n".join(
            [f"- {action['name']}({action['args']})" for action in executed_actions]
        )
        summary_prompt = SUMMARIZER_PROMPT.format(action_report=action_report)
        summary = llm.invoke(summary_prompt).content

    return Command(update={"summary": summary})
