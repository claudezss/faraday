"""
Main workflow graph definition for the Faraday agent system.
"""

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph

from faraday.schemas import State
from .control import should_continue
from .nodes import (
    cache_network,
    planner,
    executor,
    validator,
    summarizer,
    explainer,
)


def get_workflow():
    """Builds the LangGraph workflow."""
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("cache_network", cache_network)
    workflow.add_node("planner", RunnableLambda(planner))
    workflow.add_node("executor", executor)
    workflow.add_node("validator", RunnableLambda(validator))
    workflow.add_node("summarizer", RunnableLambda(summarizer))
    workflow.add_node("explainer", RunnableLambda(explainer))

    # Define workflow edges
    workflow.set_entry_point("cache_network")
    workflow.add_edge("cache_network", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "validator")
    workflow.add_conditional_edges(
        "validator",
        should_continue,
        {"planner": "planner", "summarizer": "summarizer"},
    )
    workflow.add_edge("summarizer", "explainer")
    workflow.add_edge("explainer", "__end__")

    return workflow


if __name__ == "__main__":
    workflow = get_workflow()
    graph = workflow.compile()
