"""
Main workflow graph definition for the Faraday agent system.
"""

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph

from faraday.agents.workflow.state import State
from faraday.agents.workflow.control import should_continue
from faraday.agents.workflow.nodes.cache import cache_network
from faraday.agents.workflow.nodes.planner import planner
from faraday.agents.workflow.nodes.summarizer import summarizer
from faraday.agents.workflow.nodes.explainer import explainer


def get_workflow():
    """Builds the LangGraph workflow."""
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("cache_network", cache_network)
    workflow.add_node("planner", RunnableLambda(planner))
    workflow.add_node("summarizer", RunnableLambda(summarizer))
    workflow.add_node("explainer", RunnableLambda(explainer))

    # Define workflow edges
    workflow.set_entry_point("cache_network")
    workflow.add_edge("cache_network", "planner")
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {"planner": "planner", "summarizer": "summarizer"},
    )
    workflow.add_edge("summarizer", "explainer")
    workflow.add_edge("explainer", "__end__")

    return workflow


if __name__ == "__main__":
    workflow = get_workflow()
    graph = workflow.compile()
