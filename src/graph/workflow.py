"""Workflow definition for LangGraph agent pipeline."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.nodes import evaluator_node, response_node, retriever_node
from src.graph.state import GraphState


def build_graph() -> CompiledStateGraph:
    """
    Construct and compile the linear multi-agent workflow graph:
    START -> retriever -> response -> evaluator -> END
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("response", response_node)
    workflow.add_node("evaluator", evaluator_node)

    # Define linear execution flow
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "response")
    workflow.add_edge("response", "evaluator")
    workflow.add_edge("evaluator", END)

    return workflow.compile()
