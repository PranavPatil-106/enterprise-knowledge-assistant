"""Workflow definition for LangGraph agent pipeline."""

from typing import Any
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import Settings
from src.graph.nodes import evaluator_node, response_node, retriever_node
from src.graph.state import GraphState
from src.guardrails import validate_question


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


def run_workflow(
    question: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    cleaned_question = validate_question(question)
    graph = build_graph()
    result = graph.invoke({"question": cleaned_question})

    if not result.get("answer"):
        raise ValueError("No answer was generated.")

    return result
