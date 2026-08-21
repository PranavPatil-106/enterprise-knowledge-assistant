from typing import Any
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import Settings
from src.graph.nodes import (
    evaluator_node,
    response_node,
    retriever_node,
    supervisor_node,
)
from src.graph.state import GraphState
from src.guardrails import validate_question


def route_step(state: GraphState) -> str:
    return state.get("next_step", "END")


def build_graph() -> CompiledStateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("response", response_node)
    workflow.add_node("evaluator", evaluator_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_step,
        {
            "retriever": "retriever",
            "response": "response",
            "evaluator": "evaluator",
            "END": END,
        },
    )

    workflow.add_edge("retriever", "supervisor")
    workflow.add_edge("response", "supervisor")
    workflow.add_edge("evaluator", "supervisor")

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
