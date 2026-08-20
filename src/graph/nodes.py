"""Workflow node functions for the LangGraph agent graph."""

from pathlib import Path
from typing import Any
from src.answering import format_context, generate_answer, get_source_names
from src.config import get_settings
from src.evaluation import evaluate_answer
from src.graph.state import GraphState
from src.llm_factory import create_chat_model
from src.mcp_client import read_text_file_via_mcp
from src.rag.retriever import retrieve_documents


def retriever_node(state: GraphState) -> dict[str, Any]:
    """Retriever Agent node: fetches matching documents from ChromaDB and reads the top source via Filesystem MCP."""
    print("[Node: Retriever Agent] Fetching relevant documents from ChromaDB...")
    question = state["question"]
    documents = retrieve_documents(question=question)
    context = format_context(documents)
    sources = get_source_names(documents)

    settings = get_settings()
    mcp_context = ""

    if documents:
        print("[Node: Retriever Agent] Reading top source through Filesystem MCP...")
        top_doc = documents[0]
        top_source = top_doc.metadata.get("source") or top_doc.metadata.get("file_name") or ""

        if top_source:
            source_path = Path(top_source)
            if not source_path.is_absolute():
                candidate = settings.mcp_allowed_directory / source_path.name
                if candidate.exists():
                    target_file = candidate
                else:
                    target_file = settings.mcp_allowed_directory / source_path
            else:
                target_file = source_path

            try:
                mcp_text = read_text_file_via_mcp(
                    file_path=target_file,
                    allowed_directory=settings.mcp_allowed_directory,
                )
                mcp_context = mcp_text
                context = (
                    f"{context}\n\n"
                    f"=== Additional Content via Filesystem MCP Server ({target_file.name}) ===\n"
                    f"{mcp_text}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Filesystem MCP retrieval failed for top document '{target_file.name}': {e}"
                ) from e

    return {
        "documents": documents,
        "context": context,
        "mcp_context": mcp_context,
        "sources": sources,
    }


def response_node(state: GraphState) -> dict[str, Any]:
    """Response Agent node: generates a grounded answer using the configured chat LLM."""
    print("[Node: Response Agent] Generating grounded answer from context...")
    question = state["question"]
    context = state.get("context", "")

    settings = get_settings()
    model = create_chat_model(settings)
    answer = generate_answer(question=question, context=context, model=model, settings=settings)

    return {
        "answer": answer,
    }


def evaluator_node(state: GraphState) -> dict[str, Any]:
    """Evaluator Agent node: evaluates response faithfulness and relevancy with RAGAS."""
    print("[Node: Evaluator Agent] Evaluating response quality with RAGAS...")
    question = state["question"]
    answer = state.get("answer", "")
    documents = state.get("documents", [])

    evaluation_result = evaluate_answer(
        question=question,
        answer=answer,
        documents=documents,
    )

    return {
        "evaluation": evaluation_result,
    }
