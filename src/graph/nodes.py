from pathlib import Path
from typing import Any
from src.answering import format_context, generate_answer, get_source_names
from src.config import get_settings
from src.evaluation import evaluate_answer
from src.graph.state import GraphState
from src.llm_factory import create_chat_model
from src.mcp_client import read_text_file_via_mcp
from src.rag.retriever import retrieve_documents


def supervisor_node(state: GraphState) -> dict[str, Any]:
    if "documents" not in state or state["documents"] is None:
        print("[Supervisor] Diverting to Retriever...")
        return {"next_step": "retriever"}

    if not state.get("answer"):
        print("[Supervisor] Diverting to Response...")
        return {"next_step": "response"}

    if not state.get("evaluation"):
        print("[Supervisor] Diverting to Evaluator...")
        return {"next_step": "evaluator"}

    evaluation = state.get("evaluation")
    faithfulness = 1.0
    if isinstance(evaluation, dict):
        faithfulness = evaluation.get("faithfulness", 1.0)

    retry_count = state.get("retry_count", 0)

    # Faithfulness threshold = 70%, max 2 retries
    if faithfulness < 0.70 and retry_count < 2:
        next_retry = retry_count + 1
        print(
            f"[Supervisor] Faithfulness ({faithfulness * 100:.1f}%) < 70%. "
            f"Diverting back to Response for retry {next_retry}/2..."
        )
        return {
            "next_step": "response",
            "retry_count": next_retry,
            "answer": None,
            "evaluation": None,
        }

    print("[Supervisor] Workflow complete.")
    return {"next_step": "END"}


def retriever_node(state: GraphState) -> dict[str, Any]:
    print("[Retriever] Fetching relevant documents...")
    question = state["question"]
    documents = retrieve_documents(question=question)
    context = format_context(documents)
    sources = get_source_names(documents)

    settings = get_settings()
    mcp_context = ""

    if documents:
        top_doc = documents[0]
        top_source = top_doc.metadata.get("source") or top_doc.metadata.get("filename") or ""

        if top_source:
            source_path = Path(top_source)
            target_file = source_path if source_path.is_absolute() else settings.mcp_allowed_directory / source_path.name

            try:
                mcp_text = read_text_file_via_mcp(
                    file_path=target_file,
                    allowed_directory=settings.mcp_allowed_directory,
                )
                mcp_context = mcp_text
                context = (
                    f"{context}\n\n"
                    f"=== Additional Content via FastMCP Server ({target_file.name}) ===\n"
                    f"{mcp_text}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"FastMCP retrieval failed for top document '{target_file.name}': {e}"
                ) from e

    return {
        "documents": documents,
        "context": context,
        "mcp_context": mcp_context,
        "sources": sources,
    }


def response_node(state: GraphState) -> dict[str, Any]:
    retry_count = state.get("retry_count", 0)
    if retry_count > 0:
        print(f"[Response] Regenerating answer (attempt {retry_count + 1}) with strict grounding...")
    else:
        print("[Response] Generating answer...")

    question = state["question"]
    context = state.get("context", "")

    effective_question = question
    if retry_count > 0:
        effective_question = (
            f"{question}\n\n"
            f"[IMPORTANT NOTE: Previous attempt had low faithfulness. Strictly base your answer ONLY on the provided context below. "
            f"Do not assume, extrapolate, or add external information.]"
        )

    settings = get_settings()
    model = create_chat_model(settings)
    answer = generate_answer(question=effective_question, context=context, model=model, settings=settings)

    return {"answer": answer}


def evaluator_node(state: GraphState) -> dict[str, Any]:
    print("[Evaluator] Evaluating response with RAGAS...")
    question = state["question"]
    answer = state.get("answer", "")
    documents = list(state.get("documents") or [])
    mcp_context = state.get("mcp_context")
    context = state.get("context", "")

    if mcp_context:
        from langchain_core.documents import Document
        documents.append(Document(page_content=mcp_context))
    elif not documents and context:
        from langchain_core.documents import Document
        documents = [Document(page_content=context)]

    evaluation_result = evaluate_answer(
        question=question,
        answer=answer,
        documents=documents,
    )

    return {"evaluation": evaluation_result}
