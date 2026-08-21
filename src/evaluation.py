import math
from typing import Any
from langchain_core.documents import Document
import ragas.callbacks
import ragas.dataset_schema
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.run_config import RunConfig

from src.config import Settings, get_settings
from src.llm_factory import create_chat_model
from src.rag.indexer import get_embedding_function

# Ensure strictness=1 for providers like Gemini that support 1 candidate
answer_relevancy.strictness = 1

# Prevent IndexError in RAGAS callback parsing inside LangChain runs
_orig_parse_run_traces = ragas.callbacks.parse_run_traces


def _safe_parse_run_traces(traces: dict, parent_run_id: str | None = None) -> list:
    root_traces = [t for t in traces.values() if t.parent_run_id == parent_run_id]
    return _orig_parse_run_traces(traces, parent_run_id) if root_traces else []


ragas.callbacks.parse_run_traces = _safe_parse_run_traces
ragas.dataset_schema.parse_run_traces = _safe_parse_run_traces


def interpret_scores(faithfulness_score: float, relevancy_score: float) -> str:
    min_score = min(faithfulness_score, relevancy_score)
    if min_score >= 0.80:
        return "Strong quality: Response is highly faithful to context and directly relevant to the question."
    elif min_score >= 0.60:
        return "Acceptable quality: Response is reasonably grounded and relevant, but could improve."
    return "Needs review: Low faithfulness or relevancy detected; response may contain unsupported claims or lack focus."


def _extract_metric_score(eval_result: Any, metric_name: str) -> float:
    try:
        val = eval_result.get(metric_name) if isinstance(eval_result, dict) else eval_result[metric_name]
        if isinstance(val, (list, tuple)):
            val = val[0] if val else 0.0
        score = float(val)
        return 0.0 if math.isnan(score) else round(score, 4)
    except Exception:
        return 0.0


def evaluate_answer(
    question: str,
    answer: str,
    documents: list[Document] | list[str] | str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or get_settings()

    if not cfg.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required in .env for evaluation embeddings.")

    if isinstance(documents, list):
        contexts = [
            doc.page_content if isinstance(doc, Document) else str(doc)
            for doc in documents
            if (doc.page_content if isinstance(doc, Document) else str(doc)).strip()
        ]
    elif isinstance(documents, str) and documents.strip():
        contexts = [documents.strip()]
    else:
        contexts = []

    if not contexts:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "interpretation": "Needs review: No context documents provided for evaluation.",
        }

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    dataset = EvaluationDataset(samples=[sample])

    llm = create_chat_model(cfg)
    embeddings = get_embedding_function(cfg)
    run_config = RunConfig(timeout=180, max_retries=6, max_workers=1)

    try:
        eval_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            show_progress=False,
        )
    except Exception as e:
        raise RuntimeError(f"RAGAS evaluation failed: {e}") from e

    faithfulness_val = _extract_metric_score(eval_result, "faithfulness")
    relevancy_val = _extract_metric_score(eval_result, "answer_relevancy")

    return {
        "faithfulness": faithfulness_val,
        "answer_relevancy": relevancy_val,
        "interpretation": interpret_scores(faithfulness_val, relevancy_val),
    }
