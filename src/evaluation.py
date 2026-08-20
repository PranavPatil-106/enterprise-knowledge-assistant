"""RAGAS evaluation module for evaluating RAG answer faithfulness and relevancy."""

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

# Ensure strictness=1 so LLM providers like Gemini that only support 1 candidate (n=1) succeed
answer_relevancy.strictness = 1

# Monkey-patch parse_run_traces across ragas modules to prevent IndexError when executed inside LangGraph / LangChain runs
_orig_parse_run_traces = ragas.callbacks.parse_run_traces


def _safe_parse_run_traces(traces: dict, parent_run_id: str | None = None) -> list:
    root_traces = [
        chain_trace
        for chain_trace in traces.values()
        if chain_trace.parent_run_id == parent_run_id
    ]
    if not root_traces:
        return []
    return _orig_parse_run_traces(traces, parent_run_id)


ragas.callbacks.parse_run_traces = _safe_parse_run_traces
ragas.dataset_schema.parse_run_traces = _safe_parse_run_traces


def interpret_scores(faithfulness_score: float, relevancy_score: float) -> str:
    """
    Provide an honest interpretation based on RAGAS faithfulness and answer relevancy scores.

    - scores at or above 0.80: strong
    - scores from 0.60 to below 0.80: acceptable but could improve
    - scores below 0.60: needs review
    """
    min_score = min(faithfulness_score, relevancy_score)

    if min_score >= 0.80:
        return "Strong quality: Response is highly faithful to context and directly relevant to the question."
    elif min_score >= 0.60:
        return "Acceptable quality: Response is reasonably grounded and relevant, but could improve."
    else:
        return "Needs review: Low faithfulness or relevancy detected; response may contain unsupported claims or lack focus."


def _extract_metric_score(eval_result: Any, metric_name: str) -> float:
    """Extract float score for a metric from EvaluationResult, DataFrame, or dict."""
    val = None
    if isinstance(eval_result, dict):
        val = eval_result.get(metric_name)
    else:
        try:
            val = eval_result[metric_name]
        except (KeyError, TypeError, IndexError, AttributeError):
            val = None

    if val is None:
        return 0.0

    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return 0.0
        val = val[0]

    try:
        score = float(val)
        if math.isnan(score):
            return 0.0
        return round(score, 4)
    except (ValueError, TypeError):
        return 0.0


def evaluate_answer(
    question: str,
    answer: str,
    documents: list[Document],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Evaluate the quality of an answer using RAGAS Faithfulness and Answer Relevancy metrics.

    Uses:
      - Dynamic chat model from create_chat_model(settings) as evaluator LLM.
      - Fixed Gemini embedding function from get_embedding_function(settings).
    """
    if settings is None:
        settings = get_settings()

    if not settings.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is required for RAGAS evaluation embeddings (gemini-embedding-001). "
            "Please configure GOOGLE_API_KEY in your .env file."
        )

    if not documents:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "interpretation": "Needs review: No context documents provided for evaluation.",
        }

    contexts = [doc.page_content for doc in documents if doc.page_content.strip()]
    if not contexts:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "interpretation": "Needs review: Context documents contain no readable text.",
        }

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    dataset = EvaluationDataset(samples=[sample])

    llm = create_chat_model(settings)
    embeddings = get_embedding_function(settings)
    # max_workers=1 avoids bursting free-tier concurrency limits
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
    interpretation = interpret_scores(faithfulness_val, relevancy_val)

    return {
        "faithfulness": faithfulness_val,
        "answer_relevancy": relevancy_val,
        "interpretation": interpretation,
    }
