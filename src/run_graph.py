"""CLI entry point to execute the LangGraph multi-agent workflow."""

import argparse
import sys
from typing import Any
from src.application import run_workflow
from src.config import get_settings


def format_evaluation_report(evaluation: dict[str, Any] | str | None) -> str:
    """Format evaluation results cleanly for terminal display."""
    if not evaluation:
        return "No evaluation result available."

    if isinstance(evaluation, str):
        return evaluation

    faithfulness = evaluation.get("faithfulness", 0.0)
    relevancy = evaluation.get("answer_relevancy", 0.0)
    interpretation = evaluation.get("interpretation", "No interpretation provided.")

    return (
        f"  - Faithfulness:      {faithfulness * 100:.2f}%\n"
        f"  - Answer Relevancy:  {relevancy * 100:.2f}%\n"
        f"  - Summary:           {interpretation}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Enterprise Knowledge Assistant LangGraph workflow."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=True,
        help="The question to ask the knowledge assistant.",
    )

    args = parser.parse_args()

    try:
        settings = get_settings()
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Enterprise Knowledge Assistant - LangGraph Execution")
    print(f"Provider: {settings.llm_provider} | Model: {settings.llm_model}")
    print(f"Question: {args.question}")
    print(f"LangSmith tracing: {'enabled' if settings.langsmith_tracing else 'disabled'}")
    print("=" * 60)

    try:
        final_state = run_workflow(args.question, settings=settings)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Graph execution failed: {e}")
        sys.exit(1)

    # Output formatted results
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("-" * 60)
    print(final_state.get("answer", "No answer generated."))

    print("\n" + "-" * 60)
    print("SOURCES:")
    sources = final_state.get("sources", [])
    if sources:
        for src in sources:
            print(f"  - {src}")
    else:
        print("  None")

    print("\n" + "-" * 60)
    print("EVALUATION (RAGAS):")
    print(format_evaluation_report(final_state.get("evaluation")))
    print("=" * 60)


if __name__ == "__main__":
    main()
