import sys
from typing import Any

from src.config import get_settings
from src.graph.workflow import run_workflow


def format_evaluation_report(evaluation: dict[str, Any] | str | None) -> str:
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]

    if args:
        if args[0] in ("-q", "--question") and len(args) > 1:
            question = " ".join(args[1:])
        else:
            question = " ".join(args)
    else:
        question = input("Enter your question: ").strip()

    if not question:
        print("No question provided.")
        return

    settings = get_settings()
    print(f"Running workflow ({settings.llm_provider}/{settings.llm_model}): {question}\n")

    try:
        final_state = run_workflow(question, settings=settings)
    except Exception as e:
        print(f"Execution error: {e}")
        sys.exit(1)

    print("FINAL ANSWER:\n" + final_state.get("answer", "No answer generated.") + "\n")

    sources = final_state.get("sources", [])
    print("SOURCES:")
    if sources:
        for src in sources:
            print(f"  - {src}")
    else:
        print("  None")

    print("\nEVALUATION (RAGAS):")
    print(format_evaluation_report(final_state.get("evaluation")))


if __name__ == "__main__":
    main()
