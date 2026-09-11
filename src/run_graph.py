import sys
from typing import Any

from src.application import run_workflow
from src.config import get_settings
from src.contact_directory import format_contact_footer
from src.output_guardrails import OutputGuardrailError


def format_evaluation_report(evaluation: dict[str, Any] | str | None) -> str:
    if not evaluation:
        return "No evaluation result available."
    if isinstance(evaluation, str):
        return f"  {evaluation}"

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
    except OutputGuardrailError as e:
        print(f"Guardrail Block: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Execution error: {e}")
        sys.exit(1)

    is_redirect = bool(final_state.get("is_redirect", False))
    contact = final_state.get("contact", {})

    print("FINAL ANSWER:\n" + final_state.get("answer", "No answer generated.") + "\n")

    if is_redirect:
        print("EVALUATION:")
        eval_status = final_state.get("evaluation", "Not evaluated because no verified policy context was found.")
        print(f"  {eval_status}\n")

        if contact:
            print("NEED MORE HELP?")
            print(f"  Contact:  {contact.get('name')}, {contact.get('position')}")
            print(f"  Email:    {contact.get('email')}")
    else:
        if contact:
            print(f"{format_contact_footer(contact)}\n")

        if final_state.get("used_github_mcp"):
            print("SOURCE TYPE:\n  External source: GitHub MCP (read-only)\n")
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
