import sys
from src.answering import format_context, generate_answer, get_source_names
from src.config import get_settings
from src.llm_factory import create_chat_model
from src.rag.retriever import retrieve_documents


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
    print(f"Querying ({settings.llm_provider}/{settings.llm_model}): {question}\n")

    try:
        documents = retrieve_documents(question=question, top_k=4, settings=settings)
    except Exception as e:
        print(f"Retrieval error: {e}")
        sys.exit(1)

    if not documents:
        print("No relevant documents found.")
        return

    try:
        model = create_chat_model(settings)
        context = format_context(documents)
        answer = generate_answer(question=question, context=context, model=model, settings=settings)
        sources = get_source_names(documents)
    except Exception as e:
        print(f"Answer generation error: {e}")
        sys.exit(1)

    print("ANSWER:\n" + answer + "\n")
    print("SOURCES:")
    for src in sources:
        print(f"  - {src}")


if __name__ == "__main__":
    main()
