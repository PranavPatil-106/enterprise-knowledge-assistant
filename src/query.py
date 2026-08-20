"""Command-line query tool for direct retrieval and answering."""

import argparse
import sys
from src.answering import format_context, generate_answer, get_source_names
from src.config import get_settings
from src.llm_factory import create_chat_model
from src.rag.retriever import retrieve_documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the Enterprise Knowledge Assistant knowledge base."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=True,
        help="The question to ask the knowledge assistant.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=4,
        help="Number of document chunks to retrieve (default: 4).",
    )

    args = parser.parse_args()

    try:
        settings = get_settings()
    except Exception as e:
        print(f"[ERROR] Failed to load settings: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Enterprise Knowledge Assistant - Query")
    print(f"Provider: {settings.llm_provider} | Model: {settings.llm_model}")
    print(f"Question: {args.question}")
    print("=" * 60)

    # 1. Retrieve relevant documents
    try:
        print("Retrieving relevant documents from ChromaDB...")
        documents = retrieve_documents(
            question=args.question,
            top_k=args.top_k,
            settings=settings,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Retrieval failed: {e}")
        sys.exit(1)

    if not documents:
        print("\n[WARNING] No relevant documents found.")
        sys.exit(0)

    # 2. Generate answer
    try:
        print("Generating answer using chat model...")
        model = create_chat_model(settings)
        context = format_context(documents)
        answer = generate_answer(
            question=args.question,
            context=context,
            model=model,
            settings=settings,
        )
        sources = get_source_names(documents)
    except ValueError as e:
        print(f"\n[ERROR] Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Answer generation failed: {e}")
        sys.exit(1)

    # 3. Output results
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("-" * 60)
    print(answer)
    print("\n" + "-" * 60)
    print("SOURCES:")
    for src in sources:
        print(f"  - {src}")
    print("=" * 60)


if __name__ == "__main__":
    main()
