from app.rag.answer_generator import AnswerGenerator

PROMPT = "Ask a USCIS policy question: "

def print_sources(sources: list[dict]) -> None:
    if not sources:
        print("None")
        return

    seen = set()
    for src in sources:
        label = f"{src.get('document_title', 'Unknown')} | PDF Page {src.get('page_number', 'Unknown')}"
        if label not in seen:
            seen.add(label)
            print(f"- {label}")


def main():
    generator = AnswerGenerator()

    print("USCIS Policy Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        print()
        question = input(PROMPT).strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        result = generator.answer(question)

        print("\nAnswer:\n")
        print("------")
        print(result["answer"])


        print("\nSources")
        print("-------")
        print_sources(result.get("sources", []))

        used_chunks = result.get("used_chunks", [])
        top_similarity = used_chunks[0].get("similarity") if used_chunks else None

        print("\nRetrieval Summary")
        print("-----------------")
        print(f"- Context sent to Claude: {len(used_chunks)} chunk(s)")
        if top_similarity is not None:
            print(f"- Top similarity: {top_similarity:.4f}")
        else:
            print("- Top similarity: N/A")

        print("\nEvidence Chunks")
        print("---------------")


        for i, chunk in enumerate(result["retrieved_chunks"], start=1):
            meta = chunk["metadata"]
            preview = chunk["content"][:260].replace("\n", " ")

            similarity = chunk.get("similarity")
            distance = chunk.get("distance")

            print(
                f"{i}. {meta['document_title']} | PDF Page {meta['page_number']} | "
                f"chunk {meta['chunk_index']}"
            )

            if similarity is not None and distance is not None:
                print(
                    f"   similarity={similarity:.4f} | distance={distance:.4f}"
                )
            
            print("   Relevant Policy Text:")
            print(f"   {preview}...\n")
        
        print("-" * 80)


if __name__ == "__main__":
    main()