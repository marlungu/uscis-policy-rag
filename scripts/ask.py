from app.rag.answer_generator import AnswerGenerator


def main():
    generator = AnswerGenerator()

    question = input("Ask a USCIS policy question: ").strip()

    if not question:
        print("No question provided.")
        return

    result = generator.answer(question)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nRetrieved Chunks:\n")
    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        meta = chunk["metadata"]
        preview = chunk["content"][:220].replace("\n", " ")

        similarity = chunk.get("similarity")
        distance = chunk.get("distance")

        print(
            f"{i}. {meta['document_title']} | page {meta['page_number']} | "
            f"chunk {meta['chunk_index']}"
        )

        if similarity is not None and distance is not None:
            print(
                f"   similarity={similarity:.4f} | distance={distance:.4f}"
            )
            
        print(f"   {preview}...\n")


if __name__ == "__main__":
    main()