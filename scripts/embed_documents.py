from sqlalchemy import text
from app.db import engine
from app.ingestion.loader import load_pdf_from_s3
from app.ingestion.chunker import chunk_documents
from app.embeddings.titan_embedder import TitanEmbedder

DOCUMENT_KEY = "docs/policy-manual/uscis_policy_manual_full_2026.pdf"

def to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def main():
    print("Loading pages from S3...")
    pages = load_pdf_from_s3(DOCUMENT_KEY)
    print(f"Loaded {len(pages)} cleaned pages")

    if not pages:
        raise ValueError("No pages were loaded from S3.")

    print("Chunking full USCIS Policy Manual...")
    chunks = chunk_documents(pages)
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        raise ValueError("No chunks were created from the USCIS Policy Manual.")
    
    embedder = TitanEmbedder()

    print("Clearing existing rows from document_chunks...")
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM document_chunks"))

    print("Generating embeddings and inserting chunks...")
    inserted = 0

    with engine.begin() as connection:
        for i, chunk in enumerate(chunks, start=1):
            vector = embedder.embed_text(chunk.page_content)
            vector_literal = to_pgvector_literal(vector)

            connection.execute(
                text(
                    """
                    INSERT INTO document_chunks (
                        document_title,
                        page_number,
                        chunk_index,
                        content,
                        embedding
                    )
                    VALUES (
                        :document_title,
                        :page_number,
                        :chunk_index,
                        :content,
                        CAST(:embedding AS vector)
                    )
                    """
                ),
                {
                    "document_title": chunk.metadata["document_title"],
                    "page_number": chunk.metadata["page_number"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    "content": chunk.page_content,
                    "embedding": vector_literal,
                },
            )

            inserted += 1

            if i % 25 == 0 or i == len(chunks):
                print(f"Inserted {i}/{len(chunks)} chunks")

    print(f"Done. Inserted {inserted} chunks into document_chunks.")




if __name__ == "__main__":
    main()