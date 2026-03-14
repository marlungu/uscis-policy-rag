import json

from sqlalchemy import text
from app.db import engine
from app.ingestion.loader import load_pdf_from_s3
from app.ingestion.chunker import chunk_documents
from app.embeddings.titan_embedder import TitanEmbedder

DOCUMENT_KEY = "docs/policy-manual/uscis_policy_manual_full_2026.pdf"
VOLUME_12_MARKER = "Volume 12 - Citizenship and Naturalization"

def to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def find_volume_start_index(pages):
    for idx, page in enumerate(pages):
        if VOLUME_12_MARKER.lower() in page.page_content.lower():
            return idx
    raise ValueError(f"Could not find marker: {VOLUME_12_MARKER}")


def main():
    print("Loading pages from S3...")
    pages = load_pdf_from_s3(DOCUMENT_KEY)
    print(f"Loaded {len(pages)} pages")

    volume_start_idx = find_volume_start_index(pages)
    volume_pages = pages[volume_start_idx:]

    print(
        f"Detected Volume 12 start at page index {volume_start_idx} "
        f"(PDF page {pages[volume_start_idx].metadata.get('page_number')})"
    )
    print(f"Keeping {len(volume_pages)} pages from Volume 12 onward")

    # Relabel the corpus so retrieval can target it cleanly
    for page in volume_pages:
        page.metadata["document_title"] = "USCIS Policy Manual Volume 12 2026"

    print("Chunking documents...")
    chunks = chunk_documents(volume_pages)
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        raise ValueError("No chunks were created for Volume 12.")
    
    embedder = TitanEmbedder()

    document_title = chunks[0].metadata["document_title"]

    print("Clearing existing rows for this document...")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                DELETE FROM document_chunks
                WHERE document_title = :document_title
                """
            ),
            {"document_title": document_title},
        )

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