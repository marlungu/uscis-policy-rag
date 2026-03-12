from app.ingestion.loader import list_pdf_keys, load_pdf_from_s3
from app.ingestion.chunker import chunk_documents


if __name__ == "__main__":
    keys = list_pdf_keys()
    print(f"Found {len(keys)} PDF(s) in S3.")

    if not keys:
        raise RuntimeError("No PDFs found in S3. Upload at least one PDF first.")

    first_key = keys[0]
    print(f"Testing with: {first_key}")

    pages = load_pdf_from_s3(first_key)
    print(f"Loaded {len(pages)} page(s).")

    first_page = pages[0]
    print("\\nFirst page metadata:")
    print(first_page.metadata)

    print("\\nFirst page preview:")
    print(first_page.page_content[:500])

    chunks = chunk_documents(pages)
    print(f"\\nCreated {len(chunks)} chunk(s).")

    first_chunk = chunks[0]
    print("\\nFirst chunk metadata:")
    print(first_chunk.metadata)

    print("\\nFirst chunk preview:")
    print(first_chunk.page_content[:500])