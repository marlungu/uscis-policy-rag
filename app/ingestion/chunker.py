import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def clean_text(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    skip_patterns = [
        r"^\s*Affected Sections\s*$",
        r"^\s*Read More\s*$",
        r"^\s*Policy Manual \| USCIS\s*$",
        r"^\s*Search USCIS Policy Manual Search\s*$",
        r"^\s*Current as of.*$",
        r"^\s*\d{1,2}/\d{1,2}/\d{2,4},.*$",
        r"^\s*https?://.*$",
        r"^\s*Chapter\s+\d+\s*-\s*.*$",
        r"^\s*Part\s+[A-Z]\s*-\s*.*$",
        r"^\s*Volume\s+\d+\s*-\s*.*$",
    ]

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text).strip()
    return cleaned_text


def chunk_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    cleaned_pages = []
    for page in pages:
        cleaned = clean_text(page.page_content)

        if not cleaned or len(cleaned) < 100:
            continue

        cleaned_pages.append(
            Document(
                page_content=cleaned,
                metadata=page.metadata,
            )
        )

    chunks = splitter.split_documents(cleaned_pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks