import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


HEADING_PATTERNS = [
    r"^\s*Volume\s+\d+\b.*$",
    r"^\s*Part\s+[A-Z]\b.*$",
    r"^\s*Chapter\s+\d+\b.*$",
]


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

def should_skip_page(text: str) -> bool:
    if not text:
        return True

    normalized = text.strip()
    upper_text = normalized.upper()

    if len(normalized) < 200:
        return True

    skip_markers = [
        "TABLE OF CONTENTS",
        "POLICY ALERT",
        "USCIS IS UPDATING POLICY GUIDANCE",
        "SEARCH USCIS POLICY MANUAL SEARCH",
        "READ MORE",
        "AFFECTED SECTIONS",
    ]

    if any(marker in upper_text for marker in skip_markers):
        return True

    if normalized.count("Chapter") >= 5:
        return True

    if normalized.count("Part ") >= 5:
        return True

    return False


def extract_heading_context(text: str) -> str:
    headings = []

    for line in text.splitlines():
        stripped = line.strip()
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in HEADING_PATTERNS):
            headings.append(stripped)

    # Keep only the first few distinct headings to avoid noisy repetition
    deduped = []
    seen = set()

    for heading in headings:
        key = heading.lower()
        if key not in seen:
            deduped.append(heading)
            seen.add(key)

    return " | ".join(deduped[:3])


def add_section_context(text: str) -> str:
    heading_context = extract_heading_context(text)

    if not heading_context:
        return text

    return f"{heading_context}\n\n{text}"



def chunk_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    enriched_pages = []

    for page in pages:
        cleaned = clean_text(page.page_content)

        if should_skip_page(cleaned):
            continue

        enriched_text = add_section_context(cleaned)

        enriched_pages.append(
            Document(
                page_content=enriched_text,
                metadata=page.metadata.copy(),
            )
        )

    chunks = splitter.split_documents(enriched_pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks