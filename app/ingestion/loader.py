from pathlib import Path
import tempfile
from typing import List

import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.config import settings


def _derive_document_title(s3_key: str) -> str:
    stem = Path(s3_key).stem
    title = stem.replace("_", " ").replace("-", " ")

    replacements = {
        "uscis": "USCIS",
        "vol": "Vol",
        "part": "Part",
        "omb": "OMB",
    }

    words = []
    for word in title.split():
        lower = word.lower()

        if lower.startswith("vol") and lower != "vol":
            words.append(f"Vol {word[3:]}")
        elif lower.startswith("part") and lower != "part":
            words.append(f"Part {word[4:]}")
        elif lower in replacements:
            words.append(replacements[lower])
        else:
            words.append(word.capitalize())

    return " ".join(words)


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def list_pdf_keys(prefix: str | None = None) -> List[str]:
    s3 = get_s3_client()
    effective_prefix = prefix or settings.s3_prefix

    response = s3.list_objects_v2(
        Bucket=settings.s3_bucket_name,
        Prefix=effective_prefix,
    )

    contents = response.get("Contents", [])
    return [
        obj["Key"]
        for obj in contents
        if obj["Key"].lower().endswith(".pdf")
    ]


def load_pdf_from_s3(s3_key: str) -> List[Document]:
    s3 = get_s3_client()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        s3.download_fileobj(settings.s3_bucket_name, s3_key, tmp)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        document_title = _derive_document_title(s3_key)

        for idx, page in enumerate(pages, start=1):
            page.metadata["document_title"] = document_title
            page.metadata["page_number"] = idx
            page.metadata["source_key"] = s3_key
            page.metadata["s3_bucket"] = settings.s3_bucket_name

        return pages
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def load_all_documents(prefix: str | None = None) -> List[Document]:
    all_docs: List[Document] = []

    for s3_key in list_pdf_keys(prefix=prefix):
        pages = load_pdf_from_s3(s3_key)
        all_docs.extend(pages)

    return all_docs