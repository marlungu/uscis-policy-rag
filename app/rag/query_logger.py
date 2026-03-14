import json

from sqlalchemy import text

from app.db import engine
from app.config import settings


def log_query(question: str, answer: str, retrieved_chunks: list[dict], top_k: int) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO query_logs (
                    question,
                    answer,
                    model_id,
                    top_k,
                    retrieved_chunks
                )
                VALUES (
                    :question,
                    :answer,
                    :model_id,
                    :top_k,
                    CAST(:retrieved_chunks AS JSONB)
                )
                """
            ),
            {
                "question": question,
                "answer": answer,
                "model_id": settings.chat_model_id,
                "top_k": top_k,
                "retrieved_chunks": json.dumps(retrieved_chunks),
            },
        )