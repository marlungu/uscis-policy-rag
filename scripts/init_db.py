from sqlalchemy import text
from app.db import engine


def initialize_database() -> None:
    with engine.begin() as connection:

        connection.execute(
            text(
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_title TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_index INTEGER,
                    content TEXT NOT NULL,
                    embedding VECTOR(1024)
                );
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
                ON document_chunks
                USING ivfflat (embedding vector_cosine_ops);
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id BIGSERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    retrieved_chunks JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """
            )
        )

        print("Database schema initialized successfully.")


if __name__ == "__main__":
    initialize_database()