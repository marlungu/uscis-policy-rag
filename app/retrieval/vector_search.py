from typing import List, Dict
import math

import psycopg

from app.config import settings
from app.embeddings.titan_embedder import TitanEmbedder


class VectorSearcher:
    def __init__(self):
        self.embedder = TitanEmbedder()

    def _to_vector_literal(self, values: list[float]) -> str:
        clean_values = []
        for value in values:
            if not math.isfinite(value):
                raise ValueError("Query embedding contains a non-finite value.")
            clean_values.append(f"{value:.12f}")
        return "[" + ",".join(clean_values) + "]"


    def search(self, query: str, k: int = 5) -> List[Dict]:
        query_embedding = self.embedder.embed_text(query)
        vector_literal = self._to_vector_literal(query_embedding)


        psycopg_url = settings.postgres_url.replace(
            "postgresql+psycopg://",
            "postgresql://",
            1,
        )

        sql = f"""
        SELECT
            document_title,
            page_number,
            chunk_index,
            content,
            embedding <=> '{vector_literal}'::vector AS distance
        FROM document_chunks
        ORDER BY embedding <=> '{vector_literal}'::vector
        LIMIT {int(k)}
        """

        with psycopg.connect(psycopg_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "content": row[3],
                    "metadata": {
                        "document_title": row[0],
                        "page_number": row[1],
                        "chunk_index": row[2],
                    },
                    "score": float(row[4]),
                }
            )
        return results