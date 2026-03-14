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

    def _expand_query(self, query: str) -> list[str]:
        query = query.strip()

        expansions = [
            query,
            f"{query} requirements",
            f"{query} eligibility",
            f"{query} criteria",
            f"requirements for {query}",
            f"eligibility for {query}",
        ]

        seen = set()
        unique_expansions = []

        for q in expansions:
            normalized = q.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_expansions.append(q)

        return unique_expansions


    def search(self, query: str, k: int = 5) -> List[Dict]:
        expanded_queries = self._expand_query(query)

        psycopg_url = settings.postgres_url.replace(
            "postgresql+psycopg://",
            "postgresql://",
            1,
        )

        sql = """
        SELECT
            document_title,
            page_number,
            chunk_index,
            content,
            embedding <=> %s::vector AS distance
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        all_results = []

        with psycopg.connect(psycopg_url) as conn:
            with conn.cursor() as cur:
                for expanded_query in expanded_queries:
                    query_embedding = self.embedder.embed_text(expanded_query)
                    vector_literal = self._to_vector_literal(query_embedding)
                    
                    cur.execute(sql, (vector_literal, vector_literal, int(k)))
                    rows = cur.fetchall()

                    for row in rows:
                        distance = float(row[4])
                        all_results.append(
                            {
                                "content": row[3],
                                "metadata": {
                                    "document_title": row[0],
                                    "page_number": row[1],
                                    "chunk_index": row[2],
                                },
                                "distance": distance,
                                "similarity": 1 - distance,
                                "matched_query": expanded_query,
                            }
                        )

        all_results.sort(key=lambda r: r["distance"])

        deduped_results = []
        seen_keys = set()

        for result in all_results:
            meta = result["metadata"]
            key = (
                meta["document_title"],
                meta["page_number"],
                meta["chunk_index"],
            )

            if key in seen_keys:
                continue

            seen_keys.add(key)
            deduped_results.append(result)

            if len(deduped_results) >= k:
                break

        return deduped_results