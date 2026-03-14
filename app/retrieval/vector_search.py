from typing import List, Dict, Any
import math
import re

import psycopg

from app.config import settings
from app.embeddings.titan_embedder import TitanEmbedder


class VectorSearcher:
    DEFAULT_K = 5

    def __init__(self):
        self.embedder = TitanEmbedder()

    def _to_vector_literal(self, values: list[float]) -> str:
        clean_values = []
        for value in values:
            if not math.isfinite(value):
                raise ValueError("Query embedding contains a non-finite value.")
            clean_values.append(f"{value:.12f}")
        return "[" + ",".join(clean_values) + "]"


    def _normalize_query(self, query: str) -> str:
        q = query.lower().strip()

        if (
            "become us citizen" in q
            or "become a us citizen" in q
            or "become a u.s. citizen" in q
            or "become an american citizen" in q
            or "become citizen" in q
            or "get citizenship" in q
            or "apply for citizenship" in q
        ):
            return "eligibility requirements for naturalization"

        if (
            "left the country" in q
            or "leave the country" in q
            or "out of the country" in q
            or "outside the us" in q
            or "outside the u.s." in q
            or "continuous residence" in q
        ):
            return "continuous residence requirements for naturalization"

        if (
            "english test" in q
            or "civics test" in q
            or "english requirement" in q
            or "english and civics" in q
            or "english exemption" in q
        ):
            return "english and civics requirements for naturalization"

        if (
            "good moral character" in q
            or "arrested" in q
            or "crime" in q
            or "criminal" in q
            or "lied" in q
            or "false testimony" in q
            or "disqualif" in q
        ):
            return "good moral character requirements for naturalization"

        if (
            "military" in q
            or "armed forces" in q
            or "honorable service" in q
        ):
            return "military naturalization requirements"

        return query.strip()
    

    def _build_result(self, row: tuple, matched_query: str) -> Dict[str, Any]:
        distance = float(row[4])

        return {
            "content": row[3],
            "metadata": {
                "document_title": row[0],
                "page_number": row[1],
                "chunk_index": row[2],
            },
            "distance": distance,
            "similarity": 1 - distance,
            "matched_query": matched_query,
        }


    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        stop_words = {
            "the", "a", "an", "and", "or", "of", "to", "in", "for", "on",
            "at", "by", "is", "are", "was", "were", "be", "do", "does",
            "did", "how", "what", "when", "where", "why", "can", "i", "you",
            "it", "this", "that", "with", "as", "from", "your"
        }
        return {word for word in words if word not in stop_words and len(word) > 1}


    def _keyword_overlap_score(self, query: str, content: str) -> float:
        query_tokens = self._tokenize(query)
        content_tokens = self._tokenize(content)

        if not query_tokens:
            return 0.0

        overlap = query_tokens.intersection(content_tokens)
        return len(overlap) / len(query_tokens)


    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reranked = []

        for result in results:
            similarity = result.get("similarity", 0.0) or 0.0
            overlap_score = self._keyword_overlap_score(query, result["content"])
            hybrid_score = (0.8 * similarity) + (0.2 * overlap_score)

            updated = dict(result)
            updated["overlap_score"] = overlap_score
            updated["hybrid_score"] = hybrid_score
            reranked.append(updated)

        reranked.sort(key=lambda r: r["hybrid_score"], reverse=True)
        return reranked


    def search(self, query: str, k:int = DEFAULT_K) -> List[Dict[str, Any]]:
        normalized_query = self._normalize_query(query)

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
        
        query_embedding = self.embedder.embed_text(normalized_query)
        vector_literal = self._to_vector_literal(query_embedding)


        with psycopg.connect(psycopg_url) as conn:
            with conn.cursor() as cur:   
                cur.execute(sql, (vector_literal, vector_literal, int(k)))
                rows = cur.fetchall()

        return [self._build_result(row, normalized_query) for row in rows]