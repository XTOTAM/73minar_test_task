import faiss
import numpy as np
from openai import OpenAI

from app.models import RetrievedChunk
from app.pipeline.indexer import KnowledgeIndex, tokenize


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score == min_score:
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - min_score) / (max_score - min_score)).astype(np.float32)


class HybridRetriever:
    def __init__(
        self,
        openai_client: OpenAI,
        embedding_model: str,
        top_k: int = 5,
        min_score_threshold: float = 0.3,
        hybrid_alpha: float = 0.7,
    ) -> None:
        self._client = openai_client
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._min_score_threshold = min_score_threshold
        self._hybrid_alpha = hybrid_alpha

    def _embed_query(self, query: str) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=[query],
        )
        vector = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        return vector

    def retrieve(self, query: str, index: KnowledgeIndex) -> list[RetrievedChunk]:
        query_vector = self._embed_query(query)
        candidate_k = min(max(self._top_k * 2, self._top_k), len(index.chunks))

        vector_scores = np.zeros(len(index.chunks), dtype=np.float32)
        distances, indices = index.faiss_index.search(query_vector, candidate_k)
        for rank, chunk_idx in enumerate(indices[0]):
            if chunk_idx < 0:
                continue
            vector_scores[chunk_idx] = float(distances[0][rank])

        bm25_raw = np.array(index.bm25.get_scores(tokenize(query)), dtype=np.float32)
        vector_norm = _normalize_scores(vector_scores)
        bm25_norm = _normalize_scores(bm25_raw)

        hybrid_scores = (
            self._hybrid_alpha * vector_norm + (1.0 - self._hybrid_alpha) * bm25_norm
        )

        ranked_indices = np.argsort(hybrid_scores)[::-1]
        results: list[RetrievedChunk] = []

        for chunk_idx in ranked_indices:
            score = float(hybrid_scores[chunk_idx])
            if score < self._min_score_threshold:
                continue
            results.append(
                RetrievedChunk(
                    chunk=index.chunks[int(chunk_idx)],
                    score=round(score, 4),
                    vector_score=round(float(vector_norm[chunk_idx]), 4),
                    bm25_score=round(float(bm25_norm[chunk_idx]), 4),
                )
            )
            if len(results) >= self._top_k:
                break

        return results
