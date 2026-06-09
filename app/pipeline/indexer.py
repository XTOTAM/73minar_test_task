import re
from dataclasses import dataclass

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from app.models import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@dataclass
class KnowledgeIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray
    faiss_index: faiss.IndexFlatIP
    bm25: BM25Okapi
    tokenized_corpus: list[list[str]]


class IndexBuilder:
    def __init__(self, openai_client: OpenAI, embedding_model: str) -> None:
        self._client = openai_client
        self._embedding_model = embedding_model

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        matrix = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        return matrix

    def build(self, chunks: list[Chunk]) -> KnowledgeIndex:
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        texts = [chunk.text for chunk in chunks]
        embeddings = self._embed_texts(texts)
        dimension = embeddings.shape[1]

        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings)

        tokenized_corpus = [tokenize(text) for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)

        return KnowledgeIndex(
            chunks=chunks,
            embeddings=embeddings,
            faiss_index=faiss_index,
            bm25=bm25,
            tokenized_corpus=tokenized_corpus,
        )
