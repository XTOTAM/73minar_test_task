from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Source(BaseModel):
    section: str
    chunk: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: Literal["high", "medium", "low"]
    fallback_reason: str | None
    trace_id: str
    latency_ms: int


class Chunk(BaseModel):
    chunk_id: str
    section: str
    text: str
    language: Literal["uk", "en", "mixed"]


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    vector_score: float
    bm25_score: float


class LLMResult(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    insufficient_context: bool
    fallback_reason: str | None = None


class PipelineStage(BaseModel):
    name: str
    duration_ms: int
    details: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    trace_id: str
    timestamp: str
    question: str
    stages: list[PipelineStage]
    sources: list[Source]
    answer: str
    confidence: str
    fallback_reason: str | None
    latency_ms: int
    error: str | None = None
