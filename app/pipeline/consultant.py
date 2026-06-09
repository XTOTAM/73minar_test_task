import time
import uuid

from openai import OpenAI

from app.config import Settings
from app.models import AskResponse, PipelineStage, Source, TraceRecord
from app.pipeline.chunker import chunk_knowledge_base
from app.pipeline.indexer import IndexBuilder, KnowledgeIndex
from app.pipeline.llm import LLMClient
from app.pipeline.retriever import HybridRetriever
from app.pipeline.tracer import TraceLogger
from app.pipeline.validator import AnswerValidator


class ConsultantPipeline:
    def __init__(self, settings: Settings, openai_client: OpenAI) -> None:
        self._settings = settings
        self._index: KnowledgeIndex | None = None
        self._retriever = HybridRetriever(
            openai_client=openai_client,
            embedding_model=settings.openai_embedding_model,
            top_k=settings.retrieval_top_k,
            min_score_threshold=settings.min_score_threshold,
            hybrid_alpha=settings.hybrid_alpha,
        )
        self._llm = LLMClient(openai_client, settings.openai_model)
        self._validator = AnswerValidator(settings.min_score_threshold)
        self._tracer = TraceLogger(settings.traces_path)
        self._index_builder = IndexBuilder(openai_client, settings.openai_embedding_model)

    def build_index(self) -> None:
        chunks = chunk_knowledge_base(
            self._settings.knowledge_base_path,
            max_chunk_tokens=self._settings.max_chunk_tokens,
        )
        self._index = self._index_builder.build(chunks)

    async def ask(self, question: str) -> AskResponse:
        if self._index is None:
            raise RuntimeError("Knowledge index is not initialized")

        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        stages: list[PipelineStage] = []
        error: str | None = None

        try:
            receive_started = time.perf_counter()
            stages.append(
                PipelineStage(
                    name="receive_request",
                    duration_ms=int((time.perf_counter() - receive_started) * 1000),
                    details={"question_length": len(question)},
                )
            )

            retrieve_started = time.perf_counter()
            retrieved = self._retriever.retrieve(question, self._index)
            stages.append(
                PipelineStage(
                    name="retrieve_context",
                    duration_ms=int((time.perf_counter() - retrieve_started) * 1000),
                    details={
                        "chunks_found": len(retrieved),
                        "top_scores": [item.score for item in retrieved[:3]],
                        "sections": [item.chunk.section for item in retrieved],
                    },
                )
            )

            generate_started = time.perf_counter()
            llm_result = self._llm.generate(question, retrieved)
            stages.append(
                PipelineStage(
                    name="generate_answer",
                    duration_ms=int((time.perf_counter() - generate_started) * 1000),
                    details={"model": self._settings.openai_model},
                )
            )

            validate_started = time.perf_counter()
            validated = self._validator.validate(question, retrieved, llm_result)
            stages.append(
                PipelineStage(
                    name="validate_answer",
                    duration_ms=int((time.perf_counter() - validate_started) * 1000),
                    details={
                        "passed": not validated.insufficient_context,
                        "insufficient_context": validated.insufficient_context,
                    },
                )
            )

            sources = [
                Source(section=item.chunk.section, chunk=item.chunk.text, score=item.score)
                for item in retrieved
            ]

            latency_ms = int((time.perf_counter() - started) * 1000)
            stages.append(
                PipelineStage(
                    name="return_response",
                    duration_ms=0,
                    details={"latency_ms": latency_ms},
                )
            )

            response = AskResponse(
                answer=validated.answer,
                sources=sources,
                confidence=validated.confidence,
                fallback_reason=validated.fallback_reason,
                trace_id=trace_id,
                latency_ms=latency_ms,
            )

            await self._tracer.log(
                TraceRecord(
                    trace_id=trace_id,
                    timestamp=TraceLogger.now_iso(),
                    question=question,
                    stages=stages,
                    sources=sources,
                    answer=response.answer,
                    confidence=response.confidence,
                    fallback_reason=response.fallback_reason,
                    latency_ms=latency_ms,
                    error=None,
                )
            )
            return response

        except Exception as exc:
            error = str(exc)
            latency_ms = int((time.perf_counter() - started) * 1000)
            fallback_answer = (
                "Виникла технічна помилка під час обробки запиту. "
                "Спробуйте повторити пізніше."
            )
            response = AskResponse(
                answer=fallback_answer,
                sources=[],
                confidence="low",
                fallback_reason="processing_error",
                trace_id=trace_id,
                latency_ms=latency_ms,
            )
            await self._tracer.log(
                TraceRecord(
                    trace_id=trace_id,
                    timestamp=TraceLogger.now_iso(),
                    question=question,
                    stages=stages,
                    sources=[],
                    answer=fallback_answer,
                    confidence="low",
                    fallback_reason="processing_error",
                    latency_ms=latency_ms,
                    error=error,
                )
            )
            return response
