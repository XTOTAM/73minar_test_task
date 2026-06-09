import json
from pathlib import Path

from openai import OpenAI

from app.models import Chunk, LLMResult, RetrievedChunk

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system.txt"


class LLMClient:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model
        self._system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for item in chunks:
            parts.append(
                f"[Секція: {item.chunk.section}]\n{item.chunk.text}\n"
                f"(релевантність: {item.score})"
            )
        return "\n\n---\n\n".join(parts)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> LLMResult:
        if not chunks:
            return LLMResult(
                answer=(
                    "За наданою базою знань недостатньо інформації для точної відповіді "
                    "на ваше питання. Релевантні фрагменти контексту не знайдено."
                ),
                confidence="low",
                insufficient_context=True,
                fallback_reason="no_relevant_context",
            )

        context = self._format_context(chunks)
        user_message = (
            f"Питання користувача:\n{question}\n\n"
            f"Контекст з бази знань:\n{context}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)

        return LLMResult(
            answer=payload.get("answer", ""),
            confidence=payload.get("confidence", "medium"),
            insufficient_context=bool(payload.get("insufficient_context", False)),
            fallback_reason=payload.get("fallback_reason"),
        )
