import re

from app.models import LLMResult, RetrievedChunk

CALCULATION_PATTERN = re.compile(
    r"(порахуй|розрахуй|обчисли|calculate|скільки буде|сума індексації)",
    re.IGNORECASE,
)
EXACT_DATE_PATTERN = re.compile(
    r"(точн[ауюі]\s+дат|яка дата|коли саме|exact date|which date)",
    re.IGNORECASE,
)


class AnswerValidator:
    def __init__(self, min_score_threshold: float) -> None:
        self._min_score_threshold = min_score_threshold

    def validate(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        llm_result: LLMResult,
    ) -> LLMResult:
        if not retrieved:
            return LLMResult(
                answer=(
                    "За наданою базою знань недостатньо інформації для точної відповіді "
                    "на ваше питання. Релевантні фрагменти контексту не знайдено."
                ),
                confidence="low",
                insufficient_context=True,
                fallback_reason="no_relevant_context",
            )

        top_score = retrieved[0].score
        if top_score < self._min_score_threshold:
            return LLMResult(
                answer=(
                    "За наданою базою знань недостатньо інформації для точної відповіді "
                    "на ваше питання. Знайдений контекст має низьку релевантність."
                ),
                confidence="low",
                insufficient_context=True,
                fallback_reason="no_relevant_context",
            )

        context_text = " ".join(item.chunk.text.lower() for item in retrieved)

        if CALCULATION_PATTERN.search(question):
            has_calc_rules = any(
                phrase in context_text
                for phrase in (
                    "немає достатніх числових",
                    "does not provide all required numeric",
                    "cannot be completed",
                    "розрахунку конкретної суми",
                )
            )
            if has_calc_rules or llm_result.insufficient_context:
                return LLMResult(
                    answer=llm_result.answer
                    or (
                        "За наданою базою знань недостатньо інформації для точної відповіді. "
                        "Розрахунок неможливий без усіх необхідних числових даних."
                    ),
                    confidence="low",
                    insufficient_context=True,
                    fallback_reason="calculation_not_possible",
                )

        if EXACT_DATE_PATTERN.search(question):
            has_no_dates = any(
                phrase in context_text
                for phrase in (
                    "точні календарні дати",
                    "не наведені",
                    "does not contain",
                    "exact date",
                )
            )
            if has_no_dates or llm_result.insufficient_context:
                return LLMResult(
                    answer=llm_result.answer
                    or (
                        "За наданою базою знань недостатньо інформації для точної відповіді. "
                        "Точні календарні дати у наданому джерелі відсутні."
                    ),
                    confidence="low",
                    insufficient_context=True,
                    fallback_reason="exact_date_not_in_source",
                )

        if llm_result.insufficient_context:
            return LLMResult(
                answer=llm_result.answer,
                confidence="low",
                insufficient_context=True,
                fallback_reason=llm_result.fallback_reason or "insufficient_knowledge_base",
            )

        confidence = llm_result.confidence
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        return LLMResult(
            answer=llm_result.answer,
            confidence=confidence,  # type: ignore[arg-type]
            insufficient_context=False,
            fallback_reason=None,
        )
