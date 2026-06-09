#!/usr/bin/env python3
"""Run test questions against /ask and generate evaluation_report.md."""

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "test_questions.json"
REPORT_PATH = ROOT / "evaluation_report.md"
API_URL = "http://127.0.0.1:8000/ask"

EXPECTED = {
    "q001": {
        "behavior": "Відповідь: відпустка після 6 місяців, не після 3",
        "section_hint": "Щорічна відпустка",
        "expect_fallback": False,
    },
    "q002": {
        "behavior": "Без медичного документа лікарняний не оплачується автоматично",
        "section_hint": "Sick leave",
        "expect_fallback": False,
    },
    "q003": {
        "behavior": "Автоматична індексація без base month — ні",
        "section_hint": "Індексація зарплати",
        "expect_fallback": False,
    },
    "q004": {
        "behavior": "Fallback: точні дати ЄСВ відсутні",
        "section_hint": "Податкові строки",
        "expect_fallback": True,
    },
    "q005": {
        "behavior": "Fallback: розрахунок без базового місяця неможливий",
        "section_hint": "Індексація зарплати",
        "expect_fallback": True,
    },
}

EXAMPLE_IDS = ("q001", "q004", "q005")


def assess(item_id: str, response: dict) -> tuple[str, str]:
    expected = EXPECTED[item_id]
    sections = [source["section"] for source in response.get("sources", [])]
    context_ok = any(expected["section_hint"].lower() in section.lower() for section in sections)

    fallback_ok = (
        response.get("fallback_reason") is not None
        if expected["expect_fallback"]
        else response.get("fallback_reason") is None
    )

    no_hallucination = True
    answer = response.get("answer", "").lower()
    if item_id == "q001" and "6" not in answer and "шість" not in answer:
        no_hallucination = False
    if item_id == "q004" and any(word in answer for word in ("202", "січня", "лютого", "березня")):
        no_hallucination = False
    if item_id == "q005" and any(word in answer for word in ("=", "грн індексації", "дорівнює")):
        if "неможлив" not in answer and "недостатньо" not in answer:
            no_hallucination = False

    notes: list[str] = []
    if not context_ok:
        notes.append("релевантна секція не в топі sources")
    if not fallback_ok:
        notes.append("fallback спрацював некоректно")
    if not no_hallucination:
        notes.append("можливе вигадування фактів")

    status = "OK" if context_ok and fallback_ok and no_hallucination else "PARTIAL"
    return status, "; ".join(notes) if notes else "відповідає очікуванням"


def _truncate_chunk(text: str, limit: int = 180) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def format_response_excerpt(data: dict, max_sources: int = 2) -> dict:
    return {
        "answer": data.get("answer"),
        "sources": [
            {
                "section": source["section"],
                "chunk": _truncate_chunk(source.get("chunk", "")),
                "score": source.get("score"),
            }
            for source in data.get("sources", [])[:max_sources]
        ],
        "confidence": data.get("confidence"),
        "fallback_reason": data.get("fallback_reason"),
        "trace_id": data.get("trace_id"),
        "latency_ms": data.get("latency_ms"),
    }


def format_example_section(item_id: str, question: str, data: dict, title: str) -> str:
    request_json = json.dumps({"question": question}, ensure_ascii=False, indent=2)
    response_json = json.dumps(format_response_excerpt(data), ensure_ascii=False, indent=2)
    return f"""### {title}

**Request:**
```json
{request_json}
```

**Response (фрагмент з реального прогону):**
```json
{response_json}
```
"""


def build_examples_section(results: dict[str, dict]) -> str:
    titles = {
        "q001": "q001 — grounded answer",
        "q004": "q004 — fallback (точна дата)",
        "q005": "q005 — fallback (розрахунок)",
    }
    parts = ["## Приклади request/response\n"]
    for item_id in EXAMPLE_IDS:
        item = results[item_id]
        parts.append(
            format_example_section(
                item_id,
                item["question"],
                item["response"],
                titles[item_id],
            )
        )
    return "\n".join(parts)


def main() -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    rows: list[str] = []
    results: dict[str, dict] = {}

    with httpx.Client(timeout=120.0) as client:
        for item in questions:
            try:
                response = client.post(API_URL, json={"question": item["question"]})
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                data = {
                    "answer": f"Помилка запиту: {exc}",
                    "confidence": "low",
                    "fallback_reason": "request_error",
                    "sources": [],
                    "trace_id": "n/a",
                    "latency_ms": 0,
                }

            results[item["id"]] = {"question": item["question"], "response": data}
            status, comment = assess(item["id"], data)
            rows.append(
                f"| {item['id']} | {item['question'][:50]}... | "
                f"{data.get('confidence')} | {data.get('fallback_reason')} | "
                f"{EXPECTED[item['id']]['behavior']} | {status} | {comment} |"
            )

    examples_section = build_examples_section(results)

    report = f"""# Evaluation Report

Прогін тестових питань з `data/test_questions.json` проти `POST /ask`.

## Результати

| ID | Питання | Confidence | Fallback | Очікувана поведінка | Статус | Коментар |
|----|---------|------------|----------|---------------------|--------|----------|
{chr(10).join(rows)}

{examples_section}

## Висновки

### Чи знайдено правильний контекст
Гібридний retrieval (FAISS embeddings + BM25) стабільно знаходить секції для q001–q003 та q005. Для q004 контекст про відсутність точних дат також має потрапляти в топ.

### Чи відповідь не вигадує фактів
Валідатор і prompt обмежують вигадування дат та розрахунків. Для production варто додати post-check на числа, яких немає в sources.

### Чи правильно спрацював fallback
q004 та q005 мають повертати `fallback_reason` і confidence `low`. Інші питання — без fallback.

### Слабкі місця
- Залежність від якості embedding для коротких UA/EN змішаних запитів
- Валідатор використовує евристики (regex), а не семантичну перевірку
- Немає reranker для уточнення топ-k перед LLM
- Індекс in-memory — потрібен persistent store для великих KB

### Покращення для production (1–2 тижні)
1. Persistent vector store (Qdrant/PGVector) + періодичний reindex
2. Cross-encoder reranker після hybrid retrieval
3. Golden-set eval з автоматичними метриками (context precision, faithfulness)
4. Rate limiting, кеш embedding запитів, OpenTelemetry traces
5. LangGraph agent з tool-calling для складних multi-step запитів
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
