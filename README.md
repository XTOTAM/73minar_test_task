# Controlled AI Consultant Pipeline

Мінімальний API-сервіс AI-консультанта: приймає питання українською, знаходить релевантний контекст у `knowledge_base.md` і повертає structured JSON-відповідь.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Додайте OPENAI_API_KEY у .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Перевірка: `GET http://localhost:8000/health`

## Endpoint

**URL:** `POST http://localhost:8000/ask`

**Request:**
```json
{
  "question": "Чи може працівник взяти щорічну відпустку після 3 місяців роботи?"
}
```

**Response:**
```json
{
  "answer": "За наданою базою знань, працівник може використати щорічну оплачувану відпустку після 6 місяців безперервної роботи у компанії. Після 3 місяців таке право в наданому контексті не підтверджене.",
  "sources": [
    {
      "section": "1. Щорічна відпустка",
      "chunk": "Працівник може використати щорічну оплачувану відпустку після 6 місяців безперервної роботи у компанії.",
      "score": 0.92
    }
  ],
  "confidence": "high",
  "fallback_reason": null,
  "trace_id": "uuid-or-generated-id",
  "latency_ms": 850
}
```

**Fallback приклад (q004):**
```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Яка точна дата сплати ЄСВ у цьому місяці?"}'
```

## Індексація та retrieval

1. **Chunking:** `knowledge_base.md` парситься за заголовками `##`. Великі секції додатково діляться за абзацами з overlap.
2. **Embeddings:** OpenAI `text-embedding-3-small` при старті сервісу.
3. **Vector index:** FAISS `IndexFlatIP` (cosine через L2-нормалізацію).
4. **BM25:** keyword-пошук для точних термінів (ЄСВ, 6 місяців).
5. **Hybrid score:** `0.7 * vector + 0.3 * BM25`, top-5 chunks для LLM.

Підхід масштабується на великі KB: FAISS IVFFlat/HNSW, persistent store (Qdrant), batch embedding.

## Confidence

| Рівень | Коли |
|--------|------|
| `high` | Релевантний контекст знайдено, LLM дає чітку grounded-відповідь |
| `medium` | Є контекст, але з обмеженнями бази знань |
| `low` | Fallback: недостатньо контексту, низький score, неможливий розрахунок/дата |

## Fallback

Спрацьовує коли:
- `no_relevant_context` — chunks нижче порогу релевантності
- `insufficient_knowledge_base` — LLM визнав недостатність даних
- `calculation_not_possible` — запит на розрахунок без чисел у KB
- `exact_date_not_in_source` — запит точної дати, якої немає в KB

## LLM API

- **Генерація:** OpenAI `gpt-4o-mini` (налаштовується через `OPENAI_MODEL`)
- **Embeddings:** OpenAI `text-embedding-3-small`
- Structured JSON output через `response_format: json_object`

## Trace logging

Кожен запит пишеться в `traces.jsonl` (JSON Lines):
- `trace_id`, `timestamp`, `question`
- `stages`: receive_request → retrieve_context → generate_answer → validate_answer → return_response
- знайдені `sources`, фінальна `answer`, `confidence`, `fallback_reason`, `latency_ms`, `error`

## Evaluation

```bash
# Сервер має бути запущений
python scripts/run_evaluation.py
```

Генерує `evaluation_report.md` з результатами прогону `data/test_questions.json`.

## Інтеграція з Laravel API + Python AI Layer

```
Laravel Controller → HTTP POST /ask → Python FastAPI → OpenAI
                 ← JSON AskResponse ←
```

Laravel приклад:
```php
$response = Http::timeout(30)->post('http://ai-service:8000/ask', [
    'question' => $request->input('question'),
]);
return $response->json();
```

Python-шар ізольований за HTTP-контрактом — можна замінити на LangChain/LangGraph без змін у Laravel.

## Покращення за 1–2 тижні

- Persistent vector store + scheduled reindex
- Cross-encoder reranker
- Automated eval harness з golden answers
- Rate limiting, embedding cache, OpenTelemetry
- LangGraph multi-step agent для складних кейсів
