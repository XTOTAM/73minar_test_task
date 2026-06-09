from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from openai import OpenAI

from app.config import settings
from app.models import AskRequest, AskResponse
from app.pipeline.consultant import ConsultantPipeline

pipeline: ConsultantPipeline | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipeline
    client = OpenAI(api_key=settings.openai_api_key)
    pipeline = ConsultantPipeline(settings, client)
    pipeline.build_index()
    yield


app = FastAPI(
    title="AI Consultant API",
    description="Controlled AI Consultant Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized")
    return await pipeline.ask(request.question.strip())
