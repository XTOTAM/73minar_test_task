from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    knowledge_base_path: Path = Path("data/knowledge_base.md")
    traces_path: Path = Path("traces.jsonl")
    retrieval_top_k: int = 5
    min_score_threshold: float = 0.3
    hybrid_alpha: float = 0.7
    max_chunk_tokens: int = 500


settings = Settings()
