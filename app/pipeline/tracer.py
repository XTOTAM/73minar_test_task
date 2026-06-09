import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.models import TraceRecord


class TraceLogger:
    def __init__(self, traces_path: Path) -> None:
        self._traces_path = traces_path
        self._lock = asyncio.Lock()
        self._traces_path.parent.mkdir(parents=True, exist_ok=True)

    async def log(self, record: TraceRecord) -> None:
        line = record.model_dump_json(ensure_ascii=False)
        async with self._lock:
            with self._traces_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
