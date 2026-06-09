import re
import uuid
from pathlib import Path

from app.models import Chunk

SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF]")


def detect_language(text: str) -> str:
    has_cyrillic = bool(CYRILLIC_PATTERN.search(text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_cyrillic and has_latin:
        return "mixed"
    if has_cyrillic:
        return "uk"
    return "en"


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return paragraphs or [text.strip()]


def _subchunk_paragraphs(
    section: str,
    paragraphs: list[str],
    max_tokens: int,
) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = _estimate_tokens(paragraph)
        if current and current_tokens + paragraph_tokens > max_tokens:
            chunks.append((section, "\n\n".join(current)))
            overlap = current[-1] if current else ""
            current = [overlap, paragraph] if overlap else [paragraph]
            current_tokens = _estimate_tokens("\n\n".join(current))
        else:
            current.append(paragraph)
            current_tokens += paragraph_tokens

    if current:
        chunks.append((section, "\n\n".join(current)))

    return chunks


def chunk_knowledge_base(path: Path, max_chunk_tokens: int = 500) -> list[Chunk]:
    content = path.read_text(encoding="utf-8")
    matches = list(SECTION_PATTERN.finditer(content))
    if not matches:
        return []

    chunks: list[Chunk] = []
    for index, match in enumerate(matches):
        section_title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        section_body = content[start:end].strip()
        if not section_body:
            continue

        paragraphs = _split_paragraphs(section_body)
        if _estimate_tokens(section_body) <= max_chunk_tokens:
            section_chunks = [(section_title, section_body)]
        else:
            section_chunks = _subchunk_paragraphs(section_title, paragraphs, max_chunk_tokens)

        for section, text in section_chunks:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    section=section,
                    text=text,
                    language=detect_language(text),  # type: ignore[arg-type]
                )
            )

    return chunks
