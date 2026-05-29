import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    source: str
    title: str
    section: str
    text: str
    word_count: int

    def to_dict(self) -> dict:
        return asdict(self)


HEADING_RE = re.compile(
    r"^(#{1,6}\s+.+|[A-ZÀ-Ỵ0-9][^.!?]{0,100}:?|"
    r"\d+(\.\d+)*\s+[^.!?]{3,120})$"
)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def split_metadata(raw_text: str) -> tuple[str, str, str]:
    source = ""
    title = ""
    body = raw_text

    source_match = re.search(r"^SOURCE:\s*(.+)$", raw_text, flags=re.MULTILINE)
    title_match = re.search(r"^TITLE:\s*(.+)$", raw_text, flags=re.MULTILINE)
    if source_match:
        source = source_match.group(1).strip()
    if title_match:
        title = title_match.group(1).strip()

    separator = "-" * 30
    if separator in raw_text:
        body = raw_text.split(separator, 1)[1]
    else:
        body = re.sub(r"^SOURCE:.*$", "", body, flags=re.MULTILINE)
        body = re.sub(r"^TITLE:.*$", "", body, flags=re.MULTILINE)

    return source, title, normalize_text(body)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading(line: str) -> bool:
    clean = line.strip()
    if len(clean) < 4 or len(clean) > 140:
        return False
    if clean.startswith("|"):
        return False
    if clean.endswith(".") and len(clean.split()) > 8:
        return False
    return bool(HEADING_RE.match(clean))


def structural_units(text: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    current_section = "Document"
    buffer: list[str] = []

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue

        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) == 1 and is_heading(lines[0]):
            if buffer:
                units.append((current_section, "\n".join(buffer).strip()))
                buffer = []
            current_section = lines[0].lstrip("#").strip()
            continue

        if block.startswith("[Data Table]") or block.startswith("[TABLE"):
            if buffer:
                units.append((current_section, "\n".join(buffer).strip()))
                buffer = []
            units.append((current_section, block))
            continue

        buffer.append(block)

    if buffer:
        units.append((current_section, "\n".join(buffer).strip()))

    return units


def words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = words(text)
    if len(tokens) <= chunk_size:
        return [text]

    parts = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        parts.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return parts


def make_chunks_for_file(path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    raw_text = read_text_file(path)
    source, title, body = split_metadata(raw_text)
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_section = "Document"
    current_words = 0

    def flush() -> None:
        nonlocal current_texts, current_words
        if not current_texts:
            return
        text = "\n\n".join(current_texts).strip()
        for split_text in split_long_text(text, chunk_size, overlap):
            chunk_no = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}:{chunk_no}",
                    source_file=path.name,
                    source=source,
                    title=title,
                    section=current_section,
                    text=split_text,
                    word_count=len(words(split_text)),
                )
            )
        current_texts = []
        current_words = 0

    for section, unit_text in structural_units(body):
        unit_words = len(words(unit_text))
        if current_texts and (section != current_section or current_words + unit_words > chunk_size):
            flush()
        current_section = section
        current_texts.append(unit_text)
        current_words += unit_words

    flush()
    return chunks


def make_chunks(knowledge_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.txt")):
        all_chunks.extend(make_chunks_for_file(path, chunk_size, overlap))
    return all_chunks


def format_chunk_for_prompt(chunk: dict) -> str:
    source = chunk.get("source") or chunk.get("source_file", "")
    title = chunk.get("title") or chunk.get("source_file", "")
    section = chunk.get("section") or "Document"
    return (
        f"[Source: {source}]\n"
        f"[Title: {title}]\n"
        f"[Section: {section}]\n"
        f"{chunk['text']}"
    )
