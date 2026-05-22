import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    content: str
    metadata: dict


def load_documents(file_path: str | Path) -> Iterator[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    logger.info(f"Loading documents from {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield Document(
                    id=data.get("id", f"doc_{line_num}"),
                    content=data.get("content", ""),
                    metadata=data.get("metadata", {}),
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")


def _make_chunk(
    document: Document,
    chunks: list[Document],
    text: str,
    start_index: int = 0,
) -> Document:
    idx = start_index + len(chunks)
    return Document(
        id=f"{document.id}_chunk_{idx}",
        content=text,
        metadata={
            **document.metadata,
            "source_id": document.id,
            "chunk_index": idx,
        },
    )


def _parse_blocks(content: str) -> list[str]:
    content = content.replace("\r\n", "\n")

    has_headings = bool(re.search(r"^#{1,6}\s+", content, re.MULTILINE))
    if has_headings:
        sections = re.split(r"\n(?=#{1,6}\s+)", content)
        return [s.strip() for s in sections if s.strip()]

    if "\n\n" in content:
        return [p.strip() for p in content.split("\n\n") if p.strip()]

    return []


def _chunk_by_sentences(
    document: Document,
    text: str,
    chunk_size: int,
    start_index: int = 0,
) -> list[Document]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        punct = ". " if not sentence.endswith(".") else " "
        word_count = len((sentence + punct).split())

        if current_len + word_count > chunk_size and current:
            chunk_text = "".join(current).strip()
            if chunk_text:
                chunks.append(_make_chunk(document, chunks, chunk_text, start_index))
            current = [current[-1]]
            current_len = len(current[-1].split())

        current.append(sentence + punct)
        current_len += word_count

    if current:
        chunk_text = "".join(current).strip()
        if chunk_text:
            chunks.append(_make_chunk(document, chunks, chunk_text, start_index))

    return chunks


def _chunk_by_blocks(
    document: Document,
    blocks: list[str],
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    chunks = []
    current = []
    current_len = 0

    def flush(final: bool = False):
        nonlocal current, current_len
        if not current:
            return
        chunk_text = "\n\n".join(current)
        chunks.append(_make_chunk(document, chunks, chunk_text))
        if not final:
            overlap_blocks = []
            overlap_len = 0
            for b in reversed(current):
                b_len = len(b.split())
                if overlap_len + b_len <= overlap:
                    overlap_blocks.insert(0, b)
                    overlap_len += b_len
                else:
                    remaining = overlap - overlap_len
                    if remaining > 0:
                        words = b.split()
                        truncated = " ".join(words[:remaining])
                        overlap_blocks.insert(0, truncated)
                        overlap_len += remaining
                    break
            current = overlap_blocks
            current_len = overlap_len

    for block in blocks:
        block_len = len(block.split())

        if block_len > chunk_size:
            flush()
            sentence_chunks = _chunk_by_sentences(document, block, chunk_size, start_index=len(chunks))
            if sentence_chunks:
                chunks.extend(sentence_chunks)
                last_sentence_chunk_text = sentence_chunks[-1].content
                current = [last_sentence_chunk_text]
                current_len = len(last_sentence_chunk_text.split())
            continue

        if current_len + block_len > chunk_size and current:
            flush()

        current.append(block)
        current_len += block_len

    flush(final=True)
    return chunks


def semantic_chunk(
    document: Document,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Document]:
    content = document.content
    if not content:
        return []

    blocks = _parse_blocks(content)
    if not blocks:
        return _chunk_by_sentences(document, content, chunk_size)

    return _chunk_by_blocks(document, blocks, chunk_size, overlap)


def chunk_documents(
    file_path: str | Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Document]:
    logger.info(f"Chunking documents from {file_path}")
    all_chunks = []
    for doc in load_documents(file_path):
        chunks = semantic_chunk(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
        logger.debug(f"Document {doc.id} split into {len(chunks)} chunks")
    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks