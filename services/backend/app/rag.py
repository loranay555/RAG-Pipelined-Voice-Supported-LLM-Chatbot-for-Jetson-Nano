"""Qdrant-backed retrieval: document knowledge base + long-term memories."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import AsyncQdrantClient, models

from . import clients
from .config import settings

log = logging.getLogger("rag")

_NAMESPACE = uuid.UUID("6f6c6c61-6d61-4a65-b473-6f6e00000001")
_client: AsyncQdrantClient | None = None

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log"}
PDF_SUFFIXES = {".pdf"}
INGESTABLE_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def extract_text(raw: bytes, filename: str) -> str:
    """Decode an uploaded file to plain text.

    PDFs go through pypdf's text layer. Scanned PDFs have no text layer and come
    back empty -- that needs OCR, which is deliberately not part of this stack.
    """
    if Path(filename).suffix.lower() in PDF_SUFFIXES:
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for number, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                log.warning("%s page %d: %s", filename, number, exc)
                continue
            if text.strip():
                pages.append(text.strip())

        if not pages:
            raise ValueError(
                f"{filename} has no extractable text layer (scanned PDF?); "
                "OCR is not part of this stack."
            )
        return "\n\n".join(pages)

    return raw.decode("utf-8", errors="replace")


@dataclass
class Hit:
    text: str
    score: float
    source: str
    metadata: dict


def client() -> AsyncQdrantClient:
    assert _client is not None, "rag.startup() was not awaited"
    return _client


async def startup() -> None:
    global _client
    _client = AsyncQdrantClient(url=settings.qdrant_url, prefer_grpc=False, timeout=30)

    # compose only waits for qdrant's container, not for it to accept requests
    last_error: Exception | None = None
    for attempt in range(12):
        try:
            for name in (settings.documents_collection, settings.memories_collection):
                await _ensure_collection(name)
            return
        except Exception as exc:
            last_error = exc
            log.info("qdrant not ready yet (%s), retrying", type(exc).__name__)
            await asyncio.sleep(2.5)

    raise RuntimeError(f"qdrant unreachable at {settings.qdrant_url}: {last_error}")


async def shutdown() -> None:
    if _client is not None:
        await _client.close()


async def _ensure_collection(name: str) -> None:
    existing = {c.name for c in (await client().get_collections()).collections}
    if name in existing:
        return
    log.info("creating collection %s", name)
    await client().create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=settings.embedding_dim,
            distance=models.Distance.COSINE,
            on_disk=True,
        ),
        # 8 GB board: keep the payload off the heap, memory-map the vectors
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
    )
    await client().create_payload_index(
        collection_name=name,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


# ------------------------------------------------------------------ chunking
def chunk_text(text: str) -> list[str]:
    """Paragraph-aware fixed-size chunking with overlap."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        if len(buf) + len(para) + 2 <= settings.chunk_chars:
            buf = f"{buf}\n\n{para}" if buf else para
            continue
        if buf:
            chunks.append(buf)
        # a single paragraph longer than the window gets hard-split
        while len(para) > settings.chunk_chars:
            cut = para.rfind(" ", 0, settings.chunk_chars)
            cut = cut if cut > settings.chunk_chars // 2 else settings.chunk_chars
            chunks.append(para[:cut])
            para = para[max(0, cut - settings.chunk_overlap) :]
        buf = para

    if buf:
        chunks.append(buf)
    return chunks


# ----------------------------------------------------------------- ingestion
def content_digest(text: str) -> str:
    """Identity of an ingested file.

    The chunking parameters are part of it on purpose: tuning chunk_chars must
    invalidate everything already in the collection, otherwise a rescan is a
    silent no-op and the new setting appears to do nothing.
    """
    payload = f"{settings.chunk_chars}:{settings.chunk_overlap}:{text}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


async def ingest_text(text: str, source: str, metadata: dict | None = None) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0

    await delete_source(source)

    digest = content_digest(text)
    points: list[models.PointStruct] = []

    # embed in small batches so a big file cannot blow the embedder's memory
    for start in range(0, len(chunks), 16):
        batch = chunks[start : start + 16]
        vectors = await clients.embed(batch, kind="passage")
        for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
            idx = start + offset
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(_NAMESPACE, f"{source}:{idx}")),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": source,
                        "chunk": idx,
                        "hash": digest,
                        "ingested_at": time.time(),
                        **(metadata or {}),
                    },
                )
            )

    await client().upsert(
        collection_name=settings.documents_collection, points=points, wait=True
    )
    log.info("ingested %s (%d chunks)", source, len(points))
    return len(points)


async def delete_source(source: str) -> None:
    await client().delete(
        collection_name=settings.documents_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
            )
        ),
        wait=True,
    )


async def ingest_docs_dir() -> int:
    """Sync ./data/docs into the documents collection. Unchanged files are skipped."""
    root = Path(settings.docs_dir)
    if not root.is_dir():
        return 0

    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in INGESTABLE_SUFFIXES:
            continue
        try:
            text = extract_text(path.read_bytes(), path.name)
        except Exception as exc:  # unreadable file must not abort the whole scan
            log.warning("cannot read %s: %s", path, exc)
            continue

        source = str(path.relative_to(root))
        if await _source_hash(source) == content_digest(text):
            continue
        total += await ingest_text(text, source=source)

    return total


async def _source_hash(source: str) -> str | None:
    points, _ = await client().scroll(
        collection_name=settings.documents_collection,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
        ),
        limit=1,
        with_payload=["hash"],
        with_vectors=False,
    )
    return points[0].payload.get("hash") if points else None


# ------------------------------------------------------------------- search
async def search(
    query: str,
    collection: str | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[Hit]:
    collection = collection or settings.documents_collection
    top_k = top_k or settings.rag_top_k
    threshold = settings.rag_min_score if min_score is None else min_score

    vector = await clients.embed_one(query, kind="query")
    result = await client().query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        score_threshold=threshold,
        with_payload=True,
    )

    return [
        Hit(
            text=p.payload.get("text", ""),
            score=float(p.score),
            source=p.payload.get("source", collection),
            metadata={k: v for k, v in p.payload.items() if k not in ("text", "hash")},
        )
        for p in result.points
    ]


async def remember(text: str, tags: list[str] | None = None) -> str:
    point_id = str(uuid.uuid4())
    vector = await clients.embed_one(text, kind="passage")
    await client().upsert(
        collection_name=settings.memories_collection,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": text,
                    "source": "memory",
                    "tags": tags or [],
                    "created_at": time.time(),
                },
            )
        ],
        wait=True,
    )
    return point_id


async def stats() -> dict:
    out = {}
    for name in (settings.documents_collection, settings.memories_collection):
        try:
            info = await client().get_collection(name)
            out[name] = info.points_count
        except Exception:
            out[name] = None
    return out
