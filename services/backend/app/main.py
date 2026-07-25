"""FastAPI backend: websocket voice/text pipeline + REST admin endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import clients, llm, rag
from .audio import VadSegmenter
from .config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("backend")

# httpx logs every outbound request at INFO. With a 20 s healthcheck probing
# four services that is ~15 lines/minute of noise that buries the events you
# actually want to see while debugging the voice pipeline.
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Jetson Assistant API", docs_url="/api/docs", openapi_url="/api/openapi.json")

# The page is served from the same origin through Caddy, but keep this open so
# you can also hit the API directly from a script on the LAN.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    await clients.startup()
    await rag.startup()
    asyncio.create_task(_initial_ingest())


@app.on_event("shutdown")
async def _shutdown() -> None:
    await rag.shutdown()
    await clients.shutdown()


async def _initial_ingest() -> None:
    """Sync ./data/docs in the background so startup is not blocked by it."""
    for attempt in range(10):
        try:
            count = await rag.ingest_docs_dir()
            if count:
                log.info("ingested %d chunks from %s", count, settings.docs_dir)
            return
        except Exception as exc:
            log.warning("ingest attempt %d failed: %s", attempt + 1, exc)
            await asyncio.sleep(5)


# --------------------------------------------------------------------------
# REST
# --------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    async def probe(name: str, url: str, path: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=4.0) as c:
                r = await c.get(f"{url}{path}")
            return "ok" if r.status_code < 500 else f"http {r.status_code}"
        except Exception as exc:
            return f"down ({type(exc).__name__})"

    llm_status, stt_status, embed_status = await asyncio.gather(
        probe("llm", settings.llm_url, "/health"),
        probe("stt", settings.stt_url, "/"),
        probe("embedder", settings.embedder_url, "/health"),
    )
    try:
        collections = await rag.stats()
        qdrant_status = "ok"
    except Exception as exc:
        collections, qdrant_status = {}, f"down ({type(exc).__name__})"

    ready = all(s == "ok" for s in (llm_status, embed_status, qdrant_status))
    return {
        "status": "ok" if ready else "degraded",
        "services": {
            "llm": llm_status,
            "stt": stt_status,
            "embedder": embed_status,
            "qdrant": qdrant_status,
        },
        "collections": collections,
        "config": {
            "model": settings.llm_model_name,
            "tools_enabled": settings.tools_enabled,
            "interim_transcription": settings.interim_transcription,
            "stt_language": settings.stt_language,
            "sample_rate": settings.sample_rate,
        },
    }


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[dict] = Field(default_factory=list)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Server-sent events. Handy for curl; the web UI uses the websocket."""
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def emit(event: dict) -> None:
        await queue.put(event)

    async def run() -> None:
        try:
            await llm.run_turn(list(req.history), req.message, emit)
        except Exception as exc:
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await queue.put({"type": "done"})
            await queue.put(None)

    asyncio.create_task(run())

    async def stream():
        while True:
            event = await queue.get()
            if event is None:
                return
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)


@app.post("/api/ingest")
async def ingest(req: IngestRequest) -> dict:
    chunks = await rag.ingest_text(req.text, source=req.source)
    return {"source": req.source, "chunks": chunks}


@app.post("/api/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)) -> dict:
    """Upload a PDF or a UTF-8 text file into the knowledge base."""
    name = file.filename or "upload"
    raw = await file.read()
    try:
        text = await asyncio.to_thread(rag.extract_text, raw, name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    chunks = await rag.ingest_text(text, source=name)
    return {"source": name, "chunks": chunks, "characters": len(text)}


@app.post("/api/ingest/rescan")
async def ingest_rescan() -> dict:
    return {"chunks": await rag.ingest_docs_dir()}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict:
    """One-shot transcription of a 16 kHz mono WAV."""
    text = await clients.transcribe_wav(await file.read())
    return {"text": text}


# --------------------------------------------------------------------------
# websocket
# --------------------------------------------------------------------------
class Session:
    """Per-connection state. One websocket == one conversation."""

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self.history: list[dict] = []
        self.segmenter = VadSegmenter()
        self.turn_task: asyncio.Task | None = None
        self.interim_task: asyncio.Task | None = None
        self.last_interim = 0.0
        self.recording = False
        self.language = settings.stt_language
        self.hold_mode = False
        self._send_lock = asyncio.Lock()

    async def send(self, event: dict) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(event, ensure_ascii=False))
            except (WebSocketDisconnect, RuntimeError):
                pass

    def busy(self) -> bool:
        return self.turn_task is not None and not self.turn_task.done()

    async def cancel_turn(self) -> None:
        if self.busy():
            self.turn_task.cancel()  # type: ignore[union-attr]
            try:
                await self.turn_task  # type: ignore[arg-type]
            except (asyncio.CancelledError, Exception):
                pass
        self.turn_task = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = Session(websocket)
    await session.send({"type": "status", "state": "idle"})

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                await _on_audio(session, data)
            elif (text := message.get("text")) is not None:
                await _on_control(session, text)

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket loop failed")
    finally:
        await session.cancel_turn()
        if session.interim_task and not session.interim_task.done():
            session.interim_task.cancel()


async def _on_control(session: Session, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await session.send({"type": "error", "message": "malformed control message"})
        return

    kind = msg.get("type")

    if kind == "text":
        text = (msg.get("text") or "").strip()
        if text:
            await _start_turn(session, text)

    elif kind == "set_language":
        _set_language(session, msg.get("language"))

    elif kind == "audio_start":
        _set_language(session, msg.get("language"))
        session.hold_mode = msg.get("mode") == "hold"
        session.segmenter.reset()
        session.segmenter.auto_close = not session.hold_mode
        session.recording = True
        session.last_interim = time.monotonic()
        await session.send({"type": "status", "state": "listening"})

    elif kind == "audio_stop":
        session.recording = False
        segment = session.segmenter.flush()
        if segment is not None:
            await _transcribe_and_run(session, segment.pcm)
        elif not session.busy():
            # Nothing left in the buffer. Usually this means the VAD already
            # closed the utterance on the pause before the user let go of the
            # button -- in that case a turn is running and we must not stomp on
            # its "thinking" status with an idle.
            await session.send({"type": "transcript", "text": "", "final": True})
            await session.send({"type": "status", "state": "idle"})

    elif kind == "cancel":
        await session.cancel_turn()
        await session.send({"type": "status", "state": "idle"})
        await session.send({"type": "done", "cancelled": True})

    elif kind == "reset":
        await session.cancel_turn()
        session.history.clear()
        session.segmenter.reset()
        await session.send({"type": "status", "state": "idle"})

    elif kind == "ping":
        await session.send({"type": "pong"})


SUPPORTED_LANGUAGES = {"tr", "en", "auto"}


def _set_language(session: Session, value: object) -> None:
    if not isinstance(value, str):
        return
    language = value.strip().lower()
    if language in SUPPORTED_LANGUAGES and language != session.language:
        session.language = language
        log.info("session language -> %s", language)


async def _on_audio(session: Session, pcm: bytes) -> None:
    if not session.recording:
        return

    for segment in session.segmenter.push(pcm):
        # VAD closed an utterance on its own (user paused) -- act on it without
        # waiting for the mic button to be released.
        await _transcribe_and_run(session, segment.pcm)

    if settings.interim_transcription and session.segmenter.in_speech:
        await _maybe_interim(session)


async def _maybe_interim(session: Session) -> None:
    now = time.monotonic()
    if now - session.last_interim < settings.interim_interval_ms / 1000:
        return
    if session.interim_task is not None and not session.interim_task.done():
        return

    pcm = session.segmenter.current_pcm()
    if len(pcm) < settings.sample_rate * 2 * 0.6:  # < 600 ms of audio
        return

    session.last_interim = now

    async def run() -> None:
        try:
            text = await clients.transcribe_pcm(pcm, session.language)
            if text:
                await session.send({"type": "transcript", "text": text, "final": False})
        except Exception as exc:
            log.debug("interim transcription failed: %s", exc)

    session.interim_task = asyncio.create_task(run())


async def _transcribe_and_run(session: Session, pcm: bytes) -> None:
    await session.send({"type": "status", "state": "transcribing"})
    try:
        text = await clients.transcribe_pcm(pcm, session.language)
    except Exception as exc:
        log.warning("transcription failed: %s", exc)
        await session.send({"type": "error", "message": f"Ses yazıya çevrilemedi: {exc}"})
        await session.send({"type": "status", "state": "idle"})
        return

    log.info("transcript (%.1fs audio): %r", len(pcm) / 2 / settings.sample_rate, text)
    await session.send({"type": "transcript", "text": text, "final": True})
    if not text:
        await session.send({"type": "status", "state": "idle"})
        return

    await _start_turn(session, text)


async def _start_turn(session: Session, text: str) -> None:
    await session.cancel_turn()
    await session.send({"type": "user_message", "text": text})
    await session.send({"type": "status", "state": "thinking"})

    async def run() -> None:
        try:
            await llm.run_turn(session.history, text, session.send)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("turn failed")
            await session.send({"type": "error", "message": str(exc)})
        finally:
            await session.send({"type": "status", "state": "idle"})
            await session.send({"type": "done"})

    session.turn_task = asyncio.create_task(run())
