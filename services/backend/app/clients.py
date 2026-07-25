"""Shared HTTP clients for the sidecar services."""

from __future__ import annotations

import logging

import httpx

from .audio import pcm16_to_wav
from .config import settings

log = logging.getLogger("clients")

# One pooled client per upstream. Timeouts differ a lot: whisper on a 20 s
# utterance can take several seconds, the LLM streams for a minute.
_embed_client: httpx.AsyncClient | None = None
_stt_client: httpx.AsyncClient | None = None
_llm_client: httpx.AsyncClient | None = None
_web_client: httpx.AsyncClient | None = None


async def startup() -> None:
    global _embed_client, _stt_client, _llm_client, _web_client
    _embed_client = httpx.AsyncClient(base_url=settings.embedder_url, timeout=30.0)
    _stt_client = httpx.AsyncClient(base_url=settings.stt_url, timeout=120.0)
    _llm_client = httpx.AsyncClient(
        base_url=settings.llm_url,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
    )
    _web_client = httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": settings.http_user_agent},
    )


async def shutdown() -> None:
    for client in (_embed_client, _stt_client, _llm_client, _web_client):
        if client is not None:
            await client.aclose()


def llm_client() -> httpx.AsyncClient:
    assert _llm_client is not None, "clients.startup() was not awaited"
    return _llm_client


def web_client() -> httpx.AsyncClient:
    assert _web_client is not None, "clients.startup() was not awaited"
    return _web_client


# ------------------------------------------------------------------ embedder
async def embed(texts: list[str], kind: str = "passage") -> list[list[float]]:
    assert _embed_client is not None
    resp = await _embed_client.post("/embed", json={"texts": texts, "kind": kind})
    resp.raise_for_status()
    return resp.json()["vectors"]


async def embed_one(text: str, kind: str = "query") -> list[float]:
    return (await embed([text], kind))[0]


# ----------------------------------------------------------------------- stt
async def transcribe_pcm(pcm: bytes, language: str | None = None) -> str:
    """Send one utterance to whisper.cpp and return the text."""
    return await transcribe_wav(pcm16_to_wav(pcm, settings.sample_rate), language)


async def transcribe_wav(wav: bytes, language: str | None = None) -> str:
    assert _stt_client is not None
    data = {
        "temperature": "0.0",
        "temperature_inc": "0.2",
        "response_format": "json",
    }
    # Sent even when it is "auto": otherwise whisper-server would silently fall
    # back to its CLI default, and the UI's "Otomatik" option would not be
    # automatic at all.
    if language:
        data["language"] = language

    resp = await _stt_client.post(
        "/inference",
        files={"file": ("audio.wav", wav, "audio/wav")},
        data=data,
    )
    resp.raise_for_status()

    try:
        text = resp.json().get("text", "")
    except ValueError:
        text = resp.text

    return _clean_transcript(text)


# whisper emits these for silence/music instead of an empty string
_NOISE_MARKERS = {
    "[blank_audio]", "[ sessizlik ]", "[sessizlik]", "[music]", "[müzik]",
    "(müzik)", "(music)", "[silence]", "*", "...", "[inaudible]",
    "altyazı m.k.", "abone ol", "izlediğiniz için teşekkürler",
}


def _clean_transcript(text: str) -> str:
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    if text.lower().strip(".!? ") in _NOISE_MARKERS:
        return ""
    # a lone bracketed marker like "[BLANK_AUDIO]"
    if text.startswith("[") and text.endswith("]") and " " not in text.strip("[]"):
        return ""
    return text
