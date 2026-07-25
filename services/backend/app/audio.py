"""Voice-activity segmentation for the raw PCM stream coming off the browser.

The browser sends 16 kHz mono PCM16 over the websocket.  We cannot hand an
open-ended stream to whisper.cpp, so this splits it into utterances: speech
starts, speech ends, one WAV goes to the STT service.

Design notes
------------
* webrtcvad works on exactly 10/20/30 ms frames -- anything else raises, so
  partial frames are buffered until complete.
* A pre-roll ring buffer keeps the ~300 ms *before* VAD triggered.  Without it
  the first syllable is clipped, which is the single most common cause of
  "why did it transcribe 'ava durumu'" style errors.
* max_utterance_ms force-closes a segment so a noisy room can never grow the
  buffer without bound.
"""

from __future__ import annotations

import io
import wave
from collections import deque
from dataclasses import dataclass

import webrtcvad

from .config import settings


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


@dataclass
class Segment:
    pcm: bytes
    duration_ms: int
    reason: str  # "silence" | "max_length" | "flush"


class VadSegmenter:
    def __init__(
        self,
        sample_rate: int | None = None,
        frame_ms: int | None = None,
        aggressiveness: int | None = None,
        silence_ms: int | None = None,
        min_speech_ms: int | None = None,
        max_utterance_ms: int | None = None,
        preroll_ms: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate or settings.sample_rate
        self.frame_ms = frame_ms or settings.frame_ms
        self.frame_bytes = int(self.sample_rate * self.frame_ms / 1000) * 2

        self.vad = webrtcvad.Vad(
            aggressiveness if aggressiveness is not None else settings.vad_aggressiveness
        )
        self.silence_frames_needed = max(
            1, (silence_ms or settings.vad_silence_ms) // self.frame_ms
        )
        self.min_speech_frames = max(
            1, (min_speech_ms or settings.vad_min_speech_ms) // self.frame_ms
        )
        self.max_frames = max(
            1, (max_utterance_ms or settings.vad_max_utterance_ms) // self.frame_ms
        )
        preroll_frames = max(1, (preroll_ms or settings.vad_preroll_ms) // self.frame_ms)

        # Push-to-talk turns this off: the button already marks where the
        # utterance ends, so closing on a mid-sentence pause would cut the user
        # off while they are still holding it down. max_frames still applies.
        self.auto_close = True

        self._tail = b""
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)
        self._voiced: list[bytes] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    # ------------------------------------------------------------------ api
    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def current_pcm(self) -> bytes:
        """Audio captured for the utterance in progress (for interim decodes)."""
        return b"".join(self._voiced)

    def push(self, pcm: bytes) -> list[Segment]:
        """Feed raw PCM16; returns any utterances that closed during this chunk."""
        self._tail += pcm
        segments: list[Segment] = []

        while len(self._tail) >= self.frame_bytes:
            frame = self._tail[: self.frame_bytes]
            self._tail = self._tail[self.frame_bytes :]
            seg = self._push_frame(frame)
            if seg is not None:
                segments.append(seg)

        return segments

    def flush(self) -> Segment | None:
        """Close whatever is buffered, e.g. when the user releases the mic."""
        if self._tail:
            # pad the last partial frame with silence so it is still decodable
            pad = self.frame_bytes - len(self._tail)
            self._push_frame(self._tail + b"\x00" * pad)
            self._tail = b""

        if self._speech_frames < self.min_speech_frames:
            self.reset()
            return None

        pcm = b"".join(self._voiced)
        duration = len(pcm) // 2 * 1000 // self.sample_rate
        self.reset()
        return Segment(pcm=pcm, duration_ms=duration, reason="flush")

    def reset(self) -> None:
        self._tail = b""
        self._preroll.clear()
        self._voiced.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    # -------------------------------------------------------------- internal
    def _push_frame(self, frame: bytes) -> Segment | None:
        try:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
        except Exception:  # malformed frame length -> treat as silence
            is_speech = False

        if not self._in_speech:
            self._preroll.append(frame)
            if is_speech:
                self._in_speech = True
                self._voiced = list(self._preroll)
                self._preroll.clear()
                self._speech_frames = 1
                self._silence_frames = 0
            return None

        self._voiced.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if self.auto_close and self._silence_frames >= self.silence_frames_needed:
            return self._close("silence")
        if len(self._voiced) >= self.max_frames:
            return self._close("max_length")
        return None

    def _close(self, reason: str) -> Segment | None:
        pcm = b"".join(self._voiced)
        speech_frames = self._speech_frames
        self.reset()
        if speech_frames < self.min_speech_frames:
            return None  # a cough or a door slam, not an utterance
        duration = len(pcm) // 2 * 1000 // self.sample_rate
        return Segment(pcm=pcm, duration_ms=duration, reason=reason)
