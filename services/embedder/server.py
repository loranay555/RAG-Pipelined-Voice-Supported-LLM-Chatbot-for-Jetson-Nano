"""Tiny ONNX embedding service.

Runs intfloat/multilingual-e5-small (384 dims, 12 layers, ~118M params) on the
CPU via onnxruntime.  Chosen over a Qwen embedding model because it is ~20x
smaller, has genuinely good Turkish retrieval quality, and leaves the Orin's
GPU entirely to llama.cpp and whisper.cpp.

E5 models are asymmetric: queries must be prefixed with "query: " and indexed
text with "passage: ".  Getting this wrong silently costs a lot of recall, so
the prefix is applied here rather than left to callers.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embedder")

MODEL_DIR = Path(os.getenv("EMBED_MODEL_DIR", "/models/multilingual-e5-small"))
MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))
INTRA_OP_THREADS = int(os.getenv("ORT_INTRA_OP_THREADS", "2"))

PREFIX = {"query": "query: ", "passage": "passage: "}


def _find_onnx(model_dir: Path) -> Path:
    # EMBED_ONNX_FILE=model_quantized.onnx roughly halves latency on the Orin's
    # A78 cores at a small recall cost -- worth trying if embedding shows up in
    # the latency budget.
    preferred = os.getenv("EMBED_ONNX_FILE", "model.onnx")
    candidates = [
        model_dir / preferred,
        model_dir / "onnx" / preferred,
        model_dir / "model.onnx",
        model_dir / "onnx" / "model.onnx",
    ]
    for c in candidates:
        if c.is_file():
            return c
    found = sorted(model_dir.rglob("*.onnx"))
    if found:
        return found[0]
    raise FileNotFoundError(
        f"No .onnx file under {model_dir}. Run scripts/download_models.sh."
    )


def _find_tokenizer(model_dir: Path) -> Path:
    for c in (model_dir / "tokenizer.json", model_dir / "onnx" / "tokenizer.json"):
        if c.is_file():
            return c
    raise FileNotFoundError(f"tokenizer.json not found under {model_dir}")


class Embedder:
    def __init__(self) -> None:
        onnx_path = _find_onnx(MODEL_DIR)
        tok_path = _find_tokenizer(MODEL_DIR)
        log.info("loading %s", onnx_path)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = INTRA_OP_THREADS
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_name = self._pick_output()

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.tokenizer.enable_padding(
            pad_id=self._pad_id(), pad_token="<pad>", length=None
        )
        # onnxruntime sessions are thread safe for Run(), but the tokenizer's
        # padding config is global state -- serialise the whole thing.
        self._lock = threading.Lock()
        self.dim = self._probe_dim()
        log.info("ready: dim=%d inputs=%s output=%s", self.dim, self.input_names, self.output_name)

    def _pad_id(self) -> int:
        for token in ("<pad>", "[PAD]"):
            tid = self.tokenizer.token_to_id(token)
            if tid is not None:
                return tid
        return 0

    def _pick_output(self) -> str:
        names = [o.name for o in self.session.get_outputs()]
        for preferred in ("last_hidden_state", "token_embeddings", "hidden_states"):
            if preferred in names:
                return preferred
        return names[0]

    def _probe_dim(self) -> int:
        return int(self.encode(["boyut testi"], "passage").shape[1])

    def encode(self, texts: list[str], kind: Literal["query", "passage"]) -> np.ndarray:
        prefixed = [PREFIX[kind] + t.strip() for t in texts]
        with self._lock:
            encodings = self.tokenizer.encode_batch(prefixed)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self.input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run([self.output_name], feeds)[0]

        # masked mean pooling
        mask = attention_mask[..., None].astype(np.float32)
        summed = (outputs * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)


app = FastAPI(title="embedder", docs_url=None, redoc_url=None)
_embedder: Embedder | None = None


@app.on_event("startup")
def _startup() -> None:
    global _embedder
    _embedder = Embedder()


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    kind: Literal["query", "passage"] = "passage"


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int


@app.get("/health")
def health() -> dict:
    if _embedder is None:
        raise HTTPException(status_code=503, detail="model still loading")
    return {"status": "ok", "dim": _embedder.dim, "max_length": MAX_LENGTH}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if _embedder is None:
        raise HTTPException(status_code=503, detail="model still loading")
    vectors = _embedder.encode(req.texts, req.kind)
    return EmbedResponse(vectors=vectors.tolist(), dim=_embedder.dim)
