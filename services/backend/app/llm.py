"""Agent loop against llama.cpp's OpenAI-compatible endpoint.

llama.cpp is started with --jinja, so Qwen3's own chat template handles tool
calls and the server hands them back in OpenAI `tool_calls` format. We stream
tokens as they arrive and accumulate any tool-call deltas; if the turn ends in
tool calls we run them, append the results, and go around again.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from . import clients, rag, tools
from .config import settings

log = logging.getLogger("llm")

Emit = Callable[[dict], Awaitable[None]]

SYSTEM_PROMPT = """You are a local voice assistant.

Reply in the language the user used, in two or three short spoken sentences. No markdown.

Use knowledge_search for the user's own files and notes. Use get_weather for weather. Use web_search for current facts. Never answer from memory when a tool applies; answer from the tool's result.

Right now it is {today}."""

# KEEP THIS PROMPT SHORT. Measured on Qwen3-4B at temperature 0, asking about an
# uploaded file, with the tool schemas attached:
#
#   104 chars (identity only) ......... calls knowledge_search
#   249 chars (+ language rules) ...... calls knowledge_search
#   589 chars (+ a "voice style" para) . stops calling tools entirely
#   907 chars (full earlier prompt) ... stops calling tools entirely
#
# The failure is silent: the model answers "please upload the file" instead of
# searching for it. Length is the variable, not wording -- the paragraph that
# broke it explicitly told the model to use the tools. Anything added here must
# be re-measured; ~400 characters is the working budget.


def build_system_prompt() -> str:
    # The clock lives in the prompt rather than in a get_current_time tool:
    # one fewer schema measurably improves tool selection on a 4B model, and
    # the time is cheap to inline on every turn anyway.
    today = datetime.now(ZoneInfo("Europe/Istanbul")).strftime(
        "%Y-%m-%d %H:%M (%A), Europe/Istanbul"
    )
    prompt = SYSTEM_PROMPT.format(today=today)
    if settings.assistant_language and settings.assistant_language != "auto":
        prompt += f"\n\nAlways reply in {settings.assistant_language} regardless of the input language."
    return prompt


# --------------------------------------------------------------------------
async def _stream_once(messages: list[dict], use_tools: bool) -> AsyncIterator[tuple[str, object]]:
    """Yield ("token", str) as they arrive, then ("tool_calls", list) if any."""
    # Once a tool has answered, this turn is only phrasing prose, so sample
    # normally. Before that the model is still choosing a tool, and that choice
    # needs to be repeatable rather than varied.
    answering_from_tools = any(m.get("role") == "tool" for m in messages)
    temperature = (
        settings.temperature
        if answering_from_tools or not use_tools
        else settings.tool_decision_temperature
    )

    payload: dict = {
        "model": settings.llm_model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "stream": True,
    }
    if use_tools:
        payload["tools"] = tools.TOOL_SCHEMAS
        payload["tool_choice"] = "auto"

    pending: dict[int, dict] = {}

    async with clients.llm_client().stream(
        "POST", "/v1/chat/completions", json=payload
    ) as response:
        if response.status_code >= 400:
            body = (await response.aread()).decode("utf-8", "replace")
            raise RuntimeError(f"llama.cpp returned {response.status_code}: {body[:400]}")

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            content = delta.get("content")
            if content:
                yield "token", content

            for call in delta.get("tool_calls") or []:
                index = call.get("index", 0)
                slot = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]

    if pending:
        ordered = [pending[i] for i in sorted(pending)]
        for i, slot in enumerate(ordered):
            if not slot["id"]:
                slot["id"] = f"call_{i}"
        yield "tool_calls", ordered


# --------------------------------------------------------------------------
async def run_turn(history: list[dict], user_text: str, emit: Emit) -> str:
    """Run one user turn to completion. Returns the final assistant text."""
    messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]

    # Off by default: the transcript is the user's question and goes to the model
    # untouched, with nothing prepended to it. Document retrieval happens only
    # when the model decides it is needed and calls knowledge_search.
    if settings.always_on_rag:
        context = await _retrieve_context(user_text, emit)
        if context:
            messages.append({"role": "system", "content": context})

    messages.extend(history[-settings.history_turns * 2 :])
    messages.append({"role": "user", "content": user_text})

    final_text = ""

    for round_index in range(settings.max_tool_rounds + 1):
        # last round: force a plain answer so the model cannot loop forever
        use_tools = settings.tools_enabled and round_index < settings.max_tool_rounds

        parts: list[str] = []
        tool_calls: list[dict] | None = None

        async for kind, value in _stream_once(messages, use_tools):
            if kind == "token":
                parts.append(value)  # type: ignore[arg-type]
                await emit({"type": "token", "text": value})
            else:
                tool_calls = value  # type: ignore[assignment]

        content = "".join(parts)

        if not tool_calls:
            final_text = content
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": content})
            break

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in tool_calls
                ],
            }
        )

        for call in tool_calls:
            await emit(
                {
                    "type": "tool",
                    "phase": "start",
                    "name": call["name"],
                    "arguments": call["arguments"][:400],
                }
            )
            log.info("tool %s(%s)", call["name"], call["arguments"][:160])
            result = await tools.execute(call["name"], call["arguments"])
            await emit(
                {
                    "type": "tool",
                    "phase": "end",
                    "name": call["name"],
                    "preview": result[:200],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": result,
                }
            )

    return final_text


async def _retrieve_context(query: str, emit: Emit) -> str:
    """Opt-in always-on RAG (ALWAYS_ON_RAG=true).

    Kept because it is one round-trip cheaper than a tool call when the whole
    corpus is on-topic, but it is not the default: it puts an extra system
    message in front of every question, whether or not the question was about
    the documents.
    """
    try:
        hits = await rag.search(query, settings.documents_collection, settings.rag_top_k)
        memories = await rag.search(query, settings.memories_collection, 2)
    except Exception as exc:
        log.warning("retrieval failed: %s", exc)
        return ""

    hits = sorted(hits + memories, key=lambda h: h.score, reverse=True)
    if not hits:
        return ""

    await emit(
        {
            "type": "sources",
            "items": [{"source": h.source, "score": round(h.score, 3)} for h in hits],
        }
    )

    blocks = "\n\n".join(f"[{h.source}]\n{h.text}" for h in hits)
    return (
        "Relevant excerpts from the user's local knowledge base. Use them only if "
        "they actually answer the question; otherwise ignore them silently.\n\n" + blocks
    )
