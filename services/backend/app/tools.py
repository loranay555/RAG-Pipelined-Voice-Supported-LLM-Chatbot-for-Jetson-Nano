"""Tools the LLM can call.

Everything here returns a plain string: llama.cpp feeds the tool result back as
a `role: tool` message, and a compact human-readable string costs far fewer
tokens than nested JSON on a 4B model.

No API keys are required anywhere:
  * search  -> DuckDuckGo (or a self-hosted SearxNG if SEARXNG_URL is set)
  * weather -> open-meteo.com
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from bs4 import BeautifulSoup

from . import clients, rag
from .config import settings

log = logging.getLogger("tools")

# --------------------------------------------------------------------------
# schemas advertised to the model
# --------------------------------------------------------------------------
# Measured on Qwen3-4B at temperature 0, five probe questions, tool schemas of
# varying size (total schema characters in brackets):
#
#   5 tools, strong wording  [1470]  weather ok, news MISSED, price ok, docs ok
#   3 tools, strong wording  [ 993]  all five correct
#   3 tools, stronger wording[1056]  news MISSED again
#
# Two lessons are baked into the list below. First, total length dominates:
# making a description more emphatic while making it longer made things worse.
# Second, descriptions must say the model does NOT know the answer -- purely
# descriptive wording ("Weather for a city") gets ignored because the model
# believes it already knows.
#
# fetch_page and remember are implemented below but deliberately NOT advertised:
# they cost ~470 characters, which measurably broke news lookups. Re-adding
# either means re-running the probe above.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": (
                "Search the user's own uploaded documents and saved notes. "
                "Required for any question about their files, notes or "
                "anything they told you earlier -- you cannot see them otherwise."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web. You know nothing about events after your "
                "training, so always call this for news, prices, scores or "
                "anything about today."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the real current weather and 3-day forecast for a city. "
                "You do not know today's weather -- always call this instead "
                "of guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
]


# --------------------------------------------------------------------------
# implementations
# --------------------------------------------------------------------------
async def web_search(query: str, max_results: int = 0) -> str:
    limit = max(1, min(int(max_results or settings.web_search_results), 8))

    if settings.searxng_url:
        results = await _searxng(query, limit)
    else:
        results = await asyncio.to_thread(_duckduckgo, query, limit)

    if not results:
        return f"No results for '{query}'."

    lines = []
    for i, r in enumerate(results, 1):
        body = " ".join((r.get("body") or "").split())[:320]
        lines.append(f"[{i}] {r.get('title', '')}\n{r.get('href', '')}\n{body}")
    return "\n\n".join(lines)


def _duckduckgo(query: str, limit: int) -> list[dict]:
    from ddgs import DDGS

    try:
        with DDGS(timeout=15) as ddgs:
            return list(ddgs.text(query, max_results=limit))
    except Exception as exc:  # rate limit, parser change, no network
        log.warning("duckduckgo failed: %s", exc)
        return []


async def _searxng(query: str, limit: int) -> list[dict]:
    try:
        resp = await clients.web_client().get(
            f"{settings.searxng_url.rstrip('/')}/search",
            params={"q": query, "format": "json", "safesearch": 0},
        )
        resp.raise_for_status()
        return [
            {"title": r.get("title"), "href": r.get("url"), "body": r.get("content")}
            for r in resp.json().get("results", [])[:limit]
        ]
    except Exception as exc:
        log.warning("searxng failed: %s", exc)
        return []


async def fetch_page(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://"
    try:
        resp = await clients.web_client().get(url)
        resp.raise_for_status()
    except Exception as exc:
        return f"Error fetching {url}: {exc}"

    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return f"Error: {url} is {ctype or 'not text'}, cannot read it."

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else url
    text = " ".join(soup.get_text(" ").split())
    if len(text) > settings.web_fetch_max_chars:
        text = text[: settings.web_fetch_max_chars] + " …[truncated]"
    return f"{title}\n{url}\n\n{text}"


_WMO = {
    0: ("açık", "clear sky"), 1: ("çoğunlukla açık", "mainly clear"),
    2: ("parçalı bulutlu", "partly cloudy"), 3: ("kapalı", "overcast"),
    45: ("sisli", "fog"), 48: ("kırağılı sis", "rime fog"),
    51: ("hafif çisenti", "light drizzle"), 53: ("çisenti", "drizzle"),
    55: ("yoğun çisenti", "dense drizzle"),
    61: ("hafif yağmur", "slight rain"), 63: ("yağmur", "rain"),
    65: ("şiddetli yağmur", "heavy rain"),
    66: ("dondurucu yağmur", "freezing rain"), 67: ("şiddetli dondurucu yağmur", "heavy freezing rain"),
    71: ("hafif kar", "slight snow"), 73: ("kar", "snow"), 75: ("yoğun kar", "heavy snow"),
    77: ("kar taneleri", "snow grains"),
    80: ("hafif sağanak", "slight showers"), 81: ("sağanak", "showers"),
    82: ("şiddetli sağanak", "violent showers"),
    85: ("hafif kar sağanağı", "slight snow showers"), 86: ("yoğun kar sağanağı", "heavy snow showers"),
    95: ("gök gürültülü fırtına", "thunderstorm"),
    96: ("dolulu fırtına", "thunderstorm with hail"),
    99: ("şiddetli dolulu fırtına", "thunderstorm with heavy hail"),
}


async def get_weather(location: str) -> str:
    web = clients.web_client()
    try:
        geo = await web.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "tr", "format": "json"},
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
    except Exception as exc:
        return f"Error resolving location '{location}': {exc}"

    if not results:
        return f"Location '{location}' not found."

    place = results[0]
    name = ", ".join(filter(None, [place.get("name"), place.get("admin1"), place.get("country")]))

    try:
        resp = await web.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                           "precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                         "precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Error fetching weather: {exc}"

    cur = data.get("current", {})
    code_tr, code_en = _WMO.get(cur.get("weather_code"), ("bilinmiyor", "unknown"))
    lines = [
        f"Weather for {name} (local time {cur.get('time')}):",
        f"  now: {cur.get('temperature_2m')}°C (feels {cur.get('apparent_temperature')}°C), "
        f"{code_en} / {code_tr}, humidity {cur.get('relative_humidity_2m')}%, "
        f"wind {cur.get('wind_speed_10m')} km/h, precipitation {cur.get('precipitation')} mm",
    ]

    daily = data.get("daily", {})
    for i, day in enumerate(daily.get("time", [])):
        d_tr, d_en = _WMO.get(daily["weather_code"][i], ("bilinmiyor", "unknown"))
        lines.append(
            f"  {day}: {daily['temperature_2m_min'][i]}..{daily['temperature_2m_max'][i]}°C, "
            f"{d_en} / {d_tr}, rain chance {daily['precipitation_probability_max'][i]}%"
        )
    return "\n".join(lines)


async def get_current_time(timezone: str = "Europe/Istanbul") -> str:
    try:
        tz = ZoneInfo(timezone or "Europe/Istanbul")
    except (ZoneInfoNotFoundError, ValueError):
        return f"Unknown timezone '{timezone}'. Use an IANA name like Europe/Istanbul."
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


async def knowledge_search(query: str, top_k: int = 0) -> str:
    limit = max(1, min(int(top_k or settings.rag_top_k), 8))
    floor = settings.rag_tool_min_score

    docs = await rag.search(query, settings.documents_collection, limit, min_score=floor)
    mems = await rag.search(
        query, settings.memories_collection, max(2, limit // 2), min_score=floor
    )

    hits = sorted(docs + mems, key=lambda h: h.score, reverse=True)[:limit]
    if not hits:
        return "Nothing relevant found in the user's documents or memories."

    header = (
        "Excerpts from the user's own files, best match first. Similarity is "
        "0-1; below ~0.80 the match is weak, so ignore an excerpt that does not "
        "actually answer the question instead of forcing it.\n\n"
    )
    return header + "\n\n".join(
        f"[{h.source}] (similarity {h.score:.2f})\n{h.text}" for h in hits
    )


async def remember(text: str) -> str:
    text = text.strip()
    if not text:
        return "Error: nothing to remember."
    await rag.remember(text)
    return f"Saved to long-term memory: {text}"


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
# fetch_page / get_current_time / remember stay callable so re-advertising them
# is a one-line change, but they are not in TOOL_SCHEMAS -- see the note there.
_HANDLERS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "knowledge_search": knowledge_search,
    "remember": remember,
}


async def execute(name: str, raw_arguments: str) -> str:
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'."

    try:
        args = json.loads(raw_arguments) if raw_arguments.strip() else {}
    except json.JSONDecodeError:
        return f"Error: arguments for '{name}' were not valid JSON: {raw_arguments[:200]}"

    if not isinstance(args, dict):
        return f"Error: arguments for '{name}' must be a JSON object."

    try:
        return await asyncio.wait_for(handler(**args), timeout=45)
    except asyncio.TimeoutError:
        return f"Error: tool '{name}' timed out."
    except TypeError as exc:
        return f"Error: bad arguments for '{name}': {exc}"
    except Exception as exc:  # never let a tool kill the turn
        log.exception("tool %s failed", name)
        return f"Error running '{name}': {exc}"
