"""Vendor STT and streamed-LLM clients for the /demo voice agent. API keys never leave this module's requests.

Endpoints, auth headers, form fields and default model ids were checked against the vendors' docs on 2026-09-29.
"""
import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.core.config import settings

log = logging.getLogger(__name__)
STT_ORDER = ("sarvam", "openai", "elevenlabs")
LLM_ORDER = ("openai", "gemini", "sarvam")
# ponytail: one shared client for TLS keep-alive (saves a handshake per turn); never closed, process-lifetime.
_client = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5))


class ProviderError(Exception):
    """Vendor call failed; str() is safe to show the user (key scrubbed, body truncated)."""


def _key(provider: str) -> str:
    return getattr(settings, f"platform_{provider}_api_key")


def available_providers() -> dict[str, list[str]]:
    return {"stt": [p for p in STT_ORDER if _key(p)], "llm": [p for p in LLM_ORDER if _key(p)]}


def default_provider(kind: str) -> str:
    avail = available_providers()[kind]
    want = getattr(settings, f"demo_{kind}")
    return want if want in avail else (avail[0] if avail else "")


def _scrub(provider: str, text: str) -> str:
    return text.replace(_key(provider) or "\0", "***")


def _brief(provider: str, raw: object) -> str:
    """Vendor error payload -> one short key-scrubbed line for the browser (the full body goes to the server log)."""
    if isinstance(raw, dict):
        raw = raw.get("error") or raw.get("detail") or raw
        if isinstance(raw, dict):
            raw = raw.get("message") or raw
    return _scrub(provider, raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False))[:160]


def _error(provider: str, r: httpx.Response) -> ProviderError:
    log.warning("vendor error", extra={"extra_fields": {"provider": provider, "status": r.status_code, "body": _scrub(provider, r.text)[:2000]}})
    try:
        msg = _brief(provider, r.json())
    except ValueError:
        msg = _brief(provider, r.text)
    return ProviderError(f"{provider} HTTP {r.status_code}: {msg}")


async def transcribe(provider: str, wav: bytes) -> str:
    """WAV (16 kHz mono PCM16) -> transcript text."""
    files = {"file": ("audio.wav", wav, "audio/wav")}
    key = _key(provider)
    if provider == "sarvam":
        # https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe (sync REST: up to 30 s of audio)
        url, headers, field = "https://api.sarvam.ai/speech-to-text", {"api-subscription-key": key}, "transcript"
        data = {"model": settings.demo_sarvam_stt_model, "language_code": "hi-IN"}  # "unknown" = auto-detect
    elif provider == "openai":
        # https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
        url, headers, field = "https://api.openai.com/v1/audio/transcriptions", {"Authorization": f"Bearer {key}"}, "text"
        data = {"model": settings.demo_openai_stt_model}
        if settings.demo_openai_stt_model.startswith("gpt-transcribe"):
            data["languages[]"] = ["hi"]  # repeated form field; this model replaced `language` with a list
        else:
            data["language"] = "hi"
    elif provider == "elevenlabs":
        # https://elevenlabs.io/docs/api-reference/speech-to-text/convert (ISO-639-1 or -3 code; Hindi = hi / hin)
        url, headers, field = "https://api.elevenlabs.io/v1/speech-to-text", {"xi-api-key": key}, "text"
        data = {"model_id": settings.demo_elevenlabs_stt_model, "language_code": "hi", "tag_audio_events": "false"}
    else:
        raise ProviderError(f"unknown STT provider {provider!r}")
    try:
        r = await _client.post(url, headers=headers, files=files, data=data)
    except httpx.HTTPError as e:
        raise ProviderError(f"{provider} STT request failed: {type(e).__name__}") from e
    if r.is_error:
        raise _error(provider, r)
    try:
        return str(r.json()[field]).strip()
    except (ValueError, KeyError) as e:
        log.warning("unexpected STT response", extra={"extra_fields": {"provider": provider, "body": _scrub(provider, r.text)[:2000]}})
        raise ProviderError(f"{provider} STT: unexpected response") from e


def _llm_request(provider: str) -> tuple[str, dict, dict]:
    key = _key(provider)
    if provider == "openai":
        # https://developers.openai.com/api/docs/guides/reasoning : reasoning_effort "none" = lowest latency
        return ("https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {key}"},
                {"model": settings.demo_openai_llm_model, "reasoning_effort": "none"})
    if provider == "gemini":
        # https://ai.google.dev/gemini-api/docs/openai : "minimal" is the lowest thinking level of the Flash-Lite models
        return ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                {"Authorization": f"Bearer {key}"},
                {"model": settings.demo_gemini_llm_model, "reasoning_effort": "minimal"})
    if provider == "sarvam":
        # https://docs.sarvam.ai/api-reference-docs/chat/chat-completions : Bearer or api-subscription-key; null disables reasoning
        return ("https://api.sarvam.ai/v1/chat/completions",
                {"Authorization": f"Bearer {key}", "api-subscription-key": key},
                {"model": settings.demo_sarvam_llm_model, "reasoning_effort": None})
    raise ProviderError(f"unknown LLM provider {provider!r}")


async def chat_stream(provider: str, messages: list[dict]) -> AsyncIterator[str]:
    """Yield assistant text deltas from an OpenAI-format streamed chat completion."""
    url, headers, body = _llm_request(provider)
    # OpenAI's reasoning-era models reject the old `max_tokens`; the others still take it.
    body[("max_completion_tokens" if provider == "openai" else "max_tokens")] = settings.demo_max_tokens
    try:
        async with _client.stream("POST", url, headers=headers, json={**body, "messages": messages, "stream": True}) as r:
            if r.is_error:
                await r.aread()
                raise _error(provider, r)
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                obj = json.loads(data)
                if "error" in obj:
                    log.warning("vendor stream error", extra={"extra_fields": {"provider": provider, "error": _scrub(provider, json.dumps(obj["error"], ensure_ascii=False))[:2000]}})
                    raise ProviderError(f"{provider} LLM: {_brief(provider, obj['error'])}")
                for choice in obj.get("choices") or ():
                    if text := (choice.get("delta") or {}).get("content"):
                        yield text
    except httpx.HTTPError as e:
        raise ProviderError(f"{provider} LLM request failed: {type(e).__name__}") from e
    except ValueError as e:
        raise ProviderError(f"{provider} LLM: malformed stream") from e
