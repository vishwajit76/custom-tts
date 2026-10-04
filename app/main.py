import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import demo, health, lab, speakers, speech, voices, ws
from app.core.config import settings
from app.core.limits import BodyLimitMiddleware
from app.core.logging import setup_logging
from app.services import tts

setup_logging(settings.log_level)
log = logging.getLogger("app")


def _sweep_retention() -> None:
    from app.services.speaker_registry import get_registry

    if settings.speakers_dir.is_dir():
        get_registry().purge_expired()  # per-speaker reference retention + interrupted consent revocations


async def _retention_loop() -> None:
    """Retention was only enforced at startup and when a speaker was read, so an idle speaker kept expired raw audio for
    as long as the process ran. Sweep periodically (off the event loop: it fsyncs and overwrites files)."""
    while True:
        await asyncio.sleep(settings.retention_sweep_minutes * 60)
        try:
            await asyncio.to_thread(_sweep_retention)
        except Exception:
            log.exception("speaker retention sweep failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not settings.keys:
        log.warning("API_KEYS empty: authentication DISABLED")
    tts.load()
    try:
        _sweep_retention()
    except Exception:
        log.exception("speaker retention purge failed")
    sweeper = asyncio.create_task(_retention_loop()) if settings.retention_sweep_minutes > 0 else None
    try:
        yield
    finally:
        if sweeper:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper


app = FastAPI(title="Hindi TTS", version="2.0.0", lifespan=lifespan)
for r in (health.router, speech.router, voices.router, voices.capabilities_router, speakers.router, ws.router, demo.router, lab.router):
    app.include_router(r)


app.add_middleware(BodyLimitMiddleware)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error", extra={"extra_fields": {"path": request.url.path}})
    return JSONResponse({"error": "internal error"}, status_code=500)
