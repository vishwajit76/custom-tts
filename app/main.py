import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import demo, health, speakers, speech, voices, ws
from app.core.config import settings
from app.core.limits import BodyLimitMiddleware
from app.core.logging import setup_logging
from app.services import tts

setup_logging(settings.log_level)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not settings.keys:
        log.warning("API_KEYS empty: authentication DISABLED")
    tts.load()
    try:
        from app.services.speaker_registry import get_registry

        if settings.speakers_dir.is_dir():
            get_registry().purge_expired()  # honour per-speaker reference retention
    except Exception:
        log.exception("speaker retention purge failed")
    yield


app = FastAPI(title="Hindi TTS", version="2.0.0", lifespan=lifespan)
for r in (health.router, speech.router, voices.router, voices.capabilities_router, speakers.router, ws.router, demo.router):
    app.include_router(r)


app.add_middleware(BodyLimitMiddleware)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error", extra={"extra_fields": {"path": request.url.path}})
    return JSONResponse({"error": "internal error"}, status_code=500)
