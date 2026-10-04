import sys

try:
    import resource
except ImportError:  # Windows
    resource = None

import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.config import settings
from app.core.security import require_api_key
from app.services import tts

router = APIRouter()


def _rss_bytes() -> int:
    if resource is None:
        return 0
    try:  # Linux: current RSS
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * resource.getpagesize()
    except OSError:  # macOS: peak RSS, in bytes
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024


def _load() -> dict:
    s = tts.scheduler
    return {
        "streams_active": tts.stats["streams_active"], "max_streams": settings.max_streams,
        "queue_depth": s.depth if s else 0, "workers_busy": s.busy if s else 0, "workers": s.workers if s else 0,
    }


@router.get("/health")
def health():
    ready = tts.engine.ready
    body = {"status": "ok" if ready else "loading", "engine": settings.engines, "model": settings.model_name, **_load()}
    return JSONResponse(body, status_code=200 if ready else 503)


@router.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus text format."""
    lines = [f"tts_{k} {v}" for k, v in {**tts.stats, **_load()}.items()]
    if tts.ttfa_recent:
        for q in (0.5, 0.95, 0.99):
            lines.append(f'tts_ttfa_seconds{{quantile="{q}"}} {np.quantile(tts.ttfa_recent, q):.4f}')
    lines.append(f"process_resident_memory_bytes {_rss_bytes()}")
    return "\n".join(lines) + "\n"


@router.get("/v1/models", dependencies=[Depends(require_api_key)])
def models():
    return {"object": "list", "data": [{"id": settings.model_name, "object": "model", "owned_by": "custom-tts", "engine": settings.engines}]}
