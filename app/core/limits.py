"""Request body size cap (413) for upload routes and the JSON speech routes. Checks Content-Length and counts streamed
bytes (chunked bodies). Without it a single POST of any size was buffered in memory before validation ran."""
import re

from app.core.config import settings

UPLOAD_PATHS = re.compile(r"^/v1/(speakers/[^/]+/references|voices)/?$")
SPEECH_PATHS = re.compile(r"^/v1/audio/speech(/stream)?/?$")


def body_limit(path: str) -> int | None:
    """Max body bytes for this path, None = not capped here. The JSON speech routes carry text (max_input_chars, at most 6 bytes
    per char once JSON-escaped) and, only on a cloning engine, a base64 reference clip (4/3 of upload_max_bytes)."""
    if UPLOAD_PATHS.match(path):
        return settings.upload_max_bytes
    if SPEECH_PATHS.match(path):
        from app.services import tts  # late: tts builds the engine at import

        text = settings.max_input_chars * 6 + 65536
        return text + settings.upload_max_bytes * 4 // 3 if getattr(tts.engine, "supports_cloning", False) else text
    return None


class BodyLimitMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        limit = body_limit(scope["path"]) if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH") else None
        if limit is None:
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            return await self._reject(send)
        seen = 0
        over = False

        async def counted():
            nonlocal seen, over
            if over:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > limit:
                    over = True
                    return {"type": "http.disconnect"}
            return msg

        started = False

        async def guarded_send(msg):
            nonlocal started
            if not over:
                return await send(msg)
            if not started:  # the app is failing on the aborted body: answer 413 instead
                started = True
                await self._reject(send)

        await self.app(scope, counted, guarded_send)

    @staticmethod
    async def _reject(send):
        body = b'{"error":"request body too large"}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
