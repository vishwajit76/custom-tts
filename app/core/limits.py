"""Request body size cap for upload routes (413). Checks Content-Length and counts streamed bytes (chunked bodies)."""
import re

from app.core.config import settings

UPLOAD_PATHS = re.compile(r"^/v1/(speakers/[^/]+/references|voices)/?$")


class BodyLimitMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH") or not UPLOAD_PATHS.match(scope["path"]):
            return await self.app(scope, receive, send)
        limit = settings.upload_max_bytes
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
