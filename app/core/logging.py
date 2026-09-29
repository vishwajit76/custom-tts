import json
import logging
import re
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        data.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


_API_KEY = re.compile(r"(api_key=)[^&\s\"]+")


class RedactApiKey(logging.Filter):
    """uvicorn.access logs the request URL; keep ?api_key=... out of it. Args are edited in place (uvicorn's formatter reads them)."""

    def filter(self, record: logging.LogRecord) -> bool:
        redact = lambda v: _API_KEY.sub(r"\1***", v) if isinstance(v, str) else v  # noqa: E731
        record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(map(redact, record.args))
        return True


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for name in ("uvicorn.access", "uvicorn.error"):  # HTTP requests / WebSocket handshakes log their URL to these
        logging.getLogger(name).addFilter(RedactApiKey())
