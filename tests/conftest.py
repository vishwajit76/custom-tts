import pytest

from app.core import security


@pytest.fixture(autouse=True)
def _fresh_rate_limiter(monkeypatch):
    # the limiter is process-global: without a reset, a long suite exceeds rate_limit_per_minute and later API tests get 429
    monkeypatch.setattr(security, "_hits", type(security._hits)(security._hits.default_factory))
