from datetime import UTC, datetime

import pandas as pd
import pytest

from optionsflow.provider import ProviderError, YahooProvider, clean, envelope


def test_retry_cache_and_terminal_failure(monkeypatch):
    provider = YahooProvider(retries=2, cache_seconds=60)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("slow")
        return {"ok": True}

    monkeypatch.setattr("optionsflow.provider.time.sleep", lambda _: None)
    assert provider._call(("x",), flaky) == {"ok": True}
    assert len(calls) == 3
    assert provider._call(("x",), lambda: None) == {"ok": True}
    with pytest.raises(ProviderError):
        provider._call(("y",), lambda: (_ for _ in ()).throw(ConnectionError("down")))
    timeout_provider = YahooProvider(retries=0, timeout=0.001)
    with pytest.raises(ProviderError):
        timeout_provider._call(("slow",), lambda: __import__("threading").Event().wait(0.05))


def test_provider_accessors_and_clean():
    class T:
        def __init__(self):
            self.options = ["2027-01-01"]

        def get_info(self):
            return {"x": 1}

        def option_chain(self, expiration):
            return expiration

    p = YahooProvider(retries=0)
    t = T()
    assert p.info("X", t) == {"x": 1}
    assert p.expirations("X", t) == ["2027-01-01"]
    assert p.chain("X", "2027-01-01", t) == "2027-01-01"
    cleaned = clean(
        {
            "time": datetime(2026, 1, 1, tzinfo=UTC),
            "stamp": pd.Timestamp("2026-01-01"),
            "items": [float("inf")],
            "num": pd.Series([1]).iloc[0],
        }
    )
    assert cleaned["items"] == [None] and cleaned["num"] == 1
    assert envelope({"x": 1}, [])["provider"]["real_time"] is False
