import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFException, YFRateLimitError

from .models import ResponseEnvelope


class ProviderError(RuntimeError):
    pass


class YahooProvider:
    def __init__(self, retries: int = 2, cache_seconds: float = 30, timeout: float = 15):
        self.retries, self.cache_seconds, self.timeout = retries, cache_seconds, timeout
        self._cache: dict[tuple[str, ...], tuple[float, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="optionsflow-yahoo")

    def _call(self, key: tuple[str, ...], operation: Callable[[], Any], cache: bool = True) -> Any:
        existing = self._cache.get(key)
        if cache and existing and time.monotonic() - existing[0] < self.cache_seconds:
            return existing[1]
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                value = self._executor.submit(operation).result(timeout=self.timeout)
                if cache:
                    self._cache[key] = (time.monotonic(), value)
                return value
            except (TimeoutError, ConnectionError, OSError, YFRateLimitError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.1 * 2**attempt)
            except YFException as exc:
                raise ProviderError(str(exc)) from exc
        assert last is not None
        raise ProviderError(str(last)) from last

    def ticker(self, symbol: str):
        return yf.Ticker(symbol)

    def info(self, symbol: str, ticker: Any) -> dict[str, Any]:
        return self._call((symbol, "info"), lambda: ticker.get_info())

    def expirations(self, symbol: str, ticker: Any) -> list[str]:
        return list(self._call((symbol, "expirations"), lambda: ticker.options))

    def chain(self, symbol: str, expiration: str, ticker: Any) -> Any:
        return self._call((symbol, "chain", expiration), lambda: ticker.option_chain(expiration))


def clean(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def envelope(data: Any, warnings: list[str]) -> ResponseEnvelope:
    now = datetime.now(UTC).isoformat()
    return {
        "success": True,
        "timestamp": now,
        "data": clean(data),
        "provider": {"name": "Yahoo Finance via yfinance", "as_of": now, "real_time": False},
        "warnings": warnings,
    }
