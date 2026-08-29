import time
from datetime import UTC, datetime

import yfinance as yf


class YahooProvider:
    def __init__(self, retries=2):
        self.retries = retries

    def ticker(self, symbol):
        return yf.Ticker(symbol)

    def retry(self, fn):
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except Exception:
                if attempt == self.retries:
                    raise
                time.sleep(0.2 * 2**attempt)


def envelope(data, warnings):
    now = datetime.now(UTC).isoformat()
    return {
        "success": True,
        "timestamp": now,
        "data": data,
        "provider": {"name": "Yahoo Finance via yfinance", "as_of": now, "real_time": False},
        "warnings": warnings,
    }
