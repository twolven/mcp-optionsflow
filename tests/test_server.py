from types import SimpleNamespace

import pandas as pd
import pytest

from optionsflow import server


def chains():
    def frame(kind):
        strikes = [105, 110] if kind == "call" else [95, 90]
        return pd.DataFrame(
            {
                "strike": strikes,
                "bid": [3, 1],
                "ask": [3.1, 1.1],
                "volume": [100, 100],
                "openInterest": [100, 100],
                "impliedVolatility": [0.2, 0.2],
            }
        )

    return SimpleNamespace(calls=frame("call"), puts=frame("put"))


@pytest.mark.parametrize("strategy", ["ccs", "pcs", "csp", "cc"])
def test_every_legacy_strategy(monkeypatch, strategy):
    monkeypatch.setattr(server.provider, "ticker", lambda symbol: object())
    monkeypatch.setattr(server.provider, "expirations", lambda *args: ["2027-01-15"])
    monkeypatch.setattr(
        server.provider, "info", lambda *args: {"currentPrice": 100, "dividendYield": 0.5}
    )
    monkeypatch.setattr(server.provider, "chain", lambda *args: chains())
    result = server.analyze_basic_strategies(" aapl ", strategy, "2027-01-15", 0.3, 0.05)
    assert result["success"] and result["data"]["strategy"] == strategy.upper()
    assert result["data"]["risk_free_rate"]["instrument"] is None


def test_expiration_day_boundary_and_dividend_percent_normalization():
    from datetime import UTC, date, datetime

    assert (
        server.years_to_expiration(date(2026, 8, 29), datetime(2026, 8, 29, 15, tzinfo=UTC))[0] == 0
    )
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError):
        server.years_to_expiration(date(2026, 8, 29), datetime(2026, 8, 30, tzinfo=UTC))
