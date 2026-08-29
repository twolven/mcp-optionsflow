import math
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .domain import cash_secured_put, choose_spread, covered_call, greeks
from .models import DeltaTarget, Expiration, ResponseEnvelope, Strategy, Symbol, WidthPct
from .provider import ProviderError, YahooProvider, envelope

mcp = FastMCP("optionsflow")
provider = YahooProvider()


def years_to_expiration(expiration: date, now: datetime | None = None) -> tuple[int, float]:
    current = now or datetime.now(UTC)
    close = datetime.combine(expiration, time(16), ZoneInfo("America/New_York"))
    remaining = (close.astimezone(UTC) - current.astimezone(UTC)).total_seconds()
    if remaining <= 0:
        raise ToolError("Expiration has already passed its 4:00 PM America/New_York close")
    return max(
        (expiration - current.astimezone(ZoneInfo("America/New_York")).date()).days, 0
    ), remaining / (365 * 86400)


@mcp.tool
def analyze_basic_strategies(
    symbol: Symbol,
    strategy: Strategy,
    expiration_date: Expiration,
    delta_target: DeltaTarget = 0.3,
    width_pct: WidthPct = 0.05,
) -> ResponseEnvelope:
    """Analyze one legacy options strategy using executable quotes and European Black-Scholes estimates."""
    normalized = symbol.strip().upper()
    ticker = provider.ticker(normalized)
    try:
        expirations = provider.expirations(normalized, ticker)
        if expiration_date not in expirations:
            raise ToolError(f"Expiration {expiration_date} is unavailable")
        expiration = date.fromisoformat(expiration_date)
        dte, years = years_to_expiration(expiration)
        info = provider.info(normalized, ticker) or {}
        spot = info.get("currentPrice") or info.get("regularMarketPrice")
        if not spot or not math.isfinite(float(spot)):
            raise ToolError("No valid underlying quote")
        dividend = float(info.get("trailingAnnualDividendYield") or info.get("dividendYield") or 0)
        if dividend > 0.25:
            dividend /= 100
        if not math.isfinite(dividend) or not 0 <= dividend <= 0.25:
            dividend = 0
        chain = provider.chain(normalized, expiration_date, ticker)
        rate = 0.04

        def prepare(frame, kind):
            output = frame.copy()
            output["greeks"] = [
                greeks(
                    float(spot),
                    float(row.strike),
                    years,
                    rate,
                    float(row.impliedVolatility),
                    kind,
                    dividend,
                )
                for row in output.itertuples()
            ]
            return output[output.greeks.notna()]

        calls, puts = prepare(chain.calls, "call"), prepare(chain.puts, "put")
        if strategy == "ccs":
            analysis = choose_spread(calls, float(spot), "call", width_pct, years, rate, dividend)
        elif strategy == "pcs":
            analysis = choose_spread(puts, float(spot), "put", width_pct, years, rate, dividend)
        elif strategy == "csp":
            analysis = cash_secured_put(puts, float(spot), delta_target)
        else:
            analysis = covered_call(calls, float(spot), delta_target)
        data = {
            "symbol": normalized,
            "strategy": strategy.upper(),
            "current_price": float(spot),
            "underlying_price": float(spot),
            "expiration": expiration_date,
            "expiration_date": expiration_date,
            "days_to_expiration": dte,
            "delta_target": delta_target,
            "width_pct": width_pct,
            "risk_free_rate": {"value": rate, "source": "configured fallback", "instrument": None},
            "analysis": analysis,
        }
        return envelope(
            data,
            [
                "Black-Scholes values are theoretical European-model estimates.",
                "Dividends and American-style early assignment can materially change realized outcomes.",
                "Yahoo Finance data may be delayed, incomplete, or rate-limited; this is not investment advice.",
            ],
        )
    except ToolError:
        raise
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    except (ProviderError, TimeoutError, ConnectionError, OSError) as exc:
        raise ToolError(f"Yahoo Finance request failed after bounded retries: {exc}") from exc


def main():
    mcp.run(transport="stdio", show_banner=False)
