import math
from datetime import UTC, date, datetime

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .domain import choose_spread, covered_call, greeks
from .models import AnalyzeInput
from .provider import YahooProvider, envelope

mcp = FastMCP("optionsflow")
provider = YahooProvider()


@mcp.tool
def analyze_basic_strategies(
    symbol: str, expiration_date: str | None = None, width_pct: float = 0.05
) -> dict:
    """Analyze call/put credit spreads and a covered call using theoretical European Greeks."""
    args = AnalyzeInput(symbol=symbol, expiration_date=expiration_date, width_pct=width_pct)
    ticker = provider.ticker(args.symbol)
    try:
        expirations = list(ticker.options or [])
        if not expirations:
            raise ToolError(f"No options data available for {args.symbol}")
        expiration = args.expiration_date or expirations[0]
        if expiration not in expirations:
            raise ToolError(f"Expiration {expiration} is unavailable")
        exp = date.fromisoformat(expiration)
        today = datetime.now(UTC).date()
        dte = (exp - today).days
        if dte < 1:
            raise ToolError("Expiration must be at least one day away")
        info = ticker.info or {}
        spot = info.get("currentPrice") or info.get("regularMarketPrice")
        if not spot or not math.isfinite(float(spot)):
            raise ToolError("No valid underlying quote")
        chain = ticker.option_chain(expiration)
        rate = 0.04
        dividend = float(info.get("dividendYield") or 0)
        years = dte / 365

        def prepare(frame, kind):
            out = frame.copy()
            out["greeks"] = [
                greeks(
                    float(spot),
                    float(r.strike),
                    years,
                    rate,
                    float(r.impliedVolatility),
                    kind,
                    dividend,
                )
                for r in out.itertuples()
            ]
            return out[out.greeks.notna()]

        calls, puts = prepare(chain.calls, "call"), prepare(chain.puts, "put")
        strategies = {}
        errors = {}
        for name, fn in {
            "credit_call_spread": lambda: choose_spread(calls, float(spot), "call", args.width_pct),
            "put_credit_spread": lambda: choose_spread(puts, float(spot), "put", args.width_pct),
            "covered_call": lambda: covered_call(calls, float(spot)),
        }.items():
            try:
                strategies[name] = fn()
            except ValueError as exc:
                errors[name] = str(exc)
        if not strategies:
            raise ToolError("No valid strategies: " + "; ".join(errors.values()))
        return envelope(
            {
                "symbol": args.symbol,
                "underlying_price": float(spot),
                "expiration_date": expiration,
                "days_to_expiration": dte,
                "risk_free_rate": {
                    "value": rate,
                    "source": "configured fallback",
                    "instrument": None,
                },
                "width_pct": args.width_pct,
                "strategies": strategies,
                "rejections": errors,
            },
            [
                "Black-Scholes values are theoretical European-model estimates.",
                "Dividends and American-style early assignment can materially change realized outcomes.",
                "Yahoo Finance data may be delayed, incomplete, or rate-limited; this is not investment advice.",
            ],
        )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Yahoo Finance request failed: {exc}") from exc


def main():
    mcp.run(transport="stdio", show_banner=False)
