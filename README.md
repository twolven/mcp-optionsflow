# OptionsFlow MCP

A typed FastMCP stdio server that analyzes a call credit spread, put credit spread, and covered call from Yahoo Finance option-chain data.

## Tool

`analyze_basic_strategies(symbol, strategy, expiration_date, delta_target=0.3, width_pct=0.05)` preserves the original public tool name, required arguments, defaults, and strategy values: `ccs`, `pcs`, `csp`, and `cc`. Prices and payoff values distinguish per-share amounts from the standard 100-share contract multiplier. Results include breakeven, maximum profit/loss, return on capital, probability evaluated at the actual breakeven, position Greeks, provider/as-of metadata, warnings, and rejection reasons.

```powershell
uv sync --locked
uv run python optionsflow.py
```

The server uses stdio and writes no protocol-unrelated content to stdout. Black-Scholes values are theoretical European-model estimates; dividends and American-style early assignment may make them differ materially from realized outcomes. Yahoo Finance is an unofficial personal-use source and may be delayed, incomplete, rate-limited, or structurally changed. Nothing returned is investment advice or guaranteed real-time data.

Run validation with `uv lock --check`, `uv run ruff check .`, `uv run mypy .`, `uv run pytest`, `uv build`, and `uv run python scripts/verify_wheel.py`. Domain/provider branch coverage is gated at 90%. Set `YFINANCE_LIVE=1` to opt into non-price-asserting live smoke tests.
