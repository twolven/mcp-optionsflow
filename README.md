# OptionsFlow MCP

A typed FastMCP stdio server that analyzes a call credit spread, put credit spread, and covered call from Yahoo Finance option-chain data.

## Tool

`analyze_basic_strategies(symbol, expiration_date=None, width_pct=0.05)` preserves the original public tool name and inputs. Prices and payoff values distinguish per-share amounts from the standard 100-share contract multiplier. Results include breakeven, maximum profit/loss, return on capital, risk-neutral probability at breakeven, position Greeks, provider/as-of metadata, warnings, and rejection reasons.

```powershell
uv sync --locked
uv run python optionsflow.py
```

The server uses stdio and writes no protocol-unrelated content to stdout. Black-Scholes values are theoretical European-model estimates; dividends and American-style early assignment may make them differ materially from realized outcomes. Yahoo Finance is an unofficial personal-use source and may be delayed, incomplete, rate-limited, or structurally changed. Nothing returned is investment advice or guaranteed real-time data.

Run validation with `uv run ruff check .`, `uv run pytest`, and `uv build`.
