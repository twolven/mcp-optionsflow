# OptionsFlow MCP

A typed FastMCP server that analyzes option strategies from Yahoo Finance data. It supports local stdio and containerized Streamable HTTP transports.

## Tool

`analyze_basic_strategies(symbol, strategy, expiration_date, delta_target=0.3, width_pct=0.05)` preserves the original public tool name, required arguments, defaults, and strategy values: `ccs`, `pcs`, `csp`, and `cc`. Prices and payoff values distinguish per-share amounts from the standard 100-share contract multiplier. Results include breakeven, maximum profit/loss, return on capital, probability evaluated at the actual breakeven, position Greeks, provider/as-of metadata, warnings, and rejection reasons.

```powershell
uv sync --locked
uv run python optionsflow.py
```

The server uses stdio and writes no protocol-unrelated content to stdout. Black-Scholes values are theoretical European-model estimates; dividends and American-style early assignment may make them differ materially from realized outcomes. Yahoo Finance is an unofficial personal-use source and may be delayed, incomplete, rate-limited, or structurally changed. Nothing returned is investment advice or guaranteed real-time data.

## Docker / Streamable HTTP

The container runs as an unprivileged user, installs the locked production dependencies, and serves MCP at `http://127.0.0.1:8000/mcp`. Start it with:

```powershell
docker compose up --build -d
Invoke-RestMethod http://127.0.0.1:8000/health
```

Connect a Streamable HTTP-capable MCP client to `http://127.0.0.1:8000/mcp`. To avoid a port collision when running multiple servers, set `MCP_HOST_PORT` before starting Compose, for example `$env:MCP_HOST_PORT=8001`. Stop and remove the container with `docker compose down`.

The Compose mapping intentionally binds to localhost. The endpoint has no authentication or TLS and must not be exposed to an untrusted network without a properly configured reverse proxy and access control. Running `uv run python optionsflow.py` remains the stdio-compatible default outside Docker.

Run validation with `uv lock --check`, `uv run ruff check .`, `uv run mypy .`, `uv run pytest`, `uv build`, and `uv run python scripts/verify_wheel.py`. CI also builds the container and performs health plus MCP tool-discovery checks over Streamable HTTP. Domain/provider branch coverage is gated at 90%. Set `YFINANCE_LIVE=1` to opt into non-price-asserting live smoke tests.
