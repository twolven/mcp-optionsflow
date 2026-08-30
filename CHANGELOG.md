# Changelog
## 2.0.0
- Migrated to typed FastMCP with corrected liquidity, contract multipliers, spread signs, width selection, covered-call delta selection, payoff metrics, and model disclosures.
- Restored legacy strategy/delta arguments and CSP, corrected probability-at-breakeven math, non-finite normalization, expiration-day handling, provider retries/cache, schemas, fixtures, and branch-coverage gates.
- Made the fallback `dividendYield` percent convention explicit, including low-yield values.
- Captured the live yfinance dividend-field unit contract, retain genuine yields above 25% with an explicit sensitivity warning, and reject invalid negative/non-finite values.
- Added a non-root Docker/Compose deployment with localhost-bound Streamable HTTP, health checks, and an end-to-end container contract gate.
