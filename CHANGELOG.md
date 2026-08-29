# Changelog
## 2.0.0
- Migrated to typed FastMCP with corrected liquidity, contract multipliers, spread signs, width selection, covered-call delta selection, payoff metrics, and model disclosures.
- Restored legacy strategy/delta arguments and CSP, corrected probability-at-breakeven math, non-finite normalization, expiration-day handling, provider retries/cache, schemas, fixtures, and branch-coverage gates.
- Made the fallback `dividendYield` percent convention explicit, including low-yield values.
