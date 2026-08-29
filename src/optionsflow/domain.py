import math
from statistics import NormalDist

import pandas as pd

N = NormalDist()
MULTIPLIER = 100


def greeks(
    spot: float,
    strike: float,
    years: float,
    rate: float,
    vol: float,
    kind: str,
    dividend: float = 0,
) -> dict[str, float] | None:
    if min(spot, strike, years, vol) <= 0:
        return None
    root = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate - dividend + vol**2 / 2) * years) / (vol * root)
    d2 = d1 - vol * root
    pdf = N.pdf(d1)
    call = kind == "call"
    delta = math.exp(-dividend * years) * (N.cdf(d1) if call else N.cdf(d1) - 1)
    gamma = math.exp(-dividend * years) * pdf / (spot * vol * root)
    theta1 = -(spot * vol * math.exp(-dividend * years) * pdf) / (2 * root)
    theta = (
        (
            theta1
            - rate * strike * math.exp(-rate * years) * N.cdf(d2)
            + dividend * spot * math.exp(-dividend * years) * N.cdf(d1)
        )
        if call
        else (
            theta1
            + rate * strike * math.exp(-rate * years) * N.cdf(-d2)
            - dividend * spot * math.exp(-dividend * years) * N.cdf(-d1)
        )
    )
    rho = strike * years * math.exp(-rate * years) * (N.cdf(d2) if call else -N.cdf(-d2))
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365,
        "vega": spot * math.exp(-dividend * years) * pdf * root / 100,
        "rho": rho / 100,
        "prob_itm": N.cdf(d2) if call else N.cdf(-d2),
    }


def liquid(row: pd.Series) -> tuple[bool, str | None]:
    bid, ask = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
    if bid <= 0 or ask <= 0:
        return False, "Invalid bid/ask prices"
    if ask < bid:
        return False, "Ask price lower than bid price"
    spread = (ask - bid) / ask
    limit = 0.10 if min(bid, ask) > 10 else 0.15 if min(bid, ask) < 5 else 0.20
    if spread > limit:
        return False, f"Spread ({spread:.1%}) exceeds liquidity limit ({limit:.1%})"
    if float(row.get("volume", 0) or 0) < 5 or float(row.get("openInterest", 0) or 0) < 5:
        return False, "Insufficient volume or open interest"
    return True, None


def choose_spread(chain: pd.DataFrame, spot: float, kind: str, width_pct: float) -> dict:
    calls = kind == "call"
    candidates = chain[chain.strike > spot] if calls else chain[chain.strike < spot]
    candidates = candidates.sort_values("strike", ascending=calls)
    valid = []
    rejected = []
    for _, row in candidates.iterrows():
        ok, reason = liquid(row)
        if ok:
            valid.append(row)
        else:
            rejected.append({"strike": float(row.strike), "reason": reason})
    if len(valid) < 2:
        raise ValueError("Not enough liquid strikes for spread")
    short = valid[0]
    target = float(short.strike) * (1 + width_pct if calls else 1 - width_pct)
    longs = [
        r for r in valid[1:] if (r.strike > short.strike if calls else r.strike < short.strike)
    ]
    if not longs:
        raise ValueError("No protective strike available")
    long = min(longs, key=lambda r: abs(float(r.strike) - target))
    credit = float(short.bid) - float(long.ask)
    if credit <= 0:
        raise ValueError("No positive executable credit")
    width = abs(float(long.strike) - float(short.strike))
    max_profit = credit * MULTIPLIER
    max_loss = (width - credit) * MULTIPLIER
    break_even = float(short.strike) + (credit if calls else -credit)
    sg = short.greeks
    lg = long.greeks
    return {
        "strikes": {"short_strike": float(short.strike), "long_strike": float(long.strike)},
        "metrics": {
            "credit_per_share": credit,
            "contract_multiplier": MULTIPLIER,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": break_even,
            "return_on_capital": None if max_loss <= 0 else max_profit / max_loss,
            "risk_neutral_probability_at_breakeven": 1 - float(sg["prob_itm"]),
        },
        "greeks": {
            f"net_{k}": MULTIPLIER * (-float(sg[k]) + float(lg[k]))
            for k in ("delta", "gamma", "theta", "vega", "rho")
        },
        "rejected": rejected,
    }


def covered_call(chain: pd.DataFrame, spot: float) -> dict:
    rows = chain[chain.strike > spot].copy()
    rows = rows[rows.apply(lambda r: liquid(r)[0], axis=1)]
    if rows.empty:
        raise ValueError("No liquid covered-call candidate")
    row = min((r for _, r in rows.iterrows()), key=lambda r: abs(float(r.greeks["delta"]) - 0.30))
    premium = float(row.bid)
    return {
        "strike": float(row.strike),
        "premium_per_share": premium,
        "contract_multiplier": MULTIPLIER,
        "premium": premium * MULTIPLIER,
        "breakeven": spot - premium,
        "max_profit": ((float(row.strike) - spot) + premium) * MULTIPLIER,
        "position_greeks": {
            "net_delta": MULTIPLIER * (1 - float(row.greeks["delta"])),
            "net_gamma": -MULTIPLIER * float(row.greeks["gamma"]),
            "net_theta": -MULTIPLIER * float(row.greeks["theta"]),
            "net_vega": -MULTIPLIER * float(row.greeks["vega"]),
        },
    }
