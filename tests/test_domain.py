import pandas as pd

from optionsflow.domain import MULTIPLIER, choose_spread, covered_call, greeks, liquid


def row(strike, bid, ask, delta):
    g = greeks(100, strike, 30 / 365, 0.04, 0.2, "call")
    g["delta"] = delta
    return {
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "volume": 100,
        "openInterest": 100,
        "greeks": g,
    }


def test_greek_signs():
    c = greeks(100, 100, 1, 0.04, 0.2, "call")
    p = greeks(100, 100, 1, 0.04, 0.2, "put")
    assert c["delta"] > 0 > p["delta"] and c["gamma"] > 0 and c["theta"] < 0


def test_liquidity_tuple():
    assert liquid(pd.Series({"bid": 1, "ask": 1.1, "volume": 9, "openInterest": 9})) == (True, None)
    assert not liquid(pd.Series({"bid": 0, "ask": 1, "volume": 9, "openInterest": 9}))[0]


def test_spread_payoff_and_width():
    frame = pd.DataFrame([row(105, 3, 3.1, 0.4), row(106, 2.5, 2.6, 0.35), row(110, 1, 1.1, 0.2)])
    result = choose_spread(frame, 100, "call", 0.05)
    assert result["strikes"]["long_strike"] == 110
    assert (
        result["metrics"]["max_profit"] == 1.9 * MULTIPLIER
        and result["metrics"]["breakeven"] == 106.9
    )
    assert result["greeks"]["net_delta"] < 0


def test_covered_call_position():
    result = covered_call(pd.DataFrame([row(105, 3, 3.1, 0.4), row(110, 1, 1.1, 0.3)]), 100)
    assert result["strike"] == 110 and result["position_greeks"]["net_delta"] == 70
