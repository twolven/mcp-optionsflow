import json
import math

import pandas as pd
import pytest

from optionsflow.domain import (
    MULTIPLIER,
    cash_secured_put,
    choose_spread,
    covered_call,
    greeks,
    liquid,
    probability_profit,
)
from optionsflow.provider import clean


def option(strike, bid, ask, delta, kind="call", iv=0.2, volume=100, interest=100):
    model = greeks(100, strike, 30 / 365, 0.04, iv, kind)
    if model:
        model["delta"] = delta
    return {
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "volume": volume,
        "openInterest": interest,
        "greeks": model,
    }


@pytest.mark.parametrize("kind,delta_sign,rho_sign", [("call", 1, 1), ("put", -1, -1)])
def test_black_scholes_regression_and_units(kind, delta_sign, rho_sign):
    model = greeks(100, 100, 1, 0.04, 0.2, kind)
    assert model and math.copysign(1, model["delta"]) == delta_sign and model["gamma"] > 0
    assert (
        math.copysign(1, model["rho"]) == rho_sign
        and abs(model["vega"] - 0.381) < 0.01
        and model["theta"] < 0
    )
    if kind == "call":
        assert model["theta"] == pytest.approx(-0.016132936149781025, rel=1e-12)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_model_inputs(value):
    assert greeks(100, 100, 1, 0.04, value, "call") is None


def test_liquidity_all_rejections():
    assert liquid(pd.Series({"bid": 1, "ask": 1.1, "volume": 9, "openInterest": 9})) == (True, None)
    for row in (
        {"bid": 0, "ask": 1, "volume": 9, "openInterest": 9},
        {"bid": 2, "ask": 1, "volume": 9, "openInterest": 9},
        {"bid": 1, "ask": 2, "volume": 9, "openInterest": 9},
        {"bid": 1, "ask": 1.1, "volume": 0, "openInterest": 9},
    ):
        assert not liquid(pd.Series(row))[0]


def test_call_spread_payoff_probability_and_width():
    frame = pd.DataFrame(
        [option(105, 3, 3.1, 0.4), option(106, 2.5, 2.6, 0.35), option(110, 1, 1.1, 0.2)]
    )
    result = choose_spread(frame, 100, "call", 0.05, 30 / 365, 0.04, 0)
    assert result["strikes"]["long_strike"] == 110
    assert result["metrics"]["max_profit"] == pytest.approx(1.9 * MULTIPLIER)
    assert result["metrics"]["max_loss"] == pytest.approx(3.1 * MULTIPLIER)
    assert result["metrics"]["breakeven"] == pytest.approx(106.9)
    assert result["metrics"]["risk_neutral_probability_at_breakeven"] == pytest.approx(
        probability_profit(100, 106.9, 30 / 365, 0.04, 0.2, "call", 0)
    )
    assert result["greeks"]["net_delta"] < 0


def test_put_spread_and_single_leg_strategies():
    puts = pd.DataFrame(
        [
            option(95, 3, 3.1, -0.4, "put"),
            option(94, 2.4, 2.5, -0.35, "put"),
            option(90, 1, 1.1, -0.2, "put"),
        ]
    )
    spread = choose_spread(puts, 100, "put", 0.05, 30 / 365, 0.04, 0)
    assert spread["metrics"]["breakeven"] < 95 and spread["metrics"]["max_loss"] > 0
    csp = cash_secured_put(puts, 100, 0.35)
    assert csp["strike"] == 94 and csp["cash_collateral"] == 9400
    call = covered_call(
        pd.DataFrame([option(105, 3, 3.1, 0.4), option(110, 1, 1.1, 0.3)]), 100, 0.3
    )
    assert (
        call["strike"] == 110
        and call["position_greeks"]["net_delta"] == pytest.approx(70)
        and "net_rho" in call["position_greeks"]
    )


def test_sparse_and_no_credit_failures():
    with pytest.raises(ValueError):
        choose_spread(pd.DataFrame([option(105, 1, 1.1, 0.3)]), 100, "call", 0.05, 0.1, 0.04, 0)
    with pytest.raises(ValueError):
        choose_spread(
            pd.DataFrame([option(105, 1, 1.1, 0.3), option(110, 2, 2.1, 0.2)]),
            100,
            "call",
            0.05,
            0.1,
            0.04,
            0,
        )
    assert json.dumps(clean({"x": float("nan")}), allow_nan=False)
