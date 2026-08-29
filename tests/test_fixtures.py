import json
from pathlib import Path

import pandas as pd

from optionsflow.domain import greeks


def test_captured_equity_hyphen_index_and_chain_shapes():
    fixture = json.loads((Path(__file__).parent / "fixtures/yfinance_options.json").read_text())
    assert {fixture[key]["symbol"] for key in ("equity", "hyphenated", "index")} == {
        "AAPL",
        "BRK-B",
        "^GSPC",
    }
    for kind in ("calls", "puts"):
        frame = pd.DataFrame(fixture[kind])
        assert not frame.empty
        assert greeks(
            fixture["equity"]["currentPrice"],
            frame.strike.iloc[0],
            0.25,
            0.04,
            frame.impliedVolatility.iloc[0],
            "call" if kind == "calls" else "put",
        )
