import os

import pytest
import yfinance as yf

pytestmark = pytest.mark.skipif(os.getenv("YFINANCE_LIVE") != "1", reason="set YFINANCE_LIVE=1")


def test_live_representative_option_contract():
    ticker = yf.Ticker("AAPL")
    assert ticker.get_info()
    expirations = ticker.options
    if expirations:
        chain = ticker.option_chain(expirations[0])
        assert hasattr(chain, "calls") and hasattr(chain, "puts")
