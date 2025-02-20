# OptionsFlow MCP Server

A Model Context Protocol (MCP) server providing advanced options analysis and strategy evaluation through Yahoo Finance. Enables LLMs to analyze options chains, calculate Greeks, and evaluate basic options strategies with comprehensive risk metrics.

## Features

### Options Analysis
- Complete options chain data processing
- Greeks calculation (delta, gamma, theta, vega, rho)
- Implied volatility analysis
- Probability calculations
- Risk/reward metrics

### Strategy Analysis
- Credit Call Spreads (CCS)
- Put Credit Spreads (PCS)
- Cash Secured Puts (CSP)
- Covered Calls (CC)
- Position Greeks evaluation
- Liquidity analysis
- Risk metrics calculation

### Risk Management
- Bid-ask spread analysis
- Volume and open interest validation
- Position sizing recommendations
- Maximum loss calculations
- Probability of profit estimates

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Clone the repository
git clone https://github.com/twolven/mcp-optionsflow.git
cd mcp-optionsflow
```

## Usage

Add to your Claude configuration:
In your `claude-desktop-config.json`, add the following to the `mcpServers` section:

```json
{
    "mcpServers": {
        "optionsflow": {
            "command": "python",
            "args": ["path/to/optionsflow.py"]
        }
    }
}
```

Replace "path/to/optionsflow.py" with the full path to where you saved the optionsflow.py file.

## Available Tools

1. `analyze_basic_strategies`
```python
{
    "symbol": str,                    # Required: Stock symbol
    "strategy": str,                  # Required: "ccs", "pcs", "csp", or "cc"
    "expiration_date": str,          # Required: "YYYY-MM-DD"
    "delta_target": float,           # Optional: Target delta for CSP/CC (default: 0.3)
    "width_pct": float              # Optional: Width for spreads (default: 0.05)
}
```

### Strategy Analysis Response Format

```python
{
    "symbol": str,
    "strategy": str,
    "current_price": float,
    "expiration": str,
    "days_to_expiration": int,
    "analysis": {
        # Credit Call Spread / Put Credit Spread
        "strikes": {
            "short_strike": float,
            "long_strike": float
        },
        "metrics": {
            "credit": float,
            "max_loss": float,
            "max_profit": float,
            "probability_of_profit": float,
            "risk_reward_ratio": float
        },
        "greeks": {
            "net_delta": float,
            "net_theta": float,
            "net_gamma": float
        }
        
        # Cash Secured Put
        "strike": float,
        "metrics": {
            "premium": float,
            "max_loss": float,
            "assigned_cost_basis": float,
            "return_if_otm": float,
            "downside_protection": float
        },
        "greeks": {
            "delta": float,
            "theta": float,
            "gamma": float
        }
        
        # Covered Call
        "strike": float,
        "metrics": {
            "premium": float,
            "max_profit": float,
            "max_profit_percent": float,
            "upside_cap": float,
            "premium_yield": float
        },
        "greeks": {
            "position_delta": float,
            "theta": float,
            "gamma": float
        }
    }
}
```

## Requirements

- Python 3.12+
- mcp
- yfinance
- pandas
- numpy
- scipy

## Limitations

- Data sourced from Yahoo Finance with potential delays
- Options data availability depends on market hours
- Rate limits based on Yahoo Finance API restrictions
- Greeks calculations are theoretical and based on Black-Scholes model
- Early assignment risk not factored into probability calculations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Todd Wolven - (https://github.com/twolven)

## Acknowledgments

- Built with the Model Context Protocol (MCP) by Anthropic
- Data provided by [Yahoo Finance](https://finance.yahoo.com/)
- Developed for use with Anthropic's Claude
