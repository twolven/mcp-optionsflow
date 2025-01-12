#!/usr/bin/env python3

import logging
import asyncio
import yfinance as yf
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server
import json
import traceback
import re
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.interpolate import griddata
import datetime
from functools import wraps
import time
from typing import List, Dict, Optional, Any, Tuple

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failing functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {str(e)}\n{traceback.format_exc()}")
            raise last_error
        return wrapper
    return decorator


def get_risk_free_rate() -> float:
    """Simple way to get a recent risk-free rate (using 1-year treasury yield).
       Consider more robust methods for production."""
    try:
        tbill = yf.Ticker("^IRX")  # Ticker for 13-week T-Bill
        hist = tbill.history(period="5d")
        if not hist.empty:
            return hist['Close'].iloc[-1] / 100.0  # Convert percentage to decimal
        logger.warning("Could not fetch T-Bill rate, using default")
        return 0.04  # Default if data fetch fails
    except Exception as e:
        logger.warning(f"Error fetching risk-free rate: {e}")
        return 0.04  # Default rate if there's an error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("options_analytics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("options-analytics")

class OptionsError(Exception):
    pass

class ValidationError(OptionsError):
    pass

class APIError(OptionsError):
    pass

class GreeksCalculator:
    def __init__(self):
        self.MIN_SIGMA = 0.0001  # Minimum volatility to prevent division by zero
        self.MIN_TIME = 1/365    # Minimum time (1 day) to prevent time issues
        
    @staticmethod
    def calculate_d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
        """Calculate d1 component of Black-Scholes with dividend yield"""
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return float('nan')
            return (np.log(S/K) + (r - q + (sigma**2)/2)*T) / (sigma*np.sqrt(T))
        except Exception as e:
            logger.error(f"Error in d1 calculation: {e}")
            return float('nan')

    @staticmethod
    def calculate_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
        """Calculate d2 component of Black-Scholes"""
        try:
            if T <= 0 or sigma <= 0:
                return float('nan')
            d1 = GreeksCalculator.calculate_d1(S, K, T, r, sigma, q)
            return d1 - sigma*np.sqrt(T)
        except Exception as e:
            logger.error(f"Error in d2 calculation: {e}")
            return float('nan')

    def calculate_greeks(self, S: float, K: float, T: float, r: float,
                        sigma: float, q: float, option_type: str) -> Dict[str, float]:
        """
        Calculate option Greeks using Black-Scholes model
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate (as decimal)
        sigma: Implied volatility (as decimal)
        q: Dividend yield (as decimal)
        option_type: 'CALL' or 'PUT'
        """
        try:
            # Input validation with detailed logging
            logger.debug(f"Inputs: S={S}, K={K}, T={T}, r={r}, sigma={sigma}, q={q}, type={option_type}")
            
            if pd.isna(sigma) or sigma <= 0:
                logger.warning(f"Invalid volatility: {sigma}")
                return {greek: float('nan') for greek in 
                    ['delta', 'gamma', 'theta', 'vega', 'rho']}

            # Ensure minimum values to prevent numerical issues
            T = max(T, self.MIN_TIME)
            sigma = max(sigma, self.MIN_SIGMA)

            if S <= 0 or K <= 0:
                logger.warning(f"Invalid price or strike: S={S}, K={K}")
                return {greek: float('nan') for greek in 
                    ['delta', 'gamma', 'theta', 'vega', 'rho']}

            # Base calculations with logging
            d1 = self.calculate_d1(S, K, T, r, sigma, q)
            d2 = self.calculate_d2(S, K, T, r, sigma, q)
            
            logger.debug(f"d1={d1}, d2={d2}")

            if np.isnan(d1) or np.isnan(d2):
                logger.warning("d1 or d2 calculation failed")
                return {greek: float('nan') for greek in 
                    ['delta', 'gamma', 'theta', 'vega', 'rho']}

            is_call = option_type.upper() == 'CALL'

            # Standard normal calculations
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            n_d1 = norm.pdf(d1)
            
            logger.debug(f"N_d1={N_d1}, N_d2={N_d2}, n_d1={n_d1}")

            # Delta calculation
            if is_call:
                delta = np.exp(-q*T) * N_d1
            else:
                delta = np.exp(-q*T) * (N_d1 - 1)  # Simplified put delta formula

            # Gamma calculation (same for calls and puts)
            gamma = np.exp(-q*T) * n_d1 / (S * sigma * np.sqrt(T))

            # Theta calculation
            theta_term1 = -(S * sigma * np.exp(-q*T) * n_d1) / (2 * np.sqrt(T))
            if is_call:
                theta = theta_term1 - r*K*np.exp(-r*T)*N_d2 + q*S*np.exp(-q*T)*N_d1
            else:
                theta = theta_term1 + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1)

            # Vega calculation (same for calls and puts)
            vega = S * np.exp(-q*T) * np.sqrt(T) * n_d1

            # Rho calculation
            if is_call:
                rho = K * T * np.exp(-r*T) * N_d2
            else:
                rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)

            # Log calculated values before adjustments
            logger.debug(f"Raw values: delta={delta}, gamma={gamma}, theta={theta}, vega={vega}, rho={rho}")

            # Return adjusted values
            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta/365),  # Convert to daily theta
                'vega': float(vega/100),    # Per 1% change in vol
                'rho': float(rho/100)       # Per 1% change in rates
            }

        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return {greek: float('nan') for greek in 
                ['delta', 'gamma', 'theta', 'vega', 'rho']}

class OptionsStrategyAnalyzer:
    """Analyzes basic options strategies"""
    
    def __init__(self):
        self.greeks_calculator = GreeksCalculator()
        self.MIN_DTE = 1  # Move to class initialization
        self.last_error = None
        self.MIN_VOLUME = 5
        self.MIN_OPEN_INTEREST = 5
        self.MAX_SPREAD_PCT = 0.20

    def _validate_option_liquidity(self, option: pd.Series) -> Tuple[bool, Optional[str]]:
        if option['bid'] <= 0 or option['ask'] <= 0:
            return False, "Invalid bid/ask prices"
        if option['ask'] < option['bid']:
            return False, "Ask price lower than bid price"
            
        spread_pct = (option['ask'] - option['bid']) / option['ask']
        min_price = min(option['bid'], option['ask'])
        
        # First check overall maximum spread threshold
        if spread_pct > self.MAX_SPREAD_PCT:
            return False, f"Spread ({spread_pct:.1%}) exceeds maximum threshold ({self.MAX_SPREAD_PCT:.1%})"
        
        # Additional tiered checks for different price ranges
        if min_price < 1.0 and spread_pct > 0.25:
            return False, f"Spread ({spread_pct:.1%}) too wide for sub-$1 option"
        elif min_price < 5.0 and spread_pct > 0.15:
            return False, f"Spread ({spread_pct:.1%}) too wide for $1-$5 option"
        elif min_price > 10.0 and spread_pct > 0.10:
            return False, f"Spread ({spread_pct:.1%}) too wide for $10+ option"
            
        return True, None

    def _validate_option_activity(self, option: pd.Series) -> bool:
        """Validate option has sufficient trading activity"""
        return (
            option['volume'] >= self.MIN_VOLUME and 
            option['openInterest'] >= self.MIN_OPEN_INTEREST
        )

    def analyze_credit_call_spread(self, chain: pd.DataFrame, width_pct: float = 0.05) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            if 'underlying_price' not in chain.columns:
                return None, "Missing price data in options chain"
                
            if 'dte' not in chain.columns:
                return None, "Missing DTE calculation in options chain"
                
            dte = chain['dte'].iloc[0]
            if dte < self.MIN_DTE:
                return None, f"Expiration too close. Minimum DTE: {self.MIN_DTE}"

            current_price = chain['underlying_price'].iloc[0]
            # Find closest OTM strikes
            otm_calls = chain[chain['strike'] > current_price].copy()
            if otm_calls.empty:
                return None, "No valid OTM strikes found"
                
            # Target first and second OTM strikes with sufficient volume
            valid_strikes = otm_calls[
                (otm_calls['volume'] >= self.MIN_VOLUME) & 
                (otm_calls['openInterest'] >= self.MIN_OPEN_INTEREST)
            ]['strike'].sort_values()
            
            if len(valid_strikes) < 2:
                return None, "Not enough liquid strikes for spread"
                
            target_short_strike = valid_strikes.iloc[0]
            target_long_strike = valid_strikes.iloc[1]
            
            # Find closest strikes
            short_options = chain[chain['strike'] >= target_short_strike]
            if short_options.empty:
                return None, f"No valid strikes found above {target_short_strike}"
                
            short_strike = short_options['strike'].iloc[0]
            long_options = chain[chain['strike'] >= target_long_strike]
            if long_options.empty:
                return None, f"No valid strikes found above {target_long_strike}"
                
            long_strike = long_options['strike'].iloc[0]
            
            short_option = chain[chain['strike'] == short_strike].iloc[0]
            long_option = chain[chain['strike'] == long_strike].iloc[0]

            # Validate liquidity
            if not self._validate_option_liquidity(short_option):
                return None, f"Short strike {short_strike} has insufficient liquidity (wide bid-ask spread)"
            if not self._validate_option_liquidity(long_option):
                return None, f"Long strike {long_strike} has insufficient liquidity (wide bid-ask spread)"
            
            # Validate activity
            if not self._validate_option_activity(short_option):
                return None, f"Short strike {short_strike} has insufficient volume (min: {self.MIN_VOLUME}) or open interest (min: {self.MIN_OPEN_INTEREST})"
            if not self._validate_option_activity(long_option):
                return None, f"Long strike {long_strike} has insufficient volume (min: {self.MIN_VOLUME}) or open interest (min: {self.MIN_OPEN_INTEREST})"
            
            credit = float(short_option['bid'] - long_option['ask'])
            if credit <= 0:
                return None, f"No valid credit found for strike combination {short_strike}/{long_strike}"
                
            max_loss = float(long_strike - short_strike - credit)
            probability_otm = 1 - float(short_option['prob_itm'])
            
            try:
                net_delta = float(short_option['delta'] - long_option['delta'])
                net_theta = float(short_option['theta'] - long_option['theta'])
                net_gamma = float(short_option['gamma'] - long_option['gamma'])
                
                # Validate Greeks are not zero or NaN
                if all(abs(greek) < 1e-10 for greek in [net_delta, net_theta, net_gamma]):
                    return None, "Invalid Greeks calculation for spread"
                    
            except (ValueError, TypeError) as e:
                return None, f"Error calculating spread Greeks: {str(e)}"

            return {
                'strikes': {
                    'short_strike': float(short_strike),
                    'long_strike': float(long_strike)
                },
                'metrics': {
                    'credit': credit,
                    'max_loss': max_loss,
                    'max_profit': credit,
                    'probability_of_profit': probability_otm,
                    'risk_reward_ratio': abs(max_loss/credit) if credit != 0 else float('inf')
                },
                'greeks': {
                    'net_delta': net_delta,
                    'net_theta': net_theta,
                    'net_gamma': net_gamma
                }
            }, None
            
        except Exception as e:
            error_msg = f"Error analyzing CCS: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def analyze_put_credit_spread(self, chain: pd.DataFrame, width_pct: float = 0.05) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            if 'underlying_price' not in chain.columns:
                return None, "Missing price data in options chain"
                
            if 'dte' not in chain.columns:
                return None, "Missing DTE calculation in options chain"
                
            dte = chain['dte'].iloc[0]
            if dte < self.MIN_DTE:
                return None, f"Expiration too close. Minimum DTE: {self.MIN_DTE}"

            current_price = chain['underlying_price'].iloc[0]
            
            below_current = chain[chain['strike'] < current_price]
            if below_current.empty:
                return None, f"No valid strikes found below current price {current_price}"
                
            short_strike = below_current['strike'].iloc[-1]
            below_short = chain[chain['strike'] < short_strike]
            if below_short.empty:
                return None, f"No valid strikes found below {short_strike}"
                
            long_strike = below_short['strike'].iloc[-1]
            
            short_option = chain[chain['strike'] == short_strike].iloc[0]
            long_option = chain[chain['strike'] == long_strike].iloc[0]

            # Validate liquidity
            if not self._validate_option_liquidity(short_option):
                return None, f"Short strike {short_strike} has insufficient liquidity (wide bid-ask spread)"
            if not self._validate_option_liquidity(long_option):
                return None, f"Long strike {long_strike} has insufficient liquidity (wide bid-ask spread)"
            
            # Validate activity
            if not self._validate_option_activity(short_option):
                return None, f"Short strike {short_strike} has insufficient volume (min: {self.MIN_VOLUME}) or open interest (min: {self.MIN_OPEN_INTEREST})"
            if not self._validate_option_activity(long_option):
                return None, f"Long strike {long_strike} has insufficient volume (min: {self.MIN_VOLUME}) or open interest (min: {self.MIN_OPEN_INTEREST})"
            
            credit = float(short_option['bid'] - long_option['ask'])
            if credit <= 0:
                return None, f"No valid credit found for strike combination {short_strike}/{long_strike}"
                
            max_loss = float(short_strike - long_strike - credit)
            probability_otm = 1 - float(short_option['prob_itm'])
            
            return {
                'strikes': {
                    'short_strike': float(short_strike),
                    'long_strike': float(long_strike)
                },
                'metrics': {
                    'credit': credit,
                    'max_loss': max_loss,
                    'max_profit': credit,
                    'probability_of_profit': probability_otm,
                    'risk_reward_ratio': abs(max_loss/credit) if credit != 0 else float('inf')
                },
                'greeks': {
                    'net_delta': float(short_option['delta'] - long_option['delta']),
                    'net_theta': float(short_option['theta'] - long_option['theta']),
                    'net_gamma': float(short_option['gamma'] - long_option['gamma'])
                }
            }, None
            
        except Exception as e:
            error_msg = f"Error analyzing PCS: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def analyze_cash_secured_put(self, chain: pd.DataFrame, delta_target: float = 0.3) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            if 'underlying_price' not in chain.columns:
                return None, "Missing price data in options chain"
                
            if 'dte' not in chain.columns:
                return None, "Missing DTE calculation in options chain"
                
            dte = chain['dte'].iloc[0]
            if dte < self.MIN_DTE:
                return None, f"Expiration too close. Minimum DTE: {self.MIN_DTE}"

            current_price = chain['underlying_price'].iloc[0]
            put_options = chain[chain['option_type'] == 'put']
            
            if put_options.empty:
                return None, "No valid put options found for this expiration"
                
            # For puts, find closest to -delta_target since put deltas are negative
            target_put = put_options.iloc[(put_options['delta'] + delta_target).abs().argsort()[:1]].iloc[0]
            
            # Validate liquidity
            if not self._validate_option_liquidity(target_put):
                return None, f"Strike {target_put['strike']} has insufficient liquidity (wide bid-ask spread)"
            
            # Validate activity
            if not self._validate_option_activity(target_put):
                return None, f"Strike {target_put['strike']} has insufficient volume (min: {self.MIN_VOLUME}) or open interest (min: {self.MIN_OPEN_INTEREST})"
            
            premium = float(target_put['bid'])
            max_loss = float(target_put['strike'] - premium)
            assigned_cost_basis = float(target_put['strike'] - premium)
            
            return {
                'strike': float(target_put['strike']),
                'metrics': {
                    'premium': premium,
                    'max_loss': max_loss,
                    'assigned_cost_basis': assigned_cost_basis,
                    'return_if_otm': float(premium / target_put['strike'] * 100),
                    'downside_protection': float((1 - assigned_cost_basis/current_price) * 100)
                },
                'greeks': {
                    'delta': float(target_put['delta']),
                    'theta': float(target_put['theta']),
                    'gamma': float(target_put['gamma'])
                }
            }, None
            
        except Exception as e:
            error_msg = f"Error analyzing CSP: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def analyze_covered_call(self, chain: pd.DataFrame, delta_target: float = 0.3) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            if 'underlying_price' not in chain.columns:
                return None, "Missing price data in options chain"
                
            if 'dte' not in chain.columns:
                return None, "Missing DTE calculation in options chain"
                
            dte = chain['dte'].iloc[0]
            if dte < self.MIN_DTE:
                return None, f"Expiration too close. Minimum DTE: {self.MIN_DTE}"

            current_price = chain['underlying_price'].iloc[0]
            call_options = chain[chain['option_type'] == 'call']
            if call_options.empty:
                return None, "No valid call options found for this expiration"

            # Debug info
            logger.info(f"Current price: {current_price}")
            logger.info(f"Available strikes: {call_options['strike'].tolist()}")
            logger.info(f"Deltas: {call_options['delta'].tolist()}")

            otm_calls = call_options[call_options['strike'] >= current_price]
            if otm_calls.empty:
                return None, "No valid OTM strikes found"

            logger.info(f"OTM strikes: {otm_calls['strike'].tolist()}")
            logger.info(f"OTM deltas: {otm_calls['delta'].tolist()}")

            # Find the strike with delta closest to our target
            target_delta = 1 - delta_target  # For 0.3 target, we want 0.7 delta
            target_call = otm_calls.iloc[(otm_calls['delta'] - target_delta).abs().argsort()[:1]].iloc[0]
            
            logger.info(f"Selected strike: {target_call['strike']}")
            logger.info(f"Selected delta: {target_call['delta']}")
            
            # Validate liquidity
            is_liquid, liquidity_error = self._validate_option_liquidity(target_call)
            if not is_liquid:
                return None, f"Strike {target_call['strike']} {liquidity_error}"
            
            # Validate activity
            if not self._validate_option_activity(target_call):
                return None, f"Strike {target_call['strike']} has insufficient volume (min: {self.MIN_VOLUME}) or open interest (min: {self.MIN_OPEN_INTEREST})"
            
            premium = float(target_call['bid'])
            max_profit = float(target_call['strike'] - current_price + premium)
            called_away_return = float((max_profit / current_price) * 100)
            
            return {
                'strike': float(target_call['strike']),
                'metrics': {
                    'premium': premium,
                    'max_profit': max_profit,
                    'max_profit_percent': called_away_return,
                    'upside_cap': float(target_call['strike']),
                    'premium_yield': float(premium / current_price * 100)
                },
                'greeks': {
                    'position_delta': float(target_call['delta']),  # Delta is already correct from BS calc
                    'theta': float(target_call['theta']),
                    'gamma': float(target_call['gamma'])
                }
            }, None
            
        except Exception as e:
            error_msg = f"Error analyzing CC: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

def format_response(data: Any, error: Optional[str] = None) -> List[TextContent]:
    """Format API response"""
    response = {
        "success": error is None,
        "timestamp": time.time(),
        "data": data if error is None else None,
        "error": error
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2)
    )]

# Initialize server and analyzers
app = Server("options-analytics")
greeks_calculator = GreeksCalculator()
strategy_analyzer = OptionsStrategyAnalyzer()


def process_option_chain(chain: pd.DataFrame, current_price: float, risk_free_rate: Optional[float] = None) -> pd.DataFrame:
    """Process option chain and calculate Greeks"""
    
    # Get risk-free rate if not provided
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate()
        logger.info(f"Using risk-free rate: {risk_free_rate:.4f}")
        
    # Extract symbol from contract
    contract_symbol = chain['contractSymbol'].iloc[0]
    symbol_match = re.match(r'^[A-Za-z]+', contract_symbol)
    if not symbol_match:
        raise ValueError(f"Could not extract symbol from contract: {contract_symbol}")
    symbol = symbol_match.group()
    
    # Get dividend yield
    try:
        ticker = yf.Ticker(symbol)
        div_yield = ticker.info.get('dividendYield', 0)
        if div_yield is None:
            div_yield = 0
    except Exception as e:
        logger.warning(f"Could not get dividend yield for {symbol}: {e}")
        div_yield = 0
        
    logger.info(f"Processing chain for {symbol} with div_yield={div_yield}")
        
    # Ensure we have the required columns
    if 'underlying_price' not in chain.columns:
        chain['underlying_price'] = current_price
    
    # Convert expiry to datetime and handle timezone
    chain['expiry'] = pd.to_datetime(chain['expiry'])
    
    # Calculate time to expiration
    now = datetime.datetime.now()
    chain['expiry'] = pd.to_datetime(chain['expiry'])
    chain['dte'] = (chain['expiry'] - now).dt.total_seconds() / (24 * 60 * 60)  # Exact DTE in days
    
     # Initialize Greeks calculator
    calculator = GreeksCalculator()
    
    # Calculate Greeks for each option
    for idx, row in chain.iterrows():
        try:
            # Skip if invalid IV
            if pd.isna(row['impliedVolatility']) or row['impliedVolatility'] <= 0:
                logger.warning(f"Skipping row {idx} due to invalid IV: {row['impliedVolatility']}")
                continue
                
            # Log key inputs
            logger.debug(f"Processing option: Strike={row['strike']}, IV={row['impliedVolatility']}, DTE={row['dte']}")
            
            # Calculate Greeks
            greeks = calculator.calculate_greeks(
                float(current_price),
                float(row['strike']),
                float(row['dte']) / 365,  # Convert DTE to years
                float(risk_free_rate),
                float(row['impliedVolatility']),
                float(div_yield),
                'CALL' if row['option_type'] == 'call' else 'PUT'
            )
            
            # Update DataFrame with Greeks
            for greek, value in greeks.items():
                chain.loc[idx, greek] = value
                
            # Log results
            logger.debug(f"Calculated Greeks for row {idx}: {greeks}")
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            # Set Greeks to NaN on error
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                chain.loc[idx, greek] = np.nan
    
    # Calculate probability ITM based on delta
    chain['prob_itm'] = chain.apply(
        lambda row: abs(row['delta']) if not pd.isna(row['delta']) else 0,
        axis=1
    )
    
    return chain

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="analyze_basic_strategies",
            description="Analyze basic options strategies (CCS, PCS, CSP, CC)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol"},
                    "strategy": {
                        "type": "string",
                        "enum": ["ccs", "pcs", "csp", "cc"],
                        "description": "Options strategy to analyze"
                    },
                    "delta_target": {
                        "type": "number",
                        "description": "Target delta for CSP/CC (default: 0.3)",
                        "default": 0.3
                    },
                    "width_pct": {
                        "type": "number",
                        "description": "Width for spreads as decimal (default: 0.05)",
                        "default": 0.05
                    },
                    "expiration_date": {
                        "type": "string",
                        "description": "Options expiration date (YYYY-MM-DD)"
                    }
                },
                "required": ["symbol", "strategy", "expiration_date"]
            }
        )
    ]

@app.call_tool()
@retry_on_error(max_retries=3, delay=1.0)
async def call_tool(name: str, arguments: dict):
    try:
        if name == "analyze_basic_strategies":
            symbol = arguments['symbol'].strip().upper()
            strategy = arguments['strategy'].lower()
            delta_target = arguments.get('delta_target', 0.3)
            width_pct = arguments.get('width_pct', 0.05)
            requested_expiry = arguments['expiration_date']
            
            # Get ticker data
            ticker = yf.Ticker(symbol)
            
            # Get current price
            try:
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
            except:
                try:
                    info = ticker.info
                    current_price = info.get('regularMarketPrice') or info.get('currentPrice')
                except:
                    raise APIError(f"Could not get current price for {symbol}")
            if not current_price:
                raise APIError(f"Could not get current price for {symbol}")
            
            # Get expiration dates and validate requested date
            exp_dates = ticker.options
            if not exp_dates:
                raise APIError(f"No options available for {symbol}")
                
            # Validate expiration exists
            if requested_expiry not in exp_dates:
                raise ValidationError(f"Expiration {requested_expiry} not available. Available dates: {', '.join(exp_dates[:5])}")

            # Calculate DTE
            today = pd.Timestamp.now().normalize()
            expiry_date = pd.to_datetime(requested_expiry).normalize()
            dte = (expiry_date - today).days

            # Initialize response with basic info
            response = {
                "symbol": symbol,
                "strategy": strategy.upper(),
                "current_price": current_price,
                "expiration": requested_expiry,
                "days_to_expiration": dte
            }

            # Add warning for short-dated options
            if dte < 30:  # Less than 30 DTE
                # Find a suggested date with better premium potential
                valid_dates = [date for date in exp_dates 
                            if (pd.to_datetime(date) - pd.Timestamp.now()).days >= 30]
                if valid_dates:
                    suggested_date = valid_dates[0]
                    response["warning"] = f"Warning: Short-dated option selected. Consider {suggested_date} for better premium collection."
            
            if dte < 1:
                raise ValidationError(f"Expiration too soon. DTE must be at least 1, got {dte}")
                
            # Get the chain
            chain = ticker.option_chain(requested_expiry)
            if not hasattr(chain, 'calls') or not hasattr(chain, 'puts'):
                raise APIError("Invalid options chain data")
            
            # Process chains
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            
            calls['option_type'] = 'call'
            puts['option_type'] = 'put'
            calls['underlying_price'] = current_price
            puts['underlying_price'] = current_price
            calls['expiry'] = expiry_date
            puts['expiry'] = expiry_date
            
            # Get risk-free rate
            risk_free_rate = get_risk_free_rate()
            logger.info(f"Using risk-free rate: {risk_free_rate:.4f}")
            
            # Process chains with the risk-free rate
            calls_processed = process_option_chain(calls, current_price, risk_free_rate)
            puts_processed = process_option_chain(puts, current_price, risk_free_rate)
            
            # Run strategy analysis based on type
            if strategy == "ccs":
                analysis, error = strategy_analyzer.analyze_credit_call_spread(
                    calls_processed, 
                    width_pct=width_pct
                )
            elif strategy == "pcs":
                analysis, error = strategy_analyzer.analyze_put_credit_spread(
                    puts_processed, 
                    width_pct=width_pct
                )
            elif strategy == "csp":
                analysis, error = strategy_analyzer.analyze_cash_secured_put(
                    puts_processed, 
                    delta_target=delta_target
                )
            elif strategy == "cc":
                analysis, error = strategy_analyzer.analyze_covered_call(
                    calls_processed, 
                    delta_target=delta_target
                )
            else:
                raise ValidationError(f"Invalid strategy: {strategy}")

            if error:
                raise APIError(error)
            
            if not analysis:
                raise APIError(f"Could not analyze {strategy.upper()} strategy - no valid options found")
            
            response["analysis"] = analysis
            
            return format_response(response)
            
    except ValidationError as e:
        logger.error(f"Validation error in {name}: {str(e)}")
        return format_response(None, f"Validation error: {str(e)}")
        
    except APIError as e:
        logger.error(f"API error in {name}: {str(e)}\n{traceback.format_exc()}")
        return format_response(None, f"API error: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error in {name}: {str(e)}\n{traceback.format_exc()}")
        return format_response(None, f"Internal error: {str(e)}")

async def main():    
    logger.info("Starting Options Analytics server...")
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(main())