# OPTIONS PRICING, RISK & STRATEGY ANALYSIS PLATFORM
# 1. All Core Pricing Models
# 2. Greeks Calculator
# 3. Portfolio Manager
# 4. Market Data Handler
# 5. Volatility Models
# 6. Strategy Builder
# 7. REST API (FastAPI)
# 8. CLI Interface
# 9. Streamlit Dashboard Code

# For fast mathematical operations on large arrays and matrices (e.g., Black-Scholes calculations).
import numpy as np
# For handling tabular data, datasets, and dataframes (e.g., input/output of trading/payoff tables).
import pandas as pd
# For probability and statistical functions (like cumulative normal distribution, useful in financial models)
from scipy.stats import norm
# For advanced math: finding roots, numerical derivatives (e.g., implied volatility computations).
from scipy.optimize import brentq, approx_fprime
# Allows use of abstract base classes and interfaces (for structured OOP design).
from abc import ABC, abstractmethod
# Used to show or control Python warning messages (e.g., for deprecated functions or edge cases).
import warnings
# To record, format, and manage runtime logs and errors (using file/stream handlers).
import logging
import logging.handlers
import os  # To interact with file system, environment variables, directories.
# For date/time manipulations and scheduling (option expiry, logging timestamps).
from datetime import datetime, timedelta
# For type hinting in code; improves readability and robustness.
from typing import List, Dict, Optional, Tuple
# Handles reading/writing JSON data (often for API requests, config, or caching).
import json
# Controls Python system operations, arguments (e.g., command-line parsing in CLI/API).
import sys

# SECTION 1: LOGGING CONFIGURATION


def setup_logging(log_file="options_platform.log", log_level=logging.INFO):
    """Configure logging for the entire application"""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(
        f"logs/{log_file}",
        maxBytes=10485760,
        backupCount=5
    )
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()

# SECTION 2: CUSTOM EXCEPTIONS & VALIDATORS


class CustomException(Exception):
    """Base custom exception"""
    pass


class PricingException(CustomException):
    """Exception in pricing calculation"""
    pass


class DataFetchException(CustomException):
    """Exception in data fetching"""
    pass


class ValidationException(CustomException):
    """Exception in input validation"""
    pass


class InputValidator:
    """Centralized input validation"""

    @staticmethod
    def validate_option_parameters(S, K, T, r, sigma, option_type):
        """Validate option pricing parameters"""
        errors = []

        if S <= 0:
            errors.append("Underlying price (S) must be positive")
        if K <= 0:
            errors.append("Strike price (K) must be positive")
        if T <= 0:
            errors.append("Time to expiry (T) must be positive")
        if sigma < 0:
            errors.append("Volatility (σ) cannot be negative")
        if sigma == 0:
            errors.append("Volatility (σ) should be positive")
        if option_type.lower() not in ['call', 'put']:
            errors.append("Option type must be 'call' or 'put'")
        if abs(r) > 1:
            logger.warning(f"Interest rate {r} seems unusually high/low")

        if errors:
            return False, "; ".join(errors)
        return True, None

    @staticmethod
    def validate_market_data(data_dict):
        """Validate market data dictionary"""
        required_fields = ['underlying_price', 'risk_free_rate', 'volatility']
        for field in required_fields:
            if field not in data_dict or data_dict[field] is None:
                return False, f"Missing or invalid field: {field}"
        return True, None

# SECTION 3: PRICING MODELS


class PricingModel(ABC):
    """Abstract base class for pricing models"""

    @abstractmethod
    def price(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option price"""
        pass


class BlackScholesModel(PricingModel):
    """Black-Scholes option pricing model"""

    def __init__(self):
        self.name = "Black-Scholes"

    def price(self, S, K, T, r, sigma, option_type='call'):
        """Black-Scholes pricing formula"""
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("All parameters must be positive")

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return price

    def get_d1_d2(self, S, K, T, r, sigma):
        """Helper method to calculate d1 and d2"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2


class BinomialTreeModel(PricingModel):
    """Binomial Tree pricing model for American and European options"""

    def __init__(self, steps=100):
        self.name = "Binomial Tree"
        self.steps = steps

    def price(self, S, K, T, r, sigma, option_type='call', american=False):
        """Binomial Tree option pricing"""
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("All parameters must be positive")

        dt = T / self.steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(r * dt) - d) / (u - d)

        stock_prices = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            stock_prices[self.steps, i] = S * \
                (u ** i) * (d ** (self.steps - i))

        option_values = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            if option_type.lower() == 'call':
                option_values[self.steps, i] = max(
                    stock_prices[self.steps, i] - K, 0)
            else:
                option_values[self.steps, i] = max(
                    K - stock_prices[self.steps, i], 0)

        for step in range(self.steps - 1, -1, -1):
            for i in range(step + 1):
                option_values[step, i] = np.exp(-r * dt) * (
                    q * option_values[step + 1, i] +
                    (1 - q) * option_values[step + 1, i + 1]
                )

                if american:
                    stock_prices[step, i] = S * (u ** i) * (d ** (step - i))
                    if option_type.lower() == 'call':
                        intrinsic = max(stock_prices[step, i] - K, 0)
                    else:
                        intrinsic = max(K - stock_prices[step, i], 0)
                    option_values[step, i] = max(
                        option_values[step, i], intrinsic)

        return option_values[0, 0]


class MonteCarloModel(PricingModel):
    """Monte Carlo simulation model for option pricing"""

    def __init__(self, simulations=10000, seed=42):
        self.name = "Monte Carlo"
        self.simulations = simulations
        self.seed = seed

    def price(self, S, K, T, r, sigma, option_type='call'):
        """Monte Carlo option pricing"""
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or self.simulations <= 0:
            raise ValueError("All parameters must be positive")

        np.random.seed(self.seed)

        dt = T / 252
        steps = int(252 * T)
        Z = np.random.standard_normal((self.simulations, steps))

        S_T = np.zeros(self.simulations)

        for sim in range(self.simulations):
            S_current = S
            for step in range(steps):
                S_current *= np.exp((r - 0.5 * sigma**2) *
                                    dt + sigma * np.sqrt(dt) * Z[sim, step])
            S_T[sim] = S_current

        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        option_price = np.exp(-r * T) * np.mean(payoffs)
        return option_price

    def get_paths(self, S, K, T, r, sigma, num_paths=100):
        """Generate simulated price paths"""
        np.random.seed(self.seed)
        dt = T / 252
        steps = int(252 * T)
        Z = np.random.standard_normal((num_paths, steps))

        paths = np.zeros((num_paths, steps + 1))
        paths[:, 0] = S

        for sim in range(num_paths):
            for step in range(steps):
                paths[sim, step + 1] = paths[sim, step] * np.exp(
                    (r - 0.5 * sigma**2) * dt +
                    sigma * np.sqrt(dt) * Z[sim, step]
                )

        return paths


class PricingModelFactory:
    """Factory for creating pricing model instances"""

    @staticmethod
    def create_model(model_name, **kwargs):
        """Create and return appropriate pricing model"""
        models = {
            'black-scholes': BlackScholesModel,
            'binomial': BinomialTreeModel,
            'monte-carlo': MonteCarloModel
        }

        if model_name.lower() not in models:
            raise ValueError(f"Model {model_name} not supported")

        return models[model_name.lower()](**kwargs)

# SECTION 4: GREEKS CALCULATOR


class GreeksCalculator:
    """Comprehensive Greeks calculation"""

    def __init__(self, pricing_model):
        self.model = pricing_model
        self.eps = 1e-4

    def delta_analytical(self, S, K, T, r, sigma, option_type='call'):
        """Delta (analytical)"""
        if T <= 0 or sigma <= 0:
            return 0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma_analytical(self, S, K, T, r, sigma, option_type='call'):
        """Gamma (analytical)"""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def theta_analytical(self, S, K, T, r, sigma, option_type='call'):
        """Theta (analytical)"""
        if T <= 0 or sigma <= 0:
            return 0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                     r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                     r * K * np.exp(-r * T) * norm.cdf(-d2))

        return theta / 365

    def vega_analytical(self, S, K, T, r, sigma, option_type='call'):
        """Vega (analytical)"""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T) / 100

    def rho_analytical(self, S, K, T, r, sigma, option_type='call'):
        """Rho (analytical)"""
        if T <= 0 or sigma <= 0:
            return 0

        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return rho

    def delta_numerical(self, S, K, T, r, sigma, option_type='call'):
        """Delta (numerical)"""
        S_up = S + self.eps
        S_down = S - self.eps

        price_up = self.model.price(S_up, K, T, r, sigma, option_type)
        price_down = self.model.price(S_down, K, T, r, sigma, option_type)

        return (price_up - price_down) / (2 * self.eps)

    def gamma_numerical(self, S, K, T, r, sigma, option_type='call'):
        """Gamma (numerical)"""
        S_up = S + self.eps
        S_down = S - self.eps

        price_up = self.model.price(S_up, K, T, r, sigma, option_type)
        price_center = self.model.price(S, K, T, r, sigma, option_type)
        price_down = self.model.price(S_down, K, T, r, sigma, option_type)

        return (price_up - 2 * price_center + price_down) / (self.eps ** 2)

    def theta_numerical(self, S, K, T, r, sigma, option_type='call'):
        """Theta (numerical)"""
        T_down = max(T - self.eps, 0.001)

        price_now = self.model.price(S, K, T, r, sigma, option_type)
        price_tomorrow = self.model.price(S, K, T_down, r, sigma, option_type)

        theta = (price_tomorrow - price_now) / self.eps
        return theta / 365

    def vega_numerical(self, S, K, T, r, sigma, option_type='call'):
        """Vega (numerical)"""
        sigma_up = sigma + self.eps
        sigma_down = sigma - self.eps

        price_up = self.model.price(S, K, T, r, sigma_up, option_type)
        price_down = self.model.price(S, K, T, r, sigma_down, option_type)

        vega_raw = (price_up - price_down) / (2 * self.eps)
        return vega_raw / 100

    def rho_numerical(self, S, K, T, r, sigma, option_type='call'):
        """Rho (numerical)"""
        r_up = r + self.eps
        r_down = r - self.eps

        price_up = self.model.price(S, K, T, r_up, sigma, option_type)
        price_down = self.model.price(S, K, T, r_down, sigma, option_type)

        rho_raw = (price_up - price_down) / (2 * self.eps)
        return rho_raw / 100

    def compute_all_greeks_analytical(self, S, K, T, r, sigma, option_type='call'):
        """Compute all Greeks analytically"""
        return {
            'delta': self.delta_analytical(S, K, T, r, sigma, option_type),
            'gamma': self.gamma_analytical(S, K, T, r, sigma, option_type),
            'theta': self.theta_analytical(S, K, T, r, sigma, option_type),
            'vega': self.vega_analytical(S, K, T, r, sigma, option_type),
            'rho': self.rho_analytical(S, K, T, r, sigma, option_type)
        }

    def compute_all_greeks_numerical(self, S, K, T, r, sigma, option_type='call'):
        """Compute all Greeks numerically"""
        return {
            'delta': self.delta_numerical(S, K, T, r, sigma, option_type),
            'gamma': self.gamma_numerical(S, K, T, r, sigma, option_type),
            'theta': self.theta_numerical(S, K, T, r, sigma, option_type),
            'vega': self.vega_numerical(S, K, T, r, sigma, option_type),
            'rho': self.rho_numerical(S, K, T, r, sigma, option_type)
        }


# SECTION 6: VOLATILITY MODELS


class ImpliedVolatilitySolver:
    """Solves for implied volatility"""

    def __init__(self, pricing_model, max_iterations=100, tolerance=1e-6):
        self.pricing_model = pricing_model
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def calculate_iv_brent(self, market_price, S, K, T, r, option_type='call',
                           sigma_bounds=(0.01, 5.0)):
        """Calculate implied volatility using Brent's method"""
        try:
            def objective(sigma):
                model_price = self.pricing_model.price(
                    S, K, T, r, sigma, option_type)
                return model_price - market_price

            iv = brentq(objective, sigma_bounds[0], sigma_bounds[1],
                        xtol=self.tolerance, maxiter=self.max_iterations)

            return iv

        except ValueError:
            logger.warning(
                f"Could not solve for IV (market_price={market_price})")
            return None

    def calculate_iv_newton_raphson(self, market_price, S, K, T, r,
                                    option_type='call', initial_guess=0.2):
        """Calculate IV using Newton-Raphson method"""
        sigma = initial_guess

        for iteration in range(self.max_iterations):
            model_price = self.pricing_model.price(
                S, K, T, r, sigma, option_type)
            vega = self._calculate_vega(S, K, T, r, sigma, option_type)

            if abs(vega) < 1e-8:
                logger.warning("Vega too small in Newton-Raphson")
                break

            sigma_new = sigma - (model_price - market_price) / vega

            if abs(sigma_new - sigma) < self.tolerance:
                return sigma_new

            sigma = sigma_new

        logger.warning(f"Newton-Raphson did not converge")
        return sigma

    def _calculate_vega(self, S, K, T, r, sigma, option_type):
        """Calculate vega"""
        epsilon = 1e-4
        price_up = self.pricing_model.price(
            S, K, T, r, sigma + epsilon, option_type)
        price_down = self.pricing_model.price(
            S, K, T, r, sigma - epsilon, option_type)
        return (price_up - price_down) / (2 * epsilon)


class VolatilitySurfaceGenerator:
    """Generates volatility surface"""

    def __init__(self, iv_solver):
        self.iv_solver = iv_solver
        self.surface_data = pd.DataFrame()

    def generate_surface(self, S, option_chain_df, r):
        """Generate volatility surface"""
        surface_data = []

        for expiration in option_chain_df['expiration'].unique():
            exp_data = option_chain_df[option_chain_df['expiration']
                                       == expiration]

            T = pd.to_datetime(expiration)
            T_years = (T - pd.Timestamp.now()).days / 365

            if T_years <= 0:
                continue

            for _, row in exp_data.iterrows():
                K = row['strike']

                if pd.notna(row.get('impliedVolatility')):
                    iv = row['impliedVolatility']
                else:
                    market_price = row['lastPrice']
                    option_type = 'call' if 'call' in str(
                        row.get('type', '')).lower() else 'put'

                    iv = self.iv_solver.calculate_iv_brent(
                        market_price, S, K, T_years, r, option_type
                    )

                if iv is not None:
                    surface_data.append({
                        'strike': K,
                        'maturity': T_years,
                        'implied_vol': iv,
                        'expiration': expiration
                    })

        self.surface_data = pd.DataFrame(surface_data)
        logger.info(
            f"Generated volatility surface with {len(surface_data)} points")
        return self.surface_data

    def get_surface_grid(self, strike_range=None, maturity_range=None):
        """Return surface data as 2D grid"""
        if self.surface_data.empty:
            return None

        data = self.surface_data.copy()

        if strike_range:
            data = data[(data['strike'] >= strike_range[0]) &
                        (data['strike'] <= strike_range[1])]

        if maturity_range:
            data = data[(data['maturity'] >= maturity_range[0]) &
                        (data['maturity'] <= maturity_range[1])]

        grid = data.pivot_table(
            index='strike',
            columns='maturity',
            values='implied_vol',
            aggfunc='mean'
        )

        return grid

    def analyze_skew(self):
        """Analyze volatility skew"""
        if self.surface_data.empty:
            return None

        skew_analysis = {}

        for maturity in self.surface_data['maturity'].unique():
            maturity_data = self.surface_data[self.surface_data['maturity'] == maturity]
            vols = maturity_data.sort_values('strike')['implied_vol'].values

            if len(vols) >= 3:
                skew = vols[0] - vols[-1]
                skew_analysis[f"{maturity:.2f}Y"] = skew

        return skew_analysis

# SECTION 7: STRATEGY BUILDER


class Strategy:
    """Represents a multi-leg option strategy"""

    def __init__(self, name):
        self.name = name
        self.legs = []

    def add_leg(self, option_type, strike, quantity, premium_paid):
        """Add a leg to strategy"""
        leg = {
            'type': option_type,
            'strike': strike,
            'quantity': quantity,
            'premium': premium_paid
        }
        self.legs.append(leg)
        logger.info(f"Added leg to {self.name}: {option_type} @ {strike}")

    def calculate_payoff_at_expiry(self, stock_prices):
        """Calculate strategy payoff at expiry"""
        payoff = np.zeros_like(stock_prices, dtype=float)
        cost_basis = 0

        for leg in self.legs:
            option_type = leg['type']
            strike = leg['strike']
            quantity = leg['quantity']
            premium = leg['premium']

            if 'call' in option_type.lower():
                leg_payoff = np.maximum(stock_prices - strike, 0) * quantity
            else:
                leg_payoff = np.maximum(strike - stock_prices, 0) * quantity

            if 'short' in option_type.lower():
                leg_payoff = -leg_payoff

            if 'long' in option_type.lower():
                cost_basis += premium * quantity * 100
            else:
                cost_basis -= premium * quantity * 100

            payoff += leg_payoff * 100

        net_payoff = payoff - cost_basis

        return {
            'stock_prices': stock_prices,
            'gross_payoff': payoff,
            'cost_basis': cost_basis,
            'net_payoff': net_payoff
        }

    def calculate_max_profit_loss(self, payoff_data):
        """Calculate maximum profit and loss"""
        net_payoff = payoff_data['net_payoff']

        max_profit = np.max(net_payoff)
        max_loss = np.min(net_payoff)

        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward_ratio': abs(max_profit / max_loss) if max_loss != 0 else float('inf')
        }

    def find_breakeven_points(self, payoff_data):
        """Find breakeven stock prices"""
        stock_prices = payoff_data['stock_prices']
        net_payoff = payoff_data['net_payoff']

        breakeven_points = []

        for i in range(len(net_payoff) - 1):
            if net_payoff[i] * net_payoff[i + 1] < 0:
                be_price = stock_prices[i] - net_payoff[i] * (
                    (stock_prices[i + 1] - stock_prices[i]) /
                    (net_payoff[i + 1] - net_payoff[i])
                )
                breakeven_points.append(be_price)

        return breakeven_points


class StrategyBuilder:
    """Builder for common option strategies"""

    @staticmethod
    def create_bull_call_spread(long_strike, short_strike, long_premium, short_premium):
        """Bull Call Spread"""
        strategy = Strategy("Bull Call Spread")
        strategy.add_leg('long_call', long_strike, 1, long_premium)
        strategy.add_leg('short_call', short_strike, 1, short_premium)
        return strategy

    @staticmethod
    def create_bear_put_spread(short_strike, long_strike, short_premium, long_premium):
        """Bear Put Spread"""
        strategy = Strategy("Bear Put Spread")
        strategy.add_leg('short_put', short_strike, 1, short_premium)
        strategy.add_leg('long_put', long_strike, 1, long_premium)
        return strategy

    @staticmethod
    def create_straddle(strike, call_premium, put_premium):
        """Straddle"""
        strategy = Strategy("Straddle")
        strategy.add_leg('long_call', strike, 1, call_premium)
        strategy.add_leg('long_put', strike, 1, put_premium)
        return strategy

    @staticmethod
    def create_strangle(call_strike, put_strike, call_premium, put_premium):
        """Strangle"""
        strategy = Strategy("Strangle")
        strategy.add_leg('long_call', call_strike, 1, call_premium)
        strategy.add_leg('long_put', put_strike, 1, put_premium)
        return strategy

    @staticmethod
    def create_iron_condor(long_call_strike, short_call_strike,
                           short_put_strike, long_put_strike,
                           lc_premium, sc_premium, sp_premium, lp_premium):
        """Iron Condor"""
        strategy = Strategy("Iron Condor")
        strategy.add_leg('long_call', long_call_strike, 1, lc_premium)
        strategy.add_leg('short_call', short_call_strike, 1, sc_premium)
        strategy.add_leg('short_put', short_put_strike, 1, sp_premium)
        strategy.add_leg('long_put', long_put_strike, 1, lp_premium)
        return strategy

# SECTION 8: CLI INTERFACE


class CLIInterface:
    """Command-line interface for the platform"""

    def __init__(self):
        self.pricing_factory = PricingModelFactory()
        self.bs_model = self.pricing_factory.create_model('black-scholes')
        self.greeks_calc = GreeksCalculator(self.bs_model)
        self.portfolio = Portfolio("CLI Portfolio")

    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*70)
        print("OPTIONS PRICING & RISK ANALYSIS PLATFORM")
        print("="*70)
        print("1. Price an Option")
        print("2. Calculate Greeks")
        print("3. Portfolio Analysis")
        print("4. Strategy Analysis")
        print("5. Scenario Analysis")
        print("6. Exit")
        print("="*70)

    def price_option_cli(self):
        """Price option via CLI"""
        print("\n--- OPTION PRICING ---")
        try:
            S = float(input("Underlying Price: "))
            K = float(input("Strike Price: "))
            T = float(input("Time to Expiry (years): "))
            r = float(input("Risk-free Rate: "))
            sigma = float(input("Volatility: "))
            option_type = input("Option Type (call/put): ")

            # Validate
            is_valid, error = InputValidator.validate_option_parameters(
                S, K, T, r, sigma, option_type)
            if not is_valid:
                print(f"Error: {error}")
                return

            # Price with all models
            bs_price = self.bs_model.price(S, K, T, r, sigma, option_type)
            binomial_model = self.pricing_factory.create_model(
                'binomial', steps=50)
            binomial_price = binomial_model.price(
                S, K, T, r, sigma, option_type)
            mc_model = self.pricing_factory.create_model(
                'monte-carlo', simulations=5000)
            mc_price = mc_model.price(S, K, T, r, sigma, option_type)

            print(f"\nBlack-Scholes: ${bs_price:.4f}")
            print(f"Binomial Tree: ${binomial_price:.4f}")
            print(f"Monte Carlo: ${mc_price:.4f}")

        except Exception as e:
            print(f"Error: {str(e)}")

    def calculate_greeks_cli(self):
        """Calculate Greeks via CLI"""
        print("\n--- GREEKS CALCULATION ---")
        try:
            S = float(input("Underlying Price: "))
            K = float(input("Strike Price: "))
            T = float(input("Time to Expiry (years): "))
            r = float(input("Risk-free Rate: "))
            sigma = float(input("Volatility: "))
            option_type = input("Option Type (call/put): ")

            greeks = self.greeks_calc.compute_all_greeks_analytical(
                S, K, T, r, sigma, option_type)

            print(f"\nDelta: {greeks['delta']:.6f}")
            print(f"Gamma: {greeks['gamma']:.6f}")
            print(f"Theta: {greeks['theta']:.6f}")
            print(f"Vega: {greeks['vega']:.6f}")
            print(f"Rho: {greeks['rho']:.6f}")

        except Exception as e:
            print(f"Error: {str(e)}")

    def strategy_analysis_cli(self):
        """Analyze strategy via CLI"""
        print("\n--- STRATEGY ANALYSIS ---")
        print("1. Bull Call Spread")
        print("2. Bear Put Spread")
        print("3. Straddle")
        print("4. Iron Condor")

        choice = input("Select strategy: ")

        try:
            if choice == '1':
                long_strike = float(input("Long Call Strike: "))
                short_strike = float(input("Short Call Strike: "))
                long_premium = float(input("Long Call Premium: "))
                short_premium = float(input("Short Call Premium: "))
                strategy = StrategyBuilder.create_bull_call_spread(
                    long_strike, short_strike, long_premium, short_premium
                )
            elif choice == '2':
                short_strike = float(input("Short Put Strike: "))
                long_strike = float(input("Long Put Strike: "))
                short_premium = float(input("Short Put Premium: "))
                long_premium = float(input("Long Put Premium: "))
                strategy = StrategyBuilder.create_bear_put_spread(
                    short_strike, long_strike, short_premium, long_premium
                )
            else:
                print("Invalid choice")
                return

            # Calculate payoff
            stock_prices = np.linspace(
                long_strike - 10, short_strike + 10, 100)
            payoff = strategy.calculate_payoff_at_expiry(stock_prices)
            pnl_metrics = strategy.calculate_max_profit_loss(payoff)
            breakevens = strategy.find_breakeven_points(payoff)

            print(f"\nStrategy: {strategy.name}")
            print(f"Max Profit: ${pnl_metrics['max_profit']:.2f}")
            print(f"Max Loss: ${pnl_metrics['max_loss']:.2f}")
            print(f"Risk/Reward Ratio: {pnl_metrics['risk_reward_ratio']:.2f}")
            print(f"Breakeven Points: {[f'${be:.2f}' for be in breakevens]}")

        except Exception as e:
            print(f"Error: {str(e)}")

    def run(self):
        """Main CLI loop"""
        while True:
            self.display_menu()
            choice = input("Select option: ")

            if choice == '1':
                self.price_option_cli()
            elif choice == '2':
                self.calculate_greeks_cli()
            elif choice == '3':
                print("Portfolio analysis - coming soon")
            elif choice == '4':
                self.strategy_analysis_cli()
            elif choice == '5':
                print("Scenario analysis - coming soon")
            elif choice == '6':
                print("Exiting...")
                break
            else:
                print("Invalid choice")

# SECTION 9: FASTAPI REST API


try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    # Define request/response models
    class OptionPricingRequest(BaseModel):
        underlying_price: float
        strike_price: float
        time_to_expiry: float
        risk_free_rate: float
        volatility: float
        option_type: str
        model: str = "black-scholes"

    class OptionPricingResponse(BaseModel):
        option_price: float
        delta: float
        gamma: float
        theta: float
        vega: float
        rho: float
        model_used: str
        timestamp: str

    def create_api_app(pricing_factory, greeks_calculator):
        """Create FastAPI application"""
        app = FastAPI(title="Options Analytics API", version="1.0.0")

        @app.get("/health")
        def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @app.post("/price-option", response_model=OptionPricingResponse)
        def price_option(request: OptionPricingRequest):
            """Price an option"""
            try:
                if request.underlying_price <= 0 or request.strike_price <= 0:
                    raise ValueError("Prices must be positive")
                if request.time_to_expiry <= 0 or request.volatility <= 0:
                    raise ValueError("Time and volatility must be positive")

                model = pricing_factory.create_model(request.model)

                option_price = model.price(
                    request.underlying_price,
                    request.strike_price,
                    request.time_to_expiry,
                    request.risk_free_rate,
                    request.volatility,
                    request.option_type
                )

                greeks = greeks_calculator.compute_all_greeks_analytical(
                    request.underlying_price,
                    request.strike_price,
                    request.time_to_expiry,
                    request.risk_free_rate,
                    request.volatility,
                    request.option_type
                )

                return OptionPricingResponse(
                    option_price=option_price,
                    delta=greeks['delta'],
                    gamma=greeks['gamma'],
                    theta=greeks['theta'],
                    vega=greeks['vega'],
                    rho=greeks['rho'],
                    model_used=request.model,
                    timestamp=datetime.now().isoformat()
                )

            except Exception as e:
                logger.error(f"Error pricing option: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))

        @app.post("/compute-greeks")
        def compute_greeks(request: OptionPricingRequest):
            """Compute Greeks"""
            try:
                greeks = greeks_calculator.compute_all_greeks_analytical(
                    request.underlying_price,
                    request.strike_price,
                    request.time_to_expiry,
                    request.risk_free_rate,
                    request.volatility,
                    request.option_type
                )

                return {
                    "greeks": greeks,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Error computing Greeks: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))

        return app

except ImportError:
    logger.warning(
        "FastAPI not installed. API functionality will be disabled.")

# SECTION 10: DEMO & MAIN FUNCTION


def run_demo():
    """Run comprehensive demo"""

    print("\n" + "="*80)
    print("OPTIONS PRICING, RISK & STRATEGY ANALYSIS PLATFORM - DEMO")
    print("="*80 + "\n")

    # Initialize models
    pricing_factory = PricingModelFactory()
    bs_model = pricing_factory.create_model('black-scholes')
    greeks_calc = GreeksCalculator(bs_model)

    # Example parameters
    S = 100
    K = 100
    T = 0.25
    r = 0.05
    sigma = 0.2

    print("\n" + "-"*80)
    print("1. OPTION PRICING")
    print("-"*80)

    bs_call = bs_model.price(S, K, T, r, sigma, 'call')
    bs_put = bs_model.price(S, K, T, r, sigma, 'put')
    print(f"Black-Scholes Call: ${bs_call:.4f}")
    print(f"Black-Scholes Put: ${bs_put:.4f}")

    binomial_model = pricing_factory.create_model('binomial', steps=50)
    binomial_call = binomial_model.price(S, K, T, r, sigma, 'call')
    print(f"Binomial Call (50 steps): ${binomial_call:.4f}")

    mc_model = pricing_factory.create_model('monte-carlo', simulations=5000)
    mc_call = mc_model.price(S, K, T, r, sigma, 'call')
    print(f"Monte Carlo Call: ${mc_call:.4f}")

    print("\n" + "-"*80)
    print("2. GREEKS CALCULATION")
    print("-"*80)

    greeks = greeks_calc.compute_all_greeks_analytical(
        S, K, T, r, sigma, 'call')
    print(f"Delta: {greeks['delta']:.6f}")
    print(f"Gamma: {greeks['gamma']:.6f}")
    print(f"Theta: {greeks['theta']:.6f}")
    print(f"Vega: {greeks['vega']:.6f}")
    print(f"Rho: {greeks['rho']:.6f}")

    print("\n" + "-"*80)
    print("3. STRATEGY ANALYSIS - BULL CALL SPREAD")
    print("-"*80)

    bull_call = StrategyBuilder.create_bull_call_spread(
        long_strike=100,
        short_strike=110,
        long_premium=5.0,
        short_premium=2.0
    )

    stock_prices = np.linspace(80, 120, 100)
    payoff = bull_call.calculate_payoff_at_expiry(stock_prices)
    pnl_metrics = bull_call.calculate_max_profit_loss(payoff)
    breakevens = bull_call.find_breakeven_points(payoff)

    print(f"Strategy: {bull_call.name}")
    print(f"Max Profit: ${pnl_metrics['max_profit']:.2f}")
    print(f"Max Loss: ${pnl_metrics['max_loss']:.2f}")
    print(f"Breakeven Points: {[f'${be:.2f}' for be in breakevens]}")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80 + "\n")


def main():
    """Main entry point"""
    # Check for command-line arguments (used by Docker to start API directly)
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\n" + "="*80)
        print("OPTIONS PRICING PLATFORM - MAIN MENU")
        print("="*80)
        print("\n1. Run Demo")
        print("2. CLI Interface")
        print("3. Start API Server (FastAPI)")
        print("4. Exit")
        choice = input("\nSelect option: ")

    if choice == '1':
        run_demo()
    elif choice == '2':
        cli = CLIInterface()
        cli.run()
    elif choice == '3':
        try:
            print("\nStarting API Server on http://localhost:8000...")
            print("Press Ctrl+C to stop the server")
            print("API Endpoints:")
            print("  - GET  /health")
            print("  - POST /price-option")
            print("  - POST /compute-greeks")
            pricing_factory = PricingModelFactory()
            bs_model = pricing_factory.create_model('black-scholes')
            greeks_calc = GreeksCalculator(bs_model)
            app = create_api_app(pricing_factory, greeks_calc)
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        except NameError:
            print("FastAPI is not installed. Install with: pip install fastapi uvicorn")
        except KeyboardInterrupt:
            print("\nServer stopped.")
    elif choice == '4':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
