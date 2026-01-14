"""
Pytest configuration and shared fixtures for the backtest test suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Creates sample price data for testing."""
    dates = pd.date_range('2020-01-01', periods=252, freq='B')  # 1 year of trading days
    np.random.seed(42)
    
    # Generate random walk prices
    returns = np.random.normal(0.0005, 0.02, (252, 3))  # Mean 0.05%/day, 2% vol
    prices = 100 * np.cumprod(1 + returns, axis=0)
    
    return pd.DataFrame(
        prices,
        index=dates,
        columns=['StrategyA', 'StrategyB', 'SPY']
    )


@pytest.fixture
def sample_weights() -> pd.Series:
    """Creates sample portfolio weights."""
    return pd.Series({
        'AAPL': 0.3,
        'MSFT': 0.3,
        'GOOGL': 0.2,
        'AMZN': 0.2
    })


@pytest.fixture
def sample_asset_prices() -> pd.DataFrame:
    """Creates sample asset prices for portfolio analysis."""
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    np.random.seed(123)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    returns = np.random.normal(0.0005, 0.015, (252, 4))
    prices = 100 * np.cumprod(1 + returns, axis=0)
    
    return pd.DataFrame(prices, index=dates, columns=tickers)
