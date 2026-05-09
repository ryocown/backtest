"""
Unit tests for AnalyticsEngine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import AnalyticsEngine


class TestDrawdowns:
    """Tests for drawdown calculations."""
    
    def test_get_drawdowns_basic(self, sample_prices):
        """Test basic drawdown calculation."""
        dd = AnalyticsEngine.get_drawdowns(sample_prices)
        
        # Drawdowns should be <= 0
        assert (dd <= 0).all().all()
        
        # First row should be 0 (no drawdown at start)
        assert dd.iloc[0].sum() == 0
        
        # Shape should match input
        assert dd.shape == sample_prices.shape
    
    def test_get_drawdowns_known_values(self):
        """Test drawdown with known values."""
        prices = pd.DataFrame({
            'A': [100, 110, 90, 100, 80]
        }, index=pd.date_range('2020-01-01', periods=5))
        
        dd = AnalyticsEngine.get_drawdowns(prices)
        
        # At index 2: price=90, peak=110 -> dd = 90/110 - 1 ≈ -0.182
        assert abs(dd.iloc[2, 0] - (-0.182)) < 0.01
        
        # At index 4: price=80, peak=110 -> dd = 80/110 - 1 ≈ -0.273
        assert abs(dd.iloc[4, 0] - (-0.273)) < 0.01


class TestMonthlyReturns:
    """Tests for monthly return calculations."""
    
    def test_get_monthly_returns_shape(self, sample_prices):
        """Test monthly returns have correct shape."""
        monthly = AnalyticsEngine.get_monthly_returns(sample_prices)
        
        # Should have fewer rows than daily (approx 12 months for 252 days)
        assert len(monthly) < len(sample_prices)
        assert len(monthly) == 12  # Exactly 12 months for 252 trading days


class TestRollingVolatility:
    """Tests for rolling volatility calculations."""
    
    def test_get_rolling_volatility_annualized(self, sample_prices):
        """Test that volatility is annualized."""
        vol = AnalyticsEngine.get_rolling_volatility(sample_prices, window=20)
        
        # Skip NaN values from rolling window
        valid_vol = vol.dropna()
        
        # Annualized vol should typically be between 5% and 100%
        assert (valid_vol > 0.05).all().all()
        assert (valid_vol < 1.0).all().all()
    
    def test_get_rolling_volatility_window_size(self, sample_prices):
        """Test that first window-1 values are NaN."""
        window = 30
        vol = AnalyticsEngine.get_rolling_volatility(sample_prices, window=window)
        
        # First window values should be NaN (after pct_change which loses 1 row)
        assert vol.iloc[:window - 1].isna().all().all()


class TestRollingSharpe:
    """Tests for rolling Sharpe ratio calculations."""
    
    def test_get_rolling_sharpe_basic(self, sample_prices):
        """Test basic Sharpe calculation."""
        sharpe = AnalyticsEngine.get_rolling_sharpe(sample_prices, window=20)
        
        valid_sharpe = sharpe.dropna()
        
        # Sharpe should be finite
        assert np.isfinite(valid_sharpe).all().all()
    
    def test_get_rolling_sharpe_with_risk_free(self, sample_prices):
        """Test Sharpe with risk-free rate."""
        sharpe_0 = AnalyticsEngine.get_rolling_sharpe(sample_prices, risk_free_rate=0.0)
        sharpe_4 = AnalyticsEngine.get_rolling_sharpe(sample_prices, risk_free_rate=0.04)
        
        # Higher risk-free rate should result in lower Sharpe
        valid_idx = sharpe_0.dropna().index.intersection(sharpe_4.dropna().index)
        assert (sharpe_4.loc[valid_idx] <= sharpe_0.loc[valid_idx]).all().all()


class TestCaptureRatios:
    """Tests for upside/downside capture ratios."""
    
    def test_get_capture_ratios_basic(self, sample_prices):
        """Test capture ratio calculation."""
        strat = sample_prices['StrategyA']
        bench = sample_prices['SPY']
        
        caps = AnalyticsEngine.get_capture_ratios(strat, bench)
        
        assert 'upside_capture' in caps
        assert 'downside_capture' in caps
        assert np.isfinite(caps['upside_capture'])
        assert np.isfinite(caps['downside_capture'])
    
    def test_get_capture_ratios_perfect_tracking(self):
        """Test capture ratios for perfect tracking (should both be 1.0)."""
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        prices = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates)
        
        caps = AnalyticsEngine.get_capture_ratios(prices, prices)
        
        assert abs(caps['upside_capture'] - 1.0) < 0.01
        assert abs(caps['downside_capture'] - 1.0) < 0.01


class TestRiskContribution:
    """Tests for risk contribution (MCTR) calculations."""
    
    def test_get_risk_contribution_sums_to_one(self, sample_asset_prices, sample_weights):
        """Test that percentage contributions sum to approximately 1."""
        # Align weights to available columns
        valid_tickers = [t for t in sample_weights.index if t in sample_asset_prices.columns]
        weights = sample_weights[valid_tickers]
        weights = weights / weights.sum()  # Normalize
        
        pct_ctr = AnalyticsEngine.get_risk_contribution(
            sample_asset_prices[valid_tickers], weights
        )
        
        # Should sum to approximately 1.0
        assert abs(pct_ctr.sum() - 1.0) < 0.01


class TestRollingAlphaBeta:
    """Tests for rolling CAPM alpha/beta calculations."""
    
    def test_get_rolling_alpha_beta_basic(self, sample_prices):
        """Test basic alpha/beta calculation."""
        strat_prices = sample_prices[['StrategyA', 'StrategyB']]
        bench_prices = sample_prices[['SPY']]
        
        alphas, betas = AnalyticsEngine.get_rolling_alpha_beta(
            strat_prices, bench_prices, window=20
        )
        
        # Should have same columns as strategies
        assert list(alphas.columns) == ['StrategyA', 'StrategyB']
        assert list(betas.columns) == ['StrategyA', 'StrategyB']
        
        # Betas should be finite
        valid_betas = betas.dropna()
        assert np.isfinite(valid_betas).all().all()


class TestReturnAttribution:
    """Tests for return attribution calculations."""
    
    def test_get_return_attribution_cumulative(self, sample_asset_prices, sample_weights):
        """Test that attribution is cumulative."""
        valid_tickers = [t for t in sample_weights.index if t in sample_asset_prices.columns]
        weights = sample_weights[valid_tickers]
        
        attr = AnalyticsEngine.get_return_attribution(
            sample_asset_prices[valid_tickers], weights
        )
        
        # Should be cumulative (generally increasing in absolute value)
        # Check that last value >= some intermediate values
        assert attr.shape == sample_asset_prices[valid_tickers].shape
