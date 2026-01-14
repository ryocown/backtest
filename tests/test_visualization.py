"""
Smoke tests for visualization functions.

These tests verify that plotting functions execute without error.
They don't validate visual output - that requires manual inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockResult:
    """Mock bt.Result for testing visualization functions."""
    
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self._weights = {}
    
    def get_security_weights(self, name: str) -> pd.DataFrame:
        """Returns mock security weights."""
        if name in self._weights:
            return self._weights[name]
        # Return equal weights for all columns
        weights = pd.DataFrame(
            1.0 / len(self.prices.columns),
            index=self.prices.index,
            columns=self.prices.columns
        )
        return weights


@pytest.fixture
def mock_result(sample_prices) -> MockResult:
    """Creates a mock bt.Result object."""
    return MockResult(sample_prices)


class TestRollingPlots:
    """Smoke tests for rolling metric plots."""
    
    @patch('matplotlib.pyplot.show')
    def test_plot_drawdowns(self, mock_show, mock_result):
        """Test drawdown plot executes without error."""
        from visualization.rolling import plot_drawdowns
        
        plot_drawdowns(mock_result)
        # No assertion needed - we just verify no exception
    
    @patch('matplotlib.pyplot.show')
    def test_plot_rolling_volatility(self, mock_show, mock_result):
        """Test rolling volatility plot executes without error."""
        from visualization.rolling import plot_rolling_volatility
        
        plot_rolling_volatility(mock_result, window=20)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_rolling_sharpe(self, mock_show, mock_result):
        """Test rolling Sharpe plot executes without error."""
        from visualization.rolling import plot_rolling_sharpe
        
        plot_rolling_sharpe(mock_result, window=20)


class TestStaticPlots:
    """Smoke tests for static chart plots."""
    
    @patch('matplotlib.pyplot.show')
    def test_plot_return_distribution(self, mock_show, mock_result):
        """Test return distribution plot executes without error."""
        from visualization.static import plot_return_distribution
        
        plot_return_distribution(mock_result)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_correlation_matrix(self, mock_show, mock_result):
        """Test correlation matrix plot executes without error."""
        from visualization.static import plot_correlation_matrix
        
        plot_correlation_matrix(mock_result)


class TestHeatmapPlots:
    """Smoke tests for heatmap plots."""
    
    @patch('matplotlib.pyplot.show')
    def test_plot_monthly_returns_heatmap(self, mock_show, mock_result):
        """Test monthly returns heatmap executes without error."""
        from visualization.heatmaps import plot_monthly_returns_heatmap
        
        plot_monthly_returns_heatmap(mock_result, 'StrategyA')
    
    @patch('matplotlib.pyplot.show')
    def test_plot_grouped_correlation_matrix(self, mock_show, sample_asset_prices):
        """Test grouped correlation matrix executes without error."""
        from visualization.heatmaps import plot_grouped_correlation_matrix
        
        group_map = {
            'AAPL': 'Tech',
            'MSFT': 'Tech',
            'GOOGL': 'Tech',
            'AMZN': 'Consumer'
        }
        
        plot_grouped_correlation_matrix(sample_asset_prices, group_map)


class TestCursor:
    """Tests for cursor functionality."""
    
    @patch('matplotlib.pyplot.gcf')
    def test_add_financial_cursor(self, mock_gcf):
        """Test financial cursor attachment."""
        from visualization.cursor import add_financial_cursor
        
        # Create a mock figure
        mock_fig = MagicMock()
        mock_fig.axes = []
        mock_fig.canvas = MagicMock()
        mock_gcf.return_value = mock_fig
        
        cursor = add_financial_cursor(mock_fig)
        
        # Cursor should be created
        assert cursor is not None
