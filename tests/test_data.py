"""
Unit tests for DataEngine in data.py.
"""
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from src.data import DataEngine

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Fixture to create a temporary cache directory."""
    d = tmp_path / "data_cache"
    d.mkdir()
    return str(d)

class TestDataEngine:
    """Tests for DataEngine."""

    def test_init(self, temp_cache_dir):
        """Test initialization creates directory."""
        # We need to mock environment variables for client init in __init__
        with patch.dict(os.environ, {'ALPACA_API_KEY': 'fake', 'ALPACA_SECRET_KEY': 'fake'}):
            engine = DataEngine(cache_dir=temp_cache_dir)
        assert os.path.exists(temp_cache_dir)
        assert engine.cache_dir == temp_cache_dir

    @patch('src.data.StockHistoricalDataClient')
    def test_get_ticker_ohlc_cache_miss(self, mock_alpaca, temp_cache_dir):
        """Test cache miss calls API."""
        # Setup mock for Alpaca
        mock_client = MagicMock()
        mock_alpaca.return_value = mock_client
        
        # Mock bars response
        mock_bars = MagicMock()
        mock_df_multi = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [101.0], 'volume': [1000]
        }, index=pd.MultiIndex.from_tuples([('AAPL', pd.Timestamp('2020-01-01'))], names=['symbol', 'timestamp']))
        mock_bars.df = mock_df_multi
        mock_client.get_stock_bars.return_value = mock_bars
        
        with patch.dict(os.environ, {'ALPACA_API_KEY': 'fake', 'ALPACA_SECRET_KEY': 'fake'}):
            engine = DataEngine(cache_dir=temp_cache_dir)
            df = engine.get_ticker_ohlc('AAPL', '2020-01-01', '2020-01-01')
            
        assert df is not None
        assert not df.empty
        assert 'close' in df.columns
        # The index in get_ticker_ohlc is normalized and timezone naive
        assert df.index[0] == pd.Timestamp('2020-01-01')
        
        # Verify cache file created
        cache_path = os.path.join(temp_cache_dir, "AAPL.csv")
        assert os.path.exists(cache_path)

    def test_get_ticker_ohlc_cache_hit(self, temp_cache_dir):
        """Test cache hit does not call API."""
        with patch.dict(os.environ, {'ALPACA_API_KEY': 'fake', 'ALPACA_SECRET_KEY': 'fake'}):
            engine = DataEngine(cache_dir=temp_cache_dir)
        
        # Create dummy cache file
        cache_path = os.path.join(temp_cache_dir, "AAPL.csv")
        mock_df = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [101.0], 'volume': [1000]
        }, index=pd.DatetimeIndex(['2020-01-01']))
        mock_df.to_csv(cache_path)
        
        # We should patch _fetch_bars to ensure it's not called
        with patch.object(engine, '_fetch_bars') as mock_fetch:
            df = engine.get_ticker_ohlc('AAPL', '2020-01-01', '2020-01-01')
            
        assert df is not None
        assert df.index[0] == pd.Timestamp('2020-01-01')
        mock_fetch.assert_not_called()
