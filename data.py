import os
import logging
import yfinance as yf
import json
import pandas as pd
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

logger = logging.getLogger(__name__)


class DataEngine:
    """Handles fetching and caching of historical stock data and metadata."""
    
    def __init__(self, cache_dir='data_cache'):
        self.cache_dir = os.path.join(os.path.dirname(__file__), cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.cache_dir, "metadata.json")
        self.metadata = self._load_json(self.metadata_path, default={})
        self.client = self._init_client()
        logger.debug(f"DataEngine initialized with cache_dir: {self.cache_dir}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def get_historical_data(self, tickers, start_date, end_date, fail_on_missing=False):
        """Fetches daily close prices for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            fail_on_missing: If True, returns list of failed tickers for caller to handle
            
        Returns:
            If fail_on_missing=True: (DataFrame, list of failed tickers)
            If fail_on_missing=False: DataFrame only (backward compatible)
        """
        all_data = {}
        failed_tickers = []
        
        for ticker in tickers:
            data = self._get_ticker_data(ticker, start_date, end_date)
            if data is not None:
                all_data[ticker] = data
            else:
                logger.warning(f"No data retrieved for {ticker}")
                failed_tickers.append(ticker)
        
        df = pd.DataFrame(all_data).sort_index()
        df = self._ensure_tz_naive(df)
        
        if fail_on_missing:
            return df, failed_tickers
        return df

    def get_metadata(self, tickers):
        """Fetches sector metadata for tickers, utilizing local cache."""
        updated = False
        for ticker in tickers:
            if ticker not in self.metadata:
                self.metadata[ticker] = self._fetch_ticker_metadata(ticker)
                updated = True
        
        if updated:
            self._save_json(self.metadata_path, self.metadata)
        return self.metadata

    # -------------------------------------------------------------------------
    # Alpaca Client
    # -------------------------------------------------------------------------
    
    def _init_client(self):
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables are required."
            )
        
        logger.debug("Alpaca client initialized.")
        return StockHistoricalDataClient(api_key, secret_key)

    def _fetch_bars(self, ticker, start, end):
        """Fetches daily bars from Alpaca for a single ticker."""
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start if isinstance(start, datetime) else start.to_pydatetime(),
            end=(end if isinstance(end, datetime) else end.to_pydatetime()) + pd.Timedelta(days=1),
            adjustment=Adjustment.ALL
        )
        bars = self.client.get_stock_bars(request)
        if bars.df.empty:
            return None
        
        df = bars.df.xs(ticker, level=0)
        return self._ensure_tz_naive(df)

    # -------------------------------------------------------------------------
    # Price Data Caching
    # -------------------------------------------------------------------------
    
    def _get_ticker_data(self, ticker, start_date, end_date):
        """Gets ticker data, using cache when possible."""
        cache_path = self._cache_path(ticker)
        req_start, req_end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
        
        # Try to load from cache
        cached_df = self._load_cache(cache_path)
        
        if cached_df is not None:
            cache_start, cache_end = cached_df.index.min().normalize(), cached_df.index.max().normalize()
            
            # Full cache hit
            if cache_start <= req_start and cache_end >= req_end:
                logger.debug(f"Cache hit for {ticker}")
                return cached_df['close']
            
            # Partial hit: check if the 'gaps' relative to data actually contain market days
            has_gap = False
            if req_start < cache_start:
                if not self._get_market_days(req_start, cache_start - pd.Timedelta(days=1)).empty:
                    has_gap = True
            if req_end > cache_end:
                if not self._get_market_days(cache_end + pd.Timedelta(days=1), req_end).empty:
                    has_gap = True
            
            if not has_gap:
                logger.debug(f"Cache effective hit for {ticker} (no market days in gaps)")
                return cached_df['close']

            # Partial cache hit - fetch missing ranges
            logger.info(f"Cache partial hit for {ticker}: have {cache_start.date()}-{cache_end.date()}")
            cached_df = self._fill_cache_gaps(ticker, cached_df, cache_start, cache_end, req_start, req_end)
            self._save_cache(cache_path, cached_df)
            return cached_df['close']
        
        # No cache - fetch full range
        logger.info(f"Fetching {ticker} ({start_date} to {end_date})...")
        new_df = self._fetch_bars(ticker, start_date, end_date)
        if new_df is not None:
            self._save_cache(cache_path, new_df)
            return new_df['close']
        
        return None

    def _fill_cache_gaps(self, ticker, cached_df, cache_start, cache_end, req_start, req_end):
        """Fetches and merges missing date ranges into cached data."""
        fetch_ranges = []
        if cache_start > req_start:
            fetch_ranges.append((req_start, cache_start - pd.Timedelta(days=1)))
        if cache_end < req_end:
            fetch_ranges.append((cache_end + pd.Timedelta(days=1), req_end))
        
        for start, end in fetch_ranges:
            if end < start:
                continue  # Skip invalid ranges
            
            # Only fetch if there are market days in the range
            market_days = self._get_market_days(start, end)
            if market_days.empty:
                continue
            
            logger.info(f"Fetching {ticker} {start.date()} to {end.date()}...")
            try:
                new_df = self._fetch_bars(ticker, start, end)
                if new_df is not None:
                    cached_df = pd.concat([cached_df, new_df])
                    cached_df = cached_df[~cached_df.index.duplicated(keep='last')].sort_index()
            except Exception as e:
                logger.warning(f"Error fetching {ticker} for {start.date()}-{end.date()}: {e}")
        
        return cached_df

    def _get_market_days(self, start, end):
        """Helper to get actual market days in a range using USFederalHolidayCalendar."""
        from pandas.tseries.holiday import USFederalHolidayCalendar
        biz_days = pd.bdate_range(start, end)
        holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
        return biz_days[~biz_days.isin(holidays)]

    def _cache_path(self, ticker):
        return os.path.join(self.cache_dir, f"{ticker}.csv")

    def _load_cache(self, path):
        """Loads and validates a cache file."""
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if 'close' not in df.columns or df.empty:
                return None
            return self._ensure_tz_naive(df)
        except Exception as e:
            logger.warning(f"Error reading cache {path}: {e}")
            return None

    def _save_cache(self, path, df):
        """Saves data to CSV."""
        df.to_csv(path)
        logger.debug(f"Saved cache to {path}")

    # -------------------------------------------------------------------------
    # Metadata (yfinance)
    # -------------------------------------------------------------------------
    
    def _fetch_ticker_metadata(self, ticker):
        """Fetches sector/name metadata from yfinance."""
        logger.info(f"Fetching metadata for {ticker}...")
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Other')
            # ETFs use 'category' instead of 'sector'
            if sector == 'Other' and info.get('quoteType') == 'ETF':
                sector = info.get('category', 'ETF')
            return {'sector': sector, 'name': info.get('longName', ticker)}
        except Exception as e:
            logger.error(f"Error fetching metadata for {ticker}: {e}")
            return {'sector': 'Other', 'name': ticker}

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _ensure_tz_naive(df):
        """Ensures DataFrame index is timezone-naive."""
        if hasattr(df, 'index') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    @staticmethod
    def _load_json(path, default=None):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
        return default if default is not None else {}

    @staticmethod
    def _save_json(path, data):
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {path}: {e}")
