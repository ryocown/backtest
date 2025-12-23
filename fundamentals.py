import os
import logging
import pandas as pd
import numpy as np
import simfin as sf

logger = logging.getLogger(__name__)

class FundamentalEngine:
    """Handles fetching and processing of historical fundamental data using SimFin (Free Tier)."""
    
    def __init__(self, cache_dir='data_cache'):
        self.cache_dir = os.path.join(os.path.dirname(__file__), cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # SimFin Setup
        api_key = os.getenv('SIMFIN_API_KEY')
        assert api_key is not None, "SimFin key not provided"
        sf.set_api_key(api_key)
        sf.set_data_dir(os.path.join(self.cache_dir, 'simfin'))
        
        self.income_df = None
        self._eps_cache = {}

    def load_data(self):
        """Pre-loads the SimFin dataset."""
        if self.income_df is None:
            logger.info("Loading SimFin quarterly income data (Free US dataset)...")
            try:
                # This uses the bulk download which is free for quarterly income
                self.income_df = sf.load_income(variant='quarterly', market='us')
            except Exception as e:
                logger.error(f"Failed to load SimFin data: {e}")
                self.income_df = pd.DataFrame()

    def get_historical_ttm_eps(self, ticker):
        """Calculates TTM EPS series for a single ticker."""
        if ticker in self._eps_cache:
            return self._eps_cache[ticker]
            
        if self.income_df is None or self.income_df.empty:
            return pd.Series(dtype=float)
            
        try:
            if ticker not in self.income_df.index.get_level_values(0):
                return pd.Series(dtype=float)
                
            ticker_data = self.income_df.loc[ticker].copy()
            ticker_data = ticker_data.sort_values('Publish Date')
            
            # TTM Calculation
            ticker_data['TTM_Net_Income'] = ticker_data['Net Income'].rolling(window=4).sum()
            ticker_data['EPS_TTM'] = ticker_data['TTM_Net_Income'] / ticker_data['Shares (Diluted)']
            
            # Use Publish Date as index and normalize to midnight for alignment
            ticker_data['Publish Date'] = pd.to_datetime(ticker_data['Publish Date']).dt.normalize()
            eps_series = ticker_data.set_index('Publish Date')['EPS_TTM'].dropna()
            
            # If multiple reports published on same day (restatements), take the last
            eps_series = eps_series[~eps_series.index.duplicated(keep='last')]
            
            # Store in cache
            self._eps_cache[ticker] = eps_series
            return eps_series
        except Exception as e:
            logger.warning(f"Error calculating EPS for {ticker}: {e}")
            return pd.Series(dtype=float)

    def get_portfolio_pe_series(self, weights_df, prices_df):
        """
        Calculates historical Weighted Harmonic Mean P/E for the portfolio.
        """
        self.load_data()
        
        # Ensure indices are normalized to date-only at midnight
        weights_df.index = pd.to_datetime(weights_df.index).normalize()
        prices_df.index = pd.to_datetime(prices_df.index).normalize()
        
        tickers = weights_df.columns
        all_eps = {}
        
        for ticker in tickers:
            eps = self.get_historical_ttm_eps(ticker)
            if not eps.empty:
                all_eps[ticker] = eps
        
        if not all_eps:
            logger.warning("No EPS data found for any portfolio constituents.")
            return pd.Series(dtype=float)
            
        # Combine all EPS series into a matrix and align with prices
        eps_matrix = pd.DataFrame(all_eps).reindex(prices_df.index).ffill()
        
        # Calculate P/E matrix
        pe_matrix = prices_df / eps_matrix
        
        # Filter outliers and negative P/Es
        valid_pe_mask = (pe_matrix > 0) & (pe_matrix < 1000)
        
        inv_pe = 1.0 / pe_matrix
        inv_pe[~valid_pe_mask] = np.nan
        
        # Align weights to price/eps indices if they differ
        weights_aligned = weights_df.reindex(prices_df.index).ffill()
        
        # Weighted inverse PE (Sum of W_i / PE_i)
        weighted_inv_pe = (weights_aligned * inv_pe).sum(axis=1, min_count=1)
        
        # Effective weight (sum of weights where P/E is valid/available)
        effective_weight = weights_aligned.where(inv_pe.notnull()).sum(axis=1, min_count=1)
        
        portfolio_pe = effective_weight / weighted_inv_pe
        
        # Final cleanup: replace infs and very high outliers that ruin plots
        portfolio_pe = portfolio_pe.replace([np.inf, -np.inf], np.nan)
        portfolio_pe = portfolio_pe.clip(lower=0, upper=500)
        
        valid_points = portfolio_pe.count()
        logger.info(f"Calculated P/E series: {valid_points} valid points out of {len(portfolio_pe)}")
        return portfolio_pe
