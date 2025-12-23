import pandas as pd
import numpy as np

class AnalyticsEngine:
    """Handles professional portfolio analytics and risk calculations."""

    @staticmethod
    def get_drawdowns(prices):
        """Calculates drawdown series for all columns."""
        return prices / prices.cummax() - 1

    @staticmethod
    def get_monthly_returns(prices):
        """Calculates compounded monthly returns."""
        daily_rets = prices.pct_change().dropna()
        return daily_rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    @staticmethod
    def get_rolling_volatility(prices, window=126):
        """Calculates annualized rolling volatility."""
        daily_rets = prices.pct_change().dropna()
        return daily_rets.rolling(window=window).std() * np.sqrt(252)

    @staticmethod
    def get_rolling_sharpe(prices, window=126, risk_free_rate=0.0):
        """Calculates annualized rolling Sharpe ratio."""
        returns = prices.pct_change().dropna()
        excess_returns = returns - (risk_free_rate / 252)
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        return (rolling_mean / rolling_std) * np.sqrt(252)

    @staticmethod
    def get_rolling_sortino(prices, window=126, risk_free_rate=0.0):
        """
        Calculates annualized rolling Sortino ratio.
        Institutional standard: Only considers downside deviation.
        """
        returns = prices.pct_change().dropna()
        excess_returns = returns - (risk_free_rate / 252)
        
        def downside_std(x):
            downside_diff = np.minimum(x, 0)
            return np.sqrt(np.mean(downside_diff**2))

        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_downside_std = returns.rolling(window=window).apply(downside_std)
        
        # Avoid division by zero
        return (rolling_mean / rolling_downside_std.replace(0, np.nan)) * np.sqrt(252)

    @staticmethod
    def get_capture_ratios(strat_prices, bench_prices):
        """
        Calculates Upside and Downside Capture ratios.
        Institutional metric for evaluating portfolio performance in different regimes.
        """
        strat_rets = strat_prices.pct_change().dropna()
        bench_rets = bench_prices.pct_change().dropna()
        
        # Align returns
        common_idx = strat_rets.index.intersection(bench_rets.index)
        strat_rets = strat_rets.loc[common_idx]
        bench_rets = bench_rets.loc[common_idx]
        
        # Ensure we are working with Series for mean() to return scalars
        if isinstance(strat_rets, pd.DataFrame):
            strat_rets = strat_rets.iloc[:, 0]
        if isinstance(bench_rets, pd.DataFrame):
            bench_rets = bench_rets.iloc[:, 0]
        
        up_months = bench_rets > 0
        down_months = bench_rets <= 0
        
        upside_capture = (strat_rets[up_months].mean() / bench_rets[up_months].mean()) if up_months.any() else 0
        downside_capture = (strat_rets[down_months].mean() / bench_rets[down_months].mean()) if down_months.any() else 0
        
        return {
            'upside_capture': upside_capture,
            'downside_capture': downside_capture
        }

    @staticmethod
    def get_drawdown_stats(prices):
        """
        Analyzes drawdown periods: Depth, Duration, and Recovery Time.
        Essential for institutional risk management.
        """
        dd = prices / prices.cummax() - 1
        stats = []
        
        for col in dd.columns:
            series = dd[col]
            is_zero = series == 0
            # Identify peaks (starts of drawdowns)
            peak_dates = series.index[is_zero]
            
            # Simple stats for the current underwater state
            max_dd = series.min()
            current_dd = series.iloc[-1]
            
            stats.append({
                'Strategy': col,
                'Max Drawdown': max_dd,
                'Current Drawdown': current_dd
            })
        return pd.DataFrame(stats).set_index('Strategy')

    @staticmethod
    def get_rolling_info_ratio(prices, benchmark_prices, window=126):
        """
        Calculates rolling Information Ratio: (Rp - Rb) / Tracking Error.
        Measures the consistency of active management.
        """
        returns = prices.pct_change().dropna()
        bench_rets = benchmark_prices.pct_change().dropna()
        
        # Align
        common_idx = returns.index.intersection(bench_rets.index)
        returns = returns.loc[common_idx]
        bench_rets = bench_rets.loc[common_idx]
        
        # Ensure bench_rets is a Series for proper broadcasting across strategy columns
        bench_rets_series = bench_rets.iloc[:, 0] if isinstance(bench_rets, pd.DataFrame) else bench_rets
        active_returns = returns.subtract(bench_rets_series, axis=0)
        tracking_error = active_returns.rolling(window=window).std()
        
        # Annualized IR
        ir = (active_returns.rolling(window=window).mean() / tracking_error) * np.sqrt(252)
        return ir

    @staticmethod
    def get_risk_contribution(prices, weights_series):
        """
        Calculates Marginal Contribution to Risk (MCTR) and Percentage Contribution to Risk.
        Helps identify which asset is 'driving' the portfolio's volatility.
        """
        returns = prices.pct_change().dropna()
        cov = returns.cov() * 252 # Annualized covariance
        
        weights = np.array(weights_series)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        
        # Marginal Contribution to Risk
        mctr = np.dot(cov, weights) / port_vol
        # Contribution to Risk
        ctr = weights * mctr
        # Percentage Contribution to Risk
        pct_ctr = ctr / port_vol
        
        return pd.Series(pct_ctr, index=returns.columns)
    @staticmethod
    def get_rolling_alpha_beta(prices, benchmark_prices, window=126):
        """
        Calculates rolling CAPM Alpha and Beta relative to a benchmark.
        Beta: Sensitivity to the benchmark.
        Alpha: Excess return not explained by the benchmark (annualized).
        """
        returns = prices.pct_change().dropna()
        bench_rets = benchmark_prices.pct_change().dropna()
        
        # Align
        common_idx = returns.index.intersection(bench_rets.index)
        returns = returns.loc[common_idx]
        bench_rets = bench_rets.loc[common_idx]
        
        # Bench_rets as Series
        y = bench_rets.iloc[:, 0] if isinstance(bench_rets, pd.DataFrame) else bench_rets
        
        def calc_beta(x):
            # Beta = Cov(Rp, Rb) / Var(Rb)
            cov = np.cov(x, y.loc[x.index])[0, 1]
            var = np.var(y.loc[x.index])
            return cov / var if var != 0 else np.nan

        betas = returns.rolling(window=window).apply(calc_beta)
        
        # Alpha = Rp - (Rf + Beta * (Rb - Rf)) 
        # For simplicity, Rf=0 here as it's typically accounted for in excess returns if needed
        # Alpha_daily = Rp - Beta * Rb
        alphas = returns.subtract(betas.multiply(y, axis=0), axis=0)
        annualized_alphas = alphas.rolling(window=window).mean() * 252
        
        return annualized_alphas, betas

    @staticmethod
    def get_return_attribution(prices, weights_series):
        """
        Calculates Contribution to Return (CTR) for each asset.
        Cumulative CTR = Sum(weight_i * return_i) over time.
        """
        returns = prices.pct_change().fillna(0)
        # Weights normalized to sum to 1 if needed, but here we assume weights_series matches prices columns
        weights = pd.Series(weights_series, index=prices.columns)
        
        # Daily Individual Contribution
        weighted_returns = returns.multiply(weights, axis=1)
        
        # Cumulative Contribution
        cumulative_ctr = weighted_returns.cumsum()
        
        return cumulative_ctr
