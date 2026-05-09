"""
Portfolio analytics calculations for the Institutional Backtester.

Provides professional-grade risk and return metrics:
- Drawdowns and recovery analysis
- Rolling risk-adjusted returns (Sharpe, Sortino, Information ratio)
- CAPM metrics (Alpha, Beta)
- Risk contribution (MCTR)
- Return attribution
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, Series


class AnalyticsEngine:
    """
    Handles professional portfolio analytics and risk calculations.
    
    All methods are static and operate on price DataFrames.
    """

    @staticmethod
    def get_drawdowns(prices: DataFrame) -> DataFrame:
        """
        Calculates drawdown series for all columns.
        
        Args:
            prices: Price DataFrame with datetime index.
            
        Returns:
            DataFrame of drawdowns (negative values).
        """
        return prices / prices.cummax() - 1

    @staticmethod
    def get_monthly_returns(prices: DataFrame) -> DataFrame:
        """
        Calculates compounded monthly returns.
        
        Args:
            prices: Price DataFrame with datetime index.
            
        Returns:
            DataFrame of monthly returns indexed by month-end.
        """
        daily_rets = prices.pct_change().dropna()
        return daily_rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    @staticmethod
    def get_rolling_volatility(prices: DataFrame, window: int = 126) -> DataFrame:
        """
        Calculates annualized rolling volatility.
        
        Args:
            prices: Price DataFrame.
            window: Rolling window in trading days (default 126 ≈ 6 months).
            
        Returns:
            DataFrame of annualized volatility.
        """
        daily_rets = prices.pct_change().dropna()
        return daily_rets.rolling(window=window).std() * np.sqrt(252)

    @staticmethod
    def get_rolling_sharpe(
        prices: DataFrame,
        window: int = 126,
        risk_free_rate: float = 0.0
    ) -> DataFrame:
        """
        Calculates annualized rolling Sharpe ratio.
        
        Args:
            prices: Price DataFrame.
            window: Rolling window in trading days.
            risk_free_rate: Annual risk-free rate (e.g., 0.04 for 4%).
            
        Returns:
            DataFrame of rolling Sharpe ratios.
        """
        returns = prices.pct_change().dropna()
        excess_returns = returns - (risk_free_rate / 252)
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        return (rolling_mean / rolling_std) * np.sqrt(252)

    @staticmethod
    def get_rolling_sortino(
        prices: DataFrame,
        window: int = 126,
        risk_free_rate: float = 0.0
    ) -> DataFrame:
        """
        Calculates annualized rolling Sortino ratio.
        
        Only considers downside deviation (negative returns).
        
        Args:
            prices: Price DataFrame.
            window: Rolling window in trading days.
            risk_free_rate: Annual risk-free rate.
            
        Returns:
            DataFrame of rolling Sortino ratios.
        """
        returns = prices.pct_change().dropna()
        excess_returns = returns - (risk_free_rate / 252)
        
        def downside_std(x: np.ndarray) -> float:
            downside_diff = np.minimum(x, 0)
            return np.sqrt(np.mean(downside_diff ** 2))

        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_downside_std = returns.rolling(window=window).apply(downside_std)
        
        return (rolling_mean / rolling_downside_std.replace(0, np.nan)) * np.sqrt(252)

    @staticmethod
    def get_capture_ratios(strat_prices: Series, bench_prices: Series) -> dict[str, float]:
        """
        Calculates Upside and Downside Capture ratios.
        
        Institutional metric for evaluating asymmetric performance.
        
        Args:
            strat_prices: Strategy price Series.
            bench_prices: Benchmark price Series.
            
        Returns:
            Dict with 'upside_capture' and 'downside_capture' ratios.
        """
        strat_rets = strat_prices.pct_change().dropna()
        bench_rets = bench_prices.pct_change().dropna()
        
        # Align returns
        common_idx = strat_rets.index.intersection(bench_rets.index)
        strat_rets = strat_rets.loc[common_idx]
        bench_rets = bench_rets.loc[common_idx]
        
        # Handle DataFrame inputs
        if isinstance(strat_rets, pd.DataFrame):
            strat_rets = strat_rets.iloc[:, 0]
        if isinstance(bench_rets, pd.DataFrame):
            bench_rets = bench_rets.iloc[:, 0]
        
        up_months = bench_rets > 0
        down_months = bench_rets <= 0
        
        upside_capture = (
            strat_rets[up_months].mean() / bench_rets[up_months].mean()
            if up_months.any() else 0.0
        )
        downside_capture = (
            strat_rets[down_months].mean() / bench_rets[down_months].mean()
            if down_months.any() else 0.0
        )
        
        return {
            'upside_capture': upside_capture,
            'downside_capture': downside_capture
        }

    @staticmethod
    def get_drawdown_stats(prices: DataFrame) -> DataFrame:
        """
        Analyzes drawdown statistics: max and current drawdown.
        
        Args:
            prices: Price DataFrame.
            
        Returns:
            DataFrame with max/current drawdown per column.
        """
        dd = prices / prices.cummax() - 1
        stats = []
        
        for col in dd.columns:
            series = dd[col]
            stats.append({
                'Strategy': col,
                'Max Drawdown': series.min(),
                'Current Drawdown': series.iloc[-1]
            })
        return pd.DataFrame(stats).set_index('Strategy')

    @staticmethod
    def get_rolling_info_ratio(
        prices: DataFrame,
        benchmark_prices: DataFrame,
        window: int = 126
    ) -> DataFrame:
        """
        Calculates rolling Information Ratio: (Rp - Rb) / Tracking Error.
        
        Measures the consistency of active management.
        
        Args:
            prices: Strategy price DataFrame.
            benchmark_prices: Benchmark price DataFrame (single column).
            window: Rolling window in trading days.
            
        Returns:
            DataFrame of rolling Information Ratios.
        """
        returns = prices.pct_change().dropna()
        bench_rets = benchmark_prices.pct_change().dropna()
        
        # Align
        common_idx = returns.index.intersection(bench_rets.index)
        returns = returns.loc[common_idx]
        bench_rets = bench_rets.loc[common_idx]
        
        bench_rets_series = (
            bench_rets.iloc[:, 0] if isinstance(bench_rets, pd.DataFrame)
            else bench_rets
        )
        active_returns = returns.subtract(bench_rets_series, axis=0)
        tracking_error = active_returns.rolling(window=window).std()
        
        return (active_returns.rolling(window=window).mean() / tracking_error) * np.sqrt(252)

    @staticmethod
    def get_risk_contribution(prices: DataFrame, weights_series: Series) -> Series:
        """
        Calculates Marginal Contribution to Risk (MCTR) and Percentage CTR.
        
        Identifies which asset is 'driving' portfolio volatility.
        
        Args:
            prices: Asset price DataFrame.
            weights_series: Portfolio weights as Series.
            
        Returns:
            Series of percentage contribution to risk per asset.
        """
        returns = prices.pct_change().dropna()
        cov = returns.cov() * 252  # Annualized covariance
        
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
    def get_rolling_alpha_beta(
        prices: DataFrame,
        benchmark_prices: DataFrame,
        window: int = 126
    ) -> tuple[DataFrame, DataFrame]:
        """
        Calculates rolling CAPM Alpha and Beta relative to a benchmark.
        
        Beta: Sensitivity to the benchmark.
        Alpha: Excess return not explained by the benchmark (annualized).
        
        Args:
            prices: Strategy price DataFrame.
            benchmark_prices: Benchmark price DataFrame (single column).
            window: Rolling window in trading days.
            
        Returns:
            Tuple of (alphas DataFrame, betas DataFrame).
        """
        returns = prices.pct_change().dropna()
        bench_rets = benchmark_prices.pct_change().dropna()
        
        # Align
        common_idx = returns.index.intersection(bench_rets.index)
        returns = returns.loc[common_idx]
        bench_rets = bench_rets.loc[common_idx]
        
        y = bench_rets.iloc[:, 0] if isinstance(bench_rets, pd.DataFrame) else bench_rets
        
        def calc_beta(x: pd.Series) -> float:
            cov = np.cov(x, y.loc[x.index])[0, 1]
            var = np.var(y.loc[x.index])
            return cov / var if var != 0 else np.nan

        betas = returns.rolling(window=window).apply(calc_beta)
        
        # Alpha = Rp - Beta * Rb (simplified, assumes Rf=0)
        alphas = returns.subtract(betas.multiply(y, axis=0), axis=0)
        annualized_alphas = alphas.rolling(window=window).mean() * 252
        
        return annualized_alphas, betas

    @staticmethod
    def get_return_attribution(prices: DataFrame, weights_series: Series) -> DataFrame:
        """
        Calculates Contribution to Return (CTR) for each asset.
        
        Cumulative CTR = Sum(weight_i * return_i) over time.
        
        Args:
            prices: Asset price DataFrame.
            weights_series: Portfolio weights as Series.
            
        Returns:
            DataFrame of cumulative contribution to return per asset.
        """
        returns = prices.pct_change().fillna(0)
        weights = pd.Series(weights_series, index=prices.columns)
        
        # Daily Individual Contribution
        weighted_returns = returns.multiply(weights, axis=1)
        
        # Cumulative Contribution
        return weighted_returns.cumsum()

    @staticmethod
    def get_robustness_metrics(
        prices: DataFrame,
        strategy_name: str,
        min_window: int = 20,
        max_window: int = 252,
        step: int = 20,
        sharpe_threshold: float = 1.0
    ) -> dict:
        """
        Calculates quantitative robustness and sensitivity metrics.
        
        Metrics computed:
        - Sharpe Stability Score: Mean(Sharpe) / StdDev(Sharpe) across all windows
        - Window Sensitivity: Avg |ΔSharpe| when window changes by step
        - Plateau Width: % of window range where Sharpe > threshold
        - Coefficient of Variation: StdDev / Mean across entire surface
        - Temporal Consistency: Avg correlation of Sharpe between adjacent windows
        
        Args:
            prices: Price DataFrame with strategy column.
            strategy_name: Name of the strategy column.
            min_window: Minimum rolling window size.
            max_window: Maximum rolling window size.
            step: Step between windows.
            sharpe_threshold: Threshold for plateau calculation.
            
        Returns:
            Dict with metrics and qualitative ratings.
        """
        if strategy_name not in prices.columns:
            return {'error': f'Strategy {strategy_name} not found'}
        
        strategy_prices = prices[[strategy_name]]
        windows = np.arange(min_window, max_window + 1, step)
        
        # Calculate Sharpe for each window
        all_sharpes = []
        for w in windows:
            s = AnalyticsEngine.get_rolling_sharpe(strategy_prices, window=int(w)).iloc[:, 0]
            all_sharpes.append(s)
        
        df_sharpe = pd.concat(all_sharpes, axis=1, keys=windows).dropna()
        
        if df_sharpe.empty or len(df_sharpe) < 10:
            return {'error': 'Insufficient data for robustness analysis'}
        
        # Flatten all Sharpe values for overall stats
        all_values = df_sharpe.values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        
        if len(all_values) == 0:
            return {'error': 'No valid Sharpe values'}
        
        # 1. Sharpe Stability Score: Mean / StdDev (across all windows at each time point)
        mean_sharpe = np.nanmean(all_values)
        std_sharpe = np.nanstd(all_values)
        stability_score = mean_sharpe / std_sharpe if std_sharpe > 0 else float('inf')
        
        # 2. Window Sensitivity: Average absolute change in Sharpe when window shifts
        window_means = df_sharpe.mean()  # Mean Sharpe for each window
        window_diffs = np.abs(np.diff(window_means.values))
        window_sensitivity = np.mean(window_diffs) if len(window_diffs) > 0 else 0.0
        
        # 3. Plateau Width: % of windows where mean Sharpe > threshold
        plateau_count = (window_means > sharpe_threshold).sum()
        plateau_width = (plateau_count / len(windows)) * 100
        
        # 4. Coefficient of Variation
        coef_of_variation = std_sharpe / mean_sharpe if mean_sharpe != 0 else float('inf')
        
        # 5. Temporal Consistency: Correlation between adjacent window columns
        correlations = []
        window_cols = list(df_sharpe.columns)
        for i in range(len(window_cols) - 1):
            corr = df_sharpe[window_cols[i]].corr(df_sharpe[window_cols[i + 1]])
            if not np.isnan(corr):
                correlations.append(corr)
        temporal_consistency = np.mean(correlations) if correlations else 0.0
        
        # Qualitative ratings
        def rate_stability(score: float) -> str:
            if score >= 2.0:
                return "Excellent"
            elif score >= 1.5:
                return "Good"
            elif score >= 1.0:
                return "Moderate"
            else:
                return "Poor"
        
        def rate_sensitivity(sens: float) -> str:
            if sens < 0.05:
                return "Low"
            elif sens < 0.15:
                return "Moderate"
            else:
                return "High"
        
        def rate_plateau(pct: float) -> str:
            if pct >= 80:
                return "Excellent"
            elif pct >= 60:
                return "Good"
            elif pct >= 40:
                return "Moderate"
            else:
                return "Narrow"
        
        def rate_cov(cov: float) -> str:
            if cov < 0.3:
                return "Excellent"
            elif cov < 0.5:
                return "Good"
            elif cov < 0.8:
                return "Moderate"
            else:
                return "High"
        
        return {
            'strategy': strategy_name,
            'mean_sharpe': mean_sharpe,
            'std_sharpe': std_sharpe,
            'stability_score': stability_score,
            'stability_rating': rate_stability(stability_score),
            'window_sensitivity': window_sensitivity,
            'sensitivity_rating': rate_sensitivity(window_sensitivity),
            'plateau_width_pct': plateau_width,
            'plateau_rating': rate_plateau(plateau_width),
            'coef_of_variation': coef_of_variation,
            'cov_rating': rate_cov(coef_of_variation),
            'temporal_consistency': temporal_consistency,
            'windows_tested': len(windows),
            'min_window': min_window,
            'max_window': max_window
        }
