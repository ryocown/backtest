"""
Rolling metric visualizations for time-series analysis.

All functions accept bt.Result objects and produce matplotlib figures.
Legends are pinned to 'upper right' to prevent auto-repositioning.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.analytics import AnalyticsEngine
from src.visualization.cursor import make_legend_interactive

if TYPE_CHECKING:
    import pandas as pd


def _pin_legend(ax, loc: str = 'upper right') -> None:
    """
    Pins the legend to a fixed location so it doesn't move with the cursor.
    
    This MUST be called after the plot is created but before make_legend_interactive.
    """
    leg = ax.get_legend()
    if leg is not None:
        # Remove the auto-positioned legend and create a new one with fixed location
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=loc)


def plot_drawdowns(res, title: str = "Drawdowns") -> None:
    """
    Plots the drawdown (underwater) chart for all strategies in the backtest result.
    
    Args:
        res: bt.Result object with prices DataFrame.
        title: Chart title.
    """
    dd = AnalyticsEngine.get_drawdowns(res.prices)
    ax = dd.plot(figsize=(12, 6), title=title)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    plt.grid(True, alpha=0.3)
    _pin_legend(ax)  # Pin before interactive
    plt.tight_layout()
    make_legend_interactive()


def plot_rolling_volatility(
    res, 
    window: int = 126, 
    title: str = "Rolling Volatility (6-Month)"
) -> None:
    """
    Plots annualized rolling volatility.
    
    Args:
        res: bt.Result object.
        window: Rolling window in trading days (default 126 ≈ 6 months).
        title: Chart title.
    """
    vol = AnalyticsEngine.get_rolling_volatility(res.prices, window=window)
    ax = vol.plot(figsize=(12, 6), title=title)
    ax.set_ylabel("Annualized Volatility")
    plt.grid(True, alpha=0.3)
    _pin_legend(ax)
    plt.tight_layout()
    make_legend_interactive()


def plot_rolling_sharpe(
    res, 
    window: int = 126, 
    risk_free_rate: float = 0.0, 
    title: str = "Rolling Sharpe Ratio (6-Month)"
) -> None:
    """
    Plots the rolling annualized Sharpe ratio.
    
    Args:
        res: bt.Result object.
        window: Rolling window in trading days.
        risk_free_rate: Annual risk-free rate (e.g., 0.04 for 4%).
        title: Chart title.
    """
    sharpe = AnalyticsEngine.get_rolling_sharpe(
        res.prices, window=window, risk_free_rate=risk_free_rate
    )
    ax = sharpe.plot(figsize=(12, 6), title=title)
    ax.set_ylabel("Rolling Sharpe Ratio")
    plt.grid(True, alpha=0.3)
    _pin_legend(ax)
    plt.tight_layout()
    make_legend_interactive()


def plot_rolling_sortino(
    res, 
    window: int = 126, 
    risk_free_rate: float = 0.0, 
    title: str = "Rolling Sortino Ratio (6-Month)"
) -> None:
    """
    Plots the rolling annualized Sortino ratio (downside risk-adjusted).
    
    Args:
        res: bt.Result object.
        window: Rolling window in trading days.
        risk_free_rate: Annual risk-free rate.
        title: Chart title.
    """
    sortino = AnalyticsEngine.get_rolling_sortino(
        res.prices, window=window, risk_free_rate=risk_free_rate
    )
    ax = sortino.plot(figsize=(12, 6), title=title)
    ax.set_ylabel("Rolling Sortino Ratio")
    plt.grid(True, alpha=0.3)
    _pin_legend(ax)
    plt.tight_layout()
    make_legend_interactive()


def plot_rolling_beta(
    res, 
    benchmark: str = 'SPY', 
    window: int = 126, 
    title: str = "Rolling Beta (6-Month)"
) -> None:
    """
    Plots the rolling beta of all strategies relative to a benchmark.
    
    Args:
        res: bt.Result object.
        benchmark: Benchmark ticker symbol.
        window: Rolling window in trading days.
        title: Chart title.
    """
    if benchmark not in res.prices.columns:
        print(f"Benchmark {benchmark} not found for rolling beta calculation.")
        return
        
    benchmark_prices = res.prices[[benchmark]]
    strategies = [col for col in res.prices.columns if col != benchmark]
    
    _, betas = AnalyticsEngine.get_rolling_alpha_beta(
        res.prices[strategies], benchmark_prices, window=window
    )
    
    ax = betas.plot(figsize=(12, 6), title=f"{title} vs {benchmark}")
    ax.set_ylabel("Beta")
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    _pin_legend(ax)
    plt.tight_layout()
    make_legend_interactive()


def plot_rolling_alpha(
    res, 
    benchmark: str = 'SPY', 
    window: int = 126, 
    title: str = "Rolling Annualized Alpha (6-Month)"
) -> None:
    """
    Plots the rolling annualized alpha of all strategies relative to a benchmark.
    
    Args:
        res: bt.Result object.
        benchmark: Benchmark ticker symbol.
        window: Rolling window in trading days.
        title: Chart title.
    """
    if benchmark not in res.prices.columns:
        return
        
    benchmark_prices = res.prices[[benchmark]]
    strategies = [col for col in res.prices.columns if col != benchmark]
    
    alphas, _ = AnalyticsEngine.get_rolling_alpha_beta(
        res.prices[strategies], benchmark_prices, window=window
    )
    
    ax = alphas.plot(figsize=(12, 6), title=f"{title} vs {benchmark}")
    ax.set_ylabel("Annualized Alpha")
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    _pin_legend(ax)
    plt.tight_layout()
    make_legend_interactive()


def plot_rolling_alpha_beta(
    strat_prices: "pd.DataFrame",
    bench_prices: "pd.DataFrame",
    window: int = 126,
    title_prefix: str = ""
) -> None:
    """
    Plots rolling Alpha and Beta in two subplots.
    
    Args:
        strat_prices: DataFrame of strategy prices.
        bench_prices: DataFrame of benchmark prices (single column).
        window: Rolling window in trading days.
        title_prefix: Prefix for subplot titles.
    """
    alphas, betas = AnalyticsEngine.get_rolling_alpha_beta(
        strat_prices, bench_prices, window=window
    )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    alphas.plot(ax=ax1)
    ax1.set_title(f"{title_prefix} Rolling Annualized Alpha")
    ax1.set_ylabel("Alpha")
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    _pin_legend(ax1)
    
    betas.plot(ax=ax2)
    ax2.set_title(f"{title_prefix} Rolling Beta")
    ax2.set_ylabel("Beta")
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    _pin_legend(ax2)
    
    plt.tight_layout()
    make_legend_interactive(fig)


def plot_rolling_info_ratio(
    res, 
    benchmark: str = 'SPY', 
    window: int = 126, 
    title: str = "Rolling Information Ratio (6-Month)"
) -> None:
    """
    Plots the rolling annualized Information Ratio.
    
    Measures risk-adjusted relative return: (Rp - Rb) / Tracking Error.
    
    Args:
        res: bt.Result object.
        benchmark: Benchmark ticker symbol.
        window: Rolling window in trading days.
        title: Chart title.
    """
    if benchmark not in res.prices.columns:
        return
        
    bench_prices = res.prices[[benchmark]]
    strategies = [col for col in res.prices.columns if col != benchmark]
    
    ir = AnalyticsEngine.get_rolling_info_ratio(
        res.prices[strategies], bench_prices, window=window
    )
    
    ax = ir.plot(figsize=(12, 6), title=f"{title} vs {benchmark}")
    ax.set_ylabel("Rolling Information Ratio")
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    _pin_legend(ax)
    plt.tight_layout()
    make_legend_interactive()
