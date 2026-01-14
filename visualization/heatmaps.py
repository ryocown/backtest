"""
Heatmap visualizations for portfolio analysis.

Provides monthly returns grids and correlation matrices with grouping.
"""
from __future__ import annotations

import calendar
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import pandas as pd

from analytics import AnalyticsEngine

# Try to import seaborn for better heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None


def plot_monthly_returns_heatmap(res, strategy_name: str) -> None:
    """
    Plots a heatmap of monthly returns for a specific strategy.
    
    Layout: Rows are years, Columns are months, plus a Total column.
    
    Args:
        res: bt.Result object.
        strategy_name: Name of the strategy to visualize.
    """
    if strategy_name not in res.prices.columns:
        return
        
    prices = res.prices[strategy_name]
    monthly_rets = AnalyticsEngine.get_monthly_returns(prices.to_frame())
    
    # Reformat for heatmap: Rows=Year, Cols=Month
    df = monthly_rets.iloc[:, 0].to_frame(name='ret')
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    pivot_df = df.pivot(index='year', columns='month', values='ret')
    
    # Use month names
    pivot_df.columns = [calendar.month_name[m] for m in pivot_df.columns]
    
    # Calculate yearly total (compounded)
    pivot_df['Total'] = pivot_df.apply(
        lambda row: (1 + row.dropna()).prod() - 1, axis=1
    )
    
    # Calculate monthly average
    avg_row = pivot_df.mean(axis=0)
    avg_row.name = 'Average'
    
    # Add Average row to the top
    pivot_df = pd.concat([pd.DataFrame([avg_row]), pivot_df])
    
    plt.figure(figsize=(12, 8))
    if HAS_SEABORN and sns is not None:
        sns.heatmap(
            pivot_df * 100, 
            annot=True, 
            fmt=".1f", 
            cmap="RdYlGn", 
            center=0, 
            cbar_kws={'label': 'Return (%)'}
        )
    else:
        plt.imshow(pivot_df, cmap='RdYlGn', aspect='auto')
        plt.colorbar(label='Return')
        
    plt.title(f"Monthly Returns Heatmap: {strategy_name}")
    plt.tight_layout()


def plot_grouped_correlation_matrix(
    prices: pd.DataFrame,
    group_map: dict[str, str],
    title: str = "Grouped Correlation Matrix"
) -> None:
    """
    Aggregates asset returns by group and plots their correlation.
    
    Helps identify correlations between Sectors or Asset Classes.
    
    Args:
        prices: Asset price DataFrame.
        group_map: Mapping of ticker -> group name.
        title: Chart title.
    """
    rets = prices.pct_change().dropna()
    
    # Create group returns as equal-weighted averages within each group
    group_rets: dict[str, pd.Series] = {}
    for group in set(group_map.values()):
        tickers = [t for t, g in group_map.items() if g == group and t in rets.columns]
        if tickers:
            group_rets[group] = rets[tickers].mean(axis=1)
            
    df_groups = pd.DataFrame(group_rets)
    corr = df_groups.corr()
    
    plt.figure(figsize=(12, 10))
    if HAS_SEABORN and sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    else:
        plt.imshow(corr, cmap='vlag', aspect='auto')
        plt.colorbar()
        
    plt.title(title)
    plt.tight_layout()
