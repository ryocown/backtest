"""
3D surface visualizations for advanced portfolio analysis.

These plots provide insight into parameter sensitivity and risk asymmetry.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource

from analytics import AnalyticsEngine
from visualization.cursor import make_legend_interactive

# Try to import seaborn for better heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None


def plot_sharpe_robustness_surface(
    res,
    strategy_name: str,
    min_window: int = 20,
    max_window: int = 252,
    step: int = 20,
    title: str = "Sharpe Robustness Surface"
) -> None:
    """
    Plots a 3D surface of the Rolling Sharpe Ratio.
    
    X-axis: Time (normalized)
    Y-axis: Window Size (Lookback period)
    Z-axis: Annualized Sharpe Ratio
    
    In the industry, this is used to identify 'Parameter Islands' or 'Plateaus'.
    If your strategy only performs well at window=126 but fails at 106 or 146, 
    it is likely OVERFIT. A good strategy has a broad 'plateau' of performance.
    
    Args:
        res: bt.Result object.
        strategy_name: Name of the strategy to analyze.
        min_window: Minimum rolling window size.
        max_window: Maximum rolling window size.
        step: Step size between window values.
        title: Chart title.
    """
    if strategy_name not in res.prices.columns:
        print(f"Strategy {strategy_name} not found in results.")
        return

    prices = res.prices[strategy_name]
    windows = np.arange(min_window, max_window + 1, step)
    
    # Calculate Sharpe for each window
    all_sharpes = []
    for w in windows:
        s = AnalyticsEngine.get_rolling_sharpe(prices.to_frame(), window=int(w)).iloc[:, 0]
        all_sharpes.append(s)
    
    # Create DataFrame to align all rolling windows
    df_sharpe_raw = pd.concat(all_sharpes, axis=1, keys=windows)
    
    # SMOOTHING: Resample to Weekly to remove daily noise
    df_sharpe = df_sharpe_raw.resample('W').mean().dropna()
    
    if df_sharpe.empty:
        print("Not enough data to generate smoothed 3D surface.")
        return

    # Prepare data for plotting
    X_dates = df_sharpe.index
    X_num = np.arange(len(X_dates))
    Y_windows = windows
    
    X, Y = np.meshgrid(X_num, Y_windows)
    Z = df_sharpe.values.T  # Shape (len(windows), len(dates))

    # Create a figure with two subplots: 3D Surface and 2D Heatmap
    fig = plt.figure(figsize=(18, 9))
    
    # Subplot 1: 3D Surface
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Add lighting for better depth perception
    ls = LightSource(270, 45)
    rgb = ls.shade(Z, cmap=plt.get_cmap('viridis'), vert_exag=0.1, blend_mode='soft')
    surf = ax1.plot_surface(
        X, Y, Z, facecolors=rgb, edgecolor='none', 
        alpha=0.8, antialiased=True, shade=False
    )
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Annualized Sharpe Ratio')
    
    ax1.set_title(f"3D Robustness Terrain: {strategy_name}", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time (Weeks)", labelpad=12, fontsize=10)
    ax1.set_ylabel("Lookback Window (Days)", labelpad=12, fontsize=10)
    ax1.set_zlabel("Sharpe Ratio", labelpad=12, fontsize=10)
    ax1.view_init(elev=30, azim=-60)
    
    # Subplot 2: 2D Sensitivity Map
    ax2 = fig.add_subplot(122)
    
    if HAS_SEABORN and sns is not None:
        sns.heatmap(
            df_sharpe.T, cmap='RdYlGn', ax=ax2, center=1.0,
            annot=False,
            cbar_kws={'label': 'Sharpe Ratio'},
            xticklabels=12, yticklabels=1
        )
        
        # Cleanup x-axis labels to show real dates
        locs = ax2.get_xticks()
        labels = [
            df_sharpe.index[int(i)].strftime('%Y-%m') 
            for i in locs if i < len(df_sharpe)
        ]
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=9)
    else:
        im = ax2.imshow(Z, cmap='RdYlGn', aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax2, label='Sharpe Ratio')
        ax2.set_yticks(np.arange(len(windows)))
        ax2.set_yticklabels(windows)

    ax2.set_title(f"2D Sensitivity Map: {strategy_name}", fontsize=12)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rolling Window (Days)")
    
    plt.suptitle(f"{title}: Testing Stability across Parameters", fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))


def plot_drawdown_recovery_surface(
    res,
    strategy_name: str,
    title: str = "Drawdown-Recovery Surface"
) -> None:
    """
    Plots a 3D surface showing the relationship between Drawdown Magnitude,
    Duration, and the Required Recovery Return.
    
    X-axis: Magnitude of Drawdown (%)
    Y-axis: Duration of the event (Months)
    Z-axis: Required Recovery Return (+%)
    
    This visualization highlights the 'Asymmetry of Loss'.
    
    Args:
        res: bt.Result object.
        strategy_name: Name of the strategy to analyze.
        title: Chart title.
    """
    if strategy_name not in res.prices.columns:
        return

    prices = res.prices[strategy_name]
    
    # Calculate Drawdowns
    cummax = prices.cummax()
    drawdown = (prices / cummax) - 1.0
    
    # Identify drawdown events
    is_in_drawdown = drawdown < 0
    starts = (is_in_drawdown & (~is_in_drawdown.shift(1).fillna(False))).infer_objects(copy=False)
    ends = ((~is_in_drawdown) & is_in_drawdown.shift(1).fillna(False)).infer_objects(copy=False)
    
    start_dates = prices.index[starts]
    end_dates = prices.index[ends]
    
    # Align starts and ends
    events = []
    for s in start_dates:
        future_ends = end_dates[end_dates > s]
        if not future_ends.empty:
            e = future_ends[0]
            period_prices = prices[s:e]
            mag = (period_prices.min() / period_prices.iloc[0]) - 1.0
            duration_months = (e - s).days / 30.44
            required_recovery = (1.0 / (1.0 + mag)) - 1.0
            
            events.append({
                'magnitude': abs(mag) * 100,
                'duration': duration_months,
                'recovery': required_recovery * 100
            })
    
    if not events:
        print(f"No completed drawdown events found for {strategy_name}")
        return

    df_events = pd.DataFrame(events)
    
    # Create Theoretical Surface
    max_mag = max(df_events['magnitude'].max(), 50)
    max_dur = max(df_events['duration'].max(), 12)
    
    x_surf = np.linspace(0, min(max_mag * 1.2, 90), 50)
    y_surf = np.linspace(0, max_dur * 1.2, 50)
    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
    Z_surf = (100.0 / (100.0 - X_surf)) * 100.0 - 100.0
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ls = LightSource(270, 45)
    rgb = ls.shade(Z_surf, cmap=plt.get_cmap('OrRd'), vert_exag=0.1, blend_mode='soft')
    
    # Plot Surface
    surf = ax.plot_surface(
        X_surf, Y_surf, Z_surf, facecolors=rgb, 
        alpha=0.5, antialiased=True, shade=False
    )
    
    # Plot Historical Events
    ax.scatter(
        df_events['magnitude'], df_events['duration'], df_events['recovery'],
        color='blue', s=50, edgecolors='white', label='Historical Drawdowns'
    )
    
    ax.set_title(f"{title}: {strategy_name}", fontsize=14)
    ax.set_xlabel("Magnitude of Drawdown (%)")
    ax.set_ylabel("Duration to Recovery (Months)")
    ax.set_zlabel("Required Recovery Return (%)")
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Theoretical Recovery Required')
    
    ax.view_init(elev=20, azim=-45)
    plt.tight_layout()


def plot_historical_pe_trend(
    res,
    fundamental_engine,
    prices: Optional[pd.DataFrame] = None,
    title: str = "Historical Trailing P/E Ratio"
) -> None:
    """
    Plots the trend of the weighted harmonic mean P/E ratio for each strategy.
    
    Args:
        res: bt.Result object.
        fundamental_engine: FundamentalEngine instance for EPS data.
        prices: Optional asset price DataFrame.
        title: Chart title.
    """
    try:
        plt.figure(figsize=(14, 7))
        
        for col in res.prices.columns:
            # Get weights over time
            try:
                weights = res.get_security_weights(col)
                if weights.empty:
                    weights = pd.DataFrame(1.0, index=res.prices.index, columns=pd.Index([col]))
            except Exception:
                weights = pd.DataFrame(1.0, index=res.prices.index, columns=pd.Index([col]))

            if prices is None:
                continue
                
            available_tickers = [t for t in weights.columns if t in prices.columns]
            if not available_tickers:
                continue

            # Calculate Portfolio PE series
            pe_series = fundamental_engine.get_portfolio_pe_series(
                weights[available_tickers], prices[available_tickers]
            )
            
            non_nan = pe_series.dropna()
            if len(non_nan) > 1:
                plt.plot(non_nan.index, non_nan, label=col)
                print(f"Plotted P/E trend for {col}: {len(non_nan)} points")

        plt.title(title)
        plt.ylabel("P/E Ratio (Harmonic Mean)")
        plt.xlabel("Date")
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        make_legend_interactive()
    except Exception as e:
        print(f"Historical P/E plot failed: {e}")
