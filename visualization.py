import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.text as mtext
import matplotlib.dates as mdates
from analytics import AnalyticsEngine
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection

# Try to import seaborn for better style, but fallback if missing
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="whitegrid")
except ImportError:
    HAS_SEABORN = False
    sns = None

# 
#     Adds a vertical crosshair and tooltip that follows the mouse across ALL subplots.
#     Industry standard for financial time-series visualization.
#     
def add_financial_cursor(fig=None):
    if fig is None:
        fig = plt.gcf()
    
    # Ensure the cursor is persisted by attaching it to the figure
    if not hasattr(fig, '_financial_cursor'):
        fig._financial_cursor = FinancialCursor(fig)
        # print(f"DEBUG: Attached FinancialCursor to figure {id(fig)}")
    
    return fig._financial_cursor

class FinancialCursor:
    def __init__(self, fig):
        self.fig = fig
        self.vlines = []
        self.annotations = []
        self.axes_list = []
        
        # Initialize for all compatible axes (those with time-series-like x-limits)
        for ax in self.fig.axes:
            # Skip 3D axes as they don't support simple vertical lines this way
            if ax.name == '3d':
                continue
            
            line = ax.axvline(x=np.nan, color='gray', linestyle='--', linewidth=0.8, alpha=0.8, zorder=10)
            self.vlines.append(line)
            
            # Annotation box (Tooltip)
            anno = ax.annotate(
                '', xy=(0, 0), xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                fontsize=8, zorder=11, fontfamily='monospace'
            )
            anno.set_visible(False)
            self.annotations.append(anno)
            self.axes_list.append(ax)
            
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fig.canvas.mpl_connect("axes_leave_event", self.on_leave)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        # Allow clearing or pinning cursors (optional future feature)
        pass

    def on_mouse_move(self, event):
        if not event.inaxes:
            return
        
        target_x = event.xdata
        
        # Find which ax we are in to get date formatting
        current_ax = event.inaxes
        try:
            date_str = pd.to_datetime(target_x, unit='D').strftime('%Y-%m-%d')
        except:
            date_str = f"{target_x:.2f}"

        for i, ax in enumerate(self.axes_list):
            self.vlines[i].set_xdata([target_x])
            
            # Find the closest data point in each axes to show values
            summary_text = f"[{date_str}]\n"
            found_val = False
            
            for line in ax.get_lines():
                if line.get_label().startswith('_'): # Skip internal lines
                    continue
                
                # Get data
                lx, ly = line.get_data()
                if len(lx) == 0: continue
                
                # Ensure lx is numeric for comparison with target_x (which is usually float days)
                try:
                    lx_numeric = mdates.date2num(lx)
                except:
                    lx_numeric = lx

                # Find index of closest x
                idx = np.searchsorted(lx_numeric, target_x)
                if idx >= len(lx_numeric): idx = len(lx_numeric) - 1
                
                if abs(lx_numeric[idx] - target_x) < 5: # Threshold for visibility
                    val = ly[idx]
                    label = line.get_label()
                    summary_text += f"{label[:10]}: {val:.2f}\n"
                    found_val = True
            
            if found_val:
                self.annotations[i].set_text(summary_text.strip())
                self.annotations[i].xy = (target_x, ax.get_ylim()[1])
                self.annotations[i].set_visible(True)
            else:
                self.annotations[i].set_visible(False)
        
        self.fig.canvas.draw_idle()

    def on_leave(self, event):
        for line in self.vlines:
            line.set_xdata([np.nan])
        for anno in self.annotations:
            anno.set_visible(False)
        self.fig.canvas.draw_idle()

# 
#     Makes all legends in the figure interactive.
#     Clicking on a legend label will toggle the visibility of the corresponding plot element.
#     
def make_legend_interactive(fig=None):
    if fig is None:
        fig = plt.gcf()
    
    # 1. Attach the Financial Cursor as well
    add_financial_cursor(fig)
    
    # 2. Extract all legends
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    
    # Map labels to lines
    legend_map = {}
    for leg in legends:
        if leg is None: continue
        
        for legline, origline in zip(leg.get_lines(), leg.axes.get_lines()):
            legline.set_picker(True) # Make legend line clickable
            legend_map[legline] = origline
            
        # Also handle text click
        for legtext, origline in zip(leg.get_texts(), leg.axes.get_lines()):
            legtext.set_picker(True)
            legend_map[legtext] = origline

    def on_pick(event):
        # find the original line mapped to the legend item
        leg_item = event.artist
        if leg_item not in legend_map:
            return
            
        origline = legend_map[leg_item]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        
        # Change alpha of legend item to show status
        if visible:
            leg_item.set_alpha(1.0)
            if hasattr(leg_item, 'set_fontweight'):
                leg_item.set_fontweight('normal')
        else:
            leg_item.set_alpha(0.2)
            if hasattr(leg_item, 'set_fontweight'):
                leg_item.set_fontweight('light')
        
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)

# 
#     Plots the drawdown (underwater) chart for all strategies in the backtest result.
#     
def plot_drawdowns(res, title="Drawdowns"):
    dd = AnalyticsEngine.get_drawdowns(res.prices)
    ax = dd.plot(figsize=(12, 6), title=title)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

# 
#     Plots a heatmap of monthly returns for a specific strategy.
#     Row: Year, Column: Month
#     
def plot_monthly_returns_heatmap(res, strategy_name):
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
    import calendar
    pivot_df.columns = [calendar.month_name[m] for m in pivot_df.columns]
    
    # Calculate yearly total (compounded)
    pivot_df['Total'] = pivot_df.apply(lambda row: (1 + row.dropna()).prod() - 1, axis=1)
    
    # Calculate monthly average
    avg_row = pivot_df.mean(axis=0)
    avg_row.name = 'Average'
    
    # Add Average row to the top
    pivot_df = pd.concat([pd.DataFrame([avg_row]), pivot_df])
    
    plt.figure(figsize=(12, 8))
    if HAS_SEABORN and sns is not None:
        sns.heatmap(pivot_df * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Return (%)'})
    else:
        plt.imshow(pivot_df, cmap='RdYlGn', aspect='auto')
        plt.colorbar(label='Return')
        
    plt.title(f"Monthly Returns Heatmap: {strategy_name}")
    plt.tight_layout()

# 
#     Plots annualized rolling volatility.
#     Default window=126 (approx 6 months).
#     
def plot_rolling_volatility(res, window=126, title="Rolling Volatility (6-Month)"):
    vol = AnalyticsEngine.get_rolling_volatility(res.prices, window=window)
    ax = vol.plot(figsize=(12, 6), title=title)
    ax.set_ylabel("Annualized Volatility")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

# 
#     Plots the distribution (histogram) of daily returns with median stats.
#     
def plot_return_distribution(res, title="Daily Return Distribution"):
    rets = res.prices.pct_change().dropna()
    
    plt.figure(figsize=(12, 6))
    if HAS_SEABORN and sns is not None:
        for col in rets.columns:
            sns.kdeplot(rets[col], fill=True, label=col, alpha=0.3)
    else:
        for col in rets.columns:
            plt.hist(rets[col], bins=50, alpha=0.5, label=col)
            
    plt.axvline(0, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    make_legend_interactive()

# 
#     Plots the rolling beta of all strategies relative to a benchmark.
#     
def plot_rolling_beta(res, benchmark='SPY', window=126, title="Rolling Beta (6-Month)"):
    if benchmark not in res.prices.columns:
        print(f"Benchmark {benchmark} not found for rolling beta calculation.")
        return
        
    benchmark_prices = res.prices[[benchmark]]
    strategies = [col for col in res.prices.columns if col != benchmark]
    
    _, betas = AnalyticsEngine.get_rolling_alpha_beta(res.prices[strategies], benchmark_prices, window=window)
    
    ax = betas.plot(figsize=(12, 6), title=f"{title} vs {benchmark}")
    ax.set_ylabel("Beta")
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

# 
#     Plots the rolling annualized alpha of all strategies relative to a benchmark.
#     Alpha = (Rp - Beta * Rm) * 252 (Simplified, assuming Rf=0 for visualization trend)
#     
def plot_rolling_alpha(res, benchmark='SPY', window=126, title="Rolling Annualized Alpha (6-Month)"):
    if benchmark not in res.prices.columns:
        return
        
    benchmark_prices = res.prices[[benchmark]]
    strategies = [col for col in res.prices.columns if col != benchmark]
    
    alphas, _ = AnalyticsEngine.get_rolling_alpha_beta(res.prices[strategies], benchmark_prices, window=window)
    
    ax = alphas.plot(figsize=(12, 6), title=f"{title} vs {benchmark}")
    ax.set_ylabel("Annualized Alpha")
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

# 
#     Plots the rolling annualized Sharpe ratio.
#     
def plot_rolling_sharpe(res, window=126, risk_free_rate=0.0, title="Rolling Sharpe Ratio (6-Month)"):
    sharpe = AnalyticsEngine.get_rolling_sharpe(res.prices, window=window, risk_free_rate=risk_free_rate)
    ax = sharpe.plot(figsize=(12, 6), title=title)
    ax.set_ylabel("Rolling Sharpe Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

# 
#     Plots the rolling annualized Sortino ratio (Downside risk-adjusted).
#     
def plot_rolling_sortino(res, window=126, risk_free_rate=0.0, title="Rolling Sortino Ratio (6-Month)"):
    sortino = AnalyticsEngine.get_rolling_sortino(res.prices, window=window, risk_free_rate=risk_free_rate)
    ax = sortino.plot(figsize=(12, 6), title=title)
    ax.set_ylabel("Rolling Sortino Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

# 
#     Plots bar chart of capture ratios relative to a benchmark.
#     Institutions use this to see 'Beta-up' vs 'Beta-down'.
#     
def plot_upside_downside_capture(res, benchmark='SPY', title="Upside/Downside Capture Ratios"):
    if benchmark not in res.prices.columns:
        return
        
    bench_prices = res.prices[[benchmark]]
    strategies = [col for col in res.prices.columns if col != benchmark]
    
    captures = []
    for strat in strategies:
        cap = AnalyticsEngine.get_capture_ratios(res.prices[strat], res.prices[benchmark])
        captures.append({
            'Strategy': strat,
            'Upside Capture': cap['upside_capture'],
            'Downside Capture': cap['downside_capture']
        })
    
    df = pd.DataFrame(captures).set_index('Strategy')
    ax = df.plot(kind='bar', figsize=(10, 6), title=f"{title} (Benchmark: {benchmark})")
    ax.set_ylabel("Capture Ratio")
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()

# 
#     Plots a heatmap of return correlations.
#     Professionals use this to identify hidden concentration/overlap.
#     
def plot_correlation_matrix(res, title="Strategy Correlation Matrix"):
    rets = res.prices.pct_change().dropna()
    corr = rets.corr()
    
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    else:
        plt.imshow(corr, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        
    plt.title(title)
    plt.tight_layout()

# 
#     Plots the rolling annualized Information Ratio.
#     Measures risk-adjusted relative return.
#     
def plot_rolling_info_ratio(res, benchmark='SPY', window=126, title="Rolling Information Ratio (6-Month)"):
    if benchmark not in res.prices.columns:
        return
        
    bench_prices = res.prices[[benchmark]]
    strategies = [col for col in res.prices.columns if col != benchmark]
    
    ir = AnalyticsEngine.get_rolling_info_ratio(res.prices[strategies], bench_prices, window=window)
    
    ax = ir.plot(figsize=(12, 6), title=f"{title} vs {benchmark}")
    ax.set_ylabel("Rolling Information Ratio")
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

# 
#     Plots the Percentage Contribution to Risk for each asset (or group) in a strategy.
#     Institutions use this to see what is 'driving' the volatility.
#     
def plot_risk_contribution(res, strategy_name, prices=None, group_map=None, title="Risk Contribution (MCTR)"):
    if strategy_name not in res.prices.columns:
        return
        
    weights = res.get_security_weights(strategy_name).iloc[-1]
    constituents = weights[weights > 0]
    
    if prices is None:
        return # Need price data to calc covariance
        
    # Align prices with constituents
    valid_tickers = [t for t in constituents.index if t in prices.columns]
    if not valid_tickers: return
    
    subset_prices = prices[valid_tickers]
    pct_ctr = AnalyticsEngine.get_risk_contribution(subset_prices, constituents[valid_tickers])
    
    if group_map:
        # Aggregate by group
        group_map_valid = {t: group_map.get(t, 'Other') for t in pct_ctr.index}
        pct_ctr = pct_ctr.groupby(group_map_valid).sum()
        
    plt.figure(figsize=(10, 6))
    # Using simplest form to bypass stub mismatch
    sorted_ctr = pct_ctr.sort_values() 
    sorted_ctr.plot(kind='barh', color='#e74c3c', alpha=0.8)
    plt.xlabel("Percentage of Total Risk")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

# 
#     Aggregates asset returns by group and plots their correlation.
#     Helps identify correlations between Sectors or Asset Classes.
#     
def plot_grouped_correlation_matrix(prices, group_map, title="Grouped Correlation Matrix"):
    rets = prices.pct_change().dropna()
    
    # Map columns to groups
    # Create group returns as weighted averages (assuming equal weight within group for simplicity)
    group_rets = {}
    for group in set(group_map.values()):
        tickers = [t for t, g in group_map.items() if g == group and t in rets.columns]
        if tickers:
            group_rets[group] = rets[tickers].mean(axis=1)
            
    df_groups = pd.DataFrame(group_rets)
    corr = df_groups.corr()
    
    plt.figure(figsize=(12, 10))
    if HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    else:
        plt.imshow(corr, cmap='vlag', aspect='auto')
        plt.colorbar()
        
    plt.title(title)
    plt.tight_layout()

# 
#     Plots rolling Alpha and Beta in two subplots.
#     
def plot_rolling_alpha_beta(strat_prices, bench_prices, window=126, title_prefix=""):
    alphas, betas = AnalyticsEngine.get_rolling_alpha_beta(strat_prices, bench_prices, window=window)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    alphas.plot(ax=ax1)
    ax1.set_title(f"{title_prefix} Rolling Annualized Alpha")
    ax1.set_ylabel("Alpha")
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    betas.plot(ax=ax2)
    ax2.set_title(f"{title_prefix} Rolling Beta")
    ax2.set_ylabel("Beta")
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    make_legend_interactive(fig)

# 
#     Rethought Performance Attribution Dashboard.
#     1. Grouped Cumulative Attribution (if group_map provided)
#     2. Total Contribution bar chart (Individual Assets)
#     
def plot_return_attribution(prices, weights_series, group_map=None, title=""):
    cum_ctr = AnalyticsEngine.get_return_attribution(prices, weights_series)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # 1. Plot Individual or Grouped Trends
    if group_map:
        # Group by the provided map
        group_map_valid = {t: group_map.get(t, 'Other') for t in cum_ctr.columns}
        grouped_cum_ctr = cum_ctr.T.groupby(group_map_valid).sum().T
        grouped_cum_ctr.plot(ax=ax1, title=f"Cumulative Return Attribution (by Group) - {title}")
    else:
        cum_ctr.plot(ax=ax1, title=f"Cumulative Return Attribution (by Asset) - {title}")
        
    ax1.set_ylabel("Cumulative Return %")
    ax1.grid(True, alpha=0.3)
    
    # 2. Final Total Attribution (Bar Chart)
    total_ctr = cum_ctr.iloc[-1]
    total_ctr.sort_values().plot(kind='barh', ax=ax2, color='#2ecc71', alpha=0.8, title="Total Contribution by Asset")
    ax2.set_xlabel("Return Contribution (%)")
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    make_legend_interactive(fig)

def plot_sharpe_robustness_surface(res, strategy_name, min_window=20, max_window=252, step=20, title="Sharpe Robustness Surface"):
    """
    Plots a 3D surface of the Rolling Sharpe Ratio.
    X-axis: Time (normalized)
    Y-axis: Window Size (Lookback period)
    Z-axis: Annualized Sharpe Ratio
    
    In the industry, this is used to identify 'Parameter Islands' or 'Plateaus'.
    If your strategy only performs well at window=126 but fails at 106 or 146, it is likely OVERFIT.
    A good strategy has a broad 'plateau' of performance.
    """
    if strategy_name not in res.prices.columns:
        print(f"Strategy {strategy_name} not found in results.")
        return

    prices = res.prices[strategy_name]
    windows = np.arange(min_window, max_window + 1, step)
    
    # Calculate Sharpe for each window
    # We'll use AnalyticsEngine.get_rolling_sharpe
    all_sharpes = []
    for w in windows:
        s = AnalyticsEngine.get_rolling_sharpe(prices.to_frame(), window=int(w)).iloc[:, 0]
        all_sharpes.append(s)
    
    # Create DataFrame to align all rolling windows
    df_sharpe_raw = pd.concat(all_sharpes, axis=1, keys=windows)
    
    # SMOOTHING for Human Brain: Resample to Weekly to remove daily noise
    # This makes the 3D surface look like "hills" rather than "static"
    df_sharpe = df_sharpe_raw.resample('W').mean().dropna()
    
    if df_sharpe.empty:
        print("Not enough data to generate smoothed 3D surface.")
        return

    # Prepare data for plotting
    X_dates = df_sharpe.index
    X_num = np.arange(len(X_dates))
    Y_windows = windows
    
    X, Y = np.meshgrid(X_num, Y_windows)
    Z = df_sharpe.values.T # Shape (len(windows), len(dates))

    # Create a figure with two subplots: 3D Surface and 2D Heatmap
    fig = plt.figure(figsize=(18, 9))
    
    # Subplot 1: 3D Surface (The "WOW" factor)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Add lighting for better depth perception
    from matplotlib.colors import LightSource
    ls = LightSource(270, 45)
    # Compute shaded colors
    rgb = ls.shade(Z, cmap=plt.get_cmap('viridis'), vert_exag=0.1, blend_mode='soft')
    surf = ax1.plot_surface(X, Y, Z, facecolors=rgb, edgecolor='none', alpha=0.8, antialiased=True, shade=False)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Annualized Sharpe Ratio')
    
    # Labeling 3D
    ax1.set_title(f"3D Robustness Terrain: {strategy_name}", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time (Weeks)", labelpad=12, fontsize=10)
    ax1.set_ylabel("Lookback Window (Days)", labelpad=12, fontsize=10)
    ax1.set_zlabel("Sharpe Ratio", labelpad=12, fontsize=10)
    ax1.view_init(elev=30, azim=-60)
    
    # Subplot 2: 2D Sensitivity Map (The "Human Brain" factor)
    # This is what professionals actually use to find "Parameter Plateaus"
    ax2 = fig.add_subplot(122)
    
    if HAS_SEABORN and sns is not None:
        # We use every 4th date for x-labels to avoid crowding
        sns.heatmap(df_sharpe.T, cmap='RdYlGn', ax=ax2, center=1.0, 
                    annot=False, # Too crowded for annotation
                    cbar_kws={'label': 'Sharpe Ratio'},
                    xticklabels=12, yticklabels=1) # Approx quarterly labels
        
        # Cleanup x-axis labels to show real dates
        locs = ax2.get_xticks()
        labels = [df_sharpe.index[int(i)].strftime('%Y-%m') for i in locs if i < len(df_sharpe)]
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

def plot_historical_pe_trend(res, fundamental_engine, prices=None, title="Historical Trailing P/E Ratio"):
    """
    Plots the trend of the weighted harmonic mean P/E ratio for each strategy.
    """
    try:
        plt.figure(figsize=(14, 7))
        
        for col in res.prices.columns:
            # 1. Get weights over time
            try:
                weights = res.get_security_weights(col)
                if weights.empty:
                    # Fallback for benchmarks that might not have weights recorded
                    weights = pd.DataFrame(1.0, index=res.prices.index, columns=pd.Index([col]))
            except Exception:
                weights = pd.DataFrame(1.0, index=res.prices.index, columns=pd.Index([col]))

            # 2. Get price data for these constituents
            if prices is None:
                continue
                
            available_tickers = [t for t in weights.columns if t in prices.columns]
            if not available_tickers:
                continue

            # 3. Calculate Portfolio PE series
            pe_series = fundamental_engine.get_portfolio_pe_series(weights[available_tickers], prices[available_tickers])
            
            non_nan = pe_series.dropna()
            if len(non_nan) > 1:
                plt.plot(non_nan.index, non_nan, label=col)
                print(f"Plotted P/E trend for {col}: {len(non_nan)} points from {non_nan.index[0].date()} to {non_nan.index[-1].date()}")
            else:
                print(f"Skipping P/E trend for {col}: Not enough valid data points.")

        plt.title(title)
        plt.ylabel("P/E Ratio (Harmonic Mean)")
        plt.xlabel("Date")
        
        # Force better date formatting to prevent "hourly" axis bugs
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        make_legend_interactive()
    except Exception as e:
        print(f"Historical P/E plot failed: {e}")

def plot_valuation_comparison(res, metadata, title="Portfolio Valuation Metrics"):
    """
    Compares Trailing PE, Forward PE, and PEG ratios across strategies and benchmarks.
    Calculates weighted harmonic mean of constituent metrics.
    """
    try:
        metrics = ['trailingPE', 'forwardPE', 'trailingPegRatio']
        plot_labels = ['Trailing P/E', 'Forward P/E', 'PEG Ratio']
        data = []
        
        for col in res.prices.columns:
            try:
                weights = res.get_security_weights(col).iloc[-1]
                constituents = weights[weights > 0.0001]
            except Exception:
                constituents = pd.Series({col: 1.0})

            row = {'Strategy': col}
            for metric in metrics:
                sum_inv_metric = 0
                valid_weight = 0
                for ticker, weight in constituents.items():
                    m_val = metadata.get(ticker, {}).get(metric)
                    if m_val is not None and not np.isnan(m_val) and m_val > 0.01:
                        sum_inv_metric += weight / m_val
                        valid_weight += weight
                
                row[metric] = valid_weight / sum_inv_metric if sum_inv_metric > 0 else np.nan
            data.append(row)
        
        df = pd.DataFrame(data).set_index('Strategy')
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if len(metrics) > 1 else axes
            df[metric].plot(kind='bar', ax=ax, color='#3498db', alpha=0.8)
            ax.set_title(f"{plot_labels[i]} comparison")
            ax.set_ylabel("Ratio")
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            for p in ax.patches:
                val = p.get_height()
                if not np.isnan(val):
                    ax.annotate(f"{val:.2f}", (p.get_x() + p.get_width() / 2., val),
                                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        make_legend_interactive()
    except Exception as e:
        print(f"Valuation comparison plot failed: {e}")

def plot_drawdown_recovery_surface(res, strategy_name, title="Drawdown-Recovery Surface"):
    """
    Plots a 3D surface showing the relationship between Drawdown Magnitude, 
    Duration, and the Required Recovery Return.
    
    X-axis: Magnitude of Drawdown (%)
    Y-axis: Duration of the event (Months)
    Z-axis: Required Recovery Return (+%)
    
    This visualization highlights the 'Asymmetry of Loss'.
    """
    if strategy_name not in res.prices.columns:
        return

    prices = res.prices[strategy_name]
    
    # 1. Calculate Drawdowns
    # We need to find peak-to-recovery periods
    cummax = prices.cummax()
    drawdown = (prices / cummax) - 1.0
    
    # Identify drawdown events
    is_in_drawdown = drawdown < 0
    # Find transitions - using infer_objects to avoid future warnings
    starts = (is_in_drawdown & (~is_in_drawdown.shift(1).fillna(False))).infer_objects(copy=False)
    ends = ((~is_in_drawdown) & is_in_drawdown.shift(1).fillna(False)).infer_objects(copy=False)
    
    start_dates = prices.index[starts]
    end_dates = prices.index[ends]
    
    # Align starts and ends
    events = []
    for s in start_dates:
        # Find the first end after this start
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
    
    # 2. Create Theoretical Surface
    # X: Magnitude 0 to 60% (common range) or max observed
    max_mag = max(df_events['magnitude'].max(), 50)
    max_dur = max(df_events['duration'].max(), 12)
    
    x_surf = np.linspace(0, min(max_mag * 1.2, 90), 50)
    y_surf = np.linspace(0, max_dur * 1.2, 50)
    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
    # Z is purely a function of X: Recovery = 1/(1-X) - 1
    Z_surf = (100.0 / (100.0 - X_surf)) * 100.0 - 100.0
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add lighting for better depth perception
    from matplotlib.colors import LightSource
    ls = LightSource(270, 45)
    rgb = ls.shade(Z_surf, cmap=plt.get_cmap('OrRd'), vert_exag=0.1, blend_mode='soft')
    
    # Plot Surface
    surf = ax.plot_surface(X_surf, Y_surf, Z_surf, facecolors=rgb, alpha=0.5, antialiased=True, shade=False)
    
    # Plot Historical Events
    ax.scatter(df_events['magnitude'], df_events['duration'], df_events['recovery'], 
               color='blue', s=50, edgecolors='white', label='Historical Drawdowns')
    
    # Labeling
    ax.set_title(f"{title}: {strategy_name}", fontsize=14)
    ax.set_xlabel("Magnitude of Drawdown (%)")
    ax.set_ylabel("Duration to Recovery (Months)")
    ax.set_zlabel("Required Recovery Return (%)")
    
    # Add colorbar for the surface
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Theoretical Recovery Required')
    
    ax.view_init(elev=20, azim=-45)
    plt.tight_layout()
