import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.text as mtext
from analytics import AnalyticsEngine

# Try to import seaborn for better style, but fallback if missing
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="whitegrid")
except ImportError:
    HAS_SEABORN = False
    sns = None
def add_financial_cursor(fig=None):
    """
    Adds a vertical crosshair and tooltip that follows the mouse across ALL subplots.
    Industry standard for financial time-series visualization.
    """
    if fig is None: fig = plt.gcf()

    class FinancialCursor:
        def __init__(self, fig):
            self.fig = fig
            self.vlines = []
            self.annotations = []
            self.axes_list = []
            
            # Initialize for all compatible axes
            for ax in fig.get_axes():
                if not ax.get_lines(): continue
                
                xlim = ax.get_xlim()
                vl = ax.axvline(x=xlim[0], color='gray', linestyle='--', linewidth=1, visible=False, zorder=90)
                ann = ax.annotate("", xy=(xlim[0], 0), xytext=(10, 10),
                                 textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="#fcfcfc", alpha=0.9, ec="#bdc3c7"),
                                 fontsize=9, visible=False, zorder=100)
                
                self.vlines.append(vl)
                self.annotations.append(ann)
                self.axes_list.append(ax)
                ax.set_xlim(xlim) # Prevent snapping stretching
            
            self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
            self.fig.canvas.mpl_connect("axes_leave_event", self.on_leave)
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        def on_click(self, event):
            if not event.inaxes: return
            # Find the match from our controlled axes
            for ax, ann in zip(self.axes_list, self.annotations):
                if ax == event.inaxes:
                    text = ann.get_text()
                    if text: print(f"\n--- Data Point at Cursor ---\n{text}\n---------------------------")

        def on_mouse_move(self, event):
            if not event.inaxes or event.inaxes not in self.axes_list:
                self.on_leave(event)
                return

            x_data = event.xdata
            import matplotlib.dates as mdates
            
            # Format Date for Header
            try:
                date_str = mdates.num2date(x_data).strftime('%Y-%m-%d')
                header = f"Date: {date_str}"
            except:
                header = f"X: {x_data:.2f}"

            # Update ALL axes simultaneously for a "synchronized crosshair" feel
            for ax, vl, ann in zip(self.axes_list, self.vlines, self.annotations):
                lines = ax.get_lines()
                text_lines = [header]
                found_val = False
                
                for line in lines:
                    # Skip the vertical line and invisible lines
                    if line == vl or not line.get_visible(): continue
                    
                    lx, ly = line.get_data()
                    if len(lx) == 0: continue
                    
                    # Convert to numeric for binary search
                    try:
                        # Improved date handling: check if it's already numeric (float)
                        first_val = lx[0]
                        if hasattr(first_val, 'to_datetime64') or isinstance(first_val, (pd.Timestamp, datetime, np.datetime64)):
                             lx_num = mdates.date2num(lx)
                        else:
                             lx_num = np.asarray(lx, dtype=float)
                    except:
                        try:
                            lx_num = mdates.date2num(pd.to_datetime(lx))
                        except:
                            lx_num = np.asarray(lx)

                    idx = np.searchsorted(lx_num, x_data)
                    idx = np.clip(idx, 0, len(lx_num)-1)
                    
                    # Check proximity (approx 2 weeks range to be more forgiving on weekly/monthly data)
                    if abs(lx_num[idx] - x_data) < 14:
                        val = ly[idx]
                        label = line.get_label()
                        if not label or label.startswith('_'): label = 'Value'
                        text_lines.append(f"{label}: {val:.2f}")
                        found_val = True

                if found_val:
                    vl.set_xdata([x_data])
                    vl.set_visible(True)
                    ann.set_text("\n".join(text_lines))
                    # Place annotation near the cursor but inside axes
                    # If it's the active axis, follow the mouse Y, otherwise stick to top/bottom
                    if ax == event.inaxes:
                        ann.xy = (x_data, event.ydata)
                    else:
                        ann.xy = (x_data, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1)
                    ann.set_visible(True)
                else:
                    vl.set_visible(False)
                    ann.set_visible(False)

            self.fig.canvas.draw_idle()

        def on_leave(self, event):
            for vl, ann in zip(self.vlines, self.annotations):
                vl.set_visible(False)
                ann.set_visible(False)
            self.fig.canvas.draw_idle()

    # Store reference to prevent garbage collection
    if not hasattr(fig, '_financial_cursor'):
        from datetime import datetime
        setattr(fig, '_financial_cursor', FinancialCursor(fig))

def make_legend_interactive(fig=None):
    """
    Makes all legends in the figure interactive.
    Clicking on a legend label will toggle the visibility of the corresponding plot element.
    """
    if fig is None:
        fig = plt.gcf()
    
    # Add the crosshair cursor for exact value inspection
    add_financial_cursor(fig)

    for ax in fig.get_axes():
        leg = ax.get_legend()
        if not leg:
            continue

        texts = leg.get_texts()
        for text in texts:
            text.set_picker(True)
        
        def on_pick(event):
            artist = event.artist
            if not isinstance(artist, mtext.Text):
                return
            
            label = artist.get_text()
            target_visible = None
            
            # Find all artists with this label across all possible containers
            # Search children (Lines, simple artists)
            objs = list(ax.get_children())
            for ax_obj in objs:
                if hasattr(ax_obj, 'get_label') and ax_obj.get_label() == label:
                    if target_visible is None:
                        target_visible = not ax_obj.get_visible()
                    ax_obj.set_visible(target_visible)
            
            # Search collections (PolygonCollection, PathCollection - often used in fills)
            for coll in ax.collections:
                if coll.get_label() == label:
                    if target_visible is None:
                        target_visible = not coll.get_visible()
                    coll.set_visible(target_visible)
            
            # Search containers (BarContainer - used in histograms)
            for cont in ax.containers:
                if cont.get_label() == label:
                    if target_visible is None:
                        # Containers don't have get_visible, we check first child
                        target_visible = not cont[0].get_visible() if len(cont) > 0 else True
                    for patch in cont:
                        patch.set_visible(target_visible)

            # Dim the legend text if hidden
            if target_visible is not None:
                artist.set_alpha(1.0 if target_visible else 0.2)
                fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)

def plot_drawdowns(res, title="Drawdowns"):
    """
    Plots the drawdown (underwater) chart for all strategies in the backtest result.
    """
    drawdowns = AnalyticsEngine.get_drawdowns(res.prices)
    
    plt.figure(figsize=(12, 6))
    for col in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[col], label=col)
        
    plt.fill_between(drawdowns.index, 0, drawdowns.min(axis=1), alpha=0.1, color='red')
    plt.title(title)
    plt.ylabel('Drawdown')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

def plot_monthly_returns_heatmap(res, strategy_name):
    """
    Plots a heatmap of monthly returns for a specific strategy.
    Row: Year, Column: Month
    """
    if strategy_name not in res.prices.columns:
        print(f"Strategy {strategy_name} not found in results.")
        return

    # Use AnalyticsEngine for compounded monthly returns
    monthly_rets = AnalyticsEngine.get_monthly_returns(res.prices[[strategy_name]])
    monthly_rets = monthly_rets.iloc[:, 0].to_frame(name='return')
    
    monthly_rets['Year'] = monthly_rets.index.year
    monthly_rets['Month'] = monthly_rets.index.month
    
    pivot_table = monthly_rets.pivot(index='Year', columns='Month', values='return')
    
    # Map month numbers to names for display
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    
    # Ensure all months are present
    for m in range(1, 13):
        if m not in pivot_table.columns:
            pivot_table[m] = np.nan
            
    pivot_table = pivot_table.sort_index(axis=1)
    columns_labels = [month_map[m] for m in pivot_table.columns]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, len(pivot_table) * 0.5 + 2))
    
    if HAS_SEABORN:
        sns.heatmap(pivot_table, annot=True, fmt=".1%", cmap="RdYlGn", center=0, cbar_kws={'label': 'Monthly Return'}, ax=ax)
        ax.set_xticklabels(columns_labels)
    else:
        # PURE MATPLOTLIB FALLBACK
        im = ax.imshow(pivot_table, cmap="RdYlGn", aspect='auto')
        
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(columns_labels)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        
        ax.set_xticklabels(columns_labels)
        ax.set_yticklabels(pivot_table.index)
        
        # Loop over data dimensions and create text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                val = pivot_table.iloc[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f"{val:.1%}",
                                   ha="center", va="center", color="black" if abs(val) < 0.1 else "white")
                                   
        plt.colorbar(im, label='Monthly Return')

    plt.title(f"Monthly Returns Heatmap: {strategy_name}")
    plt.tight_layout()
    make_legend_interactive()

def plot_rolling_volatility(res, window=126, title="Rolling Volatility (6-Month)"):
    """
    Plots annualized rolling volatility.
    Default window=126 (approx 6 months).
    """
    rolling_vol = AnalyticsEngine.get_rolling_volatility(res.prices, window=window)
    
    plt.figure(figsize=(12, 6))
    for col in rolling_vol.columns:
        plt.plot(rolling_vol.index, rolling_vol[col], label=col)
        
    plt.title(title)
    plt.ylabel('Annualized Volatility')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

def plot_return_distribution(res, title="Daily Return Distribution"):
    """
    Plots the distribution (histogram) of daily returns with median stats.
    """
    daily_rets = res.prices.pct_change().dropna()
    
    plt.figure(figsize=(14, 8))
    
    median_stats = []
    
    if HAS_SEABORN:
        for col in daily_rets.columns:
            median = daily_rets[col].median()
            median_stats.append(f"{col}: {median:.2%}")
            
            p = sns.histplot(daily_rets[col], kde=True, label=col, alpha=0.3, element="step")
            # Get color from the last added line or patch
            color = p.get_lines()[-1].get_color() if p.get_lines() else None
            
            # Add median line with a specific label for toggling
            plt.axvline(median, color=color, linestyle='--', alpha=0.8, linewidth=2, label=f"{col} Median")
    else:
        # PURE MATPLOTLIB
        for col in daily_rets.columns:
            median = daily_rets[col].median()
            median_stats.append(f"{col}: {median:.2%}")
            
            n, bins, patches = plt.hist(daily_rets[col], bins=50, alpha=0.4, label=col, density=True)
            color = patches[0].get_facecolor() if len(patches) > 0 else None
            
            plt.axvline(median, color=color, linestyle='--', alpha=0.8, linewidth=2, label=f"{col} Median")
            
    # Add stats textbox
    stats_text = "Medians (Daily):\n" + "\n".join(median_stats)
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.title(title)
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency (Density)')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

def plot_rolling_beta(res, benchmark='SPY', window=126, title="Rolling Beta (6-Month)"):
    """
    Plots the rolling beta of all strategies relative to a benchmark.
    """
    if benchmark not in res.prices.columns:
        print(f"Benchmark {benchmark} not found in results. Skipping Beta plot.")
        return

    returns = res.prices.pct_change().dropna()
    bench_rets = returns[benchmark]
    
    plt.figure(figsize=(12, 6))
    
    for col in returns.columns:
        if col == benchmark: continue
        
        rolling_cov = returns[col].rolling(window=window).cov(bench_rets)
        rolling_var = bench_rets.rolling(window=window).var()
        beta = rolling_cov / rolling_var
        
        plt.plot(beta.index, beta, label=col)
        
    plt.title(f"{title} vs {benchmark}")
    plt.ylabel('Beta')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

def plot_rolling_alpha(res, benchmark='SPY', window=126, title="Rolling Annualized Alpha (6-Month)"):
    """
    Plots the rolling annualized alpha of all strategies relative to a benchmark.
    Alpha = (Rp - Beta * Rm) * 252 (Simplified, assuming Rf=0 for visualization trend)
    """
    if benchmark not in res.prices.columns:
        print(f"Benchmark {benchmark} not found in results. Skipping Alpha plot.")
        return

    returns = res.prices.pct_change().dropna()
    bench_rets = returns[benchmark]
    
    plt.figure(figsize=(12, 6))
    
    for col in returns.columns:
        if col == benchmark: continue
        
        # Calculate Beta first
        rolling_cov = returns[col].rolling(window=window).cov(bench_rets)
        rolling_var = bench_rets.rolling(window=window).var()
        beta = rolling_cov / rolling_var
        
        # Calculate Alpha
        # Rp_mean - Beta * Rm_mean
        # We use simple rolling means (geometric might be more precise but arithmetic is standard for this viz)
        rolling_ret_strat = returns[col].rolling(window=window).mean()
        rolling_ret_bench = bench_rets.rolling(window=window).mean()
        
        alpha = (rolling_ret_strat - beta * rolling_ret_bench) * 252
        
        plt.plot(alpha.index, alpha, label=col)
        
    plt.title(f"{title} vs {benchmark}")
    plt.ylabel('Annualized Alpha')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

def plot_rolling_sharpe(res, window=126, risk_free_rate=0.0, title="Rolling Sharpe Ratio (6-Month)"):
    """
    Plots the rolling annualized Sharpe ratio.
    """
    sharpe = AnalyticsEngine.get_rolling_sharpe(res.prices, window=window, risk_free_rate=risk_free_rate)
    
    plt.figure(figsize=(12, 6))
    for col in sharpe.columns:
        plt.plot(sharpe.index, sharpe[col], label=col)
        
    plt.title(title)
    plt.ylabel('Sharpe Ratio')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

def plot_rolling_sortino(res, window=126, risk_free_rate=0.0, title="Rolling Sortino Ratio (6-Month)"):
    """
    Plots the rolling annualized Sortino ratio (Downside risk-adjusted).
    """
    sortino = AnalyticsEngine.get_rolling_sortino(res.prices, window=window, risk_free_rate=risk_free_rate)
    
    plt.figure(figsize=(12, 6))
    for col in sortino.columns:
        plt.plot(sortino.index, sortino[col], label=col)
        
    plt.title(title)
    plt.ylabel('Sortino Ratio')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

def plot_upside_downside_capture(res, benchmark='SPY', title="Upside/Downside Capture Ratios"):
    """
    Plots bar chart of capture ratios relative to a benchmark.
    Institutions use this to see 'Beta-up' vs 'Beta-down'.
    """
    if benchmark not in res.prices.columns:
        print(f"Benchmark {benchmark} not found in results. Skipping Capture plot.")
        return

    data = []
    strategies = [c for c in res.prices.columns if c != benchmark]
    
    for strat in strategies:
        capture = AnalyticsEngine.get_capture_ratios(res.prices[strat], res.prices[benchmark])
        data.append({
            'Strategy': strat,
            'Upside Capture': capture['upside_capture'],
            'Downside Capture': capture['downside_capture']
        })
    
    df_plot = pd.DataFrame(data).set_index('Strategy')
    
    ax = df_plot.plot(kind='bar', figsize=(10, 6), color=['#2ecc71', '#e74c3c'])
    plt.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    plt.title(f"{title} (vs {benchmark})")
    plt.ylabel('Ratio')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()


def plot_correlation_matrix(res, title="Strategy Correlation Matrix"):
    """
    Plots a heatmap of return correlations.
    Professionals use this to identify hidden concentration/overlap.
    """
    returns = res.prices.pct_change().dropna()
    corr = returns.corr()
    
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    else:
        plt.imshow(corr, cmap="coolwarm", interpolation='nearest')
        plt.colorbar()
        plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(np.arange(len(corr.columns)), corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
                
    plt.title(title)
    plt.tight_layout()
    make_legend_interactive()

def plot_rolling_info_ratio(res, benchmark='SPY', window=126, title="Rolling Information Ratio (6-Month)"):
    """
    Plots the rolling annualized Information Ratio.
    Measures risk-adjusted relative return.
    """
    if benchmark not in res.prices.columns:
        print(f"Benchmark {benchmark} not found in results. Skipping IR plot.")
        return

    # Strategies excluding benchmarks
    strategies = [c for c in res.prices.columns if c != benchmark]
    ir = AnalyticsEngine.get_rolling_info_ratio(res.prices[strategies], res.prices[[benchmark]], window=window)
    
    plt.figure(figsize=(12, 6))
    for col in ir.columns:
        plt.plot(ir.index, ir[col], label=col)
        
    plt.title(f"{title} (vs {benchmark})")
    plt.ylabel('Information Ratio')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()

def plot_risk_contribution(res, strategy_name, prices=None, group_map=None, title="Risk Contribution (MCTR)"):
    """
    Plots the Percentage Contribution to Risk for each asset (or group) in a strategy.
    Institutions use this to see what is 'driving' the volatility.
    """
    try:
        # Get weights from bt results
        weights = res.get_security_weights(strategy_name).iloc[-1]
        constituents = weights[weights > 0].index.tolist()
        
        if not constituents:
            return

        # Use explicitly passed prices or fallback to result prices (though result prices usually only have strategy curves)
        data_prices = prices if prices is not None else res.prices
        
        available = [c for c in constituents if c in data_prices.columns]
        if not available:
            print(f"No price data available for constituents of {strategy_name}: {constituents}")
            return
            
        pct_ctr = AnalyticsEngine.get_risk_contribution(data_prices[available], weights[available])
        
        # If group_map is provided, sum contributions by group
        if group_map:
            mapping = {asset: group_map.get(asset, 'Other') for asset in available}
            pct_ctr = pct_ctr.groupby(mapping).sum()
        
        # Sort descending for the bar chart
        pct_ctr = pct_ctr.sort_values(ascending=True)
        
        plt.figure(figsize=(12, 8))
        pct_ctr.plot(kind='barh')
        plt.title(f"{title}: {strategy_name}")
        plt.xlabel('Contribution to Total Portfolio Volatility (Normalized)')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        make_legend_interactive()
    except Exception as e:
        print(f"Risk contribution plot failed for {strategy_name}: {e}")

def plot_grouped_correlation_matrix(prices, group_map, title="Grouped Correlation Matrix"):
    """
    Aggregates asset returns by group and plots their correlation.
    Helps identify correlations between Sectors or Asset Classes.
    """
    try:
        returns = prices.pct_change().dropna()
        # Group columns by the map
        available_assets = [c for c in returns.columns if c in group_map]
        if not available_assets:
            return
            
        # Create group returns: average of constituent returns
        grouped_returns = pd.DataFrame()
        groups = set(group_map.values())
        
        for group in groups:
            group_assets = [a for a in available_assets if group_map[a] == group]
            if group_assets:
                grouped_returns[group] = returns[group_assets].mean(axis=1)
        
        if grouped_returns.empty:
            return
            
        corr = grouped_returns.corr()
        
        plt.figure(figsize=(12, 10))
        if HAS_SEABORN:
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        else:
            plt.imshow(corr, cmap="coolwarm", interpolation='nearest')
            plt.colorbar()
            plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(np.arange(len(corr.columns)), corr.columns)
            
        plt.title(title)
        plt.tight_layout()
        make_legend_interactive()
    except Exception as e:
        print(f"Grouped correlation plot failed: {e}")

def plot_rolling_alpha_beta(strat_prices, bench_prices, window=126, title_prefix=""):
    """
    Plots rolling Alpha and Beta in two subplots.
    """
    try:
        alphas, betas = AnalyticsEngine.get_rolling_alpha_beta(strat_prices, bench_prices, window)
        
        # We assume strat_prices only has one strategy for this specific plot for clarity
        # If multiple, we plot all.
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        for col in betas.columns:
            ax1.plot(betas[col], label=f"{col} Beta")
        ax1.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title(f"{title_prefix} Rolling Beta ({window}d)")
        ax1.set_ylabel("Beta")
        ax1.legend()
        
        for col in alphas.columns:
            ax2.plot(alphas[col] * 100, label=f"{col} Alpha (Ann.)")
        ax2.axhline(0.0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title(f"{title_prefix} Rolling Alpha ({window}d, Annualized %)")
        ax2.set_ylabel("Alpha (%)")
        ax2.legend()
        
        plt.tight_layout()
        make_legend_interactive()
    except Exception as e:
        print(f"Rolling Alpha/Beta plot failed: {e}")

def plot_return_attribution(prices, weights_series, group_map=None, title=""):
    """
    Rethought Performance Attribution Dashboard.
    1. Grouped Cumulative Attribution (if group_map provided)
    2. Total Contribution bar chart (Individual Assets)
    """
    try:
        attribution = AnalyticsEngine.get_return_attribution(prices, weights_series)
        total_attr = attribution.iloc[-1] * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
        
        # --- Panel 1: Cumulative Contribution ---
        if group_map:
            # Aggregate by group
            available = [c for c in attribution.columns if c in group_map]
            mapping = {asset: group_map.get(asset, 'Other') for asset in available}
            attr_grouped = (attribution[available] * 100).groupby(mapping, axis=1).sum()
            
            # Sort groups by final contribution for better visual order
            sorted_groups = attr_grouped.iloc[-1].sort_values(ascending=False).index
            attr_grouped = attr_grouped[sorted_groups]
            
            # Stackplot works better when values are mostly positive or we just want the trend.
            # For complex attribution, grouped area is cleaner.
            ax1.stackplot(attr_grouped.index, attr_grouped.T, labels=attr_grouped.columns, alpha=0.8)
            ax1.set_title(f"{title} - Cumulative Contribution by Group")
        else:
            # If no group map, show Top 10 + 'Other' if many assets
            if len(attribution.columns) > 12:
                top_cols = total_attr.sort_values(ascending=False).head(10).index
                other_sum = attribution.drop(columns=top_cols).sum(axis=1)
                attr_summary = attribution[top_cols].copy()
                attr_summary['Others (Combined)'] = other_sum
                
                # Sort for stackplot
                sorted_cols = attr_summary.iloc[-1].sort_values(ascending=False).index
                attr_summary = attr_summary[sorted_cols]
                
                ax1.stackplot(attr_summary.index, attr_summary.T * 100, labels=attr_summary.columns, alpha=0.8)
                ax1.set_title(f"{title} - Top 10 Contributors vs Others")
            else:
                sorted_cols = total_attr.sort_values(ascending=False).index
                ax1.stackplot(attribution.index, attribution[sorted_cols].T * 100, labels=sorted_cols, alpha=0.8)
                ax1.set_title(f"{title} - Individual Cumulative View")

        ax1.set_ylabel("Cumulative Contribution (%)")
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # --- Panel 2: Total Contribution Bar Chart ---
        # Show Top 10 and Bottom 5 for clarity if many assets
        if len(total_attr) > 15:
            top10 = total_attr.sort_values(ascending=False).head(10)
            bottom5 = total_attr.sort_values(ascending=False).tail(5)
            top_bottom = pd.concat([top10, bottom5])
            top_bottom = top_bottom[~top_bottom.index.duplicated(keep='first')].sort_values(ascending=True)
            ax2.set_title("Individual Summary: Top 10 Winners & Bottom 5 Losers")
        else:
            top_bottom = total_attr.sort_values(ascending=True)
            ax2.set_title("Total Individual Contribution")
            
        colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in top_bottom]
        top_bottom.plot(kind='barh', ax=ax2, color=colors)
        ax2.set_xlabel("Total Contribution to Portfolio Return (%)")
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        make_legend_interactive()
    except Exception as e:
        print(f"Return attribution plot failed: {e}")

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
