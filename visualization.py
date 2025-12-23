import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.text as mtext

# Try to import seaborn for better style, but fallback if missing
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="whitegrid")
except ImportError:
    HAS_SEABORN = False

def make_legend_interactive(fig=None):
    """
    Makes all legends in the figure interactive.
    Clicking on a legend label will toggle the visibility of the corresponding plot element.
    """
    if fig is None:
        fig = plt.gcf()

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
    # Calculate drawdown manually from prices
    prices = res.prices
    drawdowns = prices / prices.cummax() - 1
    
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

    # Extract daily returns
    daily_rets = res.prices[strategy_name].pct_change().dropna()
    
    # Resample to monthly returns (compounded)
    monthly_rets = daily_rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table: Year vs Month
    monthly_rets = monthly_rets.to_frame(name='return')
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
    daily_rets = res.prices.pct_change().dropna()
    rolling_vol = daily_rets.rolling(window=window).std() * np.sqrt(252)
    
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
    Plots the distribution (histogram) of daily returns.
    """
    daily_rets = res.prices.pct_change().dropna()
    
    plt.figure(figsize=(12, 6))
    
    if HAS_SEABORN:
        for col in daily_rets.columns:
            sns.histplot(daily_rets[col], kde=True, label=col, alpha=0.3, element="step")
    else:
        # PURE MATPLOTLIB
        for col in daily_rets.columns:
            plt.hist(daily_rets[col], bins=50, alpha=0.5, label=col, density=True)
            
    plt.title(title)
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency (Density)')
    plt.legend()
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
    returns = res.prices.pct_change().dropna()
    
    plt.figure(figsize=(12, 6))
    
    for col in returns.columns:
        # Excess returns
        excess_returns = returns[col] - (risk_free_rate / 252)
        
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = returns[col].rolling(window=window).std()
        
        # Annualized Sharpe
        sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        plt.plot(sharpe.index, sharpe, label=col)
        
    plt.title(title)
    plt.ylabel('Sharpe Ratio')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    make_legend_interactive()
