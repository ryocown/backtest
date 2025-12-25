import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tda_engine import TDAManager
from data import DataEngine
import visualization as viz
from datetime import datetime, timedelta
import argparse
import sys
from scipy.spatial import Delaunay
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive TDA Market Analysis")
    parser.add_argument("--start", type=str, default="2020-02-18", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2020-04-06", help="End date (YYYY-MM-DD)")
    parser.add_argument("--window", type=int, default=6, help="Window size in months")
    return parser.parse_args()

class TDAExplorer:
    def __init__(self, available_dates, precomputed_results, spy_ohlc):
        self.available_dates = available_dates
        self.results = precomputed_results
        self.spy_ohlc = spy_ohlc
        
        # Setup Figure
        self.fig = plt.figure(figsize=(18, 12))
        # 4 rows: 3D/Heatmap/Metrics, SPY Chart, Slider, Buttons
        self.gs = self.fig.add_gridspec(4, 5, height_ratios=[1, 1, 0.4, 0.1], 
                                        width_ratios=[1.2, 1, 0.05, 0.3, 0.3])
        
        self.ax_3d = self.fig.add_subplot(self.gs[0:2, 0], projection='3d')
        self.ax_heatmap = self.fig.add_subplot(self.gs[0:2, 1])
        self.ax_cbar = self.fig.add_subplot(self.gs[0:2, 2])
        self.ax_metrics = self.fig.add_subplot(self.gs[0:2, 3:])
        
        self.ax_spy = self.fig.add_subplot(self.gs[2, :])
        
        self.ax_slider = self.fig.add_subplot(self.gs[3, 0:2])
        self.ax_prev = self.fig.add_subplot(self.gs[3, 3])
        self.ax_next = self.fig.add_subplot(self.gs[3, 4])
        
        self.slider = Slider(
            ax=self.ax_slider,
            label='Date ',
            valmin=0,
            valmax=len(available_dates) - 1,
            valinit=0,
            valstep=1
        )
        
        self.btn_prev = Button(self.ax_prev, 'Prev')
        self.btn_next = Button(self.ax_next, 'Next')
        
        self.slider.on_changed(self.update)
        self.btn_prev.on_clicked(self.go_prev)
        self.btn_next.on_clicked(self.go_next)
        
        # Initial Plot
        self.render_spy_chart()
        self.current_highlight = None
        self.update(0)

    def render_spy_chart(self):
        """Renders the static part of the SPY candlestick chart."""
        self.ax_spy.clear()
        
        # Draw candlesticks
        width = 0.6
        width2 = 0.05
        
        up = self.spy_ohlc[self.spy_ohlc.close >= self.spy_ohlc.open]
        down = self.spy_ohlc[self.spy_ohlc.close < self.spy_ohlc.open]
        
        # Up candles
        self.ax_spy.bar(up.index, up.close - up.open, width, bottom=up.open, color='green', alpha=0.6)
        self.ax_spy.vlines(up.index, up.low, up.high, color='green', linewidth=1, alpha=0.6)
        
        # Down candles
        self.ax_spy.bar(down.index, down.close - down.open, width, bottom=down.open, color='red', alpha=0.6)
        self.ax_spy.vlines(down.index, down.low, down.high, color='red', linewidth=1, alpha=0.6)
        
        self.ax_spy.set_title("SPY Price Context", fontsize=12, fontweight='bold')
        self.ax_spy.grid(True, alpha=0.3)
        self.ax_spy.set_ylabel("Price ($)")

    def update(self, val):
        idx = int(self.slider.val)
        res = self.results[idx]
        
        if res is None:
            return
            
        current_date = res['date']
        coords = res['coords']
        dist_matrix = res['dist']
        corr_matrix = res['corr']
        betti = res['betti']
        chi = res['chi']
        avg_corr = res['avg_corr']
        
        # Update 3D Plot
        self.ax_3d.clear()
        self.ax_3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='red', s=40, alpha=0.8, edgecolors='k')
        
        threshold = 1.0
        n = len(coords)
        # Edges
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix.iloc[i, j] <= threshold:
                    self.ax_3d.plot([coords[i, 0], coords[j, 0]], 
                                    [coords[i, 1], coords[j, 1]], 
                                    [coords[i, 2], coords[j, 2]], 
                                    color='cyan', alpha=0.2, linewidth=0.8)
        
        # Faces
        try:
            tri = Delaunay(coords)
            valid_faces = []
            for s in tri.simplices:
                if (dist_matrix.iloc[s[0], s[1]] <= threshold and 
                    dist_matrix.iloc[s[1], s[2]] <= threshold and 
                    dist_matrix.iloc[s[2], s[0]] <= threshold):
                    valid_faces.append(s)
            if valid_faces:
                self.ax_3d.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], 
                                        triangles=valid_faces, color='cyan', alpha=0.15, shade=True)
        except:
            pass
            
        self.ax_3d.set_title(f"Market Topology: {current_date.date()}", fontsize=14, fontweight='bold')
        self.ax_3d.set_axis_off()
        
        # Update Heatmap
        self.ax_heatmap.clear()
        self.ax_cbar.clear()
        sns.heatmap(corr_matrix, ax=self.ax_heatmap, cbar_ax=self.ax_cbar, cmap='coolwarm', center=0, cbar=True, 
                    xticklabels=False, yticklabels=False)
        self.ax_heatmap.set_title(f"Correlation Matrix (Avg: {avg_corr:.3f})", fontsize=12)
        
        # Update Metrics Text
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        
        regime = "CRASH" if avg_corr > 0.7 else "STRESS" if avg_corr > 0.5 else "NORMAL"
        regime_color = "red" if regime == "CRASH" else "orange" if regime == "STRESS" else "green"
        
        metrics_text = (
            f"DATE: {current_date.date()}\n"
            f"{'='*25}\n\n"
            f"Avg Correlation: {avg_corr:.3f}\n\n"
            f"Betti 0 (Components): {betti[0]}\n"
            f"Betti 1 (Cycles):     {betti[1] if len(betti) > 1 else 0}\n"
            f"Euler (χ):           {chi}\n\n"
            f"MARKET REGIME:\n"
        )
        
        self.ax_metrics.text(0.05, 0.7, metrics_text, fontsize=13, family='monospace', verticalalignment='top')
        self.ax_metrics.text(0.05, 0.35, f"  >>> {regime} <<<  ", fontsize=18, family='monospace', 
                             fontweight='bold', color=regime_color, bbox=dict(facecolor='white', alpha=0.5))
        
        interpretation = (
            "Interpretation:\n"
            f"{'Liquidity collapse. Systemic risk high.' if regime == 'CRASH' else 
               'Rising synchronization. Hedges narrowing.' if regime == 'STRESS' else 
               'Healthy diversification. Idiosyncratic risk.'}"
        )
        self.ax_metrics.text(0.05, 0.15, interpretation, fontsize=11, style='italic', verticalalignment='top')
        
        # Update SPY Highlight
        if self.current_highlight:
            self.current_highlight.remove()
        
        self.current_highlight = self.ax_spy.axvspan(
            current_date - pd.Timedelta(hours=12), 
            current_date + pd.Timedelta(hours=12), 
            color='yellow', alpha=0.3, zorder=-1
        )
        
        self.fig.canvas.draw_idle()

    def go_prev(self, event):
        val = int(self.slider.val) - 1
        if val >= 0:
            self.slider.set_val(val)

    def go_next(self, event):
        val = int(self.slider.val) + 1
        if val < len(self.available_dates):
            self.slider.set_val(val)

    def show(self):
        plt.show()

def main():
    args = parse_args()
    
    # 1. Setup Data and Engine
    data_engine = DataEngine()
    tda_manager = TDAManager(window_months=args.window)
    
    tickers = [
        'AAPL', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'BLK', 'BRK.B', 'BX', 'CAT', 'COST', 
        'GE', 'GLD', 'GOOGL', 'GS', 'JPM', 'KRBN', 'LIN', 'LLY', 'MA', 'META', 
        'MS', 'MSFT', 'NVDA', 'QCOM', 'QQQ', 'SGOV', 'SPGI', 'SPY', 'TSLA', 'TXN', 
        'UNH', 'UNP', 'V', 'VTI', 'WMT', 'XLG', 'XOM'
    ]
    
    analysis_start = pd.to_datetime(args.start)
    analysis_end = pd.to_datetime(args.end)
    data_start = (analysis_start - pd.DateOffset(months=args.window)).strftime('%Y-%m-%d')
    
    print(f"Loading data for {len(tickers)} tickers from {data_start} to {args.end}...")
    df = data_engine.get_historical_data(tickers, data_start, args.end)
    
    if df.empty:
        print("No data found for the specified range.")
        return

    df = df.dropna(axis=1)
    print(f"Using {len(df.columns)} tickers after cleaning.")

    available_dates = df.index[(df.index >= analysis_start) & (df.index <= analysis_end)]
    
    if len(available_dates) == 0:
        print("No trading days found in the requested interactive range.")
        return

    # Fetch SPY OHLC with 5-day buffer
    spy_start = (analysis_start - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    spy_end = (analysis_end + pd.Timedelta(days=10)).strftime('%Y-%m-%d') # A bit more buffer for end
    print(f"Fetching SPY OHLC from {spy_start} to {spy_end}...")
    spy_ohlc = data_engine.get_ticker_ohlc('SPY', spy_start, spy_end)

    # 2. Precomputation
    print(f"Precomputing TDA results for {len(available_dates)} days...")
    precomputed_results = []
    
    for i, current_date in enumerate(available_dates):
        window_start = current_date - pd.DateOffset(months=args.window)
        sub_df = df.loc[window_start:current_date]
        
        if sub_df.empty or len(sub_df) < 20:
            precomputed_results.append(None)
            continue
            
        corr_matrix = sub_df.corr()
        dist_matrix = tda_manager.correlation_to_distance(corr_matrix)
        coords = tda_manager.get_3d_projection(dist_matrix)
        dgms = tda_manager.compute_persistence(dist_matrix)
        betti = tda_manager.calculate_betti_numbers(dgms, threshold=0.5)
        chi = tda_manager.calculate_euler_characteristic(betti)
        avg_corr = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean()
        
        precomputed_results.append({
            'date': current_date,
            'corr': corr_matrix,
            'dist': dist_matrix,
            'coords': coords,
            'betti': betti,
            'chi': chi,
            'avg_corr': avg_corr
        })
        if (i+1) % 5 == 0 or i == len(available_dates)-1:
            print(f"Progress: {i+1}/{len(available_dates)}", end="\r")
    print("\nPrecomputation complete.")

    # 3. Launch Explorer
    explorer = TDAExplorer(available_dates, precomputed_results, spy_ohlc)
    print("Interactive TDA ready. Use the slider or buttons to navigate.")
    explorer.show()

if __name__ == "__main__":
    main()
