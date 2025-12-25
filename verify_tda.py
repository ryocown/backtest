import os
import pandas as pd
import matplotlib.pyplot as plt
from data import DataEngine
from tda_engine import TDAManager
import visualization as viz

def main():
    # 1. Load Data
    data_engine = DataEngine()
    # Use all cached tickers for robust TDA
    tickers = [
        'AAPL', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'BLK', 'BRK.B', 'BX', 'CAT', 'COST', 
        'GE', 'GLD', 'GOOGL', 'GS', 'JPM', 'KRBN', 'LIN', 'LLY', 'MA', 'META', 
        'MS', 'MSFT', 'NVDA', 'QCOM', 'QQQ', 'SGOV', 'SPGI', 'SPY', 'TSLA', 'TXN', 
        'UNH', 'UNP', 'V', 'VTI', 'WMT', 'XLG', 'XOM'
    ]
    start_date = "2020-01-01"
    end_date = "2025-12-22"
    
    print(f"Loading data for {tickers} from {start_date} to {end_date}...")
    df = data_engine.get_historical_data(tickers, start_date, end_date)
    
    if df.empty:
        print("No data loaded. Check cache.")
        return
        
    # 2. Run TDA Analysis
    print("Running TDA Analysis (Sliding Windows)...")
    tda_manager = TDAManager(window_months=6, step_months=1)
    results = tda_manager.run_analysis(df, betti_threshold=0.5)
    
    if not results:
        print("No TDA results generated.")
        return
        
    print(f"Generated {len(results)} window results.")
    
    # 3. Visualization
    print("Generating TDA Plots...")
    
    # A. TDA Metrics Trend
    viz.plot_tda_metrics_trend(results, title="TDA Market Analysis: 2020-2024")
    plt.savefig("screenshots/tda_metrics_trend.png")
    
    # B. Persistence Diagram & Barcode for a specific window (e.g., during 2020 crash)
    # Find window starting around March 2020
    crash_window = None
    for r in results:
        if r['start'].year == 2020 and r['start'].month == 3:
            crash_window = r
            break
            
    if crash_window:
        print(f"Plotting details for window: {crash_window['start'].date()} to {crash_window['end'].date()}")
        viz.plot_persistence_diagram(crash_window['dgms'], title=f"Persistence Diagram (Crash Window: {crash_window['start'].date()})")
        plt.savefig("screenshots/tda_persistence_diagram.png")
        
        viz.plot_persistence_barcode(crash_window['dgms'], title=f"Persistence Barcode (Crash Window: {crash_window['start'].date()})")
        plt.savefig("screenshots/tda_persistence_barcode.png")
        
        # 3D Simplicial Complex
        dist_matrix = tda_manager.correlation_to_distance(crash_window['corr_matrix'])
        coords = tda_manager.get_3d_projection(dist_matrix)
        viz.plot_3d_simplicial_complex(coords, dist_matrix, threshold=1.0, title=f"3D Simplicial Complex (Crash Window: {crash_window['start'].date()})")
        plt.savefig("screenshots/tda_3d_donut_crash.png")
        
        # Also plot the LAST window for comparison
        last_window = results[-1]
        print(f"Plotting details for latest window: {last_window['start'].date()} to {last_window['end'].date()}")
        viz.plot_persistence_diagram(last_window['dgms'], title=f"Persistence Diagram (Latest Window: {last_window['start'].date()})")
        plt.savefig("screenshots/tda_persistence_diagram_latest.png")
        
        dist_matrix_latest = tda_manager.correlation_to_distance(last_window['corr_matrix'])
        coords_latest = tda_manager.get_3d_projection(dist_matrix_latest)
        viz.plot_3d_simplicial_complex(coords_latest, dist_matrix_latest, threshold=1.0, title=f"3D Simplicial Complex (Latest Window: {last_window['start'].date()})")
        plt.savefig("screenshots/tda_3d_donut_latest.png")
    else:
        # Just plot the last window
        last_window = results[-1]
        print(f"Plotting details for last window: {last_window['start'].date()} to {last_window['end'].date()}")
        viz.plot_persistence_diagram(last_window['dgms'], title=f"Persistence Diagram (Last Window: {last_window['start'].date()})")
        plt.savefig("screenshots/tda_persistence_diagram_latest.png")
        
        viz.plot_persistence_barcode(last_window['dgms'], title=f"Persistence Barcode (Last Window: {last_window['start'].date()})")
        plt.savefig("screenshots/tda_persistence_barcode_latest.png")
        
        # 3D Simplicial Complex
        dist_matrix = tda_manager.correlation_to_distance(last_window['corr_matrix'])
        coords = tda_manager.get_3d_projection(dist_matrix)
        viz.plot_3d_simplicial_complex(coords, dist_matrix, threshold=1.0, title=f"3D Simplicial Complex (Last Window: {last_window['start'].date()})")
        plt.savefig("screenshots/tda_3d_donut_latest.png")

    # C. Correlation Heatmap for the same window
    if crash_window:
        plt.figure(figsize=(10, 8))
        import seaborn as sns
        sns.heatmap(crash_window['corr_matrix'], annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title(f"Correlation Matrix (Crash Window: {crash_window['start'].date()})")
        plt.tight_layout()
        plt.savefig("screenshots/tda_correlation_heatmap.png")

    print("Verification script completed successfully. Plots saved to screenshots/tda_*.png")

if __name__ == "__main__":
    main()
