import argparse
import logging
from datetime import datetime
from engine import BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for backtesting.
    Supports CLI overrides for global settings.
    """
    parser = argparse.ArgumentParser(description="Institutional Backtester CLI")
    parser.add_argument('configs', nargs='*', default=['config.yaml'], help="Paths to YAML configuration files")
    parser.add_argument('--start_date', type=str, default="2020-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument('--benchmarks', type=str, default="SPY,QQQ,VTI", help="Comma-delimited benchmark tickers")
    parser.add_argument('--rf', type=float, default=0.04, help="Risk-free rate (e.g., 0.04 for 4%)")
    parser.add_argument('--no-graph', action='store_true', help="Do not render any graphs")
    parser.add_argument('--tda', action='store_true', help="Enable TDA Interactive Explorer")
    parser.add_argument('--tda-start', type=str, default="2020-02-18", help="TDA start date (YYYY-MM-DD)")
    parser.add_argument('--tda-end', type=str, default="2020-04-06", help="TDA end date (YYYY-MM-DD)")
    parser.add_argument('--tda-window', type=int, default=6, help="TDA window size in months")
    parser.add_argument('--web-export', action='store_true', help="Export results to JSON for web UI")
    parser.add_argument('--export-path', type=str, help="Custom path for web export JSON")
    
    args = parser.parse_args()

    # 1. Initialize Engine
    engine = BacktestEngine()
    
    # 2. Set Overrides from CLI
    overrides = {
        'start_date': datetime.strptime(args.start_date, '%Y-%m-%d'),
        'end_date': datetime.strptime(args.end_date, '%Y-%m-%d'),
        'benchmarks': [s.strip() for s in args.benchmarks.split(',') if s.strip()],
        'rf': args.rf
    }
    engine.set_overrides(overrides)
    
    # 3. Load Configurations
    engine.load_configs(args.configs)
    
    # 4. Prepare Data (Fetch, Metadata, Clean)
    engine.prepare_data()
    
    # 5. Run Backtests
    engine.run_backtests()
    
    # 6. Plot & Display Results
    engine.display_results()
    
    # --- Web Export Logic ---
    if args.web_export:
        tda_params = None
        if args.tda:
            tda_params = {
                'start': args.tda_start,
                'end': args.tda_end,
                'window': args.tda_window
            }
        engine.web_export(output_path=args.export_path, tda_params=tda_params)

    if not args.no_graph:
        engine.plot_all()
    
    # 7. Standalone TDA if requested via CLI (and not already run by web_export)
    if args.tda and not args.web_export:
        engine.run_tda_explorer(start_date=args.tda_start, end_date=args.tda_end, window_months=args.tda_window)

if __name__ == "__main__":
    main()
