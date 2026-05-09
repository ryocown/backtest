"""
Backtest Engine for the Institutional Backtester.

This module provides the core BacktestEngine class that orchestrates:
- Configuration loading and validation
- Data fetching and preparation
- Portfolio backtest execution
- Result display and export

Visualization is delegated to PlotOrchestrator.
TDA analysis is delegated to TDARunner.
"""
from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import bt
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from src.data import DataEngine

from src.analytics import AnalyticsEngine

if TYPE_CHECKING:
    from pandas import DataFrame

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Core engine for running portfolio backtests.
    
    Responsibilities:
    - Load and validate YAML portfolio configurations
    - Fetch and clean historical price data
    - Build and run bt.Backtest strategies
    - Display results and orchestrate visualizations
    """
    
    def __init__(self) -> None:
        self.data_engine = DataEngine()

        self.configs: list[dict] = []
        self.overrides: dict = {}
        self.data: Optional[DataFrame] = None
        self.metadata: Optional[dict] = None
        self.results = None

    def set_overrides(self, overrides: dict) -> None:
        """Sets global overrides from CLI."""
        self.overrides = overrides

    def load_configs(self, config_paths: list[str]) -> None:
        """Loads and validates multiple YAML configurations."""
        for path in config_paths:
            try:
                cfg = self._load_single_config(path)
                cfg_name = os.path.splitext(os.path.basename(path))[0]
                cfg['_name'] = cfg_name
                self._validate_config_weights(cfg, cfg_name)
                self.configs.append(cfg)
            except Exception as e:
                logger.error(f"Failed to load config {path}: {e}")
                raise

    def _load_single_config(self, path: str) -> dict:
        """Loads and normalizes a single YAML config."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} not found.")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Normalize configuration
        if 'settings' not in config:
            config['settings'] = {}
        if 'portfolio' not in config:
            config['portfolio'] = {}
        
        p = config['portfolio']
        p['universe'] = p.get('universe') or []
        p['exclusions'] = p.get('exclusions') or []
        p['fixed_weights'] = p.get('fixed_weights') or {}
        p['sector_weights'] = p.get('sector_weights') or {}

        if 'start_date' in config.get('settings', {}):
            config['settings']['start_date'] = datetime.strptime(
                config['settings']['start_date'], '%Y-%m-%d'
            )
        if 'end_date' in config.get('settings', {}):
            config['settings']['end_date'] = datetime.strptime(
                config['settings']['end_date'], '%Y-%m-%d'
            )
        return config

    def _validate_config_weights(self, cfg: dict, config_name: str) -> None:
        """Validates that portfolio weights sum correctly."""
        fixed_weights = cfg['portfolio'].get('fixed_weights', {}) or {}
        sector_weights = cfg['portfolio'].get('sector_weights', {}) or {}
        
        total_fixed = sum(fixed_weights.values())
        total_sector = sum(sector_weights.values())
        total = total_fixed + total_sector
        
        if sector_weights:
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"[{config_name}] Weights do not sum to 100%! ({total:.2%})")
        elif total_fixed > 1.0:
            raise ValueError(f"[{config_name}] fixed_weights ({total_fixed:.2%}) exceed 100%!")

    def prepare_data(self) -> DataFrame:
        """Aggregates tickers, fetches data, metadata, and performs cleaning."""
        if not self.configs:
            raise ValueError("No configurations loaded. Call load_configs first.")

        base_config = self.configs[0]
        start_date = self.overrides.get('start_date') or base_config['settings']['start_date']
        end_date = self.overrides.get('end_date') or base_config['settings']['end_date']
        
        # Collect benchmarks
        benchmarks: set[str] = set()
        if self.overrides.get('benchmarks'):
            benchmarks.update(self.overrides['benchmarks'])
        
        # Collect tickers from all configs
        tickers: set[str] = set()
        
        for cfg in self.configs:
            if not self.overrides.get('benchmarks'):
                benchmarks.update(cfg['settings'].get('benchmarks', []))
            
            raw_univ = cfg['portfolio'].get('universe', [])
            exclusions = set(cfg['portfolio'].get('exclusions', []))
            
            if isinstance(raw_univ, dict):
                for sector_tickers in raw_univ.values():
                    tickers.update([
                        str(t).strip() for t in sector_tickers 
                        if str(t).strip() not in exclusions
                    ])
            elif isinstance(raw_univ, str):
                tickers.update([
                    str(t).strip() for t in raw_univ.split(',')
                    if str(t).strip() not in exclusions
                ])
            else:
                tickers.update([
                    str(t).strip() for t in raw_univ
                    if str(t).strip() not in exclusions
                ])
            
            tickers.update(cfg['portfolio'].get('fixed_weights', {}).keys())

        all_tickers = sorted(list(tickers | benchmarks))
        
        logger.info(f"Fetching data for {len(all_tickers)} symbols...")
        data, failed = self.data_engine.get_historical_data(
            all_tickers, start_date, end_date, fail_on_missing=True
        )
        
        if failed:
            raise ValueError(f"Failed to fetch: {failed}")

        self.metadata = self.data_engine.get_metadata(all_tickers)
        
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Cleaning: drop high-NaN tickers
        max_nan_pct = base_config['settings'].get('max_nan_pct', 0.5)
        nan_pct = data.isna().sum() / len(data)
        drop_tickers = nan_pct[nan_pct > max_nan_pct].index.tolist()
        if drop_tickers:
            logger.warning(f"Dropping high-NaN tickers: {drop_tickers}")
            data = data.drop(columns=drop_tickers)
        
        self.data = data.ffill().bfill()
        return self.data

    def run_backtests(self):
        """Builds and runs backtests for all portfolios and benchmarks."""
        assert self.data is not None, "Data must be prepared before running backtests"
        backtests = []
        available_tickers = set(self.data.columns)

        for cfg in self.configs:
            bt_obj = self._build_portfolio_backtest(cfg, available_tickers)
            if bt_obj:
                backtests.append(bt_obj)

        # Add benchmarks
        benchmarks: set[str] = set()
        if self.overrides.get('benchmarks'):
            benchmarks.update(self.overrides['benchmarks'])
        else:
            for cfg in self.configs:
                benchmarks.update(cfg['settings'].get('benchmarks', []))
        
        for bench in sorted(list(benchmarks)):
            if bench in available_tickers:
                s = bt.Strategy(bench, [
                    bt.algos.RunOnce(),
                    bt.algos.SelectThese([bench]),
                    bt.algos.WeighEqually(),
                    bt.algos.Rebalance()
                ])
                backtests.append(bt.Backtest(s, self.data))

        logger.info(f"Running {len(backtests)} backtests...")
        self.results = bt.run(*backtests)
        
        # Set risk-free rate
        rf = self.overrides.get('rf')
        if rf is None:
            rf = self.configs[0]['settings'].get('risk_free_rate', 0.0)
        if rf != 0.0:
            self.results.set_riskfree_rate(rf)
            
        return self.results

    def _build_portfolio_backtest(self, cfg: dict, available_tickers: set[str]):
        """Builds a single portfolio backtest from config."""
        name = cfg['_name']
        raw_univ = cfg['portfolio'].get('universe', [])
        exclusions = set(cfg['portfolio'].get('exclusions', []))
        
        custom_sector_map: dict[str, str] = {}
        portfolio_tickers: list[str] = []
        
        if isinstance(raw_univ, dict):
            for sector, tickers in raw_univ.items():
                for t in tickers:
                    ticker = str(t).strip()
                    if ticker in available_tickers and ticker not in exclusions:
                        custom_sector_map[ticker] = sector
                        portfolio_tickers.append(ticker)
        else:
            if isinstance(raw_univ, str):
                raw_univ = raw_univ.split(',')
            for t in raw_univ:
                ticker = str(t).strip()
                if ticker in available_tickers and ticker not in exclusions:
                    portfolio_tickers.append(ticker)

        fixed_weights = {
            t: w for t, w in cfg['portfolio'].get('fixed_weights', {}).items()
            if t in available_tickers
        }
        sector_weights_cfg = cfg['portfolio'].get('sector_weights', {})
        
        # Build sector -> tickers mapping
        sector_to_tickers: dict[str, list[str]] = {}
        for t in portfolio_tickers:
            if t in fixed_weights:
                continue
            assert self.metadata is not None
            sec = custom_sector_map.get(t) or self.metadata.get(t, {}).get('sector', 'Other')
            sector_to_tickers.setdefault(sec, []).append(t)
            
        # Calculate target weights
        target_weights = fixed_weights.copy()
        remaining_weight = 1.0 - sum(fixed_weights.values())
        
        allocated_to_sectors = 0.0
        weighted_sectors: list[str] = []
        for sector, weight in sector_weights_cfg.items():
            if sector in sector_to_tickers:
                tickers = sector_to_tickers[sector]
                actual_weight = min(weight, max(0, remaining_weight - allocated_to_sectors))
                if actual_weight > 0:
                    dist_weight = actual_weight / len(tickers)
                    for t in tickers:
                        target_weights[t] = dist_weight
                    allocated_to_sectors += actual_weight
                    weighted_sectors.append(sector)
        
        # Distribute remaining weight equally among unweighted tickers
        remaining_after_sectors = max(0, remaining_weight - allocated_to_sectors)
        unweighted_tickers = [
            t for s, ts in sector_to_tickers.items()
            if s not in weighted_sectors
            for t in ts
        ]
        if unweighted_tickers and remaining_after_sectors > 0.001:
            dist_weight = remaining_after_sectors / len(unweighted_tickers)
            for t in unweighted_tickers:
                target_weights[t] = dist_weight
            
        if not target_weights:
            return None

        # Store for later use
        cfg['_custom_sector_map'] = custom_sector_map
        cfg['_target_weights'] = target_weights
        
        # Map rebalance frequency
        freq = cfg['settings'].get('rebalance_frequency', 'quarterly').lower()
        algo_map = {
            'daily': bt.algos.RunDaily(),
            'weekly': bt.algos.RunWeekly(),
            'monthly': bt.algos.RunMonthly(),
            'quarterly': bt.algos.RunQuarterly(),
            'yearly': bt.algos.RunYearly(),
            'once': bt.algos.RunOnce()
        }
        
        s = bt.Strategy(name, [
            algo_map.get(freq, bt.algos.RunQuarterly()),
            bt.algos.SelectThese(list(target_weights.keys())),
            bt.algos.WeighSpecified(**target_weights),
            bt.algos.Rebalance()
        ])
        return bt.Backtest(s, self.data)

    def display_results(self) -> None:
        """Displays text summary of the results."""
        if not self.results:
            return
        self.results.display()
        
        # Print Robustness Metrics for each portfolio
        self._print_robustness_metrics()

    def _print_robustness_metrics(self) -> None:
        """Prints robustness metrics summary for all portfolio strategies."""
        if self.data is None or self.results is None:
            return
        
        # Get portfolio names (exclude benchmarks)
        portfolio_names = [cfg['_name'] for cfg in self.configs]
        benchmarks: set[str] = set()
        if self.overrides.get('benchmarks'):
            benchmarks.update(self.overrides['benchmarks'])
        else:
            for cfg in self.configs:
                benchmarks.update(cfg['settings'].get('benchmarks', []))
        
        strategies_to_analyze = [
            name for name in self.results.prices.columns 
            if name in portfolio_names
        ]
        
        if not strategies_to_analyze:
            return
        
        print("\n" + "=" * 80)
        print("ROBUSTNESS METRICS SUMMARY")
        print("=" * 80)
        print("Tests Sharpe Ratio stability across rolling windows (20-252 days)")
        print("-" * 80)
        
        for strategy in strategies_to_analyze:
            metrics = AnalyticsEngine.get_robustness_metrics(
                self.results.prices, strategy
            )
            
            if 'error' in metrics:
                print(f"\n{strategy}: {metrics['error']}")
                continue
            
            print(f"\n┌─ {strategy} {'─' * (75 - len(strategy))}")
            print(f"│  Mean Sharpe:          {metrics['mean_sharpe']:>8.2f}")
            print(f"│  StdDev Sharpe:        {metrics['std_sharpe']:>8.2f}")
            print(f"├────────────────────────────────────────────────────────────────")
            print(f"│  Stability Score:      {metrics['stability_score']:>8.2f}  ({metrics['stability_rating']})")
            print(f"│  Window Sensitivity:   {metrics['window_sensitivity']:>8.3f}  ({metrics['sensitivity_rating']})")
            print(f"│  Plateau Width (>1.0): {metrics['plateau_width_pct']:>7.1f}%  ({metrics['plateau_rating']})")
            print(f"│  Coef. of Variation:   {metrics['coef_of_variation']:>8.2f}  ({metrics['cov_rating']})")
            print(f"│  Temporal Consistency: {metrics['temporal_consistency']:>8.2f}")
            print(f"└─ Windows tested: {metrics['windows_tested']} ({metrics['min_window']}-{metrics['max_window']} days)")
        
        print("\n" + "-" * 80)
        print("Interpretation:")
        print("  • Stability Score: Higher = more robust (>2.0 Excellent, >1.5 Good)")
        print("  • Window Sensitivity: Lower = less sensitive (<0.05 Low, <0.15 Moderate)")
        print("  • Plateau Width: Wider = robust across parameters (>80% Excellent)")
        print("  • CoV: Lower = more consistent (<0.3 Excellent, <0.5 Good)")
        print("=" * 80 + "\n")

    def plot_all(self) -> None:
        """Displays all visualizations using PlotOrchestrator."""
        if not self.results:
            return
        
        # Switch to hardware-accelerated backend if available
        self._try_upgrade_backend()
        
        # Get graph selection from GUI
        selection = self._get_graph_selection()
        if selection is None:
            logger.info("Graph selection cancelled.")
            return
        
        # Delegate to PlotOrchestrator
        from src.plot_orchestrator import PlotOrchestrator
        
        assert self.data is not None
        assert self.metadata is not None
        
        orchestrator = PlotOrchestrator(
            results=self.results,
            data=self.data,
            metadata=self.metadata,
            configs=self.configs,
            overrides=self.overrides
        )
        orchestrator.render_all(selection)
        
        # Launch TDA if selected
        if selection and selection.get('tda', {}).get('enabled'):
            tda_cfg = selection['tda']
            self.run_tda_explorer(
                start_date=tda_cfg['start'],
                end_date=tda_cfg['end'],
                window_months=tda_cfg['window']
            )
    
    def _try_upgrade_backend(self) -> None:
        """Attempts to switch to hardware-accelerated matplotlib backend."""
        try:
            import matplotlib
            current_backend = matplotlib.get_backend()
            if current_backend.lower() in ('agg', 'tkagg'):
                matplotlib.use('Qt5Agg', force=False)
                new_backend = matplotlib.get_backend()
                if new_backend.lower().startswith('qt'):
                    logger.info(f"Switched to hardware-accelerated backend: {new_backend}")
        except Exception as e:
            logger.debug(f"Could not switch to Qt backend: {e}")
    
    def _get_graph_selection(self) -> Optional[dict]:
        """Shows the graph selector GUI and returns selection."""
        try:
            from src.gui import GraphSelector
            from src.plot_orchestrator import PlotOrchestrator
            
            strategies = [cfg['_name'] for cfg in self.configs]
            benchmarks: set[str] = set()
            if self.overrides.get('benchmarks'):
                benchmarks.update(self.overrides['benchmarks'])
            else:
                for cfg in self.configs:
                    benchmarks.update(cfg['settings'].get('benchmarks', []))
            
            selector = GraphSelector(
                strategies,
                sorted(list(benchmarks)),
                PlotOrchestrator.GLOBAL_PLOTS,
                PlotOrchestrator.PER_ENTITY_PLOTS
            )
            return selector.get_selection()
        except Exception as e:
            logger.error(f"GUI failed: {e}. Rendering all graphs.")
            return None

    def get_tda_results(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        window_months: int = 6
    ) -> tuple[list, Optional[pd.DatetimeIndex], Optional[DataFrame]]:
        """Precomputes TDA results and returns them."""
        from src.tda_engine import TDAManager
        import pandas as pd

        # Setup Parameters
        if self.data is not None and not self.data.empty:
            last_date = self.data.index[-1]
            default_start = (last_date - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
            default_end = last_date.strftime('%Y-%m-%d')
        else:
            default_start = "2020-02-18"
            default_end = "2020-04-06"

        start_dt = pd.to_datetime(start_date or default_start)
        end_dt = pd.to_datetime(end_date or default_end)
        data_start = (start_dt - pd.DateOffset(months=window_months)).strftime('%Y-%m-%d')

        # Get Tickers
        tickers = self.data_engine.list_cached_tickers()
        if not tickers:
            logger.warning("No cached tickers found for TDA analysis.")
            return [], None, None

        logger.info(f"Loading data for {len(tickers)} tickers for TDA...")
        df = self.data_engine.get_historical_data(
            tickers, data_start, end_dt.strftime('%Y-%m-%d')
        )
        df = df.dropna(axis=1)

        tda_manager = TDAManager(window_months=window_months)
        return tda_manager.compute_tda_for_range(df, start_dt, end_dt, self.data_engine)

    def run_tda_explorer(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        window_months: int = 6
    ) -> None:
        """Launches the interactive TDA explorer."""
        from src.interactive_tda import TDAExplorer
        
        results, dates, spy_ohlc = self.get_tda_results(start_date, end_date, window_months)
        if not results:
            return

        explorer = TDAExplorer(dates, results, spy_ohlc)
        explorer.show()

    def web_export(
        self,
        output_path: Optional[str] = None,
        tda_params: Optional[dict] = None
    ) -> None:
        """Exports all results to a web-ready JSON package."""
        from src.web_export import export_to_web
        
        tda_results = None
        if tda_params:
            tda_results, _, _ = self.get_tda_results(
                start_date=tda_params.get('start'),
                end_date=tda_params.get('end'),
                window_months=tda_params.get('window', 6)
            )
            
        sector_maps = {
            cfg['_name']: cfg.get('_custom_sector_map', {})
            for cfg in self.configs
        }

        export_to_web(
            results=self.results,
            data=self.data,
            metadata=self.metadata,
            tda_results=tda_results,
            sector_maps=sector_maps,
            output_path=output_path
        )
