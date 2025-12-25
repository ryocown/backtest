import bt
import yaml
import os
import logging
import concurrent.futures
from functools import partial
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from data import DataEngine
from fundamentals import FundamentalEngine
from analytics import AnalyticsEngine
import visualization as viz

logger = logging.getLogger(__name__)

def _precompute_tda_worker(current_date, full_df, window_months):
    """Worker function for parallel TDA precomputation."""
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from tda_engine import TDAManager
    import numpy as np
    import pandas as pd
    
    tda_manager = TDAManager(window_months=window_months)
    window_start = current_date - pd.DateOffset(months=window_months)
    sub_df = full_df.loc[window_start:current_date]

    if sub_df.empty or len(sub_df) < 20:
        return None

    corr_matrix = sub_df.corr()
    dist_matrix = tda_manager.correlation_to_distance(corr_matrix)
    coords = tda_manager.get_3d_projection(dist_matrix)
    dgms = tda_manager.compute_persistence(dist_matrix)
    betti = tda_manager.calculate_betti_numbers(dgms, threshold=0.5)
    chi = tda_manager.calculate_euler_characteristic(betti)
    avg_corr = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean()

    return {
        'date': current_date,
        'corr': corr_matrix,
        'dist': dist_matrix,
        'coords': coords,
        'dgms': dgms,
        'betti': betti,
        'chi': chi,
        'avg_corr': avg_corr
    }

def _run_tda_explorer_worker(available_dates, precomputed_results, spy_ohlc):
    """Launches TDA Explorer in a subprocess."""
    import matplotlib.pyplot as plt
    from interactive_tda import TDAExplorer
    try:
        explorer = TDAExplorer(available_dates, precomputed_results, spy_ohlc)
        explorer.show()
    except Exception as e:
        print(f"TDA Explorer failed in subprocess: {e}")

def run_plot_process(task):
    """Worker function for parallel plotting."""
    func_obj, args, kwargs = task
    import matplotlib.pyplot as plt
    import visualization as viz
    try:
        if isinstance(func_obj, tuple) and len(func_obj) == 2:
            obj, method_name = func_obj
            getattr(obj, method_name)(*args, **kwargs)
        else:
            func_obj(*args, **kwargs)
        
        # Ensure legends are interactive even in subprocesses
        try:
            viz.make_legend_interactive(plt.gcf())
        except:
            pass
        plt.show() 
    except Exception as e:
        import traceback
        print(f"Plotting failed in subprocess: {e}")
        traceback.print_exc()

class BacktestEngine:
    def __init__(self):
        self.data_engine = DataEngine()
        self.fundamental_engine = FundamentalEngine()
        self.configs = []
        self.overrides = {}
        self.data = None
        self.metadata = None
        self.results = None

    def set_overrides(self, overrides):
        """Sets global overrides from CLI."""
        self.overrides = overrides

    def load_configs(self, config_paths):
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

    def _load_single_config(self, path):
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
            config['settings']['start_date'] = datetime.strptime(config['settings']['start_date'], '%Y-%m-%d')
        if 'end_date' in config.get('settings', {}):
            config['settings']['end_date'] = datetime.strptime(config['settings']['end_date'], '%Y-%m-%d')
        return config

    def _validate_config_weights(self, cfg, config_name):
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

    def prepare_data(self):
        """Aggregates tickers, fetches data, metadata, and performs cleaning."""
        if not self.configs:
            raise ValueError("No configurations loaded. Call load_configs first.")

        base_config = self.configs[0]
        start_date = self.overrides.get('start_date') or base_config['settings']['start_date']
        end_date = self.overrides.get('end_date') or base_config['settings']['end_date']
        
        benchmarks = set()
        if self.overrides.get('benchmarks'):
            benchmarks.update(self.overrides['benchmarks'])
            
        tickers = set()
        
        for cfg in self.configs:
            if not self.overrides.get('benchmarks'):
                benchmarks.update(cfg['settings'].get('benchmarks', []))
            raw_univ = cfg['portfolio'].get('universe', [])
            exclusions = set(cfg['portfolio'].get('exclusions', []))
            
            if isinstance(raw_univ, dict):
                for sector_tickers in raw_univ.values():
                    tickers.update([str(t).strip() for t in sector_tickers if str(t).strip() not in exclusions])
            elif isinstance(raw_univ, str):
                tickers.update([str(t).strip() for t in raw_univ.split(',') if str(t).strip() not in exclusions])
            else:
                tickers.update([str(t).strip() for t in raw_univ if str(t).strip() not in exclusions])
            
            tickers.update(cfg['portfolio'].get('fixed_weights', {}).keys())

        all_tickers = sorted(list(tickers | benchmarks))
        
        logger.info(f"Fetching data for {len(all_tickers)} symbols...")
        data, failed = self.data_engine.get_historical_data(all_tickers, start_date, end_date, fail_on_missing=True)
        
        if failed:
            raise ValueError(f"Failed to fetch: {failed}")

        self.metadata = self.data_engine.get_metadata(all_tickers)
        
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Cleaning
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

        # Benchmarks
        benchmarks = set()
        if self.overrides.get('benchmarks'):
            benchmarks.update(self.overrides['benchmarks'])
        else:
            for cfg in self.configs:
                benchmarks.update(cfg['settings'].get('benchmarks', []))
        
        for bench in sorted(list(benchmarks)):
            if bench in available_tickers:
                s = bt.Strategy(bench, [bt.algos.RunOnce(), bt.algos.SelectThese([bench]), bt.algos.WeighEqually(), bt.algos.Rebalance()])
                backtests.append(bt.Backtest(s, self.data))

        logger.info(f"Running {len(backtests)} backtests...")
        self.results = bt.run(*backtests)
        
        rf = self.overrides.get('rf')
        if rf is None:
            rf = self.configs[0]['settings'].get('risk_free_rate', 0.0)
            
        if rf != 0.0:
            self.results.set_riskfree_rate(rf)
            
        return self.results

    def _build_portfolio_backtest(self, cfg, available_tickers):
        name = cfg['_name']
        raw_univ = cfg['portfolio'].get('universe', [])
        exclusions = set(cfg['portfolio'].get('exclusions', []))
        
        custom_sector_map = {}
        portfolio_tickers = []
        
        if isinstance(raw_univ, dict):
            for sector, tickers in raw_univ.items():
                for t in tickers:
                    ticker = str(t).strip()
                    if ticker in available_tickers and ticker not in exclusions:
                        custom_sector_map[ticker] = sector
                        portfolio_tickers.append(ticker)
        else:
            if isinstance(raw_univ, str): raw_univ = raw_univ.split(',')
            for t in raw_univ:
                ticker = str(t).strip()
                if ticker in available_tickers and ticker not in exclusions:
                    portfolio_tickers.append(ticker)

        fixed_weights = {t: w for t, w in cfg['portfolio'].get('fixed_weights', {}).items() if t in available_tickers}
        sector_weights_cfg = cfg['portfolio'].get('sector_weights', {})
        
        sector_to_tickers = {}
        for t in portfolio_tickers:
            if t in fixed_weights: continue
            assert self.metadata is not None, "Data must be prepared before running backtests"
            sec = custom_sector_map.get(t) or self.metadata.get(t, {}).get('sector', 'Other')
            sector_to_tickers.setdefault(sec, []).append(t)
            
        target_weights = fixed_weights.copy()
        remaining_weight = 1.0 - sum(fixed_weights.values())
        
        allocated_to_sectors = 0.0
        weighted_sectors = []
        for sector, weight in sector_weights_cfg.items():
            if sector in sector_to_tickers:
                tickers = sector_to_tickers[sector]
                actual_weight = min(weight, max(0, remaining_weight - allocated_to_sectors))
                if actual_weight > 0:
                    dist_weight = actual_weight / len(tickers)
                    for t in tickers: target_weights[t] = dist_weight
                    allocated_to_sectors += actual_weight
                    weighted_sectors.append(sector)
        
        remaining_after_sectors = max(0, remaining_weight - allocated_to_sectors)
        unweighted_tickers = [t for s, ts in sector_to_tickers.items() if s not in weighted_sectors for t in ts]
        if unweighted_tickers and remaining_after_sectors > 0.001:
            dist_weight = remaining_after_sectors / len(unweighted_tickers)
            for t in unweighted_tickers: target_weights[t] = dist_weight
            
        if not target_weights:
            return None

        cfg['_custom_sector_map'] = custom_sector_map
        cfg['_target_weights'] = target_weights
        freq = cfg['settings'].get('rebalance_frequency', 'quarterly').lower()
        algo_map = {'daily': bt.algos.RunDaily(), 'weekly': bt.algos.RunWeekly(), 'monthly': bt.algos.RunMonthly(), 'quarterly': bt.algos.RunQuarterly(), 'yearly': bt.algos.RunYearly(), 'once': bt.algos.RunOnce()}
        
        s = bt.Strategy(name, [algo_map.get(freq, bt.algos.RunQuarterly()), bt.algos.SelectThese(list(target_weights.keys())), bt.algos.WeighSpecified(**target_weights), bt.algos.Rebalance()])
        return bt.Backtest(s, self.data)

    def display_results(self):
        """Displays text summary of the results."""
        if not self.results: return
        self.results.display()

    def plot_all(self):
        """Displays all visualizations."""
        if not self.results: return
        
        # Attempt to use hardware-accelerated backend (Qt) if available
        try:
            import matplotlib
            current_backend = matplotlib.get_backend()
            if current_backend.lower() == 'agg' or current_backend.lower() == 'tkagg':
                # Try to switch to Qt5 for better performance on Linux
                matplotlib.use('Qt5Agg', force=False)
                new_backend = matplotlib.get_backend()
                if new_backend.lower().startswith('qt'):
                    logger.info(f"Switched to hardware-accelerated backend: {new_backend}")
                else:
                    logger.info(f"Staying with backend: {current_backend}")
        except Exception as e:
            logger.debug(f"Could not switch to Qt backend: {e}")

        # Check for interactive backend
        backend = plt.get_backend()
        if backend.lower() == 'agg':
            logger.warning("Current Matplotlib backend is 'agg' (non-interactive). "
                           "Visualizations will NOT be displayed. "
                           "Please install an interactive backend like 'python3-tk' or 'PyQt5'.")
        
        # Prevent "More than 20 figures" memory warning
        plt.close('all')
        
        # --- GUI Selection ---
        try:
            from gui import GraphSelector
            strategies = [cfg['_name'] for cfg in self.configs]
            benchmarks = set()
            if self.overrides.get('benchmarks'):
                benchmarks.update(self.overrides['benchmarks'])
            else:
                for cfg in self.configs:
                    benchmarks.update(cfg['settings'].get('benchmarks', []))
            benchmarks = sorted(list(benchmarks))
            
            global_plots = [
                "Equity Curve", "Drawdowns", "Rolling Volatility", "Rolling Sharpe", 
                "Rolling Sortino", "Return Distribution", "Correlation Matrix", 
                "Upside/Downside Capture", "Rolling Info Ratio", "Rolling Alpha/Beta", 
                "Valuation Comparison", "Historical P/E Trend", "Global Sector Correlation"
            ]
            per_entity_plots = [
                "Heatmap", "Risk (User Class)", "Risk (Market Sector)", 
                "Correlation (User Class)", "Performance Attribution", 
                "Sharpe Robustness", "Drawdown-Recovery", "Sectoral Allocation"
            ]
            
            selector = GraphSelector(strategies, benchmarks, global_plots, per_entity_plots)
            selection = selector.get_selection()
            if not selection:
                logger.info("Graph selection cancelled.")
                return
        except Exception as e:
            logger.error(f"GUI failed: {e}. Rendering all graphs.")
            selection = None

        # --- Parallel Plot Dispatcher ---
        plot_tasks = []

        def add_task(func, *args, **kwargs):
            plot_tasks.append((func, args, kwargs))

        def is_selected(category, entity, plot):
            if not selection: return True
            if category == 'global':
                return selection['global'].get(plot, True)
            return selection['per_entity'].get(entity, {}).get(plot, True)

        if is_selected('global', None, "Equity Curve"): 
            # self.results.plot is a bound method, pass as tuple (obj, "method")
            add_task((self.results, "plot"), title="Backtest Results")
        
        if viz:
            if is_selected('global', None, "Drawdowns"): add_task(viz.plot_drawdowns, self.results)
            if is_selected('global', None, "Rolling Volatility"): add_task(viz.plot_rolling_volatility, self.results)
            if is_selected('global', None, "Rolling Sharpe"): add_task(viz.plot_rolling_sharpe, self.results)
            if is_selected('global', None, "Rolling Sortino"): add_task(viz.plot_rolling_sortino, self.results)
            if is_selected('global', None, "Return Distribution"): add_task(viz.plot_return_distribution, self.results)
            if is_selected('global', None, "Correlation Matrix"): add_task(viz.plot_correlation_matrix, self.results)
            
            # Primary benchmark selection: 
            primary_benchmark = self.configs[0]['settings'].get('primary_benchmark')
            if not primary_benchmark:
                if 'SPY' in benchmarks: primary_benchmark = 'SPY'
                elif benchmarks: primary_benchmark = benchmarks[0]
                else: primary_benchmark = 'SPY'

            if is_selected('global', None, "Upside/Downside Capture"):
                add_task(viz.plot_upside_downside_capture, self.results, benchmark=primary_benchmark)
            if is_selected('global', None, "Rolling Info Ratio"):
                add_task(viz.plot_rolling_info_ratio, self.results, benchmark=primary_benchmark)
            
            if is_selected('global', None, "Rolling Alpha/Beta"):
                add_task(viz.plot_rolling_alpha_beta,
                    self.results.prices[[cfg['_name'] for cfg in self.configs]], 
                    self.results.prices[[primary_benchmark]],
                    title_prefix="Strategies vs " + primary_benchmark
                )
                
            all_entities = strategies + benchmarks

            for name in all_entities:
                if name in self.results.prices.columns:
                    if is_selected('per_entity', name, "Heatmap"):
                        add_task(viz.plot_monthly_returns_heatmap, self.results, name)
            
            # Prepare global market sector map for correlation
            assert self.data is not None
            market_sector_map = {t: self.metadata.get(t, {}).get('sector', 'Other') for t in self.data.columns}

            for name in strategies:
                if name in self.results.prices.columns:
                    cfg = next(c for c in self.configs if c['_name'] == name)
                    custom_map = cfg.get('_custom_sector_map', {})
                    fixed_weights = (cfg['portfolio'].get('fixed_weights') or {}).keys()
                    user_class_map = custom_map.copy()
                    for fw in fixed_weights:
                        if fw not in user_class_map: user_class_map[fw] = fw

                    if is_selected('per_entity', name, "Risk (User Class)"):
                        add_task(viz.plot_risk_contribution, self.results, name, prices=self.data, group_map=user_class_map, title=f"Risk: User Asset Class ({name})")
                    if is_selected('per_entity', name, "Correlation (User Class)"):
                        add_task(viz.plot_grouped_correlation_matrix, self.data, user_class_map, title=f"Correlation: User Asset Class ({name})")
                    
                    if is_selected('per_entity', name, "Performance Attribution") and '_target_weights' in cfg:
                         add_task(viz.plot_return_attribution,
                             self.data[list(cfg['_target_weights'].keys())], 
                             cfg['_target_weights'],
                             group_map=user_class_map,
                             title=f"Performance Attribution ({name})"
                         )
                    
                    if is_selected('per_entity', name, "Risk (Market Sector)"):
                        add_task(viz.plot_risk_contribution, self.results, name, prices=self.data, group_map=market_sector_map, title=f"Risk: Market Sectors ({name})")
                    
                    if is_selected('per_entity', name, "Sharpe Robustness"):
                        add_task(viz.plot_sharpe_robustness_surface, self.results, name)
                    if is_selected('per_entity', name, "Drawdown-Recovery"):
                        add_task(viz.plot_drawdown_recovery_surface, self.results, name)
                    if is_selected('per_entity', name, "Sectoral Allocation"):
                        # self._plot_sectoral_allocation is a bound method, pass as tuple
                        add_task((self, "_plot_sectoral_allocation"), name, cfg.get('_custom_sector_map'))

            if is_selected('global', None, "Global Sector Correlation"):
                add_task(viz.plot_grouped_correlation_matrix, self.data, market_sector_map, title="Correlation: Global Market Sectors")
            
            if is_selected('global', None, "Valuation Comparison"):
                add_task(viz.plot_valuation_comparison, self.results, self.metadata)
            if is_selected('global', None, "Historical P/E Trend"):
                add_task(viz.plot_historical_pe_trend, self.results, self.fundamental_engine, prices=self.data)

        # --- Launch TDA Explorer in parallel if selected ---
        if selection and selection.get('tda', {}).get('enabled'):
            tda_cfg = selection['tda']
            logger.info("Precomputing TDA results for interactive explorer...")
            tda_results, dates, spy_ohlc = self.get_tda_results(
                start_date=tda_cfg['start'], 
                end_date=tda_cfg['end'], 
                window_months=tda_cfg['window']
            )
            if tda_results:
                add_task(_run_tda_explorer_worker, dates, tda_results, spy_ohlc)
        
        # --- Launch Parallel Plotting ---
        logger.info(f"Launching {len(plot_tasks)} visualization processes...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Wrap in list() to block until all windows are closed AND catch errors
            list(executor.map(run_plot_process, plot_tasks))

    def _plot_sectoral_allocation(self, strategy_name, custom_sector_map):
        assert self.results is not None, "Results is None when plotting sectoral allocation. This should never happen."
        assert self.metadata is not None, "Metadata is None when plotting sectoral allocation. This should never happen."

        weights = self.results.get_security_weights(strategy_name)

        sector_map = {t: (custom_sector_map.get(t) if custom_sector_map else None) or self.metadata.get(t, {}).get('sector', 'Other') for t in weights.columns}
        sector_weights = weights.T.groupby(sector_map).sum().T
        ax = sector_weights.plot(kind='area', stacked=True, figsize=(10, 6), title=f"Sector: {strategy_name}")
        ax.set_ylabel("Weight")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        
        try:
            import visualization as viz
            viz.make_legend_interactive(plt.gcf())
        except ImportError:
            pass

    def get_tda_results(self, start_date=None, end_date=None, window_months=6):
        """Precomputes TDA results and returns them as a list of dictionaries."""
        from tda_engine import TDAManager
        import numpy as np

        # 1. Setup Parameters
        if self.data is not None and not self.data.empty:
            last_date = self.data.index[-1]
            default_start = (last_date - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
            default_end = last_date.strftime('%Y-%m-%d')
        else:
            default_start = "2020-02-18"
            default_end = "2020-04-06"

        start_date = pd.to_datetime(start_date or default_start)
        end_date = pd.to_datetime(end_date or default_end)
        data_start = (start_date - pd.DateOffset(months=window_months)).strftime('%Y-%m-%d')

        # 2. Get Tickers
        tickers = self.data_engine.list_cached_tickers()
        if not tickers:
            logger.warning("No cached tickers found for TDA analysis.")
            return [], None, None

        logger.info(f"Loading data for {len(tickers)} tickers for TDA...")
        df = self.data_engine.get_historical_data(tickers, data_start, end_date.strftime('%Y-%m-%d'))
        
        # Robustness: fill minor gaps before dropping.
        # This prevents dropping tickers that miss just one or two days in the lookback window.
        df = df.ffill().bfill().dropna(axis=1)
        
        if df.empty or len(df.columns) < 2:
            logger.warning("Not enough tickers with sufficient data for TDA analysis.")
            return [], None, None
            
        logger.info(f"TDA Analysis starting with {len(df.columns)} tickers.")

        available_dates = df.index[(df.index >= start_date) & (df.index <= end_date)]
        if len(available_dates) == 0:
            logger.warning("No trading days found in the requested TDA range.")
            return [], None, None

        # 3. Fetch SPY OHLC
        spy_start = (start_date - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        # Cap end date to today to avoid "future data" fetch errors
        now_str = datetime.now().strftime('%Y-%m-%d')
        spy_end = min((end_date + pd.Timedelta(days=10)).strftime('%Y-%m-%d'), now_str)
        spy_ohlc = self.data_engine.get_ticker_ohlc('SPY', spy_start, spy_end)

        # 4. Parallel Precomputation
        logger.info(f"Precomputing TDA for {len(available_dates)} days in parallel...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            precomputed_results = list(executor.map(
                partial(_precompute_tda_worker, full_df=df, window_months=window_months), 
                available_dates
            ))
        print("\nPrecomputation complete.")
        return precomputed_results, available_dates, spy_ohlc

    def run_tda_explorer(self, start_date=None, end_date=None, window_months=6):
        """Launches the interactive TDA explorer using all cached tickers."""
        from interactive_tda import TDAExplorer
        
        results, dates, spy_ohlc = self.get_tda_results(start_date, end_date, window_months)
        if not results:
            return

        # 5. Launch Explorer
        explorer = TDAExplorer(dates, results, spy_ohlc)
        explorer.show()

    def web_export(self, output_path=None, tda_params=None):
        """Exports all results to a web-ready JSON package."""
        from web_export import export_to_web
        
        tda_results = None
        if tda_params:
            tda_results, _, _ = self.get_tda_results(
                start_date=tda_params.get('start'),
                end_date=tda_params.get('end'),
                window_months=tda_params.get('window', 6)
            )
            
        sector_maps = {cfg['_name']: cfg.get('_custom_sector_map', {}) for cfg in self.configs}

        export_to_web(
            results=self.results,
            data=self.data,
            metadata=self.metadata,
            tda_results=tda_results,
            sector_maps=sector_maps,
            output_path=output_path
        )
