import bt
import yaml
import os
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from data import DataEngine
from fundamentals import FundamentalEngine
from analytics import AnalyticsEngine

logger = logging.getLogger(__name__)

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
        
        self.results.plot(title="Backtest Results")
        
        # Advanced Visualizations need visualization module
        try:
            import visualization as viz
            viz.make_legend_interactive(plt.gcf())
        except ImportError:
            viz = None
        
        # Sectoral plots
        for cfg in self.configs:
            name = cfg['_name']
            if name in self.results.prices.columns:
                self._plot_sectoral_allocation(name, cfg.get('_custom_sector_map'))

        # Advanced Visualizations
        try:
            import visualization as viz
            viz.plot_drawdowns(self.results)
            viz.plot_rolling_volatility(self.results)
            viz.plot_rolling_sharpe(self.results)
            viz.plot_rolling_sortino(self.results)
            viz.plot_return_distribution(self.results)
            viz.plot_correlation_matrix(self.results)
            
            # Heatmaps and Capture for strategies and benchmarks
            benchmarks = set()
            for cfg in self.configs:
                benchmarks.update(cfg['settings'].get('benchmarks', []))
            
            # Primary benchmark selection: 
            # 1. Explicitly defined in config settings 
            # 2. 'SPY' if it exists in benchmarks 
            # 3. First alphabetical benchmark
            primary_benchmark = self.configs[0]['settings'].get('primary_benchmark')
            if not primary_benchmark:
                if 'SPY' in benchmarks:
                    primary_benchmark = 'SPY'
                elif benchmarks:
                    primary_benchmark = sorted(list(benchmarks))[0]
                else:
                    primary_benchmark = 'SPY'

            viz.plot_upside_downside_capture(self.results, benchmark=primary_benchmark)
            viz.plot_rolling_info_ratio(self.results, benchmark=primary_benchmark)
            
            # Global Rolling Alpha/Beta (Strategies relative to primary benchmark)
            viz.plot_rolling_alpha_beta(
                self.results.prices[[cfg['_name'] for cfg in self.configs]], 
                self.results.prices[[primary_benchmark]],
                title_prefix="Strategies vs " + primary_benchmark
            )
                
            all_plot_names = [cfg['_name'] for cfg in self.configs] + sorted(list(benchmarks))
            strategies = [cfg['_name'] for cfg in self.configs]

            for name in all_plot_names:
                if name in self.results.prices.columns:
                    viz.plot_monthly_returns_heatmap(self.results, name)
            
            # Prepare global market sector map for correlation
            assert self.data is not None
            market_sector_map = {t: self.metadata.get(t, {}).get('sector', 'Other') for t in self.data.columns}

            for name in strategies:
                if name in self.results.prices.columns:
                    # Window 1: User-defined asset classes (Universe Keys + Fixed Weights as categories)
                    cfg = next(c for c in self.configs if c['_name'] == name)
                    custom_map = cfg.get('_custom_sector_map', {})
                    fixed_weights = (cfg['portfolio'].get('fixed_weights') or {}).keys()
                    
                    # Create a map where universe keys are groups, and fixed weights are their own groups
                    user_class_map = custom_map.copy()
                    for fw in fixed_weights:
                        if fw not in user_class_map:
                            user_class_map[fw] = fw # e.g., GLD -> GLD
                    

                    viz.plot_risk_contribution(self.results, name, prices=self.data, group_map=user_class_map, title=f"Risk: User Asset Class ({name})")
                    viz.plot_grouped_correlation_matrix(self.data, user_class_map, title=f"Correlation: User Asset Class ({name})")
                    
                    # Performance Attribution for each strategy
                    if '_target_weights' in cfg:
                         viz.plot_return_attribution(
                             self.data[list(cfg['_target_weights'].keys())], 
                             cfg['_target_weights'],
                             group_map=user_class_map,
                             title=f"Performance Attribution ({name})"
                         )
                    
                    # Window 2: Market Classification (from metadata)
                    viz.plot_risk_contribution(self.results, name, prices=self.data, group_map=market_sector_map, title=f"Risk: Market Sectors ({name})")
            
            # Global Market Sector Correlation
            viz.plot_grouped_correlation_matrix(self.data, market_sector_map, title="Correlation: Global Market Sectors")
            
            # Valuation Metrics Comparison (Snapshot & Trend)
            viz.plot_valuation_comparison(self.results, self.metadata)
            viz.plot_historical_pe_trend(self.results, self.fundamental_engine, prices=self.data)
        except Exception as e:
            logger.error(f"Advanced viz failed: {e}")

        plt.show()

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
