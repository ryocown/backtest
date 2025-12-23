import bt
import yaml
import os
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from data import DataEngine

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self):
        self.data_engine = DataEngine()
        self.configs = []
        self.data = None
        self.metadata = None
        self.results = None

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
        
        config['settings']['start_date'] = datetime.strptime(config['settings']['start_date'], '%Y-%m-%d')
        config['settings']['end_date'] = datetime.strptime(config['settings']['end_date'], '%Y-%m-%d')
        return config

    def _validate_config_weights(self, cfg, config_name):
        fixed_weights = cfg['portfolio'].get('fixed_weights', {})
        sector_weights = cfg['portfolio'].get('sector_weights', {})
        
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
        start_date = base_config['settings']['start_date']
        end_date = base_config['settings']['end_date']
        
        benchmarks = set()
        tickers = set()
        
        for cfg in self.configs:
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
        for cfg in self.configs:
            benchmarks.update(cfg['settings'].get('benchmarks', []))
        
        for bench in sorted(list(benchmarks)):
            if bench in available_tickers:
                s = bt.Strategy(bench, [bt.algos.RunOnce(), bt.algos.SelectThese([bench]), bt.algos.WeighEqually(), bt.algos.Rebalance()])
                backtests.append(bt.Backtest(s, self.data))

        logger.info(f"Running {len(backtests)} backtests...")
        self.results = bt.run(*backtests)
        
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
        freq = cfg['settings'].get('rebalance_frequency', 'quarterly').lower()
        algo_map = {'daily': bt.algos.RunDaily(), 'weekly': bt.algos.RunWeekly(), 'monthly': bt.algos.RunMonthly(), 'quarterly': bt.algos.RunQuarterly(), 'yearly': bt.algos.RunYearly(), 'once': bt.algos.RunOnce()}
        
        s = bt.Strategy(name, [algo_map.get(freq, bt.algos.RunQuarterly()), bt.algos.SelectThese(list(target_weights.keys())), bt.algos.WeighSpecified(**target_weights), bt.algos.Rebalance()])
        return bt.Backtest(s, self.data)

    def plot_all(self):
        """Displays results and all visualizations."""
        if not self.results: return
        
        self.results.display()
        self.results.plot(title="Backtest Results")
        
        # Advanced Visualizations need visualization module
        try:
            import visualization as viz
            viz.make_legend_interactive()
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
            viz.plot_return_distribution(self.results)
            
            # Heatmaps for strategies and benchmarks
            benchmarks = set()
            for cfg in self.configs:
                benchmarks.update(cfg['settings'].get('benchmarks', []))
                
            all_plot_names = [cfg['_name'] for cfg in self.configs] + sorted(list(benchmarks))
            
            for name in all_plot_names:
                if name in self.results.prices.columns:
                    viz.plot_monthly_returns_heatmap(self.results, name)
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
