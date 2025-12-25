import json
import numpy as np
import pandas as pd
import os
from datetime import datetime
from analytics import AnalyticsEngine
try:
    from persim import PersLandscapeApprox
except ImportError:
    PersLandscapeApprox = None

class WebExporter:
    """
    Serializes backtest results and TDA data into a structured JSON package
    for the Angular frontend.
    """

    def __init__(self, output_path="webapp/public/data/latest_backtest.json"):
        self.output_path = output_path
        self.data_package = {
            "version": "1.0.0",
            "exported_at": datetime.now().isoformat(),
            "backtest": {},
            "tda": None,
            "metadata": {}
        }

    class WebEncoder(json.JSONEncoder):
        """Custom JSON encoder for NumPy and Pandas types."""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d')
            if isinstance(obj, pd.DataFrame):
                # Convert to record format for easier D3/Plotly consumption
                # Orientation 'records' is [ {col1: val, col2: val}, ... ]
                return obj.reset_index().to_dict(orient='records')
            if isinstance(obj, pd.Series):
                res = obj.to_dict()
                return {k.strftime('%Y-%m-%d') if isinstance(k, pd.Timestamp) else k: v for k, v in res.items()}
            return super().default(obj)

    def add_backtest_results(self, results, data, metadata=None, sector_maps=None):
        """
        Extracts key metrics and time-series from a bt.Result object.
        """
        prices = results.prices
        # Extract individual stats for each strategy/benchmark
        stats = {}
        drawdown_events = {}
        risk_attribution = {}
        return_attribution = {}
        sector_returns = {}
        sector_risk = {}
        
        sector_maps = sector_maps or {}

        # Determine benchmark (default to SPY if available, otherwise first column)
        benchmark = 'SPY' if 'SPY' in prices.columns else prices.columns[0]
        
        # Cross-strategy correlation matrix
        correlation_matrix = prices.pct_change().dropna().corr().to_dict()

        for col in prices.columns:
            # bt.Result.stats contains a wealth of metrics
            if col in results.stats.columns:
                col_stats = results.stats[col].to_dict()
                # Clean up stats to remove NaN/Inf which break JSON
                cleaned_stats = {k: v for k, v in col_stats.items() if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v))}
                stats[col] = cleaned_stats
            
            # Add advanced metrics for strategies (not benchmarks if not desired, but let's do all)
            if col in results.keys():
                drawdown_events[col] = self._get_drawdown_events(results, col)
                risk_attribution[col] = self._get_risk_attribution(results, col, data)
                
                # Cumulative return attribution
                try:
                    weights = results.get_security_weights(col).iloc[-1]
                    # Filter for symbols that actually have price data in the full asset matrix (data)
                    available_symbols = [s for s in weights[weights > 0].index if s in data.columns]
                    s_map = sector_maps.get(str(col), {})
                    
                    if available_symbols:
                        subset_prices = data[available_symbols]
                        subset_weights = weights[available_symbols]
                        attr = AnalyticsEngine.get_return_attribution(subset_prices, subset_weights)
                        # Reset index to get 'index' column for JSON
                        return_attribution[col] = attr.reset_index().to_dict(orient='records')
                        
                        # Sector grouped returns
                        if s_map:
                            # Map ticker -> sector for attribution columns
                            sector_attr = attr.rename(columns=s_map).groupby(level=0, axis=1).sum()
                            sector_returns[col] = sector_attr.reset_index().to_dict(orient='records')
                            
                            # Sector grouped risk
                            risk = risk_attribution.get(col) or self._get_risk_attribution(results, col, data)
                            if risk:
                                # Ensure it's a Series for groupby
                                s_risk = pd.Series(risk).groupby(s_map).sum()
                                sector_risk[col] = s_risk.to_dict()
                except Exception as e:
                    import traceback
                    print(f"Error calculating return attribution for {col}: {e}")
                    traceback.print_exc()

        self.data_package["backtest"] = {
            "strategies": list(prices.columns),
            "benchmark": benchmark,
            "prices": prices,
            "stats": stats,
            "weights": {col: results.get_security_weights(col) for col in prices.columns if col in results.keys()},
            "drawdown_events": drawdown_events,
            "risk_attribution": risk_attribution,
            "return_attribution": return_attribution,
            "sector_returns": sector_returns,
            "sector_risk": sector_risk,
            "sector_maps": sector_maps,
            "correlation_matrix": correlation_matrix,
            "monthly_returns": {col: self._get_monthly_returns_grid(prices[col]) for col in prices.columns},
            "rolling_stats": self._get_all_rolling_stats(prices, benchmark),
            "return_distribution": {col: self._get_return_distribution(prices[col]) for col in prices.columns},
            "capture_ratios": {col: AnalyticsEngine.get_capture_ratios(prices[col], prices[benchmark]) for col in prices.columns if col != benchmark}
        }
        
        if metadata:
            self.data_package["metadata"] = metadata

    def _get_drawdown_events(self, results, strategy_name):
        """Calculates drawdown events for the recovery surface."""
        prices = results.prices[strategy_name]
        cummax = prices.cummax()
        drawdown = (prices / cummax) - 1.0
        
        is_in_drawdown = drawdown < 0
        starts = (is_in_drawdown & (~is_in_drawdown.shift(1).fillna(False))).infer_objects(copy=False)
        ends = ((~is_in_drawdown) & is_in_drawdown.shift(1).fillna(False)).infer_objects(copy=False)
        
        start_dates = prices.index[starts]
        end_dates = prices.index[ends]
        
        events = []
        for s in start_dates:
            future_ends = end_dates[end_dates > s]
            if not future_ends.empty:
                e = future_ends[0]
                period_prices = prices[s:e]
                mag = (period_prices.min() / period_prices.iloc[0]) - 1.0
                duration_months = (e - s).days / 30.44
                required_recovery = (1.0 / (1.0 + mag)) - 1.0
                
                events.append({
                    'date': s.strftime('%Y-%m-%d'),
                    'magnitude': abs(float(mag)),
                    'duration': float(duration_months),
                    'recovery': float(required_recovery)
                })
        return events

    def _get_risk_attribution(self, results, strategy_name, asset_prices):
        """Calculates risk attribution (MCTR) for the latest weights."""
        try:
            weights = results.get_security_weights(strategy_name).iloc[-1]
            constituents = weights[weights > 0]
            valid_tickers = [t for t in constituents.index if t in asset_prices.columns]
            
            if valid_tickers:
                subset_prices = asset_prices[valid_tickers]
                pct_ctr = AnalyticsEngine.get_risk_contribution(subset_prices, constituents[valid_tickers])
                # Clean up tickers for JSON
                return {str(k): float(v) for k, v in pct_ctr.to_dict().items()}
        except Exception as e:
            print(f"Error calculating risk attribution for {strategy_name}: {e}")
        return {}

    def _get_monthly_returns_grid(self, prices):
        """Formats monthly returns into a Year -> Month -> Value grid."""
        try:
            monthly_rets = AnalyticsEngine.get_monthly_returns(prices.to_frame())
            df = monthly_rets.iloc[:, 0].to_frame(name='ret')
            df['year'] = df.index.year
            df['month'] = df.index.month
            pivot = df.pivot(index='year', columns='month', values='ret')
            return pivot.to_dict(orient='index')
        except Exception as e:
            print(f"Error calculating monthly returns grid: {e}")
            return {}

    def _get_all_rolling_stats(self, prices, benchmark):
        """Calculates all rolling stats for all strategies vs benchmark."""
        all_rolling = {}
        window = 126 # 6 months
        try:
            vol = AnalyticsEngine.get_rolling_volatility(prices, window=window)
            sharpe = AnalyticsEngine.get_rolling_sharpe(prices, window=window)
            
            non_bench = [c for c in prices.columns if c != benchmark]
            if non_bench:
                alpha, beta = AnalyticsEngine.get_rolling_alpha_beta(prices[non_bench], prices[[benchmark]], window=window)
                ir = AnalyticsEngine.get_rolling_info_ratio(prices[non_bench], prices[[benchmark]], window=window)
            else:
                alpha, beta, ir = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            for col in prices.columns:
                stats = {
                    "volatility": vol[col].dropna(),
                    "sharpe": sharpe[col].dropna()
                }
                if col in alpha.columns:
                    stats["alpha"] = alpha[col].dropna()
                    stats["beta"] = beta[col].dropna()
                    stats["info_ratio"] = ir[col].dropna()
                all_rolling[col] = stats
        except Exception as e:
            print(f"Error calculating rolling stats: {e}")
        return all_rolling

    def _get_return_distribution(self, prices):
        """Calculates histogram data for daily returns."""
        try:
            rets = prices.pct_change().dropna()
            hist, bin_edges = np.histogram(rets, bins=50)
            return {
                "values": hist.tolist(),
                "bins": bin_edges.tolist()
            }
        except Exception as e:
            print(f"Error calculating return distribution: {e}")
            return {"values": [], "bins": []}

    def add_tda_results(self, precomputed_results):
        """
        Serializes TDA precomputed results (coordinates, Betti numbers, diagrams).
        """
        serialized_tda = []
        betti_trend = []
        euler_trend = []
        avg_corr_trend = []
        dates = []

        for res in precomputed_results:
            if res is None:
                serialized_tda.append(None)
                continue
            
            # persistence diagrams (dgms) are a list of numpy arrays
            # We need to ensure they are nested lists
            dgms = []
            for dgm in res['dgms']:
                dgms.append(dgm.tolist())

            # Edge information for simplicial complex visualization (thresholded edges)
            threshold = 0.5
            edges = []
            corr = res['corr']
            dist = 1 - corr
            n = len(dist)
            for i in range(n):
                for j in range(i + 1, n):
                    if dist.iloc[i, j] <= threshold:
                        edges.append([i, j])

            win_date = res['date'].strftime('%Y-%m-%d')
            dates.append(win_date)
            betti_trend.append(res['betti'])
            euler_trend.append(int(res['chi']))
            avg_corr_trend.append(float(res['avg_corr']))

            landscapes = []
            if PersLandscapeApprox:
                try:
                    # We compute landscapes for H0 and H1 (first two diagrams)
                    for i in range(min(2, len(res['dgms']))):
                        dgm = res['dgms'][i]
                        # Remove infinite death values for landscape calculation
                        dgm_filtered = dgm[~np.any(dgm == np.inf, axis=1)]
                        if len(dgm_filtered) > 0:
                            # Use fewer steps (100) for web visualization to save space
                            pla = PersLandscapeApprox(dgms=[dgm_filtered], hom_deg=0, num_steps=100)
                            landscapes.append({
                                "hom_deg": i,
                                "start": float(pla.start),
                                "stop": float(pla.stop),
                                "values": pla.values[0:3].tolist() # Top 3 depths are usually enough
                            })
                except Exception as e:
                    print(f"Error calculating landscapes: {e}")

            serialized_tda.append({
                "date": res['date'].strftime('%Y-%m-%d'),
                "avg_corr": float(res['avg_corr']),
                "chi": int(res['chi']),
                "betti": [int(b) for b in res['betti']],
                "coords": res['coords'].tolist(), # (N, 3) array
                "dgms": dgms,
                "landscapes": landscapes,
                "edges": edges,
                "tickers": list(res['corr'].columns)
            })
        
        self.data_package["tda"] = {
            "windows": serialized_tda,
            "trends": {
                "dates": dates,
                "betti": betti_trend,
                "euler": euler_trend,
                "avg_corr": avg_corr_trend
            }
        }

    def export(self):
        """Writes the data package to the specified output path."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.data_package, f, cls=self.WebEncoder, indent=2)
        print(f"Web data package exported to: {self.output_path}")

def export_to_web(results, data, metadata, tda_results=None, sector_maps=None, output_path=None):
    """Convenience function for exporting."""
    exporter = WebExporter(output_path) if output_path else WebExporter()
    exporter.add_backtest_results(results, data, metadata, sector_maps)
    if tda_results is not None:
        exporter.add_tda_results(tda_results)
    exporter.export()
