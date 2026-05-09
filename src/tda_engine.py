import numpy as np
import pandas as pd
import logging
from ripser import ripser
from persim import plot_diagrams
from sklearn.manifold import MDS

logger = logging.getLogger(__name__)

class TDAManager:
    """
    Manages Topological Data Analysis (TDA) computations for stock market data.
    Uses Vietoris-Rips filtration to identify market regimes and "spectral reddening".
    """
    
    def __init__(self, window_months=6, step_months=1):
        self.window_months = window_months
        self.step_months = step_months
        
    def correlation_to_distance(self, corr_matrix):
        """
        Converts a correlation matrix (rho) to a distance matrix (d).
        Formula: d = sqrt(2 * (1 - rho))
        """
        # Ensure diagonal is 1.0 to avoid small negative numbers due to precision
        np.fill_diagonal(corr_matrix.values, 1.0)
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        return dist_matrix

    def get_sliding_windows(self, df):
        """
        Generator that yields (start_date, end_date, sub_df) for sliding windows.
        """
        if df.empty:
            return
            
        start_date = df.index.min()
        max_date = df.index.max()
        
        current_start = start_date
        while True:
            current_end = current_start + pd.DateOffset(months=self.window_months)
            if current_end > max_date:
                break
                
            yield current_start, current_end, df.loc[current_start:current_end]
            
            current_start = current_start + pd.DateOffset(months=self.step_months)

    def get_3d_projection(self, dist_matrix):
        """
        Uses Multidimensional Scaling (MDS) to project the distance matrix into 3D.
        Returns a (n_stocks, 3) array of coordinates.
        """
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(dist_matrix.values)
        return coords

    def compute_persistence(self, dist_matrix, max_dim=1):
        """
        Computes persistence diagrams for a given distance matrix.
        """
        # ripser expects a distance matrix if distance_matrix=True
        # We use the distance matrix directly
        result = ripser(dist_matrix.values, distance_matrix=True, maxdim=max_dim)
        return result['dgms']

    def calculate_betti_numbers(self, diagrams, threshold):
        """
        Calculates Betti numbers (beta_0, beta_1, ...) for a given filtration threshold (epsilon).
        Betti number beta_n is the count of n-dimensional holes that exist at threshold epsilon.
        """
        betti = []
        for dim, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                betti.append(0)
                continue
                
            # A hole exists at threshold epsilon if birth <= epsilon < death
            # Note: death can be np.inf
            count = np.sum((dgm[:, 0] <= threshold) & (dgm[:, 1] > threshold))
            betti.append(int(count))
        return betti

    def calculate_euler_characteristic(self, betti_numbers):
        """
        Calculates Euler Characteristic (chi) as the alternating sum of Betti numbers.
        chi = beta_0 - beta_1 + beta_2 - ...
        """
        chi = 0
        for i, b in enumerate(betti_numbers):
            chi += ((-1) ** i) * b
        return chi

    def run_analysis(self, df, betti_threshold=0.5):
        """
        Runs the full TDA analysis over sliding windows.
        Returns a list of results for each window.
        """
        results = []
        
        for start, end, sub_df in self.get_sliding_windows(df):
            # 1. Correlation Matrix
            corr = sub_df.corr()
            if corr.isnull().values.any():
                logger.warning(f"Skipping window {start.date()} to {end.date()} due to NaNs in correlation")
                continue
                
            # 2. Distance Matrix
            dist = self.correlation_to_distance(corr)
            
            # 3. Persistence Homology
            dgms = self.compute_persistence(dist)
            
            # 4. Metrics
            betti = self.calculate_betti_numbers(dgms, betti_threshold)
            chi = self.calculate_euler_characteristic(betti)
            
            results.append({
                'start': start,
                'end': end,
                'dgms': dgms,
                'betti': betti,
                'euler': chi,
                'avg_corr': corr.values[np.triu_indices(len(corr), k=1)].mean(),
                'corr_matrix': corr
            })
            
        return results

    def compute_tda_for_range(self, df, start_dt, end_dt, data_engine):
        available_dates = df.index[(df.index >= start_dt) & (df.index <= end_dt)]
        if len(available_dates) == 0:
            logger.warning("No trading days found in the requested TDA range.")
            return [], None, None

        # Fetch SPY OHLC
        spy_start = (start_dt - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        spy_end = (end_dt + pd.Timedelta(days=10)).strftime('%Y-%m-%d')
        spy_ohlc = data_engine.get_ticker_ohlc('SPY', spy_start, spy_end)

        precomputed_results = []

        logger.info(f"Precomputing TDA for {len(available_dates)} days...")
        for i, current_date in enumerate(available_dates):
            window_start = current_date - pd.DateOffset(months=self.window_months)
            sub_df = df.loc[window_start:current_date]

            if sub_df.empty or len(sub_df) < 20:
                precomputed_results.append(None)
                continue

            corr_matrix = sub_df.corr()
            dist_matrix = self.correlation_to_distance(corr_matrix)
            coords = self.get_3d_projection(dist_matrix)
            dgms = self.compute_persistence(dist_matrix)
            betti = self.calculate_betti_numbers(dgms, threshold=0.5)
            chi = self.calculate_euler_characteristic(betti)
            avg_corr = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean()

            precomputed_results.append({
                'date': current_date,
                'corr': corr_matrix,
                'dist': dist_matrix,
                'coords': coords,
                'dgms': dgms,
                'betti': betti,
                'chi': chi,
                'avg_corr': avg_corr
            })
            if (i + 1) % 5 == 0 or i == len(available_dates) - 1:
                print(f"TDA Progress: {i + 1}/{len(available_dates)}", end="\r")
        print("\nPrecomputation complete.")
        return precomputed_results, available_dates, spy_ohlc

