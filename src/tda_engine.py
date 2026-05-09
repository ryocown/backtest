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
