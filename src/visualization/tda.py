"""
Topological Data Analysis (TDA) visualizations.

Provides persistence diagrams, barcodes, and 3D simplicial complex plots.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.visualization.cursor import make_legend_interactive

# Optional TDA dependencies
try:
    from persim import plot_diagrams
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def plot_persistence_diagram(dgms: list, title: str = "Persistence Diagram") -> None:
    """
    Plots the persistence diagram using persim.
    
    Points far from the diagonal represent robust topological features.
    
    Args:
        dgms: List of persistence diagrams (numpy arrays).
        title: Chart title.
    """
    if not HAS_PERSIM:
        print("persim not installed. Cannot plot persistence diagram.")
        return
        
    plt.figure(figsize=(8, 8))
    plot_diagrams(dgms, show=False)
    plt.title(title)
    plt.tight_layout()


def plot_persistence_barcode(dgms: list, title: str = "Persistence Barcode") -> None:
    """
    Plots the persistence barcode as horizontal lines.
    
    Longer bars represent more significant topological features.
    
    Args:
        dgms: List of persistence diagrams (numpy arrays).
        title: Chart title.
    """
    fig, axes = plt.subplots(len(dgms), 1, figsize=(10, 2 * len(dgms)), sharex=True)
    if len(dgms) == 1:
        axes = [axes]
        
    for dim, (dgm, ax) in enumerate(zip(dgms, axes)):
        if len(dgm) > 0:
            persistence = dgm[:, 1] - dgm[:, 0]
            # Handle infinite death
            max_finite = np.max(persistence[~np.isinf(persistence)]) if np.any(~np.isinf(persistence)) else 1.0
            persistence[np.isinf(dgm[:, 1])] = max_finite * 1.2
            
            idx = np.argsort(persistence)
            dgm_sorted = dgm[idx]
            
            for i, (birth, death) in enumerate(dgm_sorted):
                if np.isinf(death):
                    death = np.max(dgm[~np.isinf(dgm[:, 1]), 1]) * 1.1 if np.any(~np.isinf(dgm[:, 1])) else birth + 1
                ax.plot([birth, death], [i, i], color=f"C{dim}", linewidth=2)
        
        ax.set_title(f"Dimension {dim} Barcode")
        ax.set_ylabel("Hole Index")
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Filtration Value (Distance)")
    plt.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))


def plot_tda_metrics_trend(tda_results: list[dict], title: str = "TDA Market Metrics Trend") -> None:
    """
    Plots the trend of Betti numbers and Euler Characteristic over time.
    
    Args:
        tda_results: List of TDA result dicts with keys: start, betti, euler, avg_corr.
        title: Chart title.
    """
    dates = [r['start'] for r in tda_results]
    betti_0 = [r['betti'][0] for r in tda_results]
    betti_1 = [r['betti'][1] if len(r['betti']) > 1 else 0 for r in tda_results]
    euler = [r['euler'] for r in tda_results]
    avg_corr = [r['avg_corr'] for r in tda_results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. Betti Numbers
    ax1.plot(dates, betti_0, label="Betti 0 (Connected Components)", marker='o', markersize=4)
    ax1.plot(dates, betti_1, label="Betti 1 (1D Holes/Cycles)", marker='s', markersize=4)
    ax1.set_title("Betti Numbers Trend")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Euler Characteristic
    ax2.plot(dates, euler, color='red', label="Euler Characteristic (χ)", marker='d', markersize=4)
    ax2.set_title("Euler Characteristic Trend")
    ax2.set_ylabel("χ")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Average Correlation
    ax3.plot(dates, avg_corr, color='green', label="Avg Correlation", marker='x', markersize=4)
    ax3.set_title("Market Correlation (Spectral Reddening Indicator)")
    ax3.set_ylabel("Correlation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    make_legend_interactive(fig)


def plot_3d_simplicial_complex(
    coords: np.ndarray,
    dist_matrix: pd.DataFrame,
    threshold: float = 0.5,
    title: str = "3D Simplicial Complex"
) -> None:
    """
    Visualizes the Vietoris-Rips complex in 3D.
    
    - Points: Stocks projected via MDS.
    - Edges: Connections where distance < threshold.
    - Faces: Triangles where all 3 edges < threshold.
    
    Args:
        coords: (N, 3) array of 3D coordinates from MDS projection.
        dist_matrix: Distance matrix as DataFrame.
        threshold: Maximum distance for edge/face inclusion.
        title: Chart title.
    """
    if not HAS_SCIPY:
        print("scipy not installed. Cannot plot 3D simplicial complex.")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot Nodes
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='red', s=50, label='Stocks')

    # 2. Plot Edges (1-simplexes)
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix.iloc[i, j] <= threshold:
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    [coords[i, 2], coords[j, 2]],
                    color='cyan', alpha=0.3, linewidth=1
                )

    # 3. Plot Faces (2-simplexes) using Delaunay
    tri = Delaunay(coords)
    simplices = tri.simplices
    
    valid_faces = []
    for s in simplices:
        d1 = dist_matrix.iloc[s[0], s[1]]
        d2 = dist_matrix.iloc[s[1], s[2]]
        d3 = dist_matrix.iloc[s[2], s[0]]
        if d1 <= threshold and d2 <= threshold and d3 <= threshold:
            valid_faces.append(s)
            
    if valid_faces:
        ax.plot_trisurf(
            coords[:, 0], coords[:, 1], coords[:, 2],
            triangles=valid_faces,
            color='cyan', alpha=0.2, shade=True
        )

    ax.set_title(title)
    ax.set_xlabel("MDS X")
    ax.set_ylabel("MDS Y")
    ax.set_zlabel("MDS Z")
    plt.tight_layout()
