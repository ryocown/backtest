"""
Backward compatibility shim for visualization module.

This module re-exports all functions from the new visualization/ package
to maintain backward compatibility with existing code that imports from
the old monolithic visualization.py file.

New code should import directly from the visualization package:
    from src.visualization import plot_drawdowns
    from src.visualization.rolling import plot_rolling_sharpe
"""
from __future__ import annotations

# Re-export everything from the new package
from src.visualization import *  # noqa: F401, F403
