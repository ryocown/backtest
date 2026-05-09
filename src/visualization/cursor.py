"""
Interactive cursor and legend functionality for financial charts.

Provides:
- FinancialCursor: Crosshair and tooltip that follows mouse across subplots
- make_legend_interactive: Click legend items to toggle visibility
"""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.backend_bases import Event


class FinancialCursor:
    """
    Adds a vertical crosshair and tooltip that follows the mouse across ALL subplots.
    Industry standard for financial time-series visualization.
    
    Uses weakref to avoid memory leaks when figures are closed.
    Tooltip is positioned in bottom-left to avoid legend overlap.
    """
    
    def __init__(self, fig: Figure) -> None:
        self._fig_ref = weakref.ref(fig)
        self.vlines: list = []
        self.annotations: list = []
        self.axes_list: list = []
        self._cids: list[int] = []  # Connection IDs for cleanup
        
        # Initialize for all compatible axes (skip 3D)
        for ax in fig.axes:
            if ax.name == '3d':
                continue
            
            line = ax.axvline(
                x=np.nan, color='gray', linestyle='--', 
                linewidth=0.8, alpha=0.8, zorder=10
            )
            self.vlines.append(line)
            
            # Annotation box (Tooltip) - positioned in BOTTOM LEFT to avoid legend
            anno = ax.annotate(
                '', 
                xy=(0.02, 0.02),  # Bottom-left corner
                xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9, edgecolor='gray'),
                fontsize=8, 
                zorder=100,  # High z-order to stay on top
                fontfamily='monospace',
                verticalalignment='bottom',
                horizontalalignment='left'
            )
            anno.set_visible(False)
            self.annotations.append(anno)
            self.axes_list.append(ax)
        
        # Connect events and store IDs for cleanup
        canvas = fig.canvas
        self._cids.append(canvas.mpl_connect("motion_notify_event", self.on_mouse_move))
        self._cids.append(canvas.mpl_connect("axes_leave_event", self.on_leave))
        self._cids.append(canvas.mpl_connect("close_event", self.on_close))

    @property
    def fig(self) -> Optional[Figure]:
        """Returns the figure if it still exists."""
        return self._fig_ref()

    def on_close(self, event: Event) -> None:
        """Cleanup when figure is closed."""
        fig = self.fig
        if fig is not None and fig.canvas is not None:
            for cid in self._cids:
                try:
                    fig.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._cids.clear()
        self.vlines.clear()
        self.annotations.clear()
        self.axes_list.clear()

    def on_mouse_move(self, event: Event) -> None:
        """Updates crosshair position and tooltip values."""
        fig = self.fig
        if fig is None or not event.inaxes:
            return
        
        target_x = event.xdata
        
        # Format date for display
        try:
            date_str = pd.to_datetime(target_x, unit='D').strftime('%Y-%m-%d')
        except Exception:
            date_str = f"{target_x:.2f}"

        for i, ax in enumerate(self.axes_list):
            self.vlines[i].set_xdata([target_x])
            
            # Find the closest data point in each axes to show values
            summary_text = f"[{date_str}]\n"
            found_val = False
            
            for line in ax.get_lines():
                if line.get_label().startswith('_'):  # Skip internal lines
                    continue
                
                lx, ly = line.get_data()
                if len(lx) == 0:
                    continue
                
                # Ensure lx is numeric for comparison
                try:
                    lx_numeric = mdates.date2num(lx)
                except Exception:
                    lx_numeric = np.asarray(lx)

                # Find index of closest x
                idx = np.searchsorted(lx_numeric, target_x)
                if idx >= len(lx_numeric):
                    idx = len(lx_numeric) - 1
                
                if abs(lx_numeric[idx] - target_x) < 5:  # Threshold for visibility
                    val = ly[idx]
                    label = line.get_label()
                    summary_text += f"{label[:10]}: {val:.2f}\n"
                    found_val = True
            
            if found_val:
                self.annotations[i].set_text(summary_text.strip())
                self.annotations[i].set_visible(True)
            else:
                self.annotations[i].set_visible(False)
        
        fig.canvas.draw_idle()

    def on_leave(self, event: Event) -> None:
        """Hides crosshair when mouse leaves axes."""
        fig = self.fig
        if fig is None:
            return
            
        for line in self.vlines:
            line.set_xdata([np.nan])
        for anno in self.annotations:
            anno.set_visible(False)
        fig.canvas.draw_idle()


def add_financial_cursor(fig: Optional[Figure] = None) -> Optional[FinancialCursor]:
    """
    Attaches a FinancialCursor to the figure.
    
    Uses weakref storage to avoid memory leaks. The cursor is stored
    as a private attribute on the figure.
    
    Args:
        fig: Matplotlib figure. If None, uses current figure.
        
    Returns:
        The FinancialCursor instance, or None if no figure.
    """
    if fig is None:
        fig = plt.gcf()
    
    if not hasattr(fig, '_financial_cursor') or fig._financial_cursor is None:
        fig._financial_cursor = FinancialCursor(fig)
    
    return fig._financial_cursor


def make_legend_interactive(fig: Optional[Figure] = None) -> None:
    """
    Makes all legends in the figure interactive and pins them in place.
    
    - Clicking on a legend label toggles the visibility of the corresponding plot element.
    - Legends are PINNED to upper-right corner to prevent repositioning.
    - Also attaches the FinancialCursor for crosshair functionality.
    - Supports both Line2D objects and PolyCollections (filled areas from kdeplot).
    
    Args:
        fig: Matplotlib figure. If None, uses current figure.
    """
    if fig is None:
        fig = plt.gcf()
    
    # Attach the Financial Cursor as well
    add_financial_cursor(fig)
    
    # Extract all legends and PIN them to a FIXED location
    legends = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            # FORCE legend to upper-right with fixed location code
            leg._loc = 1  # Force upper right location code
            leg.set_draggable(False)  # Prevent dragging
            
            if hasattr(leg, '_set_loc'):
                leg._set_loc(1)  # 1 = upper right
            
            legends.append(leg)
    
    # Map legend items to original artists (lines OR collections)
    legend_map: dict = {}
    
    for leg in legends:
        ax = leg.axes
        
        # Get all visible artists with labels (lines and collections)
        lines = [l for l in ax.get_lines() if not l.get_label().startswith('_')]
        
        # Get collections (PolyCollections from kdeplot fill, histograms, etc.)
        from matplotlib.collections import PolyCollection
        collections = [c for c in ax.collections 
                       if isinstance(c, PolyCollection) and not c.get_label().startswith('_')]
        
        # Combine all labeled artists
        all_artists = lines + collections
        
        # Get legend handles and texts
        legend_handles = leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles
        legend_texts = leg.get_texts()
        
        # Map legend handles to corresponding artists by matching labels
        artist_by_label = {a.get_label(): a for a in all_artists}
        
        for i, (handle, text) in enumerate(zip(legend_handles, legend_texts)):
            label = text.get_text()
            if label in artist_by_label:
                orig_artist = artist_by_label[label]
                
                # Make handle pickable
                handle.set_picker(5)
                legend_map[handle] = orig_artist
                
                # Make text pickable too
                text.set_picker(5)
                legend_map[text] = orig_artist

    def on_pick(event: Event) -> None:
        """Toggle visibility of the artist associated with the clicked legend item."""
        leg_item = event.artist
        if leg_item not in legend_map:
            return
            
        orig_artist = legend_map[leg_item]
        visible = not orig_artist.get_visible()
        orig_artist.set_visible(visible)
        
        # Change alpha of legend item to show status
        if visible:
            leg_item.set_alpha(1.0)
            if hasattr(leg_item, 'set_fontweight'):
                leg_item.set_fontweight('normal')
        else:
            leg_item.set_alpha(0.2)
            if hasattr(leg_item, 'set_fontweight'):
                leg_item.set_fontweight('light')
        
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)


def pin_legend(ax, loc: str = 'upper right') -> None:
    """
    Creates or updates the legend with a fixed, non-moving position.
    
    Call this instead of ax.legend() to ensure the legend never moves.
    
    Args:
        ax: Matplotlib axes.
        loc: Location string (default 'upper right').
    """
    # Location code mapping
    loc_codes = {
        'best': 0, 'upper right': 1, 'upper left': 2, 'lower left': 3,
        'lower right': 4, 'right': 5, 'center left': 6, 'center right': 7,
        'lower center': 8, 'upper center': 9, 'center': 10
    }
    
    leg = ax.legend(loc=loc)
    if leg is not None:
        leg._loc = loc_codes.get(loc, 1)
        leg.set_draggable(False)
    return leg
