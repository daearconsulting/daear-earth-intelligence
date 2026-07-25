"""
Shared plotting style -- every repo's figures should look like one product
line. Keep this file small and dependency-light (matplotlib only) so it's
easy to drop into any notebook.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Daear brand palette (matches the CV / one-pager accent color)
ACCENT = "#1F4E3D"       # deep evergreen
ACCENT_LIGHT = "#6B9080"
WARN = "#B45309"         # burn/erosion warning tone
MUTED = "#555555"

RESILIENCE_CMAP = LinearSegmentedColormap.from_list(
    "daear_resilience", ["#8B2E2E", "#D8A657", "#6B9080", "#1F4E3D"]
)
SEVERITY_CMAP = LinearSegmentedColormap.from_list(
    "daear_severity", ["#F5F0E6", "#D8A657", "#B45309", "#5A1E1E"]
)


def style_axes(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=13, fontweight="bold", color=ACCENT, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    return ax


def plot_raster(da, title: str = "", cmap=None, ax=None, **kwargs):
    """Consistent raster plot for any 2D (lat, lon) DataArray."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = da.plot(ax=ax, cmap=cmap or RESILIENCE_CMAP, add_colorbar=True, **kwargs)
    style_axes(ax, title=title or getattr(da, "name", ""), xlabel="Longitude", ylabel="Latitude")
    return ax
