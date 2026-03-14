"""Matplotlib styling for arXiv-quality figures across the paper series."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Model color palette (consistent across all papers)
MODEL_COLORS: dict[str, str] = {
    "google__gemini-3-flash-preview": "#4285F4",       # Google blue
    "google__gemini-3-flash-preview__no_resize": "#7BAAF7",
    "google__gemini-2.5-pro": "#0F9D58",               # Google green
    "openai__gpt-4.1": "#10A37F",                      # OpenAI green
    "anthropic__claude-haiku-4.5": "#D4A574",           # Anthropic tan
    "qwen__qwen3.5-flash-02-23": "#FF6B35",            # Orange
    "qwen__qwen3-vl-8b-instruct": "#E63946",           # Red
    "qwen__qwen3-vl-8b-thinking": "#9B2335",           # Dark red
}

# Dimension colors
DIM_COLORS: dict[str, str] = {
    "overall": "#2171B5",
    "sharpness": "#238B45",
    "color_fidelity": "#D94701",
}

# Tier colors (degradation gradient)
TIER_COLORS: dict[str, str] = {
    "ORIGINAL": "#1a9850",
    "PRISTINE": "#66bd63",
    "HIGH": "#a6d96a",
    "MEDIUM": "#fee08b",
    "LOW": "#f46d43",
    "DEGRADED": "#d73027",
}

# Engine colors
ENGINE_COLORS: dict[str, str] = {
    "tesseract": "#2171B5",
    "easyocr": "#238B45",
    "rapidocr": "#D94701",
    "gcloud_vision": "#6A3D9A",
}


def apply_arxiv_style() -> None:
    """Set matplotlib defaults for arXiv-quality figures."""
    mpl.rcParams.update({
        # Font
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (7.0, 4.5),
        # Lines and markers
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        # Borders
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Layout
        "figure.constrained_layout.use": True,
    })


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    """Save figure with tight layout at publication quality."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path} ({path.stat().st_size / 1024:.0f} KB)")
