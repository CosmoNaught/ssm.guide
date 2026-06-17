"""Shared figure style for the book: palette, typography, and helpers.

Palette mirrors styles/theme.scss. Output (SVG + PNG) is written into figures/,
the parent of this _src/ dir.
"""
from __future__ import annotations

import os
import matplotlib as mpl

mpl.use("Agg")  # headless: no GUI window, just file output
import matplotlib.pyplot as plt  # noqa: E402

# --- palette (mirrors styles/theme.scss) -----------------------------------
INK         = "#303030"   # body text
BLUE        = "#4a5c79"   # primary structure / headings
BLUE_DARK   = "#3a4a63"
ORANGE      = "#e37c47"   # accent / highlight
ORANGE_DARK = "#cf6a37"
GREY        = "#6b7280"   # secondary text
GREY_LINE   = "#c9ccd3"   # light rules / spines / cell borders
GREY_FILL   = "#f5f5f5"   # panel fill
WHITE       = "#ffffff"

# three further hues for the five-colour categorical palette
GREEN       = "#5e9c6e"   # sage green
GREEN_DARK  = "#4c8159"
PLUM        = "#8e78a8"   # lavender / plum
PLUM_DARK   = "#74608f"
ROSE        = "#d6849a"   # dusty rose
ROSE_DARK   = "#bf6c84"

# ordered blue ramp, dark -> light, for families of curves / matrix diagonals
BLUE_RAMP = ["#3a4a63", "#4a5c79", "#6e7f9c", "#94a2bb", "#bcc6d6"]

# five-colour categorical palette in cycle order; blue leads so the orange
# accent stays free to mark the key object
PALETTE      = [BLUE, ORANGE, GREEN, PLUM, ROSE]
PALETTE_DARK = [BLUE_DARK, ORANGE_DARK, GREEN_DARK, PLUM_DARK, ROSE_DARK]

# output dir = the figures/ folder (parent of this _src/ dir)
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def use_style() -> None:
    """Apply the book's rcParams. Idempotent; call before building a figure."""
    mpl.rcParams.update({
        "figure.facecolor":   WHITE,
        "savefig.facecolor":  WHITE,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.06,

        "font.family":        "sans-serif",
        "font.sans-serif":    ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size":          11,
        "axes.titlesize":     12.5,
        "axes.labelsize":     11.5,
        "mathtext.fontset":   "dejavusans",  # sans math, to match the labels

        "text.color":         INK,
        "axes.labelcolor":    INK,
        "axes.edgecolor":     GREY_LINE,
        "axes.linewidth":     1.0,
        "xtick.color":        GREY,
        "ytick.color":        GREY,
        "xtick.labelcolor":   INK,
        "ytick.labelcolor":   INK,

        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "lines.linewidth":    2.2,
        "lines.solid_capstyle": "round",
        "legend.frameon":     False,

        # default categorical colour cycle = the five-colour palette
        "axes.prop_cycle":    mpl.cycler(color=PALETTE),
    })


def figure(w: float = 6.0, h: float = 3.8):
    """A styled (fig, ax) at the given size in inches."""
    use_style()
    return plt.subplots(figsize=(w, h))


def clean_axes(ax) -> None:
    """Light spines + ticks for a data plot."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GREY_LINE)
    ax.spines["bottom"].set_color(GREY_LINE)
    ax.tick_params(length=4, width=1.0)


def blank_axes(ax) -> None:
    """No frame, equal aspect: for schematic / matrix diagrams."""
    ax.set_axis_off()
    ax.set_aspect("equal")


def palette_cycle():
    """A matplotlib cycler over the five book colours, for categorical series."""
    return mpl.cycler(color=PALETTE)


def _lum(hexc: str) -> float:
    """Perceived luminance in [0,1] for a #rrggbb colour."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def text_on(fill: str) -> str:
    """Legible label colour for text drawn on `fill` (dark ink on light fills,
    white on dark fills). Threshold chosen so the orange accent takes ink."""
    return INK if _lum(fill) > 0.55 else WHITE


def save(fig, stem: str):
    """Write <stem>.svg and <stem>.png into figures/. Returns the two paths."""
    use_style()
    svg = os.path.join(OUT_DIR, f"{stem}.svg")
    png = os.path.join(OUT_DIR, f"{stem}.png")
    fig.savefig(svg)
    fig.savefig(png)
    plt.close(fig)
    print(f"wrote {svg}")
    print(f"wrote {png}")
    return svg, png
