"""Figure 5.2: the convolution matrix T in y = T u. For chapters/02-foundations/04-recurrence-convolution-toeplitz, section 5.4.

T is lower triangular (causality zeros the strict upper triangle) and Toeplitz: each descending diagonal carries one constant kernel tap Kbar_m = C Abar^m Bbar, so equal lags share a colour.
"""
import os
import sys

import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

L = 5  # sequence length / matrix size

# per-lag colour: lag 0 (main diagonal) is the orange accent; lags 1..L-1 fade
# down the blue ramp so older taps recede.
LAG_BLUE = S.BLUE_RAMP[1:]           # dark -> light
ZERO_FILL = S.GREY_FILL             # causal zeros
RULE = S.GREY_LINE


def lag_color(m: int) -> str:
    return S.ORANGE if m == 0 else LAG_BLUE[(m - 1) % len(LAG_BLUE)]


S.use_style()
fig, ax = S.figure(9.4, 5.0)

# Geometry: matrix in columns [0, L), u to its right, y left of "=". Cell
# (row i, col j) has its lower-left corner at (j, L-1-i) so row 0 is at the top.
GAP_VEC = 0.85          # gap between matrix and the input/output vectors
VEC_W = 1.0             # width of a vector column
CELL = 1.0


def draw_bracket(ax, x_left, x_right, y_bot, y_top, tick=0.18, lw=1.8):
    """Draw a pair of square matrix/vector brackets spanning the box."""
    for xb, x_in in [(x_left, x_left + tick), (x_right, x_right - tick)]:
        ax.plot([xb, xb], [y_bot, y_top], color=S.INK, lw=lw, zorder=5)
        ax.plot([xb, x_in], [y_bot, y_bot], color=S.INK, lw=lw, zorder=5)
        ax.plot([xb, x_in], [y_top, y_top], color=S.INK, lw=lw, zorder=5)


# convolution matrix T
for i in range(L):                       # row, top -> bottom
    for j in range(L):                   # column, left -> right
        x, y = j, (L - 1 - i)
        if i >= j:                       # lower triangle: kernel tap Kbar_{i-j}
            m = i - j
            fill = lag_color(m)
            ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=fill,
                                   edgecolor=RULE, lw=1.1, zorder=2))
            label = rf"$\bar K_{{{m}}}$"
            ax.text(x + 0.5, y + 0.5, label, color=S.text_on(fill),
                    ha="center", va="center", fontsize=14, zorder=3)
        else:                            # strict upper triangle: causal zero
            ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=ZERO_FILL,
                                   edgecolor=RULE, lw=1.1, zorder=2))
            ax.text(x + 0.5, y + 0.5, r"$0$", color=S.GREY,
                    ha="center", va="center", fontsize=12, zorder=3)

draw_bracket(ax, -0.06, L + 0.06, 0, L)

# input vector u (right of the matrix)
u_x = L + GAP_VEC
for i in range(L):
    y = L - 1 - i
    ax.add_patch(Rectangle((u_x, y), VEC_W, CELL, facecolor=S.WHITE,
                           edgecolor=RULE, lw=1.1, zorder=2))
    ax.text(u_x + 0.5, y + 0.5, rf"$u_{{{i}}}$", color=S.INK,
            ha="center", va="center", fontsize=13, zorder=3)
draw_bracket(ax, u_x - 0.06, u_x + VEC_W + 0.06, 0, L)

# output vector y (left of "="), positioned to centre "=" in the gap to T
EQ_GAP = 1.7             # horizontal gap reserved for the "=" sign
y_x = -(EQ_GAP + VEC_W)
for i in range(L):
    yy = L - 1 - i
    ax.add_patch(Rectangle((y_x, yy), VEC_W, CELL, facecolor=S.WHITE,
                           edgecolor=RULE, lw=1.1, zorder=2))
    ax.text(y_x + 0.5, yy + 0.5, rf"$y_{{{i}}}$", color=S.INK,
            ha="center", va="center", fontsize=13, zorder=3)
draw_bracket(ax, y_x - 0.06, y_x + VEC_W + 0.06, 0, L)

# "=" centred between the right edge of y and the left edge of T
eq_x = 0.5 * ((y_x + VEC_W) + 0.0)
ax.text(eq_x, L / 2, r"$=$", color=S.INK,
        ha="center", va="center", fontsize=18, zorder=5)

# symbol labels under each block
ax.text(y_x + 0.5 * VEC_W, -0.55, r"$y$", color=S.INK,
        ha="center", va="top", fontsize=14)
ax.text(L / 2, -0.55, r"$T$", color=S.INK,
        ha="center", va="top", fontsize=14)
ax.text(u_x + 0.5 * VEC_W, -0.55, r"$u$", color=S.INK,
        ha="center", va="top", fontsize=14)

# limits: pad so brackets and under-block labels are not clipped
ax.set_xlim(y_x - 0.5, u_x + VEC_W + 0.5)
ax.set_ylim(-1.05, L + 0.35)
S.blank_axes(ax)

S.save(fig, "fig-5-2-toeplitz")
