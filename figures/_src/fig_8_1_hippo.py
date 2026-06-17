"""Figure 8.1: the N x N HiPPO-LegS matrix H_N as a heatmap. For chapters/03-s4/07-hippo.qmd.

Zero-based indices n, k = 0..N-1; see ssm_book.numpy_ref.hippo.legs_hippo:

    H_{nk} = sqrt((2n+1)(2k+1))   for n > k   (strictly lower),
    H_{nn} = n + 1                for n = k   (diagonal),
    H_{nk} = 0                    for n < k   (strictly upper).
"""
import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

N = 8

n = np.arange(N)
r = np.sqrt(2 * n + 1.0)
H = r[:, None] * r[None, :]          # H_{nk} = sqrt((2n+1)(2k+1))
H = np.tril(H, k=-1)                 # keep strictly-lower part
H[np.diag_indices(N)] = n + 1.0      # diagonal H_{nn} = n+1

lower = np.tril(np.ones((N, N), bool))   # diagonal + below = the filled cells

# near-white at zero, house blue ramp at the maximum
blues = LinearSegmentedColormap.from_list(
    "ssm_blues",
    ["#f3f5f8"] + S.BLUE_RAMP[::-1],
)
ZERO_FILL = "#fafbfc"   # structural zeros (upper triangle)
RULE = S.GREY_LINE

vmax = float(np.ceil(H.max()))   # ceiling 14 (true max H_{7,6}=sqrt(195)~=13.96)
norm = Normalize(vmin=0.0, vmax=vmax)

S.use_style()
fig, ax = S.figure(6.4, 5.4)

# row 0 flipped to the top so the matrix reads like its written form
for i in range(N):
    for j in range(N):
        x, y = j, (N - 1 - i)
        if lower[i, j]:
            val = H[i, j]
            fill = blues(norm(val))
            on_diag = (i == j)
            ax.add_patch(Rectangle(
                (x, y), 1, 1, facecolor=fill,
                edgecolor=RULE, lw=0.8, zorder=2))
            # label only the diagonal (1..N); colour carries the rest
            if on_diag:
                # outline so labels read on both light and dark cells
                ax.text(x + 0.5, y + 0.5, f"${i + 1}$", color=S.ORANGE_DARK,
                        ha="center", va="center", fontsize=11.5, zorder=4,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground=S.INK)])
        else:                        # strictly upper triangle: pale empty cells
            ax.add_patch(Rectangle(
                (x, y), 1, 1, facecolor=ZERO_FILL,
                edgecolor=RULE, lw=0.8, zorder=2))

# matrix brackets
for xb, x_in in [(-0.10, 0.24), (N + 0.10, N - 0.24)]:
    ax.plot([xb, xb], [0, N], color=S.INK, lw=1.8, zorder=6,
            solid_capstyle="round")
    ax.plot([xb, x_in], [0, 0], color=S.INK, lw=1.8, zorder=6,
            solid_capstyle="round")
    ax.plot([xb, x_in], [N, N], color=S.INK, lw=1.8, zorder=6,
            solid_capstyle="round")

# index ticks: n down the rows, k across the columns
for j in range(N):                   # column index k along the top
    ax.text(j + 0.5, N + 0.30, f"${j}$", color=S.GREY, ha="center",
            va="bottom", fontsize=9.5)
for i in range(N):                   # row index n down the left
    ax.text(-0.30, (N - 1 - i) + 0.5, f"${i}$", color=S.GREY, ha="right",
            va="center", fontsize=9.5)
ax.text(N / 2, N + 0.96, r"column index $k$", color=S.INK, ha="center",
        va="bottom", fontsize=10.5)
ax.text(-1.28, N / 2, r"row index $n$", color=S.INK, ha="center",
        va="center", rotation=90, fontsize=10.5)
# colourbar: vertical gradient strip in data units so it tracks the matrix height
cb_x0 = N + 0.85
cb_w = 0.42
cb_y0, cb_y1 = 0.0, float(N)
grad = np.linspace(0.0, 1.0, 256)[:, None]
ax.imshow(grad, extent=(cb_x0, cb_x0 + cb_w, cb_y0, cb_y1), aspect="auto",
          origin="lower", cmap=blues, vmin=0.0, vmax=1.0, zorder=2,
          interpolation="bilinear")
ax.add_patch(Rectangle((cb_x0, cb_y0), cb_w, cb_y1 - cb_y0,
                       fill=False, edgecolor=S.GREY_LINE, lw=0.8, zorder=3))
# bar ticks: 0 and the rounded ceiling
for frac, label in [(0.0, "$0$"), (1.0, f"${vmax:.0f}$")]:
    yt = cb_y0 + frac * (cb_y1 - cb_y0)
    ax.plot([cb_x0 + cb_w, cb_x0 + cb_w + 0.10], [yt, yt],
            color=S.GREY_LINE, lw=0.8, zorder=3)
    ax.text(cb_x0 + cb_w + 0.18, yt, label, color=S.GREY,
            ha="left", va="center", fontsize=9)
ax.text(cb_x0 + cb_w / 2, cb_y1 + 0.46, r"$H_{nk}$", color=S.INK,
        ha="center", va="bottom", fontsize=10.5)

ax.set_xlim(-1.8, cb_x0 + cb_w + 1.5)
ax.set_ylim(-0.9, N + 1.6)
S.blank_axes(ax)

S.save(fig, "fig-8-1-hippo-matrix")
