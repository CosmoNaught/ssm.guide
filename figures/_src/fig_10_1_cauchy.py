"""Figure 10.1: the Cauchy matrix C_{jn} = 1 / (s_j - lambda_n), rows indexed by
evaluation nodes s_j (L-th roots of unity), columns by poles lambda_n (diagonal
eigenvalues of A). For chapters/03-s4/10-diagonal-state-spaces.qmd.
"""
import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

L = 12                      # rows: evaluation nodes  s_0 .. s_{L-1}
N = 8                       # cols: poles             lambda_0 .. lambda_{N-1}

j = np.arange(L)
s = np.exp(2j * np.pi * j / L)              # roots of unity on the unit circle

# poles: N/2 conjugate pairs in the left half-plane (S4D initialisation)
k = np.arange(N // 2)
alpha = 0.5 * np.ones(N // 2)               # gentle, uniform decay rate alpha > 0
omega = 0.55 + 0.95 * k                     # rising frequencies
half = -alpha + 1j * omega
lam = np.empty(N, dtype=complex)
lam[0::2] = half
lam[1::2] = np.conj(half)                    # interleave conjugate pairs

C = 1.0 / (s[:, None] - lam[None, :])        # the Cauchy matrix C_{jn}
mag = np.abs(C)

# log magnitude scale spreads the wide dynamic range of 1/|.|
blues = LinearSegmentedColormap.from_list(
    "ssm_blues",
    ["#f3f5f8", "#bcc6d6", "#94a2bb", "#6e7f9c", "#4a5c79", "#3a4a63"],
)
RULE = S.GREY_LINE

vmin, vmax = mag.min(), mag.max()
norm = LogNorm(vmin=vmin, vmax=vmax)

S.use_style()
fig, ax = S.figure(7.4, 6.0)

# draw cell by cell so every cell carries a thin grey rule
for jj in range(L):                          # row index j
    for nn in range(N):                      # column index n
        x, y = nn, (L - 1 - jj)              # flip so row 0 sits at the top
        ax.add_patch(Rectangle(
            (x, y), 1, 1, facecolor=blues(norm(mag[jj, nn])),
            edgecolor=RULE, lw=0.6, zorder=2))

# matrix brackets around the L x N block
ear = 0.22
for xb, xin in [(-0.07, -0.07 + ear), (N + 0.07, N + 0.07 - ear)]:
    ax.plot([xb, xb], [0, L], color=S.INK, lw=1.8, zorder=6)
    ax.plot([xb, xin], [0, 0], color=S.INK, lw=1.8, zorder=6)
    ax.plot([xb, xin], [L, L], color=S.INK, lw=1.8, zorder=6)

# index ticks: poles lambda_n across the columns, nodes s_j down the rows
for nn in range(N):
    ax.text(nn + 0.5, L + 0.30, rf"$\lambda_{{{nn}}}$", color=S.BLUE,
            ha="center", va="bottom", fontsize=10.5, zorder=5)
ax.text(N / 2, L + 1.18, r"poles  $\lambda_n$", color=S.BLUE_DARK,
        ha="center", va="bottom", fontsize=12, zorder=5)

for jj in range(L):
    ax.text(-0.26, (L - 1 - jj) + 0.5, rf"$s_{{{jj}}}$", color=S.BLUE,
            ha="right", va="center", fontsize=10.5, zorder=5)
ax.text(-1.45, L / 2, r"nodes  $s_j$", color=S.BLUE_DARK,
        ha="center", va="center", rotation=90, fontsize=12, zorder=5)

# matrix name, set clear of the row-index axis label
ax.text(-2.55, L / 2, r"$C=$", color=S.INK, ha="center", va="center",
        fontsize=16, zorder=5)

# defining entry, to the right of the matrix
rx = N + 1.1
ax.text(rx, L - 0.7, r"$C_{jn}=\dfrac{1}{s_j-\lambda_n}$",
        color=S.ORANGE_DARK, ha="left", va="center", fontsize=15, zorder=6)
ax.text(rx, L - 2.7, r"$s_j=e^{2\pi i\,j/L}$", color=S.INK,
        ha="left", va="center", fontsize=11.5, zorder=6)

# magnitude legend: vertical gradient bar (data units) in the lower-right
cb_x0, cb_x1 = N + 1.3, N + 1.9
cb_y0, cb_y1 = 0.3, L * 0.42
grad = np.linspace(0.0, 1.0, 256)[:, None]
ax.imshow(grad, extent=(cb_x0, cb_x1, cb_y0, cb_y1), aspect="auto",
          origin="lower", cmap=blues, vmin=0.0, vmax=1.0, zorder=2,
          interpolation="bilinear")
ax.add_patch(Rectangle((cb_x0, cb_y0), cb_x1 - cb_x0, cb_y1 - cb_y0,
                       fill=False, edgecolor=S.GREY_LINE, lw=0.8, zorder=3))
ax.text(cb_x1 + 0.18, cb_y0, "low", color=S.GREY, ha="left", va="center",
        fontsize=9.5, zorder=5)
ax.text(cb_x1 + 0.18, cb_y1, "high", color=S.GREY, ha="left", va="center",
        fontsize=9.5, zorder=5)
ax.text(cb_x0 - 0.05, cb_y1 + 0.42, r"$|C_{jn}|$", color=S.INK, ha="left",
        va="bottom", fontsize=11.5, zorder=5)

ax.set_xlim(-3.0, N + 4.6)
ax.set_ylim(-0.8, L + 1.9)
S.blank_axes(ax)

S.save(fig, "fig-10-1-cauchy")
