"""Figure fig-9-1-nplr-dplr: HiPPO-LegS A as NPLR then DPLR. For chapters/03-s4/10-diagonal-state-spaces.

Three rows, the same operator rewritten:
    A          = V L V* - P P*    normal plus low rank (NPLR)
               = L     - P~ P~*   diagonal plus low rank (DPLR), conjugated by V.
"""
import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402


# numerics: HiPPO-LegS and its NPLR / DPLR decomposition
def hippo_legs(n: int) -> np.ndarray:
    """HiPPO-LegS state matrix (S4 paper), N x N, real and lower triangular."""
    A = np.zeros((n, n))
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lower = rows > cols
    A[lower] = -np.sqrt((2 * rows[lower] + 1) * (2 * cols[lower] + 1))
    np.fill_diagonal(A, -(np.arange(n) + 1.0))
    return A


def decompose(n: int):
    """Return the matrices to plot.

    A         dense HiPPO-LegS
    S         normal part, A = S - P P^T  (S is skew-symmetric plus -1/2 I)
    P         rank-one factor (real column)
    Lam       diagonal of eigenvalues (complex)
    Ptil      transformed rank-one factor (complex)
    """
    A = hippo_legs(n)
    idx = np.arange(n)
    P = np.sqrt(idx + 0.5)                 # rank-one factor
    Snormal = A + np.outer(P, P)           # normal: S + S^T = -I exactly
    w, V = np.linalg.eig(Snormal)
    Vinv = np.linalg.inv(V)
    Lam = Vinv @ Snormal @ V               # diagonal (eigenvalues)
    Ptil = Vinv @ P                        # transformed factor
    return A, Snormal, P, Lam, Ptil


N = 12
A, Snormal, P, Lam, Ptil = decompose(N)

# magnitudes drive heatmap intensity (matrices are complex / signed)
M_dense = np.abs(A)
M_normal = np.abs(Snormal)
lam = np.diag(Lam)                          # complex eigenvalues
M_diag = np.abs(np.diag(lam))               # N x N, magnitudes on the diagonal
p_dense = np.abs(P)                         # real rank-one column
p_diag = np.abs(Ptil)                       # complex rank-one column

# shared normalisation so the matrix panels are comparable
mat_norm = max(M_dense.max(), M_normal.max(), M_diag.max())
# rank-one factors share their own normalisation
vec_norm = max(p_dense.max(), p_diag.max())


# style + figure
S.use_style()

# blue ramp for structure intensity (light = small, dark = large)
BLUE_MAP = LinearSegmentedColormap.from_list(
    "ssm_blue", [S.WHITE, "#bcc6d6", S.BLUE, S.BLUE_DARK])
# orange ramp for the accent: diagonal of Lambda and the rank-one factor
ORANGE_MAP = LinearSegmentedColormap.from_list(
    "ssm_orange", [S.WHITE, S.ORANGE, S.ORANGE_DARK])

CELL = S.GREY_LINE


def draw_matrix(ax, x0, y0, mag, *, diag_emph=False, diag_only=False):
    """Draw an N x N heatmap as a grid of cells from (x0, y0), growing right and down (row 0 at top). Returns (width, height)."""
    n = mag.shape[0]
    norm = mat_norm
    for i in range(n):
        for j in range(n):
            on_diag = (i == j)
            if diag_only and not on_diag:
                face = S.WHITE
            elif on_diag and diag_emph:
                # darker blue so the diagonal Lambda reads against white off-diagonal, without stealing the accent
                face = BLUE_MAP(0.45 + 0.55 * mag[i, j] / norm)
            else:
                face = BLUE_MAP(0.12 + 0.88 * mag[i, j] / norm)
            ax.add_patch(FancyBboxPatch(
                (x0 + j, y0 - (i + 1)), 1, 1,
                boxstyle="square,pad=0", linewidth=0.6,
                edgecolor=CELL, facecolor=face, zorder=2,
                mutation_aspect=1))
    # outer frame
    ax.add_patch(FancyBboxPatch(
        (x0, y0 - n), n, n, boxstyle="square,pad=0",
        linewidth=1.3, edgecolor=S.INK, facecolor="none", zorder=4,
        mutation_aspect=1))
    return n, n


def draw_outer(ax, x0, y0, col, *, height):
    """Draw a rank-one outer product p p^* as a bent corner: column p down the left, row p^* along the top, sharing the top-left cell. Returns (width, height)."""
    n = col.shape[0]
    cw = 1.0                                 # column thickness (one cell wide)
    rh = 1.0                                 # row thickness (one cell tall)
    row_len = n - 1                          # row continues from the shared cell

    # column p : full height, top cell at y0
    for i in range(n):
        val = 0.20 + 0.80 * col[i] / vec_norm
        ax.add_patch(FancyBboxPatch(
            (x0, y0 - (i + 1)), cw, 1, boxstyle="square,pad=0",
            linewidth=0.6, edgecolor=CELL, facecolor=ORANGE_MAP(val),
            zorder=2, mutation_aspect=1))
    ax.add_patch(FancyBboxPatch(
        (x0, y0 - n), cw, n, boxstyle="square,pad=0",
        linewidth=1.3, edgecolor=S.INK, facecolor="none", zorder=4,
        mutation_aspect=1))

    # row p^* : one cell tall, extending right from the column; skip j=0 (shared corner)
    ry = y0 - rh
    rx = x0 + cw
    for j in range(1, n):
        val = 0.20 + 0.80 * col[j] / vec_norm
        ax.add_patch(FancyBboxPatch(
            (rx + (j - 1), ry), 1, rh, boxstyle="square,pad=0",
            linewidth=0.6, edgecolor=CELL, facecolor=ORANGE_MAP(val),
            zorder=2, mutation_aspect=1))
    ax.add_patch(FancyBboxPatch(
        (rx, ry), row_len, rh, boxstyle="square,pad=0",
        linewidth=1.3, edgecolor=S.INK, facecolor="none", zorder=4,
        mutation_aspect=1))

    return cw + row_len, height


# layout: three rows, each "MAT  -  (col)(row)" with row labels on the left
fig, ax = S.figure(7.6, 8.4)

row_gap = 3.4                                # vertical gap between rows
top = 0.0                                    # y of the top edge of row 0

# column anchors (x of left edge of each element)
x_lab = -6.6                                 # left equation label
x_mat = 0.0                                  # the matrix block
x_minus = N + 1.4                            # the minus sign
x_outer = x_minus + 1.9                      # the rank-one block (its column)

LBL = 16
SYM = 14
GREY_TAG = 11
HEAD = 11.5


def minus_sign(y_top):
    ax.text(x_minus, y_top - N / 2, "−", color=S.INK,
            ha="center", va="center", fontsize=24)


def eq_label(y_top, sym, tag):
    """Left equation label: symbol with a descriptor under it."""
    ax.text(x_lab, y_top - N / 2 + 0.5, sym, color=S.INK, ha="left",
            va="center", fontsize=LBL)
    ax.text(x_lab, y_top - N / 2 - 1.3, tag, color=S.GREY, ha="left",
            va="center", fontsize=GREY_TAG)


def outer_label(y_top, sym):
    """Rank-one factor symbol, set above the column."""
    ax.text(x_outer + 0.5, y_top + 0.7, sym, color=S.ORANGE_DARK,
            ha="center", va="bottom", fontsize=SYM)


# row 1: A (dense)
y0 = top
draw_matrix(ax, x_mat, y0, M_dense)
eq_label(y0, r"$A$", "dense")

# row 2: V L V* - P P* (NPLR)
y0 = top - (N + row_gap)
draw_matrix(ax, x_mat, y0, M_normal)
minus_sign(y0)
draw_outer(ax, x_outer, y0, p_dense, height=N)
eq_label(y0, r"$V\Lambda V^{*}$", "normal")
outer_label(y0, r"$P\,P^{*}$")

# row 3: L - P~ P~* (DPLR)
y0 = top - 2 * (N + row_gap)
draw_matrix(ax, x_mat, y0, M_diag, diag_emph=True, diag_only=True)
minus_sign(y0)
draw_outer(ax, x_outer, y0, p_diag, height=N)
eq_label(y0, r"$\Lambda$", "diagonal")
outer_label(y0, r"$\tilde P\,\tilde P^{*}$")

# "=" between rows, conjugation marker between rows 2 and 3
mid12 = top - (N + row_gap / 2)
ax.text(x_mat + N / 2, mid12, "=", color=S.INK, ha="center", va="center",
        fontsize=22)
mid23 = top - (N + row_gap) - (N + row_gap / 2)
ax.text(x_mat + N / 2, mid23, "=", color=S.INK, ha="center", va="center",
        fontsize=22)
# conjugation V*(.)V carrying row 2 into row 3
ax.text(x_mat + N / 2 + 1.0, mid23, r"$V^{*}(\cdot)\,V$", color=S.GREY,
        ha="left", va="center", fontsize=11.5)

# limits + framing
x_right = x_outer + 1 + 0.6 + N + 0.8
ax.set_xlim(x_lab - 0.6, x_right)
ax.set_ylim(top - 2 * (N + row_gap) - N - 1.2, top + 1.6)
S.blank_axes(ax)

S.save(fig, "fig-9-1-nplr-dplr")
