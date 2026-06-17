"""Figure 7.2: projecting a history onto a finite Legendre basis (the HiPPO idea).
For chapters/03-s4/06-memory-problem.qmd. Panels: (a) basis g_0..g_{N-1};
(b) history f_t(r) and its order-(N-1) projection; (c) coefficient vector c in R^N.

The history curve and coefficient bars are also exported as high-DPI PNGs
(figures/_tikz/fig72_*.png) for figure 8.2 to embed. PNG not PDF: dvisvgm carries
an embedded raster into the SVG but silently drops an embedded PDF form XObject.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402


# Shifted Legendre P_n(2r-1), normalised so <g_n, g_m> = delta_{nm} on [0,1].
def legendre_shifted(n, r):
    x = 2.0 * r - 1.0           # map [0,1] -> [-1,1]
    p0 = np.ones_like(x)
    if n == 0:
        leg = p0
    else:
        p1 = x
        if n == 1:
            leg = p1
        else:
            for k in range(2, n + 1):
                p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                p0, p1 = p1, p2
            leg = p1
    return np.sqrt(2 * n + 1.0) * leg   # L2-normalised on [0,1]


# A smooth shape the first N modes capture well but not exactly, so the
# projection visibly tracks the history with a small gap.
def history(r):
    return (0.50
            + 0.46 * np.cos(2.6 * np.pi * r - 0.7)
            - 0.30 * r
            + 0.22 * np.sin(3.4 * np.pi * r))


N = 4                                  # state dimension = number of coefficients
r = np.linspace(0.0, 1.0, 1000)
f = history(r)

# basis functions and projection coefficients  c_n = <f, g_n>  on [0,1]
basis = np.array([legendre_shifted(n, r) for n in range(N)])
coeffs = np.array([np.trapz(f * basis[n], r) for n in range(N)])

# reconstruction from the first N coefficients
f_hat = (coeffs[:, None] * basis).sum(axis=0)

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

fig = plt.figure(figsize=(9.4, 3.5))
gs = fig.add_gridspec(
    1, 3, width_ratios=[0.92, 1.18, 0.62], wspace=0.34,
    left=0.055, right=0.975, top=0.86, bottom=0.165,
)
axB = fig.add_subplot(gs[0, 0])   # the basis
axF = fig.add_subplot(gs[0, 1])   # history + projection
axC = fig.add_subplot(gs[0, 2])   # coefficient bars (the state)

# panel (a): basis g_0..g_{N-1}. Orange is held back for the projection, so
# each basis function colour matches its coefficient bar in panel (c).
cat = [c for c in S.PALETTE if c != S.ORANGE][:N]
cat_dark = [c for c in S.PALETTE_DARK if c != S.ORANGE_DARK][:N]
for n in range(N):
    axB.plot(r, basis[n], color=cat[n], lw=2.0, zorder=3 + (N - n) * 0.01)
axB.axhline(0.0, color=S.GREY_LINE, lw=1.0, zorder=1)

axB.set_xlim(0.0, 1.0)
axB.set_xticks([0.0, 0.5, 1.0])
# clip the y-range so high-order modes spiking at the endpoints do not stretch
# the panel.
bmax = np.percentile(np.abs(basis), 99.0)
axB.set_ylim(-1.30 * bmax, 1.45 * bmax)
axB.set_yticks([])
axB.set_xlabel(r"$r\in[0,1]$")
axB.set_title(r"(a)  basis  $g_0,\dots,g_{N-1}$", color=S.INK, fontsize=12, pad=8)
S.clean_axes(axB)
axB.spines["left"].set_visible(False)
axB.tick_params(axis="y", length=0)

# panel (b): history and its order-(N-1) projection.
axF.fill_between(r, f, f_hat, color=S.ORANGE, alpha=0.13, lw=0, zorder=1)
axF.plot(r, f, color=S.BLUE, lw=2.4, zorder=3, label=r"history  $f_t(r)$")
axF.plot(r, f_hat, color=S.ORANGE, lw=2.7, zorder=4,
         label=r"projection  $\hat f_t(r)$")

axF.set_xlim(0.0, 1.0)
axF.set_xticks([0.0, 0.5, 1.0])
ymin = min(f.min(), f_hat.min())
ymax = max(f.max(), f_hat.max())
axF.set_ylim(ymin - 0.14 * (ymax - ymin), ymax + 0.34 * (ymax - ymin))
axF.set_yticks([])
axF.set_xlabel(r"$r\in[0,1]$")
axF.set_title("(b)  history and projection", color=S.INK, fontsize=12, pad=8)
S.clean_axes(axF)
axF.spines["left"].set_visible(False)
axF.tick_params(axis="y", length=0)

axF.legend(loc="upper left", fontsize=10, frameon=True,
           facecolor=S.WHITE, edgecolor=S.GREY_LINE, framealpha=0.92,
           borderpad=0.55, handlelength=1.6, labelspacing=0.45)

# panel (c): coefficient vector c, the state that is kept.
idx = np.arange(N)
# each coefficient c_n carries the colour of its basis function g_n (panel a).
axC.bar(idx, coeffs, width=0.66, color=cat,
        edgecolor=cat_dark, lw=1.0, zorder=3)
axC.axhline(0.0, color=S.GREY_LINE, lw=1.0, zorder=2)

axC.set_xticks(idx)
axC.set_xticklabels([rf"$c_{{{n}}}$" for n in idx], fontsize=11)
cmax = np.abs(coeffs).max()
axC.set_ylim(-0.55 * cmax, 1.30 * cmax)
axC.set_xlim(-0.62, N - 0.38)
axC.set_yticks([])
axC.spines["left"].set_visible(False)
axC.tick_params(axis="x", length=0, width=1.0)
axC.tick_params(axis="y", length=0)
axC.set_title(r"(c)  state  $c\in\mathbb{R}^{N}$", color=S.INK, fontsize=12, pad=8)

# transform labels in the inter-panel gaps (figure coords)
fig.canvas.draw()


def _gap_arrow(ax_left, ax_right, label, y=0.50):
    x0 = ax_left.get_position().x1
    x1 = ax_right.get_position().x0
    arrow = plt.matplotlib.patches.FancyArrowPatch(
        (x0 + 0.008, y), (x1 - 0.008, y),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=13,
        lw=1.5, color=S.GREY, zorder=5)
    fig.patches.append(arrow)
    fig.text(0.5 * (x0 + x1), y + 0.035, label, ha="center", va="bottom",
             color=S.GREY, fontsize=11)


# basis -> history+projection: the inner product picks the coefficients
_gap_arrow(axB, axF, r"$\langle f_t,\,g_n\rangle$")
# projection -> coefficients: keep c
_gap_arrow(axF, axC, r"$c_n$")

# Export the history curve and coefficient bars so figure 8.2 embeds the
# identical artefacts. Saved into figures/_tikz so the TikZ build (cwd =
# figures/_tikz) can \includegraphics them directly.
_tikz = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_tikz")

gh = plt.figure(figsize=(2.8, 1.0))
axgh = gh.add_axes([0.0, 0.0, 1.0, 1.0]); axgh.axis("off")
axgh.plot(r, f, color=S.BLUE, lw=3.2, solid_capstyle="round")
axgh.set_xlim(0.0, 1.0)
axgh.set_ylim(f.min() - 0.08, f.max() + 0.08)
gh.savefig(os.path.join(_tikz, "fig72_history.png"), transparent=True,
           dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(gh)

gc = plt.figure(figsize=(2.2, 1.0))
axgc = gc.add_axes([0.0, 0.0, 1.0, 1.0]); axgc.axis("off")
axgc.bar(idx, coeffs, width=0.66, color=cat, edgecolor=cat_dark, lw=1.3)
axgc.axhline(0.0, color=S.GREY_LINE, lw=1.3)
axgc.set_xlim(-0.62, N - 0.38)
axgc.set_ylim(-0.55 * cmax, 1.20 * cmax)
gc.savefig(os.path.join(_tikz, "fig72_coeffs.png"), transparent=True,
           dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(gc)

S.save(fig, "fig-7-2-projection")
