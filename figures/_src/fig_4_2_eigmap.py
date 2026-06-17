"""Figure 4.2: zero-order hold maps the stable left half-plane (Re lambda < 0) into the unit disc via mu = e^{lambda*Delta}, since |mu| = e^{Delta*Re lambda}. For chapters/02-foundations/03-discretisation.

Not labelled s-/z-plane: the book z-transform convention (chapter 6) puts stable poles outside the unit circle, which would contradict it.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

S.use_style()

# discretisation step and representative conjugate pair
DELTA = 1.0
SIGMA, OMEGA = -0.7, 1.0                       # lambda = sigma + i*omega
MU = np.exp(DELTA * (SIGMA + 1j * OMEGA))      # its image, |MU| = e^{-0.7} ~ 0.5

# windows: both panels square (equal data span) so they render at the same scale
XL_MIN, XL_MAX = -2.0, 0.8
YL = 0.5 * (XL_MAX - XL_MIN)                   # 1.4: half-height, equal aspect
LIM_R = 1.4                                    # z-plane half-width (= YL)

# layout: two equal squares, origins at the same height, a gap for the map
FIG_W, FIG_H = 7.2, 3.6
BOX_IN = 2.7                                   # rendered panel size (inches)
W_FRAC, H_FRAC = BOX_IN / FIG_W, BOX_IN / FIG_H
MARG = 0.03
Y0 = 0.5 - H_FRAC / 2 + 0.04                   # nudged up: panel names sit below

fig = plt.figure(figsize=(FIG_W, FIG_H))
axL = fig.add_axes([MARG, Y0, W_FRAC, H_FRAC])
axR = fig.add_axes([1.0 - MARG - W_FRAC, Y0, W_FRAC, H_FRAC])


def plane_frame(ax, xmin, xmax, ymin, ymax, v_axis=True):
    """Draw arrowed axes through the origin, equal aspect, no box. v_axis=False skips the vertical axis (the lambda-plane draws its own, since there the imaginary axis is the boundary)."""
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    axis = dict(arrowstyle="-|>", color=S.GREY, lw=1.1, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(xmax, 0), xytext=(xmin, 0), arrowprops=axis, zorder=4)
    if v_axis:
        ax.annotate("", xy=(0, ymax), xytext=(0, ymin), arrowprops=axis, zorder=4)


# left panel (lambda-plane): stable region is the left half-plane Re lambda < 0
plane_frame(axL, XL_MIN, XL_MAX, -YL, YL, v_axis=False)

axL.axvspan(XL_MIN, 0, color=S.BLUE, alpha=0.12, zorder=0, lw=0)

# the imaginary axis is the stability boundary (it maps onto the unit circle), so it carries boundary green
axL.annotate("", xy=(0, YL), xytext=(0, -YL), zorder=5,
             arrowprops=dict(arrowstyle="-|>", color=S.GREEN_DARK, lw=2.2,
                             shrinkA=0, shrinkB=0, mutation_scale=16))

# the conjugate pair
axL.plot([SIGMA, SIGMA], [OMEGA, -OMEGA], marker="x", ms=11, mew=2.8,
         color=S.ORANGE, ls="none", zorder=8)
axL.text(SIGMA, OMEGA + 0.13, r"$\lambda$", color=S.ORANGE_DARK,
         ha="center", va="bottom", fontsize=12, zorder=9)
axL.text(SIGMA, -OMEGA - 0.13, r"$\bar\lambda$", color=S.ORANGE_DARK,
         ha="center", va="top", fontsize=12, zorder=9)

# region label
axL.text(XL_MIN + 0.13, -0.45, r"$\mathrm{Re}\,\lambda < 0$",
         color=S.BLUE_DARK, ha="left", va="center", fontsize=11, zorder=6)

# axis labels
axL.text(XL_MAX - 0.02, -0.10, r"$\mathrm{Re}\,\lambda$", color=S.GREY,
         ha="right", va="top", fontsize=10.5)
axL.text(0.09, YL - 0.02, r"$\mathrm{Im}\,\lambda$", color=S.GREY,
         ha="left", va="top", fontsize=10.5)
axL.text(0.5 * (XL_MIN + XL_MAX), -YL - 0.18, r"$\lambda$-plane",
         color=S.INK, ha="center", va="top", fontsize=12)


# right panel (mu-plane): stable region is the unit disk |mu| < 1
plane_frame(axR, -LIM_R, LIM_R, -LIM_R, LIM_R)

axR.add_patch(Circle((0, 0), 1.0, facecolor=S.BLUE, alpha=0.12,
                     edgecolor="none", zorder=0))

# the unit circle: image of the imaginary axis, same green stroke
theta = np.linspace(0, 2 * np.pi, 400)
axR.plot(np.cos(theta), np.sin(theta), color=S.GREEN_DARK, lw=2.2, zorder=5)

# the image pair mu, mubar inside the disk
axR.plot([MU.real, MU.real], [MU.imag, -MU.imag], marker="x", ms=11, mew=2.8,
         color=S.ORANGE, ls="none", zorder=8)
axR.text(MU.real, MU.imag + 0.13, r"$\mu$", color=S.ORANGE_DARK,
         ha="center", va="bottom", fontsize=12, zorder=9)
axR.text(MU.real, -MU.imag - 0.13, r"$\bar\mu$", color=S.ORANGE_DARK,
         ha="center", va="top", fontsize=12, zorder=9)

# region label inside the disk
axR.text(-0.52, -0.40, r"$|\mu| < 1$", color=S.BLUE_DARK,
         ha="center", va="center", fontsize=11, zorder=6)

# axis labels
axR.text(LIM_R - 0.02, -0.10, r"$\mathrm{Re}\,\mu$", color=S.GREY,
         ha="right", va="top", fontsize=10.5)
axR.text(0.09, LIM_R - 0.02, r"$\mathrm{Im}\,\mu$", color=S.GREY,
         ha="left", va="top", fontsize=10.5)
axR.text(0, -LIM_R - 0.18, r"$\mu$-plane", color=S.INK, ha="center", va="top",
         fontsize=12)


# cross-panel transform arrow in the gap, the map written above
gx0 = MARG + W_FRAC                  # right edge of the left panel (fig coords)
gx1 = 1.0 - MARG - W_FRAC            # left edge of the right panel
gy = Y0 + H_FRAC / 2                 # height of both origins (data y = 0)
PAD = 0.022

arr = FancyArrowPatch(
    (gx0 + PAD, gy), (gx1 - PAD, gy), transform=fig.transFigure,
    arrowstyle="-|>", mutation_scale=14, color=S.GREY, lw=1.8,
    shrinkA=0, shrinkB=0, zorder=10,
)
fig.patches.append(arr)
fig.text(0.5 * (gx0 + gx1), gy + 0.07, r"$\mu = e^{\lambda\Delta}$",
         color=S.BLUE_DARK, ha="center", va="bottom", fontsize=13.5,
         zorder=11)

S.save(fig, "fig-4-2-eigenvalue-map")
