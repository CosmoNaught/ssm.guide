"""Figure 3.4: eigenvalue lambda = -alpha + i*omega as a point in the s-plane. For chapters/02-foundations/02-continuous-time.qmd, sec 3.3.

Book convention: alpha > 0 is the decay rate, so Re(lambda) = -alpha < 0 for a stable mode (open left half-plane). A real matrix gives a conjugate pair mirrored across the real axis (orange accent).
"""
import os
import sys

import numpy as np
from matplotlib.patches import Arc, Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

# conjugate pair lambda = -alpha +- i*omega, with alpha>0 the decay rate (Re = -alpha < 0)
ALPHA, OMEGA = -1.0, 1.2                          # ALPHA is the x-coordinate = -alpha
R = float(np.hypot(ALPHA, OMEGA))                 # |lambda| = circle radius
ANG = float(np.degrees(np.arctan2(OMEGA, ALPHA)))  # pole angle from +Re axis

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = S.figure(6.4, 5.6)

# axis extents (wide enough for the whole |lambda| circle)
XMIN, XMAX = -2.40, 2.00
YMIN, YMAX = -2.00, 2.00

# stable region: shade the left half-plane Re < 0
ax.axvspan(XMIN, 0.0, color=S.BLUE, alpha=0.12, zorder=0, lw=0)

# the |lambda| circle (the poles sit on it)
ax.add_patch(Circle((0, 0), R, fill=False, edgecolor=S.GREY_LINE, lw=1.3, zorder=1))

# complex-plane axes
arrow = dict(arrowstyle="-|>", color=S.GREY, lw=1.3, shrinkA=0, shrinkB=0)
ax.annotate("", xy=(XMAX, 0), xytext=(XMIN, 0), arrowprops=arrow, zorder=2)
ax.annotate("", xy=(0, YMAX), xytext=(0, YMIN), arrowprops=arrow, zorder=2)
ax.text(XMAX - 0.03, -0.12, r"$\mathrm{Re}\,\lambda$", color=S.GREY,
        ha="right", va="top", fontsize=12)
ax.text(0.10, YMAX - 0.02, r"$\mathrm{Im}\,\lambda$", color=S.GREY,
        ha="left", va="top", fontsize=12)

# radii to the poles + |lambda| label
ax.plot([0, ALPHA], [0, OMEGA], color=S.INK, lw=1.5, zorder=4)
ax.plot([0, ALPHA], [0, -OMEGA], color=S.GREY, lw=1.0, alpha=0.7, zorder=3)
ax.text(0.52 * ALPHA + 0.22, 0.52 * OMEGA + 0.10, r"$|\lambda|$", color=S.INK,
        ha="left", va="center", fontsize=12, zorder=9)

# angle theta between the negative real axis and the radius
ax.add_patch(Arc((0, 0), 1.05, 1.05, angle=0, theta1=ANG, theta2=180,
                 color=S.ORANGE_DARK, lw=1.6, zorder=5))
ax.text(-0.34, 0.16, r"$\theta$", color=S.ORANGE_DARK,
        ha="center", va="center", fontsize=12, zorder=6)

# dotted guides: drop each pole onto the Re and Im axes
guide = dict(color=S.GREY, lw=1.0, ls=(0, (2, 2.6)), zorder=3)
for sgn in (+1, -1):
    ax.plot([ALPHA, ALPHA], [0, sgn * OMEGA], **guide)
    ax.plot([ALPHA, 0], [sgn * OMEGA, sgn * OMEGA], **guide)

# the conjugate pair (orange X markers)
ax.plot([ALPHA, ALPHA], [OMEGA, -OMEGA], marker="x", ms=12, mew=3.0,
        color=S.ORANGE, ls="none", zorder=8)
ax.text(ALPHA - 0.10, OMEGA + 0.16, r"$\lambda=-\alpha+i\omega$", color=S.ORANGE_DARK,
        ha="center", va="bottom", fontsize=12.5, zorder=9)
ax.text(ALPHA - 0.10, -OMEGA - 0.16, r"$\bar\lambda=-\alpha-i\omega$",
        color=S.ORANGE_DARK, ha="center", va="top", fontsize=12.5, zorder=9)

# ticks + symbol labels for the real / imaginary parts
# real part is -alpha (alpha>0), so the tick sits left of origin
TICK = 0.07
ax.plot([ALPHA, ALPHA], [-TICK, TICK], color=S.INK, lw=1.4, zorder=5)
# label the real-part leg below theta, off the dashed drop-line
ax.text(0.66 * ALPHA, 0.09, r"$-\alpha$", color=S.INK, ha="center", va="bottom", fontsize=11.5)
for y, lab in ((OMEGA, r"$i\omega$"), (-OMEGA, r"$-i\omega$")):
    ax.plot([-TICK, TICK], [y, y], color=S.INK, lw=1.4, zorder=5)
    ax.text(0.12, y, lab, color=S.INK, ha="left", va="center", fontsize=11)

# frame / extents
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
S.blank_axes(ax)

S.save(fig, "fig-3-4-eigenvalues")
