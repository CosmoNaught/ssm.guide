"""Figure 7.1: rescaling the growing history [0, t] onto the fixed unit interval via s = t*r, f_t(r) = u(t*r). For chapters/03-s4/07-hippo.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

# underlying signal u(s), s >= 0
def u(s):
    return (0.55 * np.exp(-((s - 1.05) ** 2) / 0.80)
            + 0.95 * np.exp(-((s - 3.05) ** 2) / 0.62)
            + 0.50 * np.exp(-((s - 4.75) ** 2) / 1.10)
            + 0.075 * s)


T = 5.2   # current time t

CURVE = S.BLUE_DARK
FILL = S.BLUE
FILL_ALPHA = 0.18
PRESENT = S.ORANGE

YMAX = 1.62
YFLOOR = -0.32

fig = plt.figure(figsize=(9.4, 3.6))
# left panel spans [0, t] wide; right panel is the narrow unit interval
gs = fig.add_gridspec(1, 3, width_ratios=[1.62, 0.40, 0.72], wspace=0.0,
                      left=0.045, right=0.985, top=0.97, bottom=0.215)
axL = fig.add_subplot(gs[0, 0])
axA = fig.add_subplot(gs[0, 1])
axR = fig.add_subplot(gs[0, 2])

# left panel: history u(s) on the growing window [0, t]
s = np.linspace(0.0, T, 700)
axL.fill_between(s, 0, u(s), color=FILL, alpha=FILL_ALPHA, lw=0, zorder=1)
axL.plot(s, u(s), color=CURVE, lw=2.5, zorder=4)

# faint signal just beyond t, suggesting the window keeps growing
sAfter = np.linspace(T, T + 1.05, 200)
axL.plot(sAfter, u(sAfter), color=S.GREY_LINE, lw=1.8, zorder=3)

# time axis
axL.plot([0, T + 1.05], [0, 0], color=S.GREY_LINE, lw=1.4, zorder=2)

# present marker at s = t
axL.plot([T, T], [0, u(T)], color=PRESENT, lw=1.7, ls=(0, (4, 3)), zorder=5)
axL.plot([T], [u(T)], "o", color=PRESENT, ms=8, mec=S.WHITE, mew=1.4, zorder=6)

# axis end labels: 0 and t
for xv in (0.0, T):
    axL.plot([xv, xv], [0, -0.055], color=S.GREY, lw=1.3, clip_on=False,
             zorder=5)
axL.text(0.0, -0.125, r"$0$", color=S.INK, ha="center", va="top", fontsize=11.5)
axL.text(T, -0.125, r"$t$", color=S.ORANGE_DARK, ha="center", va="top",
         fontsize=11.5)
axL.text(T / 2, -0.285, r"time $s$", color=S.GREY, ha="center", va="top",
         fontsize=10.5)

# signal name
axL.text(1.05, u(1.05) + 0.18, r"$u(s)$", color=CURVE, ha="center",
         va="bottom", fontsize=13)

axL.set_xlim(-0.22, T + 1.15)
axL.set_ylim(YFLOOR, YMAX)
axL.set_axis_off()
axL.set_aspect("auto")

# middle: the rescaling map f_t(r) = u(t r)
axA.set_xlim(0, 1)
axA.set_ylim(YFLOOR, YMAX)
axA.set_axis_off()
axA.set_aspect("auto")
ymid = 0.50 * (YMAX + YFLOOR) + 0.16
axA.annotate("", xy=(0.88, ymid), xytext=(0.12, ymid),
             arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.8,
                             mutation_scale=20))
axA.text(0.5, ymid + 0.055, r"$f_t(r)=u(t\,r)$", color=S.INK, ha="center",
         va="bottom", fontsize=12.5)

# right panel: rescaled signal on the fixed unit interval [0, 1]
r = np.linspace(0.0, 1.0, 600)
fr = u(T * r)   # same heights as the left curve, on a unit axis

axR.fill_between(r, 0, fr, color=FILL, alpha=FILL_ALPHA, lw=0, zorder=1)
axR.plot(r, fr, color=CURVE, lw=2.5, zorder=4)
axR.plot([0, 1], [0, 0], color=S.GREY_LINE, lw=1.4, zorder=2)

# present marker at r = 1
axR.plot([1, 1], [0, u(T)], color=PRESENT, lw=1.7, ls=(0, (4, 3)), zorder=5)
axR.plot([1], [u(T)], "o", color=PRESENT, ms=8, mec=S.WHITE, mew=1.4, zorder=6)

# axis end labels 0 and 1 (= present)
for xv in (0.0, 1.0):
    axR.plot([xv, xv], [0, -0.055], color=S.GREY, lw=1.3, clip_on=False,
             zorder=5)
axR.text(0.0, -0.125, r"$0$", color=S.INK, ha="center", va="top", fontsize=11.5)
axR.text(1.0, -0.125, r"$1$", color=S.ORANGE_DARK, ha="center", va="top",
         fontsize=11.5)
axR.text(0.5, -0.285, r"rescaled time $r$", color=S.GREY, ha="center",
         va="top", fontsize=10.5)

# signal name and the present marker r = 1
axR.text(0.205, u(T * 0.205) + 0.18, r"$f_t(r)$", color=CURVE, ha="center",
         va="bottom", fontsize=13)
axR.text(1.0, u(T) + 0.12, r"$r=1$", color=S.ORANGE_DARK, ha="right",
         va="bottom", fontsize=11)

axR.set_xlim(-0.085, 1.13)
axR.set_ylim(YFLOOR, YMAX)
axR.set_axis_off()
axR.set_aspect("auto")

S.save(fig, "fig-7-1-rescale")
