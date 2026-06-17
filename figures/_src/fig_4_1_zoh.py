"""Figure 4.1: sampling and zero-order hold. The ZOH holds each sample u_k = u(k*Delta) constant over [k*Delta, (k+1)*Delta). For chapters/02-foundations/03-discretisation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

S.use_style()
fig, ax = S.figure(6.4, 3.8)

# continuous input u(t)
def u(t):
    return 1.00 + 0.40 * np.sin(0.74 * t) + 0.34 * np.sin(1.33 * t + 0.5)

Delta = 1.0                       # sample spacing
N = 8                             # number of held intervals shown
t = np.linspace(0.0, N * Delta, 800)

# sample instants t_k = k*Delta and samples u_k = u(t_k)
tk = np.arange(0, N + 1) * Delta
uk = u(tk)

# vertical sample rules at each sample instant
for x in tk:
    ax.axvline(x, color=S.GREY_LINE, lw=0.9, ls=(0, (2.0, 2.4)), zorder=0)

ax.plot(t, u(t), color=S.BLUE, lw=2.2, zorder=3,
        label=r"continuous input $u(t)$")

# zero-order-hold staircase: one path of held levels plus risers
xs, ys = [], []
for k in range(N):
    xs += [tk[k], tk[k + 1]]
    ys += [uk[k], uk[k]]
    if k + 1 < N:                 # riser to the next held level
        xs += [tk[k + 1]]
        ys += [uk[k + 1]]
ax.plot(xs, ys, color=S.ORANGE, lw=2.4, solid_capstyle="butt",
        solid_joinstyle="miter", zorder=5,
        label=r"zero-order hold $u_{\mathrm{zoh}}(t)$")

# sample dots at the left edge of each held step
ax.plot(tk[:N], uk[:N], "o", color=S.ORANGE, ms=5.5,
        mec=S.WHITE, mew=1.2, zorder=7, label=r"samples $u_k=u(k\Delta)$")

# axes
S.clean_axes(ax)
ax.set_xlabel(r"time $t$")
ax.set_ylabel(r"input $u$")
ax.set_xlim(-0.04, N * Delta + 0.04)
ax.set_ylim(0.0, 1.78)

# x ticks at the sample instants, labelled as multiples of Delta
ax.set_xticks(tk)
ax.set_xticklabels([r"$0$", r"$\Delta$", r"$2\Delta$", r"$3\Delta$",
                    r"$4\Delta$", r"$5\Delta$", r"$6\Delta$", r"$7\Delta$",
                    r"$8\Delta$"])
ax.set_yticks([0.0, 0.5, 1.0, 1.5])

leg = ax.legend(loc="lower left", frameon=True, fontsize=9.5,
                facecolor=S.WHITE, edgecolor=S.GREY_LINE, framealpha=0.94,
                borderpad=0.7, labelspacing=0.5, handlelength=1.9)
for txt in leg.get_texts():
    txt.set_color(S.INK)

S.save(fig, "fig-4-1-zoh")
