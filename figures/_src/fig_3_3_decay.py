"""Figure 3.3: exponential decay curves e^{-alpha*tau} (alpha = -Re lambda > 0) with the half-life construction tau_1/2 = log2 / alpha on the accent curve. For chapters/02-foundations/02-continuous-time.qmd."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

S.use_style()
fig, ax = S.figure(6.4, 4.0)

tau = np.linspace(0.0, 6.0, 600)

# decay rates alpha = -Re(lambda) > 0, listed slowest-to-fastest so the legend
# reads top-to-bottom from highest curve to lowest
HI = 0.50                      # accent rate carrying the half-life construction
curves = [
    (0.25, S.BLUE_RAMP[4], 2.0, 2),
    (HI,   S.ORANGE,       2.8, 5),
    (1.00, S.BLUE_RAMP[2], 2.0, 2),
    (2.00, S.BLUE_RAMP[0], 2.0, 2),
]

for a, color, lw, z in curves:
    ax.plot(tau, np.exp(-a * tau), color=color, lw=lw, zorder=z,
            label=f"${a:g}$")

# half-life construction on the accent curve
t_half = np.log(2.0) / HI
guide = dict(color=S.GREY, lw=1.2, ls=(0, (4, 3)), zorder=3)
ax.plot([0, t_half], [0.5, 0.5], **guide)         # half-maximum level
ax.plot([t_half, t_half], [0.0, 0.5], **guide)    # drop to the lag axis
ax.plot([t_half], [0.5], "o", color=S.ORANGE, ms=7.0,
        mec=S.WHITE, mew=1.3, zorder=6)

ax.set_xlabel(r"lag  $\tau$")
ax.set_ylabel(r"fraction remaining   $e^{-\alpha\tau}$")
ax.set_xlim(0, 6.15)
ax.set_ylim(0, 1.03)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_yticks([0, 0.5, 1.0])
S.clean_axes(ax)

# mark tau_1/2 on the lag axis
ax.annotate(r"$\tau_{1/2}$", xy=(t_half, 0), xytext=(t_half, -0.085),
            ha="center", va="top", color=S.ORANGE_DARK, fontsize=11)
ax.plot([t_half, t_half], [0.0, -0.022], color=S.ORANGE_DARK, lw=1.4,
        clip_on=False, zorder=5)

leg = ax.legend(title=r"rate  $\alpha=-\mathrm{Re}\,\lambda$", loc="upper right",
                frameon=True, fontsize=10, title_fontsize=10,
                facecolor=S.WHITE, edgecolor=S.GREY_LINE, framealpha=0.95,
                borderpad=0.7, labelspacing=0.5, handlelength=1.8)
leg.get_title().set_color(S.INK)

S.save(fig, "fig-3-3-decay")
