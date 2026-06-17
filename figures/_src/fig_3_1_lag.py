"""Figure 3.1: two impulse responses h(tau)=C e^{A tau} B. For chapters/02-foundations/02-continuous-time.qmd, sec 3.2.

  (i)  scalar decay, A=[-1]: h(tau)=e^{-tau}
  (ii) decayed rotation, lambda = -0.3 +- 2i: h(tau)=e^{-0.3 tau} cos 2 tau, bounded by +-e^{-0.3 tau}

Dots mark the lags the code cell prints (0, 0.5, 1, 2, 4).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

gamma, omega = 0.3, 2.0
tau = np.linspace(0.0, 8.0, 800)
h_decay = np.exp(-tau)                                   # scalar decay, A=[-1]
h_rot = np.exp(-gamma * tau) * np.cos(omega * tau)       # decayed rotation
env = np.exp(-gamma * tau)                               # |lambda| decay envelope

ts = np.array([0.0, 0.5, 1.0, 2.0, 4.0])                 # lags the code cell prints

fig, ax = S.figure(6.6, 3.5)
S.clean_axes(ax)
ax.axhline(0.0, color=S.GREY_LINE, lw=1.0, zorder=1)

# rotation's decay envelope +-e^{-gamma tau}
ax.plot(tau, env, color=S.GREY_LINE, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.plot(tau, -env, color=S.GREY_LINE, lw=1.2, ls=(0, (4, 3)), zorder=2)

ax.plot(tau, h_decay, color=S.BLUE, lw=2.4, zorder=4)
ax.plot(tau, h_rot, color=S.ORANGE, lw=2.4, zorder=5)

# dots at the printed lags
ax.plot(ts, np.exp(-ts), ls="none", marker="o", ms=4.2, color=S.BLUE_DARK,
        zorder=6)
ax.plot(ts, np.exp(-gamma * ts) * np.cos(omega * ts), ls="none", marker="o",
        ms=4.2, color=S.ORANGE_DARK, zorder=7)

ax.set_xlim(-0.18, 8.15)
ax.set_ylim(-0.72, 1.14)
ax.set_xlabel(r"lag $\tau$")
ax.set_ylabel(r"$h(\tau)$")
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([-0.5, 0.0, 0.5, 1.0])

# inline curve labels (defining forms)
ax.text(1.5, 0.42, r"$e^{-\tau}$", color=S.BLUE, ha="left", va="bottom",
        fontsize=12.5)
ax.text(3.05, 0.70, r"$e^{-0.3\tau}\cos 2\tau$", color=S.ORANGE_DARK, ha="left",
        va="bottom", fontsize=12.5)
ax.text(5.55, 0.255, r"$\pm e^{-0.3\tau}$", color=S.GREY, ha="left", va="bottom",
        fontsize=10.5)

S.save(fig, "fig-3-1-lag")
