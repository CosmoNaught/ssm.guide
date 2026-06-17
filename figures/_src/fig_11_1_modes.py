"""Figure 11.1: a diagonal SSM kernel as a sum of decaying-oscillating modes, Kbar_m = sum_n w_n lambdabar_n^m. For chapters/03-s4/10-diagonal-state-spaces (S4D)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

S.use_style()
fig, ax = S.figure(6.6, 4.2)

# the modes
# Each mode is the real part of a conjugate-paired eigenvalue
# lambdabar_n = r_n * exp(i theta_n) with weight w_n:
# w_n * Re(lambdabar_n^m) = w_n * r_n^m * cos(theta_n * m).
# r_n in (0,1) is per-step decay; theta_n is per-step phase [rad].
#   (r_n decay, theta_n per-step phase [rad], w_n weight)
modes = [
    (0.965, 0.00,  0.38),   # slow, near-DC  -> smooth dominant long tail
    (0.945, 0.52,  0.46),   # low frequency  -> broad, persistent oscillation
    (0.90,  1.15,  0.40),   # mid frequency  -> the early shoulder/dip
    (0.85,  2.10,  0.30),   # high frequency -> fast ripple, decays first
]

m = np.arange(0, 33)                      # lag axis, integer steps
mode_curves = [w * r ** m * np.cos(th * m) for (r, th, w) in modes]
K = np.sum(mode_curves, axis=0)           # Kbar_m = sum_n w_n Re(lambdabar_n^m)

# blue ramp for the modes, slow near-DC mode most saturated; lightest ramp
# shade skipped as near-invisible against white
mode_colors = [S.BLUE_RAMP[0], S.BLUE_RAMP[1], S.BLUE_RAMP[2], S.BLUE_RAMP[3]]

# baseline so the sign changes (oscillation) read clearly
ax.axhline(0.0, color=S.GREY_LINE, lw=1.0, zorder=1)

# individual modes; only the first carries a legend entry for the family
mode_label = r"modes   $w_n\,\bar\lambda_n^{\,m}$"
for i, ((r, th, w), curve, col) in enumerate(zip(modes, mode_curves, mode_colors)):
    lbl = mode_label if i == 0 else None
    ax.plot(m, curve, color=col, lw=1.4, alpha=0.7, zorder=2, label=lbl)

# the kernel: orange sum
ax.plot(m, K, color=S.ORANGE, lw=2.8, zorder=5, solid_capstyle="round",
        label=r"kernel   $\bar K_m=\sum_n w_n\,\bar\lambda_n^{\,m}$")
ax.plot(m, K, "o", color=S.ORANGE, ms=4.6, mec=S.WHITE, mew=1.0, zorder=6)

ax.set_xlabel(r"lag $m$")
ax.set_ylabel(r"kernel weight   $\bar K_m$")
ax.set_xlim(-0.6, 32.6)
ax.set_ylim(-0.55, 1.7)
ax.set_xticks(range(0, 33, 8))
ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
S.clean_axes(ax)

leg = ax.legend(loc="upper right", frameon=True, fontsize=10.5,
                facecolor=S.WHITE, edgecolor=S.GREY_LINE, framealpha=0.96,
                borderpad=0.7, handlelength=1.7, handletextpad=0.6)
for txt in leg.get_texts():
    txt.set_color(S.INK)

S.save(fig, "fig-11-1-modes")
