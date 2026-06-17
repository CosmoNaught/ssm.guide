"""Figure 6.1: the L roots of unity and inverse-FFT kernel recovery. For chapters/02-foundations/05-transfer-functions.qmd, section 6.5.

(a) the L-th roots of unity omega_j = e^{-2 pi i j / L} on the unit circle, the DFT evaluation points; (b) the length-L kernel Kbar_m recovered as the inverse FFT of G sampled there. Worked example is a strongly damped 2D system discretised by zero-order hold, so recovery is exact up to negligible aliasing.
"""
import os
import sys

import numpy as np
from scipy.linalg import expm
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssmstyle as S  # noqa: E402

L = 8          # small L so every root and tap stays legible
J_MARK = 1     # highlight omega_1 as the orange worked example

# worked-example system -> length-L kernel
A = np.array([[-0.8, 1.0], [-1.0, -0.8]], dtype=complex)   # strongly damped rotation
B = np.array([1.0, 0.5], dtype=complex)[:, None]
C = np.array([1.0, -1.0], dtype=complex)
dt = 0.5
N = A.shape[0]
M = np.zeros((N + 1, N + 1), dtype=complex)
M[:N, :N] = A
M[:N, N:] = B
E = expm(M * dt)
Abar, Bbar = E[:N, :N], E[:N, N:].reshape(-1)

# sample G on the L roots of unity, recover the kernel by inverse FFT
j = np.arange(L)
omega = np.exp(-2j * np.pi * j / L)               # omega_j = e^{-2 pi i j / L}
I = np.eye(N, dtype=complex)
Gsamp = np.array([C @ (np.linalg.inv(I - z * Abar) @ Bbar) for z in omega])
K = np.fft.ifft(Gsamp).real                       # recovered length-L kernel K_m

# root coordinates (clockwise from +1, matching the e^{-...} convention)
ang = -2 * np.pi * j / L
xr, yr = np.cos(ang), np.sin(ang)

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(9.8, 4.7),
    gridspec_kw={"width_ratios": [1.06, 1.0], "wspace": 0.28},
)

# left panel: the L-th roots of unity on the unit circle
for spine in axL.spines.values():
    spine.set_visible(False)

# Re / Im axes through the origin: light, just past the circle, small heads
AX = 1.30
for (x0, y0, x1, y1) in [(-AX, 0, AX, 0), (0, -AX, 0, AX)]:
    axL.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                  mutation_scale=9, color=S.GREY_LINE, lw=1.0, zorder=1))
axL.text(AX + 0.05, 0.12, r"$\mathrm{Re}$", color=S.GREY,
         ha="left", va="center", fontsize=10)
axL.text(0.10, AX + 0.07, r"$\mathrm{Im}$", color=S.GREY,
         ha="left", va="bottom", fontsize=10)

# the unit circle
t = np.linspace(0, 2 * np.pi, 400)
axL.plot(np.cos(t), np.sin(t), color=S.BLUE, lw=2.0, zorder=2)

# radial spokes to each root
for x, y in zip(xr, yr):
    axL.plot([0, x], [0, y], color=S.GREY_LINE, lw=0.9, zorder=2)

# the roots: blue dots, one orange accent (the worked example omega_1)
for k in range(L):
    hi = (k == J_MARK)
    axL.plot([xr[k]], [yr[k]], "o",
             color=S.ORANGE if hi else S.BLUE_DARK,
             ms=11 if hi else 8.5, mec=S.WHITE, mew=1.4,
             zorder=6 if hi else 5)

# label every root omega_0 .. omega_{L-1}, pushed radially outward
for k in range(L):
    hi = (k == J_MARK)
    lx, ly = 1.21 * xr[k], 1.21 * yr[k]
    ha = "center" if abs(lx) < 0.25 else ("left" if lx > 0 else "right")
    va = "center" if abs(ly) < 0.25 else ("bottom" if ly > 0 else "top")
    axL.text(lx, ly, rf"$\omega_{{{k}}}$",
             color=S.ORANGE_DARK if hi else S.INK,
             ha=ha, va=va, fontsize=12 if hi else 10.5,
             fontweight="bold" if hi else "normal", zorder=7)

# the equal angular spacing, shown once as a small arc + radius on omega_1
axL.plot([0, xr[J_MARK]], [0, yr[J_MARK]], color=S.ORANGE, lw=1.5, zorder=4)
arc_t = np.linspace(0, ang[J_MARK], 60)
ar = 0.34
axL.plot(ar * np.cos(arc_t), ar * np.sin(arc_t), color=S.ORANGE, lw=1.4, zorder=4)
axL.text(0.50, -0.12, r"$\dfrac{2\pi}{L}$", color=S.ORANGE_DARK,
         ha="left", va="top", fontsize=11, zorder=7)

# the defining map, lower-left
axL.text(-1.46, -1.30, r"$\omega_j=e^{-2\pi i\,j/L}$",
         color=S.INK, ha="left", va="bottom", fontsize=12.5, zorder=7)

axL.set_xlim(-1.62, 1.50)
axL.set_ylim(-1.62, 1.46)
axL.set_aspect("equal")
axL.set_xticks([]); axL.set_yticks([])
axL.text(0.01, 0.99, "(a)", transform=axL.transAxes, ha="left", va="top",
         fontsize=13, fontweight="bold", color=S.INK, zorder=8)

# right panel: the recovered length-L kernel (stem plot)
m = np.arange(L)
axR.axhline(0, color=S.GREY_LINE, lw=1.0, zorder=1)
# stems
for mi, ki in zip(m, K):
    axR.plot([mi, mi], [0, ki], color=S.BLUE, lw=2.0,
             solid_capstyle="butt", zorder=2)
# heads: uniform blue, since the inverse FFT mixes all L samples into every tap
for mi, ki in zip(m, K):
    axR.plot([mi], [ki], "o", color=S.BLUE_DARK, ms=7, mec=S.WHITE, mew=1.2,
             zorder=3)

S.clean_axes(axR)
axR.set_xlim(-0.7, L - 0.3)
ypad = 0.10 * (K.max() - K.min())
axR.set_ylim(K.min() - 1.4 * ypad, K.max() + 2.2 * ypad)
axR.set_xticks(list(m))
axR.set_xlabel(r"$m$")
axR.set_ylabel(r"$\bar K_m$")
# x ticks stay at the figure floor, clear of negative stems; zero baseline is the grey rule above
axR.tick_params(axis="x", direction="out", pad=3)
axR.text(0.0, 1.0, "(b)", transform=axR.transAxes, ha="left", va="top",
         fontsize=13, fontweight="bold", color=S.INK, zorder=8)

fig.subplots_adjust(left=0.045, right=0.975, top=0.96, bottom=0.11)
S.save(fig, "fig-6-1-roots-of-unity")
