"""Figure 11.2: the S4D family of diagonal initialisations, for
chapters/03-s4/10-diagonal-state-spaces (after the S4D-Lin/S4D-Inv formulas).

Reference: S4D (Gu et al. 2022) Figure 4. Each scheme pins Re lambda to -1/2 and
differs only in Im lambda; we plot |Im| sorted high to low as a spacing law.
S4D-Quad uses (2n+1)^2 rescaled to the LegS range, since its bare scale is
normalisation-dependent.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# make the book package importable: <repo>/src
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))

import ssmstyle as S  # noqa: E402
from ssm_book.numpy_ref.diagonal import s4d_lin_init, s4d_inv_init  # noqa: E402
from ssm_book.numpy_ref.structured_matrices import hippo_nplr  # noqa: E402

S.use_style()

# state size; large enough that the spacing laws read as smooth curves
M = 64
n = np.arange(M)


def desc(a):
    """Imaginary magnitudes sorted high -> low (the paper's x-ordering)."""
    return np.sort(np.abs(np.asarray(a)))[::-1]


# normal HiPPO eigenvalues come in conjugate pairs, so |Im| lists each frequency
# twice; take one per pair (every other entry) for the distinct frequencies
legs = desc(hippo_nplr(M)[0].imag)[0::2]      # true HiPPO-LegS spectrum
lin = desc(s4d_lin_init(M).imag)              # linear law  pi*n
inv = desc(s4d_inv_init(M).imag)              # inverse law (M/pi)(M/(2n+1)-1)
inv2 = desc((M / np.pi) * (M / (n + 1) - 1))  # milder inverse (paper eq. 12 form)

# S4D-Quad power-2 law, rescaled to the LegS top since its bare scale is
# normalisation-dependent; only the shape (crowding at high frequency) matters
quad = (2 * n + 1.0) ** 2
quad = desc(quad / quad.max() * legs.max())

# (name, series, colour) in the reference's legend order.
series = [
    ("S4D-LegS", legs, S.BLUE),
    ("S4D-Inv", inv, S.ORANGE),
    ("S4D-Lin", lin, S.GREEN),
    ("S4D-Quad", quad, S.ROSE),
    ("S4D-Inv2", inv2, S.PLUM),
]

fig, ax = S.figure(7.4, 4.7)

# faint grid to read the widely differing frequency ranges off the axis
ax.grid(True, axis="both", color=S.GREY_LINE, lw=0.6, alpha=0.55, zorder=0)

for name, y, colour in series:
    # stretch each series across [0, M-1]; LegS has half as many distinct points
    x = np.linspace(0, M - 1, len(y))
    ax.plot(x, y, color=colour, lw=2.0, alpha=0.95, zorder=4, label=name)

ax.set_xlim(0, M - 1)
ax.set_ylim(0, 1.04 * max(legs.max(), quad.max()))
ax.set_xlabel(r"$n$-th eigenvalue")
ax.set_ylabel(r"$\mathrm{Im}\,\lambda$   (mode frequency)")

S.clean_axes(ax)

leg = ax.legend(loc="upper right", frameon=True, fontsize=11,
                facecolor=S.WHITE, edgecolor=S.GREY_LINE, framealpha=0.96,
                borderpad=0.6, handletextpad=0.6, labelspacing=0.4)
leg.set_zorder(10)

S.save(fig, "fig-11-2-s4d-init")
