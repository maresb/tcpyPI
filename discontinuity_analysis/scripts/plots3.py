"""Four-archetype figure: the three CAPE conventions on concrete profiles."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
FIG = "/home/mares/repos/tcpyPI/discontinuity_analysis/figures"

BLUE = "#2a78d6"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"

plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "lines.linewidth": 1.4, "figure.dpi": 150, "savefig.bbox": "tight",
})

t = np.linspace(0, 1.15, 4601)
dt = t[1] - t[0]


def G(c, s):
    return np.exp(-((t - c) / s) ** 2)


def conventions(b):
    """Return dict with values and terminal locations for E_top/E_max/E_reach."""
    W = np.concatenate([[0.0], np.cumsum(0.5 * (b[1:] + b[:-1]) * dt)])
    sgn = np.sign(b)
    cross = np.where(np.diff(sgn) != 0)[0]
    down = [q for q in cross if b[q] > 0 >= b[q + 1]]
    if b[-1] > 0:
        z_top, W_top = t[-1], W[-1]
    elif down:
        z_top, W_top = t[down[-1]], W[down[-1]]
    else:
        z_top, W_top = np.nan, 0.0
    E_top = max(W_top, 0.0)
    kmax = int(np.argmax(W))
    E_max, z_max = max(W[kmax], 0.0), t[kmax]
    pos = W > 1e-12
    if not pos.any():
        E_reach, z_reach = 0.0, np.nan
    else:
        k0 = int(np.argmax(pos))
        k1 = k0
        while k1 < len(W) - 1 and W[k1 + 1] > 0:
            k1 += 1
        kr = k0 + int(np.argmax(W[k0:k1 + 1]))
        E_reach, z_reach = float(W[kr]), t[kr]
    return dict(W=W, E_top=E_top, z_top=z_top, W_top=W_top,
                E_max=E_max, z_max=z_max, E_reach=E_reach, z_reach=z_reach)


CASES = [
    ("(a) single positive region:\nall three agree",
     1.0 * G(0.35, 0.18) - 0.7 * G(0.85, 0.12)),
    ("(b) dominated upper cell, no barrier:\n$E_{\\rm top}$ undercounts",
     1.0 * G(0.25, 0.10) - 0.55 * G(0.5, 0.09) + 0.35 * G(0.72, 0.08)
     - 0.8 * G(1.0, 0.10)),
    ("(c) marginal bump behind a barrier\n(the issue-77 shape): $E_{\\rm top}$ clamps to 0",
     1.0 * G(0.25, 0.10) - 1.1 * G(0.5, 0.10) + 0.12 * G(0.78, 0.05)
     - 0.5 * G(1.0, 0.08)),
    ("(d) dominant upper cell behind a barrier:\n$E_{\\rm reach}$ stalls below the others",
     0.6 * G(0.22, 0.09) - 1.0 * G(0.45, 0.10) + 1.3 * G(0.75, 0.12)
     - 0.9 * G(1.05, 0.08)),
]

fig, axs = plt.subplots(2, 2, figsize=(8.8, 5.8))
for ax, (title, b) in zip(axs.flat, CASES):
    c = conventions(b)
    W = c["W"]
    ylo, yhi = W.min() - 0.03, W.max() + 0.055
    ax.fill_between(t, ylo, yhi, where=b > 0, color=BLUE, alpha=0.07, lw=0)
    ax.plot(t, W, color=BLUE, lw=1.6)
    ax.axhline(0, color=INK2, lw=0.7)
    # markers: E_max dot, E_reach open square, E_top X (at its terminal point)
    ax.plot(c["z_max"], c["E_max"], "o", color=AQUA, ms=8, zorder=6,
            label=rf"$E_{{\max}}={c['E_max']:.3f}$")
    ax.plot(c["z_reach"], c["E_reach"], "s", mfc="none", mec=VIOLET, mew=1.6,
            ms=12, zorder=7, label=rf"$E_{{\rm reach}}={c['E_reach']:.3f}$")
    lbl = rf"$E_{{\rm top}}={c['E_top']:.3f}$"
    if c["E_top"] != c["W_top"]:
        lbl = rf"$E_{{\rm top}}=0$ (clamped from ${c['W_top']:.3f}$)"
    ax.plot(c["z_top"], c["W_top"], "X", color=RED, ms=9, zorder=8, label=lbl)
    ax.set_ylim(ylo, yhi)
    ax.set_title(title)
    ax.legend(loc="lower left", handletextpad=0.4)
    ax.set_xlabel(r"trial outflow height $t$")
    ax.set_ylabel(r"$W(t)$")

fig.tight_layout()
fig.savefig(f"{FIG}/fig_conventions.pdf")
fig.savefig(f"{FIG}/fig_conventions.png", dpi=100)

for title, b in CASES:
    c = conventions(b)
    print(title.split(':')[0].replace('\n', ' '),
          f"E_top={c['E_top']:.4f} (W_top={c['W_top']:.4f}) "
          f"E_max={c['E_max']:.4f} E_reach={c['E_reach']:.4f}")
