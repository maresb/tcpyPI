"""Redesigned toy figure + discretization figure."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
FIG = "/home/mares/repos/tcpyPI/discontinuity_analysis/figures"

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
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
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.4, "figure.dpi": 150, "savefig.bbox": "tight",
})

tgrid = np.linspace(0, 1.1, 4401)
dt = tgrid[1] - tgrid[0]


def G(c, s):
    return np.exp(-((tgrid - c) / s) ** 2)


def b_toy(eps, delta, h2=1.0):
    return 0.6 * G(0.22, 0.10) - delta * G(0.5, 0.12) + eps * G(0.8, 0.06) \
        + (h2 - 1.0) * 0.0


def b_two(delta, a2=1.0):
    return 0.6 * G(0.22, 0.10) - delta * G(0.5, 0.12) + a2 * G(0.8, 0.10)


def conv(b):
    """(E_top, E_max, E_reach, t_argmax) for buoyancy array b on tgrid."""
    W = np.concatenate([[0.0], np.cumsum(0.5 * (b[1:] + b[:-1]) * dt)])
    sgn = np.sign(b)
    cross = np.where(np.diff(sgn) != 0)[0]
    down = [q for q in cross if b[q] > 0 >= b[q + 1]]
    # pcmin convention: terminal point = crossing above the TOPMOST positive
    # point; if b > 0 at the domain top, integrate to the top (INB = last level)
    if b[-1] > 0:
        E_top = max(W[-1], 0.0)
    else:
        E_top = max(W[down[-1]], 0.0) if down else 0.0
    kmax = int(np.argmax(W))
    E_max = max(W[kmax], 0.0)
    pos = W > 1e-12
    if not pos.any():
        E_reach = 0.0
    else:
        k0 = int(np.argmax(pos))
        k1 = k0
        while k1 < len(W) - 1 and W[k1 + 1] > 0:
            k1 += 1
        E_reach = float(W[k0:k1 + 1].max())
    return E_top, E_max, E_reach, tgrid[kmax]


# ---------------- fig_toy: 2x2, two sweeps ----------------
fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.2))

# (a) epsilon family, delta=1.1 fixed
ax = axs[0, 0]
for eps, cc in [(0.05, BLUE), (0.6, YELLOW), (1.3, RED)]:
    ax.plot(tgrid, 0.6 * G(0.22, 0.10) - 1.1 * G(0.5, 0.12) + eps * G(0.8, 0.06),
            color=cc, label=rf"$\varepsilon={eps}$")
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xlabel(r"height $t$")
ax.set_ylabel(r"$b(t)$")
ax.set_title(r"(a) sweep 1: marginal upper bump, amplitude $\varepsilon$")
ax.legend(loc="upper left")

# (b) value functions vs epsilon
ax = axs[0, 1]
epss = np.linspace(-0.1, 1.5, 1601)
V = np.array([conv(0.6 * G(0.22, 0.10) - 1.1 * G(0.5, 0.12) + e * G(0.8, 0.06))
              for e in epss])
ax.plot(epss, V[:, 0], color=RED, lw=1.5, label=r"$E_{\rm top}$ (code's rule)")
ax.plot(epss, V[:, 1], color=AQUA, lw=1.9, label=r"$E_{\max}$")
ax.plot(epss, V[:, 2], color=VIOLET, ls="--", lw=1.3, label=r"$E_{\rm reach}$")
ax.set_xlabel(r"bump amplitude $\varepsilon$")
ax.set_ylabel("available energy")
ax.set_title(r"(b) $E_{\rm top}$ jumps at $\varepsilon\!\approx\!0^+$ (artificial)")
ax.legend(loc="center right")
jt = np.argmax(np.abs(np.diff(V[:, 0])))
print(f"toy sweep1: E_top jump at eps={epss[jt]:.4f}, size={V[jt,0]-V[jt+1,0]:+.4f}")
print(f"  E_max kink near eps where argmax jumps: "
      f"{epss[np.argmax(np.abs(np.diff(V[:,3])))]:.3f}")

# (c) delta family
ax = axs[1, 0]
for delta, cc in [(0.3, BLUE), (0.75, YELLOW), (1.3, RED)]:
    ax.plot(tgrid, b_two(delta), color=cc, label=rf"$\delta={delta}$")
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xlabel(r"height $t$")
ax.set_ylabel(r"$b(t)$")
ax.set_title(r"(c) sweep 2: two real cells, valley depth $\delta$")
ax.legend(loc="lower left")

# (d) value functions vs delta
ax = axs[1, 1]
deltas = np.linspace(0.2, 1.8, 1601)
V2 = np.array([conv(b_two(d)) for d in deltas])
ax.plot(deltas, V2[:, 0], color=RED, lw=1.5, label=r"$E_{\rm top}$")
ax.plot(deltas, V2[:, 1], color=AQUA, lw=1.9, label=r"$E_{\max}$")
ax.plot(deltas, V2[:, 2], color=VIOLET, ls="--", lw=1.3, label=r"$E_{\rm reach}$")
ax.set_xlabel(r"valley depth $\delta$")
ax.set_ylabel("available energy")
ax.set_title(r"(d) $E_{\rm reach}$ jumps at barrier closure (genuine)")
ax.legend(loc="upper right")
jr = np.argmax(np.abs(np.diff(V2[:, 2])))
print(f"toy sweep2: E_reach jump at delta={deltas[jr]:.4f}, size={V2[jr,2]-V2[jr+1,2]:+.4f}")
jm = np.argmax(np.abs(np.diff(V2[:, 3])))
print(f"  E_max argmax jump at delta={deltas[jm]:.4f} "
      f"(t: {V2[jm,3]:.2f}->{V2[jm+1,3]:.2f}); E_max step {V2[jm,1]-V2[jm+1,1]:+.5f}")

fig.tight_layout()
fig.savefig(f"{FIG}/fig_toy.pdf"); fig.savefig(f"{FIG}/fig_toy.png", dpi=100)
plt.close(fig)

# ---------------- fig_discretization ----------------
R = np.load(os.path.join(SCRATCH, "refinedata.npz"))

fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.2))

# (a) toy: truth vs sampled/PWL (two node phases) vs smooth recovery
ax = axs[0]
tt = np.linspace(0, 1, 2001)


def truth(t):
    return (-0.06 - 0.03 * (t - 0.5)
            + 0.10 * np.exp(-((t - 0.53) / 0.015) ** 2))


ax.plot(tt, truth(tt), color=INK, lw=1.6, label="ground truth")
for off, cc, lbl in [(0.00, BLUE, "nodes miss the bump"),
                     (0.0355, YELLOW, "node lands on the bump")]:
    nodes = np.arange(0.08, 1.0, 0.0833) + off
    nodes = nodes[nodes <= 1]
    ax.plot(nodes, truth(nodes), "o", color=cc, ms=4)
    ax.plot(nodes, truth(nodes), color=cc, lw=1.1, ls="-", alpha=0.9, label=lbl)
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xlabel(r"height $t$")
ax.set_ylabel(r"$b(t)$")
ax.set_title("(a) sampling butchers a narrow feature:\nvisibility depends on node phase")
ax.legend(loc="upper left", fontsize=7.5)

# (b) real column: native vs reconstructions, zoom
ax = axs[1]
ax.plot(R["native_b"], R["native_blev"], "o-", color=INK, ms=3.5, lw=1.3,
        label="native 37-level (PWL)")
ax.plot(R["linear_b"], R["linear_blev"], color=YELLOW, lw=1.1,
        label="linear-in-$\\log p$, 5 hPa")
ax.plot(R["pchip_b"], R["pchip_blev"], color=AQUA, lw=1.4,
        label="monotone cubic, 5 hPa")
ax.axvline(0, color=INK2, lw=0.7)
ax.set_ylim(520, 240)
ax.set_xlim(-0.6, 0.25)
ax.set_xlabel(r"buoyancy $b$ [K]")
ax.set_ylabel("pressure [hPa]")
ax.set_title("(b) issue-77 column, reconstructed:\nthe bump is real but reconstruction-dependent")
ax.legend(loc="lower left", fontsize=7.5)

# (c) residuals F = g - x for native / linear / pchip
ax = axs[2]
Dn = np.load(os.path.join(SCRATCH, "docdata.npz"))
mm = (Dn["a_pm"] > 950.3) & (Dn["a_pm"] < 952.0)


def plot_res(ax, pm, g, color, label, jump_thresh=0.25):
    F = (g - pm).astype(float)
    pmp = pm.astype(float)
    br = np.where(np.abs(np.diff(F)) > jump_thresh)[0]
    for q in br[::-1]:
        F = np.insert(F, q + 1, np.nan)
        pmp = np.insert(pmp, q + 1, np.nan)
    ax.plot(pmp, F, color=color, lw=1.4, label=label)


plot_res(ax, Dn["a_pm"][mm], Dn["a_g"][mm], INK, "native (fails)")
ml = (R["linear_pm"] > 950.3) & (R["linear_pm"] < 952.0)
plot_res(ax, R["linear_pm"][ml], R["linear_g"][ml], YELLOW, "linear 5 hPa (fails)")
mp = (R["pchip_pm"] > 950.3) & (R["pchip_pm"] < 952.0)
plot_res(ax, R["pchip_pm"][mp], R["pchip_g"][mp], AQUA, "cubic 5 hPa (converges)")
ax.axhline(0, color=INK2, lw=0.8, ls="--")
ax.set_xlabel(r"$x = P_M$ [hPa]")
ax.set_ylabel(r"$F(x) = g(x) - x$ [hPa]")
ax.set_title("(c) recovery moves the gap but cannot\nremove it: zero crossing vs gap position")
ax.legend(loc="upper right", fontsize=7.5)

fig.tight_layout()
fig.savefig(f"{FIG}/fig_discretization.pdf"); fig.savefig(f"{FIG}/fig_discretization.png", dpi=100)
plt.close(fig)
print("saved fig_toy.pdf (redesigned), fig_discretization.pdf")
