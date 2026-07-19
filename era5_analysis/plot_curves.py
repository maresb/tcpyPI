"""Spaghetti plots of buoyancy curves conditioned on topology."""

import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("ERA5_SCRATCH", str(_Path(__file__).resolve().parent / "work"))
_SRC = str(_Path(__file__).resolve().parents[1] / "src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



BLUE = "#2a78d6"
DBLUE = "#104281"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"

plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

z = np.load(f"{SCRATCH}/topology_curves.npz")
P = z["P"]
YT = [1000, 850, 700, 500, 400, 300, 200, 150, 100, 70]

groups = {}
for k in z.files:
    if k == "P":
        continue
    pname, topo, share = k.split("|")
    groups.setdefault(pname, []).append((float(share), topo, z[k]))

TITLES = {"A": "Parcel A — environmental (ambient launch)",
          "B": "Parcel B — eyewall (moisture-enriched, at converged $P_M$)",
          "C": "Parcel C — saturated core (at converged $P_M$)"}

for pname, items in groups.items():
    items.sort(key=lambda x: -x[0])
    ncol = min(3, len(items))
    nrow = int(np.ceil(len(items) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 4.1 * nrow),
                            sharey=True, squeeze=False)
    # x-limits from the troposphere below ~125 hPa, where the topology lives
    # (the stratospheric plunge to -40 K would squash the structure; curves
    # simply exit the frame near the top)
    sel = P >= 125
    allb = np.concatenate([c[:, sel].ravel() for _, _, c in items])
    lo, hi = np.nanpercentile(allb, [0.5, 99.5])
    pad = 0.10 * (hi - lo)
    for ax, (share, topo, curves) in zip(axs.flat, items):
        for c in curves:
            ax.plot(c, P, color=BLUE, alpha=0.10, lw=0.7, zorder=3)
        ax.plot(np.nanmedian(curves, axis=0), P, color=DBLUE, lw=1.8, zorder=5,
                label="pointwise median")
        ax.axvline(0, color=INK, lw=0.9, zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(1010, 66)
        ax.set_yticks(YT)
        ax.set_yticklabels([str(t) for t in YT])
        ax.minorticks_off()
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f"{topo}   ({share:.2f}%)")
        ax.set_xlabel("buoyancy [K]")
    for ax in axs[:, 0]:
        ax.set_ylabel("pressure [hPa], log scale (up = higher)")
    for ax in axs.flat[len(items):]:
        ax.set_visible(False)
    axs.flat[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(f"{TITLES[pname]} — {sum(len(c) for _, _, c in items)} sampled buoyancy profiles by topology"
             " (x-axis clipped to tropospheric range)",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(f"{SCRATCH}/curves_parcel_{pname}.png", dpi=140)
    fig.savefig(f"{SCRATCH}/curves_parcel_{pname}.pdf")
    plt.close(fig)
    print(f"saved curves_parcel_{pname}.png/.pdf")
