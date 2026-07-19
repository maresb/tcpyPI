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
YT = [1000, 850, 700, 500, 400, 300, 200, 150, 100, 70, 50, 30, 20]

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
    # x lower limit: most negative interior dip (before the curve's last
    # positive level); the terminal stratospheric plunge never returns to
    # positive and should not set the range. 10% margin.
    lo = 0.0
    hi = 0.0
    for _, _, curves_ in items:
        for c in curves_:
            fin = np.isfinite(c)
            pos = np.where(fin & (c > 0))[0]
            if len(pos):
                lo = min(lo, np.nanmin(c[: pos[-1] + 1]))
            hi = max(hi, np.nanmax(c[fin]) if fin.any() else 0.0)
    hi = min(hi, np.nanpercentile(
        np.concatenate([c.ravel() for _, _, c in items]), 99.95))
    lo *= 1.1
    hi *= 1.05
    pad = 0.0
    for ax, (share, topo, curves) in zip(axs.flat, items):
        for c in curves:
            ax.plot(c, P, color=BLUE, alpha=0.10, lw=0.7, zorder=3)
        ax.plot(np.nanmedian(curves, axis=0), P, color=DBLUE, lw=1.8, zorder=5,
                label="pointwise median")
        ax.axvline(0, color=INK, lw=0.9, zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(1010, 19)
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
             " (x-axis clipped to the interior-dip range)",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(f"{SCRATCH}/curves_parcel_{pname}.png", dpi=140)
    fig.savefig(f"{SCRATCH}/curves_parcel_{pname}.pdf")
    plt.close(fig)
    print(f"saved curves_parcel_{pname}.png/.pdf")
