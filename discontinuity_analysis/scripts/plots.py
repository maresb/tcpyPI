"""All figures for the discontinuity analysis document."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
FIG = "/home/mares/repos/tcpyPI/discontinuity_analysis/figures"
os.makedirs(FIG, exist_ok=True)

# ---- palette (dataviz reference, light mode) ----
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

D = np.load(os.path.join(SCRATCH, "docdata.npz"))
FX = np.load(os.path.join(SCRATCH, "fixdata.npz"))
H = np.load(os.path.join(SCRATCH, "humpdata.npz"))
HW = {t: np.load(os.path.join(SCRATCH, f"humpW_{t}.npz")) for t in ("lo", "at", "hi")}
H2 = np.load(os.path.join(SCRATCH, "healthy2.npz"))
A = np.load(os.path.join(SCRATCH, "branch_pi.npz"))
B = np.load(os.path.join(SCRATCH, "branch_fix_pi.npz"))


def cobweb(ax, xs, color=INK2, lw=0.8, zorder=5, arrow_last=False):
    """Draw cobweb path for iterates xs on an (x, g(x)) axes."""
    for k in range(len(xs) - 1):
        x0, x1 = xs[k], xs[k + 1]
        ax.plot([x0, x0], [x0, x1], color=color, lw=lw, zorder=zorder, alpha=0.85)
        ax.plot([x0, x1], [x1, x1], color=color, lw=lw, zorder=zorder, alpha=0.85)


def plot_map_with_jumps(ax, pm, g, jump_thresh=0.1):
    """Plot g with NaN breaks at jumps so no vertical connector is drawn."""
    gg = g.copy()
    br = np.where(np.abs(np.diff(g)) > jump_thresh)[0]
    gplot = gg.astype(float)
    pmp = pm.astype(float)
    for q in br:
        gplot = np.insert(gplot, q + 1, np.nan)
        pmp = np.insert(pmp, q + 1, np.nan)
    ax.plot(pmp, gplot, color=BLUE, lw=1.4, zorder=4)
    return br


# =====================================================================
# FIG toy map: abstract gap map with superattracting 2-cycle
# =====================================================================
fig, ax = plt.subplots(figsize=(3.4, 3.2))
xstar = 0.5
s = 0.25
a1, delta = 0.62, 0.28
xl = np.linspace(0.05, xstar, 200)
xr = np.linspace(xstar, 0.95, 200)
gl = a1 - s * (xl - xstar)
gr = a1 - delta - s * (xr - xstar)
ax.plot(xl, gl, color=BLUE, lw=1.6)
ax.plot(xr, gr, color=BLUE, lw=1.6)
ax.plot([xstar, xstar], [a1 - delta, a1], color=BLUE, lw=0.8, ls=":", alpha=0.6)
lim = [0.05, 0.95]
ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
# iterate from 0.15
xs = [0.12]
for _ in range(9):
    x = xs[-1]
    xs.append(a1 - s * (x - xstar) if x <= xstar else a1 - delta - s * (x - xstar))
cobweb(ax, xs, color=INK, lw=0.9)
ax.scatter(xs[-2:], xs[-1:] + [xs[-1]], s=0)  # noop keep
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$g(x)$")
ax.set_title("gap map: no fixed point,\nsuperattracting 2-cycle")
ax.annotate("gap straddles\nthe diagonal", xy=(xstar, a1 - delta / 2),
            xytext=(0.6, 0.35), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
fig.savefig(f"{FIG}/fig_toymap.pdf"); fig.savefig(f"{FIG}/fig_toymap.png", dpi=100)
plt.close(fig)

# =====================================================================
# FIG 1: healthy anatomy — b(p) and W(p_t), saturated parcel
# =====================================================================
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
ax = axs[0]
ax.plot(H2["c_b"], H2["c_blevels"], color=BLUE, marker="o", ms=2.5, lw=1.2)
ax.axvline(0, color=INK2, lw=0.7)
ax.set_ylim(1010, 40)
ax.set_xlabel(r"buoyancy $b_j$  [K]")
ax.set_ylabel("pressure [hPa]  (up = higher altitude)")
ax.set_title("(a) buoyancy of the lifted parcel")
ax = axs[1]
ax.plot(H2["c_W_W"], H2["c_W_p"], color=BLUE, lw=1.4)
kmax = int(np.argmax(H2["c_W_W"]))
ax.plot(H2["c_W_W"][kmax], H2["c_W_p"][kmax], "o", color=AQUA, ms=6, zorder=6)
ax.annotate("LNB / argmax\n(CAPE = max $W$)",
            xy=(H2["c_W_W"][kmax], H2["c_W_p"][kmax]), xytext=(0.35, 0.45),
            textcoords="axes fraction", fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.axvline(0, color=INK2, lw=0.7)
ax.set_ylim(1010, 40)
ax.set_xlabel(r"$W(p_t)$  [J kg$^{-1}$]")
ax.set_title("(b) running work integral (single hump)")
fig.suptitle("Generic column (January, eastern tropical Pacific, 10\u00b0N 108\u00b0W): one positive region, unambiguous LNB",
             fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_healthy_anatomy.pdf"); fig.savefig(f"{FIG}/fig_healthy_anatomy.png", dpi=100)
plt.close(fig)

# =====================================================================
# FIG 2: healthy map — g smooth at the crossing, iteration converges
# =====================================================================
fig, ax = plt.subplots(figsize=(4.6, 3.4))
plot_map_with_jumps(ax, H2["c_pm"], H2["c_g"], jump_thresh=0.15)
lim = [900, 1000]
ax.plot(lim, lim, color=INK2, lw=0.8, ls="--", label="diagonal $g(x)=x$")
cobweb(ax, list(H2["c_iter"]), color=INK, lw=0.9)
xfp = H2["c_iter"][-1]
ax.plot(xfp, xfp, "o", color=AQUA, ms=7, zorder=7)
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xlabel(r"$x=$ trial central pressure $P_M$ [hPa]")
ax.set_ylabel(r"$g(x)$ [hPa]")
ax.set_title("Generic column: contraction to a fixed point")
ax.legend(loc="lower right")
fig.savefig(f"{FIG}/fig_healthy_map.pdf"); fig.savefig(f"{FIG}/fig_healthy_map.png", dpi=100)
plt.close(fig)

# =====================================================================
# FIG 3: PR77 map, full + zoom with 2-cycle
# =====================================================================
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
ax = axs[0]
br = plot_map_with_jumps(ax, D["a_pm"], D["a_g"], jump_thresh=0.3)
lim = [946, 958]
ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
cobweb(ax, list(D["a_iter"][:14]), color=INK, lw=0.8)
ax.set_xlim(*lim)
ax.set_ylim(949.8, 953.2)
ax.set_xlabel(r"$x = P_M$ [hPa]")
ax.set_ylabel(r"$g(x)$ [hPa]")
ax.set_title("(a) issue-77 column: the map $g$")
zx0, zx1 = 950.45, 951.5
ax.add_patch(plt.Rectangle((zx0, zx0), zx1 - zx0, zx1 - zx0, fill=False,
                           edgecolor=RED, lw=0.9, zorder=7))
ax = axs[1]
m = (D["a_pm"] >= zx0) & (D["a_pm"] <= zx1)
plot_map_with_jumps(ax, D["a_pm"][m], D["a_g"][m], jump_thresh=0.3)
ax.plot([zx0, zx1], [zx0, zx1], color=INK2, lw=0.8, ls="--")
cyc = [950.6533454438, 951.2790079839]
cobweb(ax, list(D["a_iter"][6:]) + cyc * 3, color=INK, lw=0.9)
ax.plot(cyc, cyc[::-1], "s", color=RED, ms=5, zorder=8, label="2-cycle $\\{A,B\\}$")
jump_x = 950.829
ax.annotate("gap: $g(x)-x$ changes sign\nhere, but $g$ has no value\non the diagonal",
            xy=(jump_x, jump_x + 0.02), xytext=(950.95, 950.62), fontsize=8,
            color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_xlim(zx0, zx1)
ax.set_ylim(zx0, zx1)
ax.set_xlabel(r"$x = P_M$ [hPa]")
ax.set_title("(b) zoom: the diagonal crosses inside the gap")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_pr77_map.pdf"); fig.savefig(f"{FIG}/fig_pr77_map.png", dpi=100)
plt.close(fig)

# =====================================================================
# FIG 4: PR77 mechanism — buoyancy zoom + W curves at both phases
# =====================================================================
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.4))
ax = axs[0]
lv = D["a_blevels"]
ax.plot(D["a_b_hi"], lv, color=YELLOW, marker="o", ms=3, lw=1.2,
        label=r"$x=B=951.279$ (bump $<0$)")
ax.plot(D["a_b_lo"], lv, color=BLUE, marker="o", ms=3, lw=1.2,
        label=r"$x=A=950.653$ (bump $>0$)")
ax.axvline(0, color=INK2, lw=0.7)
ax.set_ylim(1010, 240)
ax.set_xlim(-4.0, 2.6)
ax.set_xlabel(r"buoyancy $b_j$ [K]")
ax.set_ylabel("pressure [hPa]")
ax.set_title("(a) eyewall-parcel buoyancy at the two phases")
ax.legend(loc="lower left", fontsize=7.5)
axz = ax.inset_axes([0.60, 0.52, 0.37, 0.40])
axz.plot(D["a_b_hi"], lv, color=YELLOW, marker="o", ms=3, lw=1.0)
axz.plot(D["a_b_lo"], lv, color=BLUE, marker="o", ms=3, lw=1.0)
axz.axvline(0, color=INK2, lw=0.6)
axz.set_xlim(-0.08, 0.05)
axz.set_ylim(410, 290)
axz.set_title("350 hPa bump: $\\pm$14 mK", fontsize=7)
axz.tick_params(labelsize=6)
axz.grid(True, color=GRID, lw=0.5)

ax = axs[1]
ax.plot(D["a_W_hi_W"], D["a_W_hi_p"], color=YELLOW, lw=1.4, label=r"$W$ at $x=B$")
ax.plot(D["a_W_lo_W"], D["a_W_lo_p"], color=BLUE, lw=1.4, label=r"$W$ at $x=A$")
ax.axvline(0, color=INK2, lw=0.7)
# selections
khi = int(np.argmax(D["a_W_hi_W"]))
ax.plot(D["a_W_hi_W"][khi], D["a_W_hi_p"][khi], "o", color=AQUA, ms=6, zorder=6)
# pcmin selection at A: topmost crossing ~ level 18/19 where W ~ -61
plo = D["a_W_lo_p"]
wlo = D["a_W_lo_W"]
ksel = np.argmin(np.abs(plo - 349.0))
ax.plot(wlo[ksel], plo[ksel], "X", color=RED, ms=8, zorder=7)
ax.annotate("code's selection at $x=A$:\n$W=-61$ J/kg $\\to$ clamped to 0",
            xy=(wlo[ksel], plo[ksel]), xytext=(0.25, 0.30),
            textcoords="axes fraction",
            fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.annotate("argmax (both phases):\n$W\\approx 222$ J/kg at 745 hPa",
            xy=(np.max(D["a_W_hi_W"]), D["a_W_hi_p"][int(np.argmax(D['a_W_hi_W']))]),
            xytext=(0.05, 0.72), textcoords="axes fraction",
            fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_ylim(1010, 240)
ax.set_xlim(-350, 330)
ax.set_xlabel(r"$W(p_t)$ [J kg$^{-1}$]")
ax.set_title("(b) work integral and the two selections")
ax.legend(loc="lower left", fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_pr77_mechanism.pdf"); fig.savefig(f"{FIG}/fig_pr77_mechanism.png", dpi=100)
plt.close(fig)

# =====================================================================
# FIG 5: CAPEM(PM) staircase + zoom on g
# =====================================================================
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.1))
ax = axs[0]
cm = D["a_capem"].copy()
pm = D["a_pm"]
br = np.where(np.abs(np.diff(cm)) > 30)[0]
cmp_, pmp = cm.astype(float), pm.astype(float)
for q in br[::-1]:
    cmp_ = np.insert(cmp_, q + 1, np.nan)
    pmp = np.insert(pmp, q + 1, np.nan)
ax.plot(pmp, cmp_, color=BLUE, lw=1.4)
ax.set_xlabel(r"$x = P_M$ [hPa]")
ax.set_ylabel(r"CAPE$_M(x)$ [J kg$^{-1}$]")
ax.set_title("(a) the discontinuous coefficient: CAPE$_M$")
ax.annotate("222 J/kg jump from a\n$10^{-4}$ K sign flicker",
            xy=(950.83, 100), xytext=(952.5, 60), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_xlim(946, 958)

ax = axs[1]
mm = (pm > 950.5) & (pm < 951.2)
ax.plot(pm[mm], D["a_g"][mm] - pm[mm], color=BLUE, lw=1.4)
ax.axhline(0, color=INK2, lw=0.8, ls="--")
ax.set_xlabel(r"$x = P_M$ [hPa]")
ax.set_ylabel(r"$F(x) = g(x) - x$ [hPa]")
ax.set_title("(b) residual changes sign only across the gap")
ax.annotate("$F>0$", xy=(950.62, 0.45), fontsize=9, color=INK2)
ax.annotate("$F<0$", xy=(951.0, -0.28), fontsize=9, color=INK2)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_pr77_capem.pdf"); fig.savefig(f"{FIG}/fig_pr77_capem.png", dpi=100)
plt.close(fig)

# =====================================================================
# FIG 6: 1995 case map
# =====================================================================
fig, ax = plt.subplots(figsize=(4.6, 3.4))
plot_map_with_jumps(ax, D["b_pm"], D["b_g"], jump_thresh=0.3)
lim = [955, 968]
ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
cobweb(ax, list(D["b_iter"][4:20]), color=INK, lw=0.8)
ax.set_xlim(959, 964)
ax.set_ylim(959, 964)
ax.set_xlabel(r"$x = P_M$ [hPa]")
ax.set_ylabel(r"$g(x)$ [hPa]")
ax.set_title("Second instance (real 1995 sounding):\nsame gap-straddling geometry")
fig.savefig(f"{FIG}/fig_1995_map.pdf"); fig.savefig(f"{FIG}/fig_1995_map.png", dpi=100)
plt.close(fig)

# =====================================================================
# FIG 7: survey + SST sweep
# =====================================================================
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.1))
ax = axs[0]
mj = D["s_maxjump"]
mj = mj[np.isfinite(mj)]
ax.hist(np.log10(mj), bins=28, color=BLUE, edgecolor="white", lw=0.5)
ax.axvline(np.log10(0.5), color=RED, lw=1.2, ls="--")
ax.annotate("convergence\ntolerance 0.5 hPa", xy=(np.log10(0.5), ax.get_ylim()[1] * 0.75),
            xytext=(0.62, 0.8), textcoords="axes fraction", fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
n_all = int(D["s_n"])
ax.set_xlabel(r"$\log_{10}$ largest jump of $g$ in column [hPa]")
ax.set_ylabel("columns")
ax.set_title(f"(a) jumps are generic: {len(mj)}/{n_all} sampled\ncolumns have $\\geq$1 jump; 0 fail")
ax = axs[1]
st = D["d_status"]
pmfp = D["d_pm"]
dss = D["d_dss"]
ax.plot(dss, pmfp, color=BLUE, lw=1.0)
bad = st == 2
ax.axvspan(dss[bad].min() - 0.005, dss[bad].max() + 0.005, color=RED,
           alpha=0.75, lw=0)
ax.annotate("the Francine (issue-77) member ($\\Delta=0$) sits\nin a $\\sim$0.04 K-wide no-fixed-point band",
            xy=(0.0, 950.5), xytext=(-2.9, 925), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
ax.set_xlabel(r"further SST perturbation $\Delta$ [K]")
ax.set_ylabel(r"fixed point $x^\ast$ [hPa]")
ax.set_title("(b) failure band under SST perturbation\n(red: no fixed point, 4/601 offsets)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_survey.pdf"); fig.savefig(f"{FIG}/fig_survey.png", dpi=100)
plt.close(fig)

# =====================================================================
# FIG 8: genuine argmax bifurcation (Berge picture), real column
# =====================================================================
fig = plt.figure(figsize=(8.6, 3.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], hspace=0.12)
ax = fig.add_subplot(gs[:, 0])
for tag, cc in [("lo", BLUE), ("at", AQUA), ("hi", YELLOW)]:
    d = HW[tag]
    ax.plot(d["W"], d["p"], color=cc, lw=1.3, label=f"$f={float(d['f']):.3f}$")
ax.axvline(0, color=INK2, lw=0.7)
ax.set_ylim(1010, 130)
ax.set_xlim(-260, 320)
ax.set_xlabel(r"$W(p_t)$ [J kg$^{-1}$]")
ax.set_ylabel("pressure [hPa]")
ax.set_title("(a) two competing humps as moisture scales")
ax.legend(loc="lower left", title="moisture scale $f$", fontsize=7.5,
          title_fontsize=7.5)
fs = H["fs"]
axt = fig.add_subplot(gs[0, 1])
axt.plot(fs, H["cape_mw"], color=AQUA, lw=1.8)
axt.set_ylabel(r"CAPE [J kg$^{-1}$]")
axt.set_title("(b) value continuous ...")
axt.tick_params(labelbottom=False)
axb = fig.add_subplot(gs[1, 1], sharex=axt)
axb.plot(fs, H["lnb_mw"], color=VIOLET, lw=1.3)
axb.invert_yaxis()
axb.set_ylabel("argmax [hPa]")
axb.set_xlabel(r"boundary-layer moisture scale $f$")
axb.set_title("(c) ... while the maximizer jumps", pad=2)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_berge.pdf"); fig.savefig(f"{FIG}/fig_berge.png", dpi=100)
plt.close(fig)

# =====================================================================
# FIG 9: the repair — g_fix continuous + population effect
# =====================================================================
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.3))
ax = axs[0]
m = (D["a_pm"] > 949.5) & (D["a_pm"] < 953)
plot_map_with_jumps(ax, D["a_pm"][m], D["a_g"][m], jump_thresh=0.3)
mf = (FX["pm"] > 949.5) & (FX["pm"] < 953)
ax.plot(FX["pm"][mf], FX["g"][mf], color=AQUA, lw=1.7, label=r"$g$ with max-$W$ CAPE")
ax.plot([949.5, 953], [949.5, 953], color=INK2, lw=0.8, ls="--")
cobweb(ax, list(FX["iters"][2:]), color=INK, lw=0.9)
xf = FX["iters"][-1]
ax.plot(xf, xf, "o", color=AQUA, ms=6, zorder=8)
ax.plot([], [], color=BLUE, lw=1.4, label=r"$g$ with pcmin CAPE")
ax.set_xlim(949.9, 952.4)
ax.set_ylim(949.9, 952.4)
ax.set_xlabel(r"$x = P_M$ [hPa]")
ax.set_ylabel(r"$g(x)$ [hPa]")
ax.set_title("(a) continuous CAPE closes the gap;\nfixed point exists, 5 iterations")
ax.legend(loc="upper right", fontsize=7.5)

ax = axs[1]
both1 = (A["IFL"] == 1) & (B["IFL"] == 1)
d = np.abs(A["VMAX"] - B["VMAX"])[both1]
d = np.sort(d[np.isfinite(d)])
dnz = np.clip(d, 1e-12, None)
ecdf = np.arange(1, len(d) + 1) / len(d)
ax.semilogx(dnz, 100 * ecdf, color=BLUE, lw=1.6)
ax.axvline(0.1, color=RED, lw=1.0, ls="--")
ax.annotate("98.1% below 0.1 m/s\n(loop-tolerance noise)", xy=(0.1, 55),
            xytext=(1.5e-8, 40), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_xlabel(r"$|\Delta V_{\max}|$ pcmin vs max-$W$  [m s$^{-1}$]")
ax.set_ylabel("cumulative share of columns [%]")
ax.set_title("(b) population effect over 7093 columns:\nflags unchanged, tail = fragile columns")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_fix.pdf"); fig.savefig(f"{FIG}/fig_fix.png", dpi=100)
plt.close(fig)

print("all figures written to", FIG)
