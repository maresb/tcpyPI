"""Generate the statistical-mechanics figures for thermodynamic_foundations.tex.

Every figure is computed from the formulas derived in the document; the
entropy-inversion figure (fig_sf_inversion) additionally reproduces, line for
line, the algebra of ``utilities.entropy_S`` / ``pi.solve_temperature_from_entropy``
(transcribed below and cross-checked against the docstring examples of the
source, so a numba-free interpreter can run this script).

Usage:  python3 foundations_figs.py        (writes ../figures/fig_sf_*.{pdf,png})
"""

import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260720)

# ----------------------------------------------------------------------------
# Style: match the lmodern/Computer-Modern look of the document; recessive
# axes/grids, thin marks, direct labels.  Categorical palette (fixed order,
# validated): blue, green, magenta, yellow.  Sequential families (increasing n)
# use a single blue hue, light -> dark.
# ----------------------------------------------------------------------------
C_BLUE, C_GREEN, C_MAGENTA, C_YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"
INK, INK2 = "#0b0b0b", "#52514e"
BLUES = ["#b9d1ef", "#7fa9e0", "#3e7fd0", "#134f96"]  # light -> dark

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.edgecolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.labelcolor": INK,
    "grid.color": INK2,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "figure.dpi": 110,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote {name}")


def lbinom(a, b):
    """log C(a, b) via lgamma (floats fine, a,b >= 0)."""
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def binom(a, b):
    return math.exp(lbinom(a, b))


# ============================================================================
# Fig 1 -- classical shell marginal: hard constraint on the whole becomes the
# Boltzmann weight on one mode as n grows.  P(eps) = ((n-1)/E)(1-eps/E)^(n-2),
# E = n kT (so the mean energy per mode is kT throughout).
# ============================================================================
def fig_classical_marginal():
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    eps = np.linspace(0, 5, 400)
    ns = [3, 10, 30, 100]
    for n, c in zip(ns, BLUES):
        E = float(n)  # kT = 1
        P = np.where(eps < E, (n - 1) / E * np.maximum(1 - eps / E, 0) ** (n - 2), 0.0)
        ax.plot(eps, P, color=c, label=f"$n={n}$")
    ax.plot(eps, np.exp(-eps), color=INK, ls="--", lw=1.3,
            label=r"$e^{-\varepsilon/k_BT}$ (limit)")
    ax.annotate(r"$P(\varepsilon)\propto\Omega_{\mathrm{bath}}(E-\varepsilon)"
                r"\propto(1-\varepsilon/E)^{\,n-2}$",
                xy=(1.95, 0.40), fontsize=8.5, color=INK2)
    ax.set_xlabel(r"single-mode energy $\varepsilon/k_BT$")
    ax.set_ylabel(r"$P(\varepsilon)\,k_BT$")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", handlelength=1.6)
    save(fig, "fig_sf_classical_marginal")


# ============================================================================
# Fig 2 -- quantum marginal: exact rho_S(k) of the microcanonical shell for
# finite n vs the geometric (Planck) limit, mu = M/n = 1.
# ============================================================================
def fig_quantum_marginal():
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    ks = np.arange(0, 13)
    for n, c in zip([4, 16, 64], BLUES[1:]):
        M = n  # mu = 1
        dR = lbinom(M + n - 1, n - 1)
        p = np.array([math.exp(lbinom(M - k + n - 2, n - 2) - dR)
                      if k <= M else np.nan for k in ks])
        ax.plot(ks, p, "o-", ms=3.6, color=c, lw=1.2, label=f"$n={n}$")
    geo = 0.5 ** (ks + 1)  # (1-x) x^k, x = mu/(1+mu) = 1/2
    ax.plot(ks, geo, "s", ms=4.5, mfc="none", mec=INK, mew=1.0, ls="--",
            color=INK, lw=1.0, label=r"$(1-x)\,x^{k}$ (limit)")
    ax.set_yscale("log")
    ax.set_xlabel(r"system occupation $k$")
    ax.set_ylabel(r"$\varrho_S(k)$")
    ax.set_xlim(-0.4, 12.4)
    ax.set_ylim(1e-5, 1)
    ax.annotate(r"$\varrho_S(k)=\binom{M-k+n-2}{n-2}/\binom{M+n-1}{n-1}$,"
                r"  $\mu=1$", xy=(0.03, 0.05), xycoords="axes fraction",
                fontsize=8.5, color=INK2)
    ax.legend(loc="upper right", handlelength=1.6)
    save(fig, "fig_sf_quantum_marginal")


# ============================================================================
# Fig 3 -- canonical typicality Monte Carlo.  For a Haar-random state of the
# shell, the vector of diagonal entries rho_S(k) is exactly Dirichlet
# distributed with parameters (N_0, ..., N_M), N_k = Omega(M-k, n-1); sample it
# and compare the trace distance to the PSW bound and to the atypical Fock
# state |M,0,...,0>.
# ============================================================================
def fig_typicality():
    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    ns = np.arange(2, 31)
    means, lo, hi, bound, atyp = [], [], [], [], []
    NSAMP = 400
    for n in ns:
        M = int(n)  # mu = 1
        ldR = lbinom(M + n - 1, n - 1)
        Nk = np.array([math.exp(lbinom(M - k + n - 2, n - 2)) for k in range(M + 1)])
        pk = np.array([math.exp(lbinom(M - k + n - 2, n - 2) - ldR)
                       for k in range(M + 1)])
        samp = RNG.dirichlet(Nk, size=NSAMP)          # exact law of diag(rho_S)
        d1 = np.abs(samp - pk).sum(axis=1)            # ||rho_S - varrho_S||_1
        means.append(d1.mean())
        lo.append(np.percentile(d1, 5))
        hi.append(np.percentile(d1, 95))
        bound.append(math.exp(0.5 * (math.log(M + 1) - ldR)))   # sqrt(d_S/d_R)
        atyp.append(2 * (1 - math.exp(-ldR)))         # 2(1 - 1/d_R)
    ax.fill_between(ns, lo, hi, color=C_BLUE, alpha=0.18, lw=0)
    ax.plot(ns, means, color=C_BLUE, label="Haar mean (5–95% band)")
    ax.plot(ns, bound, color=INK, ls="--", lw=1.2,
            label=r"PSW bound $\sqrt{d_S/d_R}$")
    ax.plot(ns, atyp, color=C_MAGENTA, ls=":", lw=1.6,
            label=r"atypical $|M,0,\dots,0\rangle$")
    ax.annotate("extrapolates to 28 orders\nof magnitude by $n=100$",
                xy=(0.80, 0.72), xycoords="axes fraction", fontsize=8,
                color=INK2, ha="center")
    ax.set_yscale("log")
    ax.set_ylim(top=30)
    ax.set_xlabel(r"number of oscillators $n$  ($\mu=1$)")
    ax.set_ylabel(r"$\|\rho_S-\varrho_S\|_1$")
    ax.set_xlim(2, 30)
    ax.legend(loc="lower left", handlelength=1.7)
    save(fig, "fig_sf_typicality")


# ============================================================================
# Fig 4 -- density of states rho(E) (delta comb, weights Omega(M,3)) vs the
# smoothed envelope rho-bar and the windowed count Omega(E).
# ============================================================================
def fig_comb():
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    n = 3
    Ms = np.arange(0, 15)
    w = np.array([binom(M + n - 1, n - 1) for M in Ms])   # C(M+2,2)
    Ee = np.linspace(0, 14.4, 300)
    env = (Ee + 1) * (Ee + 2) / 2                          # smooth envelope
    win = (8.5, 11.5)
    inwin = (Ms >= win[0]) & (Ms <= win[1])
    count = int(w[inwin].sum())
    ax.axvspan(*win, color=C_YELLOW, alpha=0.22, lw=0)
    ml, sl, bl = ax.stem(Ms, w, basefmt=" ")
    plt.setp(sl, color=C_BLUE, lw=1.3)
    plt.setp(ml, color=C_BLUE, ms=3.5)
    ax.plot(Ee, env, color=INK, ls="--", lw=1.2)
    ax.annotate(r"$\bar\rho(E)$ (smoothed envelope)", xy=(9.5, 87),
                fontsize=8.5, color=INK, ha="right")
    ax.annotate(r"$\rho(E)=\sum_M\Omega(M,n)\,\delta(E-M\hbar\omega)$",
                xy=(0.4, 100), fontsize=8.5, color=C_BLUE)
    ax.annotate(rf"window $\Delta E$:  $\Omega={count}$ states,"
                r"  $S=k_B\ln\Omega$",
                xy=(10.0, 128), fontsize=8.5, color=INK2, ha="center",
                annotation_clip=False)
    ax.set_xlabel(r"$E/\hbar\omega$   ($n=3$ oscillators)")
    ax.set_ylabel(r"level degeneracy $\Omega(M,n)$")
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(0, 125)
    save(fig, "fig_sf_comb")


# ============================================================================
# Fig 5 -- the bridge: (a) Legendre geometry on the exact SHO entropy
# sigma(mu); (b) the saddle integrand rho(E)e^{-beta E} sharpening as 1/sqrt(n).
# ============================================================================
def fig_bridge():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    mu = np.linspace(1e-3, 3.2, 400)
    sig = (1 + mu) * np.log(1 + mu) - mu * np.log(mu)
    a.plot(mu, sig, color=C_BLUE)
    mus, slope = 1.0, math.log(2)               # beta* hbar omega = ln 2 at mu=1
    sigs = 2 * math.log(2)
    tang = sigs + slope * (mu - mus)
    a.plot(mu, tang, color=INK, ls="--", lw=1.1)
    a.plot([mus], [sigs], "o", ms=4, color=C_MAGENTA, zorder=5)
    a.plot([0], [sigs - slope * mus], "o", ms=4, mfc="none", mec=C_GREEN, mew=1.2,
           zorder=5, clip_on=False)
    a.annotate(r"$S(E)/nk_B=\sigma(\mu)$", xy=(2.28, 1.80), color=C_BLUE,
               fontsize=8.5, ha="left", va="top")
    a.annotate(r"slope $=\beta^{*}\hbar\omega=\ln 2$", xy=(1.95, 2.25),
               fontsize=8.5, color=INK2, rotation=20, ha="center")
    a.annotate(r"intercept $=-F/nk_BT=\ln 2$", xy=(0.06, 0.48),
               fontsize=8.5, color=C_GREEN)
    a.set_xlabel(r"$\mu=E/n\hbar\omega$")
    a.set_ylabel(r"entropy per oscillator $/k_B$")
    a.set_xlim(0, 3.2)
    a.set_ylim(0, 3.0)
    a.set_title("(a) Legendre geometry", fontsize=9, color=INK2)

    x = np.linspace(0.4, 1.8, 600)
    for n, c in zip([10, 100, 1000], BLUES[1:]):
        expo = n * ((1 + x) * np.log(1 + x) - x * np.log(x) - x * math.log(2))
        expo -= expo.max()
        b.plot(x, np.exp(expo), color=c, label=f"$n={n}$")
    b.axvline(1.0, color=INK2, lw=0.7, ls=":")
    b.annotate(r"width $\propto 1/\sqrt{n}$", xy=(1.28, 0.55), fontsize=8.5,
               color=INK2)
    b.set_xlabel(r"$E/E^{*}$")
    b.set_ylabel(r"$\rho(E)\,e^{-\beta E}$ (norm.)")
    b.set_title("(b) the saddle sharpens", fontsize=9, color=INK2)
    b.legend(loc="upper left", handlelength=1.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_sf_bridge")


# ============================================================================
# Fig 6 -- non-concave S(E): the Legendre/canonical picture sees only the
# concave hull (Maxwell double tangent); the convex intruder is invisible to Z.
# ============================================================================
def fig_nonconcave():
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    E = np.linspace(0.02, 3.0, 900)
    S = 1.9 * np.sqrt(E) - 0.42 * np.exp(-((E - 1.15) / 0.30) ** 2)
    # upper concave hull via monotone scan of slopes
    pts = list(zip(E, S))
    hull = [pts[0]]
    for p in pts[1:]:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x2) <= (p[1] - y2) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(p)
    hx = np.array([p[0] for p in hull])
    hy = np.array([p[1] for p in hull])
    Hy = np.interp(E, hx, hy)
    gap = Hy - S > 1e-4
    i1, i2 = np.argmax(gap), len(gap) - 1 - np.argmax(gap[::-1])
    E1, E2 = E[i1], E[i2]
    ax.fill_between(E, S, Hy, where=gap, color=C_MAGENTA, alpha=0.20, lw=0)
    ax.plot(E, S, color=C_BLUE, label=r"$S(E)$ (microcanonical)")
    ax.plot(E, Hy, color=INK, ls="--", lw=1.2, label=r"concave hull $=$ what $Z$ sees")
    for Ep in (E1, E2):
        ax.plot([Ep], [np.interp(Ep, E, Hy)], "o", ms=4, color=C_GREEN, zorder=5)
    ax.annotate("", xy=(E2, 0.55), xytext=(E1, 0.55),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
    ax.annotate(r"latent heat $\Delta E$", xy=((E1 + E2) / 2, 0.60),
                ha="center", fontsize=8.5, color=INK2)
    ax.annotate(r"convex intruder" "\n" r"(lost by Legendre)",
                xy=((E1 + E2) / 2, 1.28), ha="center", fontsize=8.5,
                color="#b0447a")
    ax.annotate(r"double tangent, slope $1/T_c$", xy=(2.0, 2.36), fontsize=8.5,
                color=INK2, rotation=13)
    ax.set_xlabel(r"$E$ (arb.)")
    ax.set_ylabel(r"$S(E)$ (arb.)")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 3.4)
    ax.legend(loc="upper left", handlelength=1.7)
    save(fig, "fig_sf_nonconcave")


# ============================================================================
# Fig 7 -- ergodicity and its failure: mode energies of a 32-oscillator chain,
# harmonic (frozen) vs alpha-FPUT (mixing, with the famous near-recurrence).
# ============================================================================
def fput_modes(alpha, tmax, dt=0.05, N=32, nrec=1200):
    """Velocity-Verlet integration of the fixed-end alpha-FPUT chain, energy
    started in mode 1; returns times (in mode-1 periods) and E_k(t), k=1..4."""
    i = np.arange(1, N + 1)
    k = np.arange(1, 5)[:, None]
    modes = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * k * i / (N + 1))  # (4, N)
    om = 2 * np.sin(np.pi * np.arange(1, 5) / (2 * (N + 1)))
    x = np.sin(np.pi * i / (N + 1))          # amplitude 1 in mode 1
    v = np.zeros(N)

    def force(x):
        xp = np.concatenate(([0.0], x, [0.0]))
        d = np.diff(xp)                      # d_j = x_j - x_{j-1}
        f = d[1:] - d[:-1] + alpha * (d[1:] ** 2 - d[:-1] ** 2)
        return f

    nst = int(tmax / dt)
    every = max(1, nst // nrec)
    ts, Es = [], []
    f = force(x)
    for s in range(nst):
        v += 0.5 * dt * f
        x += dt * v
        f = force(x)
        v += 0.5 * dt * f
        if s % every == 0:
            Q = modes @ x
            P = modes @ v
            Es.append(0.5 * (P ** 2 + (om * Q[: 4].T) ** 2))
            ts.append((s + 1) * dt)
    T1 = 2 * np.pi / om[0]
    return np.array(ts) / T1, np.array(Es)


def fig_fput():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    cols = [C_BLUE, C_GREEN, C_MAGENTA, C_YELLOW]
    t0, E0 = fput_modes(alpha=0.0, tmax=30000)
    t1, E1 = fput_modes(alpha=0.25, tmax=30000)
    for k in range(4):
        a.plot(t0, E0[:, k], color=cols[k], lw=1.1)
        b.plot(t1, E1[:, k], color=cols[k], lw=1.1,
               label=f"mode {k+1}")
    a.annotate("mode 1", xy=(0.5, 0.86), xycoords="axes fraction",
               color=C_BLUE, fontsize=8.5)
    a.annotate("modes 2–4 (exactly zero forever)", xy=(0.5, 0.10),
               xycoords="axes fraction", color=INK2, fontsize=8.5, ha="center")
    a.set_ylim(0, 0.088)
    a.set_title(r"(a) harmonic ($\alpha=0$): integrable, frozen",
                fontsize=9, color=INK2)
    b.set_title(r"(b) $\alpha$-FPUT ($\alpha=1/4$): mixing $+$ recurrence",
                fontsize=9, color=INK2)
    a.set_xlabel(r"$t$ (mode-1 periods)")
    b.set_xlabel(r"$t$ (mode-1 periods)")
    a.set_ylabel(r"mode energy $E_k$")
    b.legend(loc="upper right", ncols=2, columnspacing=0.9, handlelength=1.3)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_sf_fput")


# ============================================================================
# Fig 8 -- the entropy inversion, with the code's own formulas.
# Transcribed from src/tcpyPI/utilities.py and pi.py (constants.py values);
# cross-checked below against the source docstring examples.
# ============================================================================
CPD, CPV, CL = 1005.7, 1870.0, 2500.0
CPVMCL = CPV - CL
RV, RD = 461.5, 287.04
EPS = RD / RV
ALV0 = 2.501e6


def es_cc(TC):
    return 6.112 * math.exp(17.67 * TC / (243.5 + TC))


def Lv(TC):
    return ALV0 + CPVMCL * TC


def ev(R, P):
    return R * P / (EPS + R)


def rv_(E, P):
    return EPS * E / (P - E)


def entropy_S(T, R, P):
    EV = ev(R, P)
    ES = es_cc(T - 273.15)
    RH = min(EV / ES, 1.0)
    ALV = Lv(T - 273.15)
    return (CPD + R * CL) * math.log(T) - RD * math.log(P - EV) \
        + ALV * R / T - R * RV * math.log(RH)


def s_sat(T, RP, P):
    """Saturated entropy SG exactly as in solve_temperature_from_entropy."""
    TC = T - 273.15
    ES = es_cc(TC)
    RG = rv_(ES, P)
    ALV = Lv(TC)
    EM = ev(RG, P)
    return (CPD + RP * CL) * math.log(T) - RD * math.log(P - EM) + ALV * RG / T


def newton_track(S, P, RP, T_initial):
    """The code's guarded Newton iteration, returning the iterate track."""
    TGNEW, TG, NC, track = T_initial, 0.0, 0, [T_initial]
    while abs(TGNEW - TG) > 0.001:
        TG = TGNEW
        TC = TG - 273.15
        ENEW = es_cc(TC)
        RG = rv_(ENEW, P)
        NC += 1
        ALV = Lv(TC)
        SL = (CPD + RP * CL + ALV * ALV * RG / (RV * TG * TG)) / TG
        EM = ev(RG, P)
        SG = (CPD + RP * CL) * math.log(TG) - RD * math.log(P - EM) + ALV * RG / TG
        AP = 0.3 if NC < 3 else 1.0
        TGNEW = TG + AP * (S - SG) / SL
        track.append(TGNEW)
        if NC > 500 or ENEW > P - 1:
            break
    return track


def check_transcription():
    assert abs(es_cc(20) - 23.369) < 1e-2, es_cc(20)
    assert abs(es_cc(0) - 6.112) < 1e-9
    assert abs(Lv(20) - 2488400.0) < 1e-6
    assert abs(ev(0.01, 1000) - 15.823) < 1e-2
    assert abs(rv_(15.942, 1000) - 0.010076) < 1e-5
    assert abs(entropy_S(300, 0.01, 1000) - 3987.17) < 1e-1
    tr = newton_track(4000.0, 1000.0, 0.01, 300.0)
    assert abs(tr[-1] - 292.676) < 1e-2, tr[-1]
    print("transcription cross-checks vs source docstrings: OK")


def fig_inversion():
    check_transcription()
    RP, s0 = 0.018, entropy_S(300.0, 0.018, 1000.0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.9),
                               gridspec_kw={"width_ratios": [1.35, 1]})
    T = np.linspace(232, 312, 500)
    press = [1000, 850, 700, 500, 300]
    blues5 = ["#c4d8f1", "#8fb3e5", "#5a8dd8", "#2a6ac0", "#123f7e"]
    for P, c in zip(press, blues5):
        sv = np.array([s_sat(t, RP, P) for t in T])
        a.plot(T, sv, color=c)
        if P < 900:  # mark roots only above the LCL (~950 hPa here)
            Tstar = newton_track(s0, P, RP, 300.0)[-1]
            a.plot([Tstar], [s0], "o", ms=3.8, color=c, zorder=5)
        a.annotate(f"{P}", xy=(T[-1] + 1, sv[-1]), fontsize=7.5, color=c,
                   va="center", annotation_clip=False)
    a.axhline(s0, color=INK2, lw=0.9, ls="--")
    a.annotate(r"$s_0$", xy=(234, s0 + 22), fontsize=9, color=INK2)
    a.annotate("hPa", xy=(313, s_sat(312, RP, 300)), fontsize=7.5, color=INK2,
               annotation_clip=False, va="bottom")
    a.set_xlabel(r"$T$ (K)")
    a.set_ylabel(r"$s_{\mathrm{sat}}(T,p)$  (J kg$^{-1}$K$^{-1}$)")
    a.set_title(r"(a) monotone $s_{\mathrm{sat}}$, one root per level",
                fontsize=9, color=INK2)
    a.set_xlim(232, 312)

    for P, T0, c, lab in [(500, 300.0, C_BLUE, r"start $300\,$K"),
                          (500, 240.0, C_GREEN, r"start $240\,$K")]:
        tr = np.array(newton_track(s0, P, RP, T0))
        Tstar = tr[-1]
        err = np.abs(tr - Tstar)[:-1]
        err = np.where(err > 0, err, 1e-14)
        b.semilogy(range(len(err)), err, "o-", ms=3.2, lw=1.1, color=c,
                   label=lab)
    b.axvspan(-0.4, 2.0, color=C_YELLOW, alpha=0.18, lw=0)
    b.annotate(r"guarded" "\n" r"$AP=0.3$", xy=(0.8, 3e-4), fontsize=8,
               color=INK2, ha="center")
    b.set_xlabel(r"Newton iteration")
    b.set_ylabel(r"$|T_n-T^{*}|$ (K)")
    b.set_title("(b) guarded Newton, $p=500$ hPa", fontsize=9, color=INK2)
    b.set_ylim(top=2000)
    b.legend(loc="upper right", handlelength=1.4)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_sf_inversion")


if __name__ == "__main__":
    fig_classical_marginal()
    fig_quantum_marginal()
    fig_typicality()
    fig_comb()
    fig_bridge()
    fig_nonconcave()
    fig_fput()
    fig_inversion()
    print("all figures written to", FIGDIR)
