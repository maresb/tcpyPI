"""Generate all data for the discontinuity analysis document.

Uses the COMPILED cape() from the branch tree for speed (survey, sweeps), and
the instrumented pure-python cape for per-case internals (TVRDIF, W curves).

Outputs: docdata.npz in this directory.
"""

import os
import sys

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRATCH, "branch", "src"))
sys.path.insert(0, SCRATCH)

import numpy as np
import xarray as xr
from tcpyPI import constants, utilities
from tcpyPI.pi import cape as cape_jit  # compiled pcmin-convention cape

from gmap import cape_instr  # instrumented pure-python (same convention)

EPS = constants.EPS
RD = constants.RD


def g_eval(PM, SSTK, MSL, P, T, R, ES0, CKCD=0.9, diss_flag=1):
    """One iteration body PM -> PNEW using compiled cape. Returns PNEW & parts."""
    PP = min(PM, 1000.0)
    RP = EPS * R[0] * MSL / (PP * (EPS + R[0]) - R[0] * MSL)
    CAPEM, _, _, flM = cape_jit(T[0], RP, PP, T, R, P, 0, 50, 1)
    RPS = utilities.rv(ES0, PP)
    CAPEMS, TOMS, LNBS, flS = cape_jit(SSTK, RPS, PP, T, R, P, 0, 50, 1)
    TO = TOMS
    RAT = SSTK / TO if diss_flag else 1.0
    TV0 = utilities.Trho(T[0], R[0], R[0])
    TVSST = utilities.Trho(SSTK, RPS, RPS)
    TVAV = 0.5 * (TV0 + TVSST)
    CAPEA, _, _, _ = cape_jit(T[0], R[0], P[0], T, R, P, 0, 50, 1)
    CAT = max((CAPEM - CAPEA) + 0.5 * CKCD * RAT * (CAPEMS - CAPEM), 0.0)
    return MSL * np.exp(-CAT / (RD * TVAV)), CAPEM, CAPEMS, TO


def g_scan(pms, SSTK, MSL, P, T, R, ES0):
    out = np.empty((len(pms), 4))
    for k, pm in enumerate(pms):
        out[k] = g_eval(pm, SSTK, MSL, P, T, R, ES0)
    return out


def iterate_raw(SSTK, MSL, P, T, R, ES0, x0=970.0, nmax=200, tol=0.5):
    """Raw pcmin iteration without rescue. Returns (status, iterates).
    status: 1 converged, 2 cycle/cap, 0 blowdown(PM<400)."""
    PM, PMOLD, PNEW, NP = x0, x0, 0.0, 0
    xs = []
    while abs(PNEW - PMOLD) > tol:
        PNEW = g_eval(PM, SSTK, MSL, P, T, R, ES0)[0]
        xs.append(PM)
        PMOLD, PM = PM, PNEW
        NP += 1
        if NP > nmax or PM < 400:
            return (2 if PM >= 400 else 0), np.array(xs + [PM])
    xs.append(PM)
    return 1, np.array(xs)


def W_curve(TP, RP, PP, T, R, P, ptop=50):
    """Running signed work integral W(p_t): nodes + crossing candidates.
    Returns arrays (p_t values, W values) densified, plus candidate list."""
    base = cape_instr(TP, RP, PP, T, R, P, 0, ptop)
    N = int(np.count_nonzero(P > ptop))
    Pt, Tt = P[:N], T[:N]
    b = base["TVRDIF"]
    pts = [PP]
    Ws = [0.0]
    W = RD * (PP - Pt[0]) / (PP + Pt[0]) * b[0]
    pts.append(Pt[0])
    Ws.append(W)
    for j in range(1, N):
        b0, b1 = b[j - 1], b[j]
        if b0 * b1 < 0.0:
            Pc = (Pt[j] * b0 - Pt[j - 1] * b1) / (b0 - b1)
            Wc = W + RD * b0 * (Pt[j - 1] - Pc) / (Pt[j - 1] + Pc)
            pts.append(Pc)
            Ws.append(Wc)
        W += RD * (b1 + b0) * (Pt[j - 1] - Pt[j]) / (Pt[j] + Pt[j - 1])
        pts.append(Pt[j])
        Ws.append(W)
    return np.array(pts), np.array(Ws), b, Pt


# ---------------- profiles ----------------
P77 = np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650,
                600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125,
                100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1], dtype=float)
TC77 = np.array([25.260956, 23.078949, 20.881561, 18.652649, 16.615143, 14.63147,
                 12.794586, 11.79306, 11.01236, 10.847565, 10.354492, 7.807007,
                 5.473297, 2.5278625, -1.532135, -7.3142395, -13.635345,
                 -20.613602, -28.588928, -37.270096, -45.40825, -50.047455,
                 -54.84575, -59.173737, -62.878662, -65.78009, -69.30669,
                 -65.512024, -60.76361, -55.100464, -51.38333, -43.95462,
                 -39.709717, -33.193268, -21.576843, -15.402496, -13.186676])
R77 = np.array([1.0783693e01, 1.0704287e01, 1.0680210e01, 1.0617845e01,
                1.0320683e01, 9.8811483e00, 9.1884289e00, 7.1884680e00,
                5.6963191e00, 3.5568204e00, 1.5912720e00, 1.0433695e00,
                5.9723043e-01, 4.3974420e-01, 4.8722979e-01, 5.8590513e-01,
                4.5599860e-01, 3.1293562e-01, 1.9222400e-01, 9.7611703e-02,
                3.3851895e-02, 2.4188591e-02, 1.9636340e-02, 1.3214327e-02,
                7.2453087e-03, 4.3027173e-03, 3.7014042e-03, 2.9414182e-03,
                2.7806845e-03, 2.8306348e-03, 2.9053832e-03, 2.9848625e-03,
                3.0779461e-03, 3.1315640e-03, 3.2939769e-03, 3.3872949e-03,
                3.7360021e-03])
SST77, MSL77 = 28.20263671875, 1014.9654541015625

P95 = np.array([1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 825.0, 800.0,
                775.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0,
                350.0, 300.0, 250.0, 225.0, 200.0, 175.0, 150.0, 125.0, 100.0,
                70.0, 50.0, 30.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0])
TC95 = np.array([20.266, 18.11, 15.978, 13.94, 12.576, 11.213, 9.669, 8.084,
                 6.516, 4.993, 4.374, 4.93, 1.852, -2.047, -6.917, -13.015,
                 -19.698, -26.881, -34.178, -42.822, -51.573, -54.68, -57.207,
                 -58.943, -60.751, -63.255, -63.277, -61.5, -58.453, -53.327,
                 -50.156, -46.602, -42.283, -35.389, -22.349, -14.944, -11.886])
R95 = np.array([9.65, 9.517, 9.459, 9.205, 7.784, 6.561, 5.952, 5.517, 4.976,
                4.332, 3.102, 1.207, 1.521, 1.263, 0.857, 0.556, 0.33, 0.211,
                0.104, 0.073, 0.037, 0.027, 0.018, 0.01, 0.005, 0.004, 0.003,
                0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.004, 0.004])
SST95, MSL95 = 25.267, 1012.9

save = {}


def prep(SSTC, MSL, TCp, Rp):
    SSTK = utilities.T_Ctok(SSTC)
    T = utilities.T_Ctok(TCp)
    R = Rp * 0.001
    ES0 = utilities.es_cc(SSTC)
    return SSTK, MSL, T, R, ES0


# ---------- case A: PR77 ----------
print("case A: PR77 ...")
S77 = prep(SST77, MSL77, TC77, R77)
pms = np.arange(946.0, 958.0, 0.002)
sc = g_scan(pms, S77[0], S77[1], P77, S77[2], S77[3], S77[4])
save["a_pm"] = pms
save["a_g"] = sc[:, 0]
save["a_capem"] = sc[:, 1]
save["a_capems"] = sc[:, 2]
st, xs = iterate_raw(S77[0], S77[1], P77, S77[2], S77[3], S77[4], nmax=60)
save["a_iter_status"] = st
save["a_iter"] = xs
# buoyancy + W at the two cycle phases
for tag, pm in [("lo", 950.6533454438), ("hi", 951.2790079839)]:
    PP = min(pm, 1000.0)
    RP = EPS * S77[3][0] * MSL77 / (PP * (EPS + S77[3][0]) - S77[3][0] * MSL77)
    pts, Ws, b, Pt = W_curve(S77[2][0], RP, PP, S77[2], S77[3], P77)
    save[f"a_W_{tag}_p"] = pts
    save[f"a_W_{tag}_W"] = Ws
    save[f"a_b_{tag}"] = b
save["a_blevels"] = P77[: int(np.count_nonzero(P77 > 50))]

# ---------- case B: 1995 problematic ----------
print("case B: 1995 ...")
S95 = prep(SST95, MSL95, TC95, R95)
pms = np.arange(955.0, 968.0, 0.002)
sc = g_scan(pms, S95[0], S95[1], P95, S95[2], S95[3], S95[4])
save["b_pm"] = pms
save["b_g"] = sc[:, 0]
save["b_capem"] = sc[:, 1]
st, xs = iterate_raw(S95[0], S95[1], P95, S95[2], S95[3], S95[4], nmax=60)
save["b_iter_status"] = st
save["b_iter"] = xs

# ---------- sample columns ----------
print("loading sample ...")
ds = xr.open_dataset(os.path.join(SCRATCH, "sample_data.nc"))
p_s = ds["p"].values.astype(float)
sst_s, msl_s = ds["sst"].values, ds["msl"].values
t_s, r_s = ds["t"].values, ds["r"].values
a_ref = np.load(os.path.join(SCRATCH, "branch_pi.npz"))
both1 = a_ref["IFL"] == 1


def col(m, j, i):
    T = utilities.T_Ctok(t_s[m, :, j, i].astype(float))
    R = r_s[m, :, j, i].astype(float) * 0.001
    R[np.isnan(R)] = 0.0
    SSTK = utilities.T_Ctok(sst_s[m, j, i])
    ES0 = utilities.es_cc(sst_s[m, j, i])
    return SSTK, float(msl_s[m, j, i]), T, R, ES0


# ---------- case C: healthy column (strong PI, mid-domain) ----------
# pick the column with max VMAX (deep tropics, textbook profile)
mm, jj, ii = np.unravel_index(np.nanargmax(np.where(both1, a_ref["VMAX"], -1)),
                              a_ref["VMAX"].shape)
print(f"case C: healthy col (m={mm},j={jj},i={ii}) VMAX={a_ref['VMAX'][mm,jj,ii]:.1f}")
SC = col(mm, jj, ii)
pms = np.arange(850.0, 1005.0, 0.02)
sc = g_scan(pms, SC[0], SC[1], p_s, SC[2], SC[3], SC[4])
save["c_pm"] = pms
save["c_g"] = sc[:, 0]
st, xs = iterate_raw(SC[0], SC[1], p_s, SC[2], SC[3], SC[4], nmax=60)
save["c_iter"] = xs
save["c_idx"] = np.array([mm, jj, ii])
# W curve + buoyancy for the saturated parcel at converged PM (textbook single hump)
pmc = xs[-1]
PP = min(pmc, 1000.0)
pts, Ws, b, Pt = W_curve(SC[0], utilities.rv(SC[4], PP), PP, SC[2], SC[3], p_s)
save["c_W_p"] = pts
save["c_W_W"] = Ws
save["c_b"] = b
save["c_blevels"] = p_s[: int(np.count_nonzero(p_s > 50))]

# ---------- survey: jump statistics over converging columns ----------
print("survey ...")
idxs = np.argwhere(both1)
rng = np.random.default_rng(42)
sel = idxs[rng.choice(len(idxs), size=min(1200, len(idxs)), replace=False)]
pmgrid = np.arange(900.0, 1005.0, 0.1)
maxjump = np.full(len(sel), np.nan)
njumps = np.zeros(len(sel))
fp_in_jump = np.zeros(len(sel), bool)
dist_fp_jump = np.full(len(sel), np.nan)
SMOOTH = 0.06  # smooth |dg| bound for 0.1 step given slope<=0.5
for k, (m, j, i) in enumerate(sel):
    S = col(m, j, i)
    gs = np.empty(len(pmgrid))
    for q, pm in enumerate(pmgrid):
        gs[q] = g_eval(pm, S[0], S[1], p_s, S[2], S[3], S[4])[0]
    dg = np.diff(gs)
    jmask = np.abs(dg) > SMOOTH
    njumps[k] = int(jmask.sum())
    if jmask.any():
        maxjump[k] = np.max(np.abs(dg[jmask]))
    # diagonal crossing
    F = gs - pmgrid
    scg = np.where(np.diff(np.sign(F)) != 0)[0]
    if len(scg):
        q = scg[0]
        fp_in_jump[k] = jmask[q]
        if jmask.any():
            jpos = pmgrid[:-1][jmask] + 0.05
            dist_fp_jump[k] = np.min(np.abs(jpos - pmgrid[q]))
save["s_maxjump"] = maxjump
save["s_njumps"] = njumps
save["s_fp_in_jump"] = fp_in_jump
save["s_dist"] = dist_fp_jump
save["s_n"] = len(sel)
print(f"  columns with >=1 jump: {(njumps>0).sum()}/{len(sel)}, "
      f"fp inside jump: {fp_in_jump.sum()}")

# ---------- case D: SST sweep on PR77 column ----------
print("SST sweep ...")
dss = np.arange(-3.0, 3.0001, 0.01)
sweep_status = np.zeros(len(dss))
sweep_pm = np.full(len(dss), np.nan)
for k, d in enumerate(dss):
    S = prep(SST77 + d, MSL77, TC77, R77)
    st, xs = iterate_raw(S[0], S[1], P77, S[2], S[3], S[4], nmax=200)
    sweep_status[k] = st
    if st == 1:
        sweep_pm[k] = xs[-1]
save["d_dss"] = dss
save["d_status"] = sweep_status
save["d_pm"] = sweep_pm
print(f"  failures: {(sweep_status==2).sum()}/{len(dss)}")

np.savez(os.path.join(SCRATCH, "docdata.npz"), **save)
print("saved docdata.npz")
