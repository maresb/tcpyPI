"""Find a column with two comparable local maxima of W(p_t) (saturated parcel),
and produce data showing: CAPE_maxW continuous vs SST while argmax (TO) jumps.
"""

import os
import sys

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRATCH, "branch", "src"))
sys.path.insert(0, SCRATCH)

import numpy as np
import xarray as xr
from tcpyPI import constants, utilities

from gmap import cape_instr

RD = constants.RD


def W_all(TP, RP, PP, T, R, P, ptop=50):
    base = cape_instr(TP, RP, PP, T, R, P, 0, ptop)
    N = int(np.count_nonzero(P > ptop))
    Pt, Tt = P[:N], T[:N]
    b = base["TVRDIF"]
    pts, Ws, Ts = [PP], [0.0], [TP]
    W = RD * (PP - Pt[0]) / (PP + Pt[0]) * b[0]
    pts.append(Pt[0]); Ws.append(W); Ts.append(Tt[0])
    for j in range(1, N):
        b0, b1 = b[j - 1], b[j]
        if b0 * b1 < 0.0:
            Pc = (Pt[j] * b0 - Pt[j - 1] * b1) / (b0 - b1)
            Wc = W + RD * b0 * (Pt[j - 1] - Pc) / (Pt[j - 1] + Pc)
            Tc = (Tt[j-1] * (Pc - Pt[j]) + Tt[j] * (Pt[j-1] - Pc)) / (Pt[j-1] - Pt[j])
            pts.append(Pc); Ws.append(Wc); Ts.append(Tc)
        W += RD * (b1 + b0) * (Pt[j - 1] - Pt[j]) / (Pt[j] + Pt[j - 1])
        pts.append(Pt[j]); Ws.append(W); Ts.append(Tt[j])
    return np.array(pts), np.array(Ws), np.array(Ts), b, Pt


def local_maxima(pts, Ws):
    out = []
    for k in range(1, len(Ws) - 1):
        if Ws[k] >= Ws[k - 1] and Ws[k] >= Ws[k + 1] and Ws[k] > 0:
            out.append((pts[k], Ws[k]))
    return out


ds = xr.open_dataset(os.path.join(SCRATCH, "sample_data.nc"))
p_s = ds["p"].values.astype(float)
sst_s, msl_s = ds["sst"].values, ds["msl"].values
t_s, r_s = ds["t"].values, ds["r"].values
a_ref = np.load(os.path.join(SCRATCH, "branch_pi.npz"))
both1 = a_ref["IFL"] == 1

# hunt over ENVIRONMENTAL-parcel W curves (the M call is where #77-type marginal
# humps live); require two maxima with ratio in [0.3, 1.0] and separation > 150 hPa
idxs = np.argwhere(both1)
rng = np.random.default_rng(1)
sel = idxs[rng.choice(len(idxs), size=800, replace=False)]
best = None
for m, j, i in sel:
    T = utilities.T_Ctok(t_s[m, :, j, i].astype(float))
    R = r_s[m, :, j, i].astype(float) * 0.001
    R[np.isnan(R)] = 0.0
    MSL = float(msl_s[m, j, i])
    PM = a_ref["PMIN"][m, j, i] + 20  # near converged radius pressure
    PP = min(PM, 1000.0)
    RP = constants.EPS * R[0] * MSL / (PP * (constants.EPS + R[0]) - R[0] * MSL)
    if RP < 1e-6 or np.isnan(RP):
        continue
    pts, Ws, Ts, b, Pt = W_all(T[0], RP, PP, T, R, p_s)
    lm = local_maxima(pts, Ws)
    if len(lm) >= 2:
        lm = sorted(lm, key=lambda x: -x[1])[:2]
        ratio = lm[1][1] / lm[0][1]
        sep = abs(lm[0][0] - lm[1][0])
        if 0.25 < ratio and sep > 150:
            score = ratio * sep
            if best is None or score > best[0]:
                best = (score, (m, j, i), ratio, sep)

print("best two-hump column:", best)

if best is not None:
    m, j, i = best[1]
    T = utilities.T_Ctok(t_s[m, :, j, i].astype(float))
    R = r_s[m, :, j, i].astype(float) * 0.001
    R[np.isnan(R)] = 0.0
    MSL = float(msl_s[m, j, i])
    SSTC0 = float(sst_s[m, j, i])
    PM = a_ref["PMIN"][m, j, i] + 20
    PP = min(PM, 1000.0)

    # sweep a parameter (parcel moisture scaling) to force a dominance exchange:
    # scale R[0] (boundary-layer moisture) by f in [0.9, 1.1]
    fs = np.linspace(0.94, 1.06, 601)
    cape_pc, cape_mw, lnb_mw, lnb_pc = [], [], [], []
    for f in fs:
        Rk = R.copy()
        Rk[0] = R[0] * f
        RP = constants.EPS * Rk[0] * MSL / (PP * (constants.EPS + Rk[0]) - Rk[0] * MSL)
        pts, Ws, Ts, b, Pt = W_all(T[0], RP, PP, T, Rk, p_s)
        kmax = int(np.argmax(Ws))
        cape_mw.append(max(Ws[kmax], 0.0))
        lnb_mw.append(pts[kmax] if Ws[kmax] > 0 else np.nan)
        o = cape_instr(T[0], RP, PP, T, Rk, p_s, 0, 50)
        cape_pc.append(o["CAPED"])
        lnb_pc.append(o["LNB"])
    np.savez(os.path.join(SCRATCH, "humpdata.npz"),
             fs=fs, cape_pc=np.array(cape_pc), cape_mw=np.array(cape_mw),
             lnb_mw=np.array(lnb_mw), lnb_pc=np.array(lnb_pc),
             idx=np.array([m, j, i]), sst=SSTC0, pm=PM)
    # W curves at three f values around the swap
    swap_k = int(np.argmax(np.abs(np.diff(np.array(lnb_mw))))) if len(fs) > 1 else 300
    for tag, f in [("lo", fs[max(swap_k - 60, 0)]), ("at", fs[swap_k]),
                   ("hi", fs[min(swap_k + 60, len(fs) - 1)])]:
        Rk = R.copy()
        Rk[0] = R[0] * f
        RP = constants.EPS * Rk[0] * MSL / (PP * (constants.EPS + Rk[0]) - Rk[0] * MSL)
        pts, Ws, Ts, b, Pt = W_all(T[0], RP, PP, T, Rk, p_s)
        np.savez(os.path.join(SCRATCH, f"humpW_{tag}.npz"), p=pts, W=Ws, f=f)
    print(f"saved humpdata.npz; swap near f={fs[swap_k]:.4f}; col (m,j,i)={(m,j,i)} SST={SSTC0:.2f}")
