"""Classify branch-vs-fix differences: evaluate BOTH cape definitions at
IDENTICAL inputs (same PM), removing the iteration-path effect.

For each changed column, at the branch's own converged PM:
 - compare cape_pcmin vs cape_maxw for the three calls (A, M, S)
 - classify: identical (path noise) / surface-sliver / dominated-hump
"""

import os
import sys

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRATCH, "branch", "src"))  # pcmin cape via cape_instr

import numpy as np
import xarray as xr
from tcpyPI import constants, utilities

sys.path.insert(0, SCRATCH)
from gmap import cape_instr  # pcmin-convention instrumented cape


def cape_maxw_generic(TP, RP, PP, T, R, P, ascent_flag=0, ptop=50):
    base = cape_instr(TP, RP, PP, T, R, P, ascent_flag, ptop)
    if base["INB"] == -1:
        return base
    N = int(np.count_nonzero(P > ptop))
    Pt, Tt = P[:N], T[:N]
    TVRDIF = base["TVRDIF"]
    RD = constants.RD
    W = RD * (PP - Pt[0]) / (PP + Pt[0]) * TVRDIF[0]
    BESTW, BESTP, BESTT = 0.0, 0.0, Tt[0]
    if W > BESTW:
        BESTW, BESTP, BESTT = W, Pt[0], Tt[0]
    for j in range(1, N):
        b0, b1 = TVRDIF[j - 1], TVRDIF[j]
        if b0 * b1 < 0.0:
            Pc = (Pt[j] * b0 - Pt[j - 1] * b1) / (b0 - b1)
            Wc = W + RD * b0 * (Pt[j - 1] - Pc) / (Pt[j - 1] + Pc)
            if Wc > BESTW:
                Tc = (Tt[j - 1] * (Pc - Pt[j]) + Tt[j] * (Pt[j - 1] - Pc)) / (Pt[j - 1] - Pt[j])
                BESTW, BESTP, BESTT = Wc, Pc, Tc
        W += RD * (b1 + b0) * (Pt[j - 1] - Pt[j]) / (Pt[j] + Pt[j - 1])
        if W > BESTW:
            BESTW, BESTP, BESTT = W, Pt[j], Tt[j]
    if BESTW <= 0.0:
        return dict(CAPED=0.0, TOB=Tt[0], LNB=0.0, IFLAG=1)
    return dict(CAPED=BESTW, TOB=BESTT, LNB=BESTP, IFLAG=1)


ds = xr.open_dataset(os.path.join(SCRATCH, "sample_data.nc"))
p = ds["p"].values.astype(float)
sst, msl = ds["sst"].values, ds["msl"].values
t4, r4 = ds["t"].values, ds["r"].values

a = np.load(os.path.join(SCRATCH, "branch_pi.npz"))
b = np.load(os.path.join(SCRATCH, "branch_fix_pi.npz"))
both1 = (a["IFL"] == 1) & (b["IFL"] == 1)
d_vmax = np.abs(a["VMAX"] - b["VMAX"])

chg = np.argwhere(both1 & (d_vmax > 0.1))
print(f"dissecting {len(chg)} columns with |dVMAX| > 0.1 m/s")

cls = {"A": 0, "M": 0, "S": 0, "surface_sliver": 0, "none@fixedPM": 0}
details = []
for idx in chg:
    m, j, i = idx
    T = utilities.T_Ctok(t4[m, :, j, i].astype(float))
    R = r4[m, :, j, i].astype(float) * 0.001
    R[np.isnan(R)] = 0.0
    SSTK = utilities.T_Ctok(sst[m, j, i])
    MSL = msl[m, j, i]
    ES0 = utilities.es_cc(sst[m, j, i])
    # reconstruct branch converged PM from its PMIN? PM isn't stored; instead
    # evaluate both definitions at the same probe PM = branch PMIN's implied
    # iterate is unknown, so use PM=970 start AND branch PMIN as two probes.
    diffs = set()
    sliver = False
    for PM in (970.0, max(a["PMIN"][m, j, i], 401.0)):
        PP = min(PM, 1000.0)
        RP = constants.EPS * R[0] * MSL / (PP * (constants.EPS + R[0]) - R[0] * MSL)
        for tag, (TPq, RPq, PPq) in {
            "A": (T[0], R[0], p[0]),
            "M": (T[0], RP, PP),
            "S": (SSTK, utilities.rv(ES0, PP), PP),
        }.items():
            o1 = cape_instr(TPq, RPq, PPq, T, R, p, 0, 50)
            o2 = cape_maxw_generic(TPq, RPq, PPq, T, R, p, 0, 50)
            if abs(o1["CAPED"] - o2["CAPED"]) > 1e-9 or abs((o1["LNB"] or 0) - o2["LNB"]) > 0.5:
                diffs.add(tag)
                if o1["INB"] == 0 and o2["CAPED"] > 0:
                    sliver = True
    if not diffs:
        cls["none@fixedPM"] += 1
    elif sliver:
        cls["surface_sliver"] += 1
    else:
        for tag in diffs:
            cls[tag] += 1
    details.append((tuple(idx), sorted(diffs), a["VMAX"][m, j, i], b["VMAX"][m, j, i],
                    a["TO"][m, j, i], b["TO"][m, j, i]))

print("classification counts:", cls)
print()
for d in details[:25]:
    print(f"{d[0]} calls-differing={d[1]} VMAX {d[2]:.3f}->{d[3]:.3f} TO {d[4]:.2f}->{d[5]:.2f}")

# Also: how many of ALL both1 columns have identical cape at fixed PM=970 for all 3 calls?
print()
print("=== sampling 300 unchanged columns to confirm exact equality at fixed PM ===")
ok_cols = np.argwhere(both1 & (d_vmax <= 1e-8))
rng = np.random.default_rng(0)
sel = ok_cols[rng.choice(len(ok_cols), size=300, replace=False)]
neq = 0
for idx in sel:
    m, j, i = idx
    T = utilities.T_Ctok(t4[m, :, j, i].astype(float))
    R = r4[m, :, j, i].astype(float) * 0.001
    R[np.isnan(R)] = 0.0
    SSTK = utilities.T_Ctok(sst[m, j, i])
    MSL = msl[m, j, i]
    ES0 = utilities.es_cc(sst[m, j, i])
    PM = 970.0
    PP = min(PM, 1000.0)
    RP = constants.EPS * R[0] * MSL / (PP * (constants.EPS + R[0]) - R[0] * MSL)
    for TPq, RPq, PPq in [(T[0], R[0], p[0]), (T[0], RP, PP), (SSTK, utilities.rv(ES0, PP), PP)]:
        o1 = cape_instr(TPq, RPq, PPq, T, R, p, 0, 50)
        o2 = cape_maxw_generic(TPq, RPq, PPq, T, R, p, 0, 50)
        if o1["CAPED"] != o2["CAPED"]:
            neq += 1
            break
print(f"columns (of 300 sampled unchanged) with ANY cape difference at PM=970: {neq}")
