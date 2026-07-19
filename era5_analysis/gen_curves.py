"""Sample buoyancy curves per (parcel, topology class) for spaghetti plots."""

import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("ERA5_SCRATCH", str(_Path(__file__).resolve().parent / "work"))
_SRC = str(_Path(__file__).resolve().parents[1] / "src")

import sys

import numpy as np


sys.path.insert(0, SCRATCH)
sys.path.insert(0, _SRC)

from scan_final import buoyancy, EPS, PTOP  # njit'd
from tcpyPI import utilities

NSAMP = 150

d = np.load(f"{SCRATCH}/profiles_converted.npz")
z = np.load(f"{SCRATCH}/parcel_topology_final.npz")
P_full = d["P"]
nlvl = int((P_full > PTOP).sum())
P = P_full[:nlvl].copy()
tc = (d["sst_C"] >= 26) & (d["sp_hPa"] >= 1000) & np.isfinite(z["PM"])

def topo_str(fs, n):
    c0 = "+" if fs > 0 else "-"
    return (("+-" if c0 == "+" else "-+") * ((int(n) + 1) // 2))[: int(n)]

rng = np.random.default_rng(3)
out = {}
CLASSES = {"A": 6, "B": 4, "C": 3}
for pname, nclass in CLASSES.items():
    o = z[pname]
    ok = tc & np.isfinite(o[:, 0]) & (o[:, 0] != 0)
    keys = [(fs, n) for fs, n in {(o[i, 0], o[i, 1]) for i in np.where(ok)[0]}]
    counts = {k: 0 for k in keys}
    lab = np.full(len(o), -1)
    keymap = {}
    for i in np.where(ok)[0]:
        k = (o[i, 0], o[i, 1])
        if k not in keymap:
            keymap[k] = len(keymap)
        lab[i] = keymap[k]
        counts[k] += 1
    top = sorted(counts, key=lambda k: -counts[k])[:nclass]
    Ntot = ok.sum()
    for k in top:
        rows = np.where(ok & (lab == keymap[k]))[0]
        sel = rng.choice(rows, size=min(NSAMP, len(rows)), replace=False)
        curves = np.empty((len(sel), nlvl))
        b = np.empty(nlvl)
        for m, i in enumerate(sel):
            T = d["TC"][i, :nlvl] + 273.15
            R = d["R"][i, :nlvl] * 0.001
            if pname == "A":
                f = buoyancy(T[0], R[0], P[0], T, R, P, nlvl, b)
            else:
                PP = min(z["PM"][i], 1000.0)
                if pname == "B":
                    MSL = d["sp_hPa"][i]
                    RP = EPS * R[0] * MSL / (PP * (EPS + R[0]) - R[0] * MSL)
                    f = buoyancy(T[0], RP, PP, T, R, P, nlvl, b)
                else:
                    SSTK = d["sst_C"][i] + 273.15
                    ES0 = utilities.es_cc(d["sst_C"][i])
                    f = buoyancy(SSTK, utilities.rv(ES0, PP), PP, T, R, P, nlvl, b)
            curves[m] = b if f == 1 else np.nan
        t = topo_str(*k)
        share = 100.0 * counts[k] / Ntot
        out[f"{pname}|{t}|{share:.2f}"] = curves
        print(f"parcel {pname}  {t:14s} {share:6.2f}%  sampled {len(sel)}")

np.savez_compressed(f"{SCRATCH}/topology_curves.npz", P=P,
                    **{k: v for k, v in out.items()})
print("saved topology_curves.npz")
