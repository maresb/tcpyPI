"""Assemble the JSON payload for the sounding-scope artifact."""

import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("ERA5_SCRATCH", str(_Path(__file__).resolve().parent / "work"))
_SRC = str(_Path(__file__).resolve().parents[1] / "src")

import json
import sys

import numpy as np


sys.path.insert(0, SCRATCH)
sys.path.insert(0, _SRC)

from scan_final import buoyancy, EPS, PTOP
from tcpyPI import utilities

# materialize everything: NpzFile decompresses the whole array per access
_d = np.load(f"{SCRATCH}/profiles_converted.npz")
d = {k: _d[k] for k in _d.files}
_z = np.load(f"{SCRATCH}/parcel_topology_final.npz")
z = {k: _z[k] for k in ("A", "B", "C", "PM")}
P_full = d["P"]
nlvl = int((P_full > PTOP).sum())
P = P_full[:nlvl].copy()
# per-parcel populations: A is defined everywhere; B/C wherever the
# max-work pressure iteration converged (SST > 5 C)
mA = np.isfinite(z["A"][:, 0])
mBC = np.isfinite(z["PM"]) & np.isfinite(z["B"][:, 0]) & np.isfinite(z["C"][:, 0])
popmask = {"A": mA, "B": mBC, "C": mBC}
pop = {p: np.where(popmask[p])[0] for p in "ABC"}
print("populations:", {p: len(pop[p]) for p in "ABC"})

t64 = d["time"].astype("datetime64[s]")
years = t64.astype("datetime64[Y]").astype(int) + 1970
doy = (t64.astype("datetime64[D]")
       - t64.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1


def topo_str(fs, n):
    if fs == 0:
        return "0"
    c0 = "+" if fs > 0 else "-"
    return (("+-" if c0 == "+" else "-+") * ((int(n) + 1) // 2))[: int(n)]


# per-parcel class tables (top 7 by count within tc, then "other")
tables = {}
labels = {}
allstrs = {}
for pname in "ABC":
    o = z[pname]
    rows = pop[pname]
    strs = np.array([topo_str(o[i, 0], o[i, 1]) for i in rows])
    allstrs[pname] = strs
    uniq, cnt = np.unique(strs, return_counts=True)
    order = np.argsort(-cnt)
    top = [str(uniq[k]) for k in order[:7]]
    tables[pname] = {t: i for i, t in enumerate(top)}
    labels[pname] = [[str(uniq[k]), int(cnt[k]),
                      float(round(100.0 * cnt[k] / len(rows), 2))]
                     for k in order[:7]]
    other = int(len(rows) - cnt[order[:7]].sum())
    labels[pname].append(["other", other, float(round(100.0 * other / len(rows), 2))])


def cls(pname, i):
    if not popmask[pname][i]:
        return -1
    return tables[pname].get(topo_str(z[pname][i, 0], z[pname][i, 1]), 7)


rng = np.random.default_rng(42)

# ---- player pool: ensure every top class of every parcel is represented ----
pool = set(rng.choice(pop["A"], 1500, replace=False).tolist())
for pname in "ABC":
    for t in tables[pname]:
        rows = pop[pname][allstrs[pname] == t]
        take = min(220, len(rows))
        pool.update(rng.choice(rows, take, replace=False).tolist())
pool = np.array(sorted(pool))
print("player pool:", len(pool))

curves = {"A": [], "B": [], "C": []}
b = np.empty(nlvl)
for i in pool:
    T = d["TC"][i, :nlvl] + 273.15
    R = d["R"][i, :nlvl] * 0.001
    buoyancy(T[0], R[0], P[0], T, R, P, nlvl, b)
    curves["A"].append([round(float(x), 2) for x in b])
    if mBC[i]:
        PP = min(z["PM"][i], 1000.0)
        MSL = d["sp_hPa"][i]
        RP = EPS * R[0] * MSL / (PP * (EPS + R[0]) - R[0] * MSL)
        buoyancy(T[0], RP, PP, T, R, P, nlvl, b)
        curves["B"].append([round(float(x), 2) for x in b])
        ES0 = utilities.es_cc(d["sst_C"][i])
        buoyancy(d["sst_C"][i] + 273.15, utilities.rv(ES0, PP), PP, T, R, P, nlvl, b)
        curves["C"].append([round(float(x), 2) for x in b])
    else:
        curves["B"].append([])
        curves["C"].append([])

lonw = np.where(d["lon"][pool] > 180, d["lon"][pool] - 360, d["lon"][pool])
player = {
    "lat": [round(float(x), 2) for x in d["lat"][pool]],
    "lon": [round(float(x), 2) for x in lonw],
    "year": [int(x) for x in years[pool]],
    "doy": [int(x) for x in doy[pool]],
    "sst": [round(float(x), 1) for x in d["sst_C"][pool]],
}
for pname in "ABC":
    m = popmask[pname]
    player[f"c{pname}"] = [int(cls(pname, i)) for i in pool]
    player[f"Etop{pname}"] = [int(round(z[pname][i, 2])) if m[i] else 0 for i in pool]
    player[f"Emax{pname}"] = [int(round(z[pname][i, 3])) if m[i] else 0 for i in pool]
    player[f"topo{pname}"] = [topo_str(z[pname][i, 0], z[pname][i, 1]) if m[i] else ""
                              for i in pool]
    player[f"curve{pname}"] = curves[pname]

# ---- map layer ----
msel = rng.choice(pop["A"], min(40000, len(pop["A"])), replace=False)
mlon = np.where(d["lon"][msel] > 180, d["lon"][msel] - 360, d["lon"][msel])
mappts = {
    "lat": [round(float(x), 2) for x in d["lat"][msel]],
    "lon": [round(float(x), 2) for x in mlon],
    "doy": [int(x) for x in doy[msel]],
    "year": [int(x) for x in years[msel]],
}
for pname in "ABC":
    mappts[f"c{pname}"] = [int(cls(pname, i)) for i in msel]

# per-parcel x-limits for the scope (tropospheric percentiles)
xlims = {}
for pname in "ABC":
    arr = np.array([c for c in curves[pname] if len(c)])
    sel = P >= 125
    lo, hi = np.percentile(arr[:, sel], [0.5, 99.5])
    pad = 0.1 * (hi - lo)
    xlims[pname] = [round(float(lo - pad), 1), round(float(hi + pad), 1)]

payload = {
    "P": [float(x) for x in P],
    "player": player,
    "map": mappts,
    "classes": labels,
    "xlims": xlims,
    "years": [int(years[pop["A"]].min()), int(years[pop["A"]].max())],
    "npop": {p: int(len(pop[p])) for p in "ABC"},
    "ntc": int(len(pop["A"])),
}
js = json.dumps(payload, separators=(",", ":"))
open(f"{SCRATCH}/artifact_data.json", "w").write(js)
print(f"payload: {len(js)/1e6:.2f} MB; years {payload['years']}")
for pname in "ABC":
    print(pname, labels[pname][:4])
