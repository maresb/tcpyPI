"""Assemble the JSON payload for the sounding-scope artifact.

The per-profile statistics table covers the FULL population (every column,
binary-packed little-endian, gzipped, base64) so that counts, class shares,
densities and the dVMAX scatters are exact under any marginalization. The
flip-book player carries curves for a ~20k stratified sample (curves are the
expensive part; the sample only has to be dense enough to animate).
"""

import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("ERA5_SCRATCH", str(_Path(__file__).resolve().parent / "work"))
_SRC = str(_Path(__file__).resolve().parents[1] / "src")

import base64
import gzip
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
_pc = np.load(f"{SCRATCH}/pi_conv_results.npz")
PI = {k: _pc[k] for k in ("VMAX", "PMIN", "IFL")}  # (n, 4): top,max,first,reach
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
pool = set(rng.choice(pop["A"], 14000, replace=False).tolist())
# force-include the wild legacy non-convergence columns (IFL_top=2, IFL_max=1)
legfail_rows = np.where((PI["IFL"][:, 0] == 2) & (PI["IFL"][:, 1] == 1) & mA)[0]
pool.update(legfail_rows.tolist())
print("legacy-failure columns force-included:", len(legfail_rows))
for pname in "ABC":
    for t in tables[pname]:
        rows = pop[pname][allstrs[pname] == t]
        take = min(600, len(rows))
        pool.update(rng.choice(rows, take, replace=False).tolist())
pool = np.array(sorted(pool))
print("player pool:", len(pool))

# curve rectangles: (3, npool, nlvl) int16 in centi-J/kg (0.01 precision,
# same as the old JSON rounding); zeros where B/C undefined -- gzip erases
# the constant runs, and the JS side rebuilds [] from the class sentinel
npool = len(pool)
curvebuf = np.zeros((3, npool, nlvl), np.float64)
b = np.empty(nlvl)
for k, i in enumerate(pool):
    T = d["TC"][i, :nlvl] + 273.15
    R = d["R"][i, :nlvl] * 0.001
    buoyancy(T[0], R[0], P[0], T, R, P, nlvl, b)
    curvebuf[0, k] = b
    if mBC[i]:
        PP = min(z["PM"][i], 1000.0)
        MSL = d["sp_hPa"][i]
        RP = EPS * R[0] * MSL / (PP * (EPS + R[0]) - R[0] * MSL)
        buoyancy(T[0], RP, PP, T, R, P, nlvl, b)
        curvebuf[1, k] = b
        ES0 = utilities.es_cc(d["sst_C"][i])
        buoyancy(d["sst_C"][i] + 273.15, utilities.rv(ES0, PP), PP, T, R, P, nlvl, b)
        curvebuf[2, k] = b
curves = {pn: curvebuf[j] for j, pn in enumerate("ABC")}
curve_i2 = np.round(curvebuf * 100).astype("<i2")
assert np.abs(curvebuf).max() < 327, "curve exceeds int16 centi-range"

lonw = np.where(d["lon"][pool] > 180, d["lon"][pool] - 360, d["lon"][pool])
player = {
    "lat": [round(float(x), 2) for x in d["lat"][pool]],
    "lon": [round(float(x), 2) for x in lonw],
    "year": [int(x) for x in years[pool]],
    "doy": [int(x) for x in doy[pool]],
    "hour": [int(x) for x in ((d["time"][pool] // 3600) % 24)],
    "sst": [round(float(x), 1) for x in d["sst_C"][pool]],
}
for pname in "ABC":
    m = popmask[pname]
    o = z[pname]
    player[f"c{pname}"] = [int(cls(pname, i)) for i in pool]
    for tag, col in (("Etop", 2), ("Emax", 3), ("Ereach", 5)):
        player[f"{tag}{pname}"] = [int(round(o[i, col])) if m[i] else 0 for i in pool]
    for tag, col in (("Ltop", 8), ("Lmax", 9), ("Lreach", 11)):
        player[f"{tag}{pname}"] = [round(float(o[i, col]), 1) if m[i] else 0 for i in pool]
    player[f"topo{pname}"] = [topo_str(o[i, 0], o[i, 1]) if m[i] else ""
                              for i in pool]
# PI per convention for the column (ptop=50 tcpyPI convention; -1 = missing)
for k, tag in zip((0, 1, 3), ("top", "max", "reach")):
    player[f"V{tag}"] = [round(float(PI["VMAX"][i, k]), 1)
                         if np.isfinite(PI["VMAX"][i, k]) else -1 for i in pool]
player["Ptop"] = [round(float(PI["PMIN"][i, 0]), 1)
                  if np.isfinite(PI["PMIN"][i, 0]) else -1 for i in pool]
player["Pmax"] = [round(float(PI["PMIN"][i, 1]), 1)
                  if np.isfinite(PI["PMIN"][i, 1]) else -1 for i in pool]
player["legfail"] = [1 if (PI["IFL"][i, 0] == 2 and PI["IFL"][i, 1] == 1) else 0
                     for i in pool]

# ---- statistics table: FULL population, binary-packed ----
N = len(mA)
lon_all = np.where(d["lon"] > 180, d["lon"] - 360, d["lon"])
hours_all = (d["time"] // 3600) % 24

clsv, gv = {}, {}
for pname in "ABC":
    o = z[pname]
    ci = np.full(N, -1, np.int8)
    lut = tables[pname]
    ci[pop[pname]] = np.array([lut.get(s, 7) for s in allstrs[pname]], np.int8)
    clsv[pname] = ci
    # disagreement bitmask: 1 = clamp-flip (E_top=0 while E_max>0);
    # 2 = |E_top-E_max| > 1 J/kg; 4 = |E_reach-E_max| > 1 J/kg
    with np.errstate(invalid="ignore"):
        g = ((o[:, 2] == 0.0) & (o[:, 3] > 0.0)).astype(np.uint8)
        g |= (np.abs(o[:, 2] - o[:, 3]) > 1.0).astype(np.uint8) << 1
        g |= (np.abs(o[:, 5] - o[:, 3]) > 1.0).astype(np.uint8) << 2
    g[~popmask[pname]] = 0
    gv[pname] = g

# VMAX per convention, deci-m/s; -10 encodes missing (decodes to -1.0)
vq = {}
for tag, k in (("top", 0), ("max", 1), ("reach", 3)):
    V = PI["VMAX"][:, k]
    vq[tag] = np.where(np.isfinite(V), np.round(V * 10), -10).astype("<i2")
legf = (PI["IFL"][:, 0] == 2) & (PI["IFL"][:, 1] == 1) & mA
with np.errstate(invalid="ignore"):
    dv_all = np.abs(PI["VMAX"][:, 0] - PI["VMAX"][:, 1])
    big = np.isfinite(dv_all) & (dv_all > 1.0)
gpi_all = np.where(legf, 1, np.where(big, 2, 0)).astype(np.uint8)

# int16 columns first so every offset stays 2-byte aligned
bincols = [
    ("lat", "i2", 100, np.round(d["lat"] * 100).astype("<i2")),
    ("lon", "i2", 100, np.round(lon_all * 100).astype("<i2")),
    ("doy", "i2", 1, doy.astype("<i2")),
    ("year", "i2", 1, years.astype("<i2")),
    ("Vtop", "i2", 10, vq["top"]),
    ("Vmax", "i2", 10, vq["max"]),
    ("Vreach", "i2", 10, vq["reach"]),
    ("hour", "u1", 1, hours_all.astype(np.uint8)),
    ("cA", "i1", 1, clsv["A"]), ("cB", "i1", 1, clsv["B"]),
    ("cC", "i1", 1, clsv["C"]),
    ("gA", "u1", 1, gv["A"]), ("gB", "u1", 1, gv["B"]),
    ("gC", "u1", 1, gv["C"]),
    ("gpi", "u1", 1, gpi_all),
]
blob = b"".join(a.tobytes() for _, _, _, a in bincols)
mapbin = {
    "n": int(N),
    "cols": [[name, dt, sc] for name, dt, sc, _ in bincols],
    "b64": base64.b64encode(gzip.compress(blob, 9)).decode(),
}
curvebin = {
    "n": int(npool), "nlvl": int(nlvl), "scale": 100,
    "b64": base64.b64encode(gzip.compress(curve_i2.tobytes(), 9)).decode(),
}
print(f"table blob: {len(blob)/1e6:.1f} MB raw -> "
      f"{len(mapbin['b64'])/1e6:.1f} MB b64gz; "
      f"curves: {curve_i2.nbytes/1e6:.1f} MB raw -> "
      f"{len(curvebin['b64'])/1e6:.1f} MB b64gz")

# per-parcel x-limits for the scope (tropospheric percentiles)
# x-limits: lower bound = the most negative any curve goes BEFORE its last
# positive level (interior dips only; the terminal stratospheric plunge never
# returns to positive and should not set the range), with a 10% margin.
xlims = {}
poolBC = mBC[pool]
for pname in "ABC":
    arr = curves[pname] if pname == "A" else curves[pname][poolBC]
    lo = 0.0
    for c in arr:
        pos = np.where(c > 0)[0]
        if len(pos) == 0:
            continue
        m = c[: pos[-1] + 1].min()
        if m < lo:
            lo = m
    hi = float(np.percentile(arr, 99.9))
    xlims[pname] = [round(float(lo * 1.1), 1), round(hi * 1.05 + 0.5, 1)]

payload = {
    "P": [float(x) for x in P],
    "player": player,
    "mapbin": mapbin,
    "curvebin": curvebin,
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
