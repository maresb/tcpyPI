"""Build the per-profile results parquet and print population statistics."""

import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("ERA5_SCRATCH", str(_Path(__file__).resolve().parent / "work"))
_SRC = str(_Path(__file__).resolve().parents[1] / "src")

import numpy as np
import pandas as pd



d = np.load(f"{SCRATCH}/profiles_converted.npz")
res = np.load(f"{SCRATCH}/buoyancy_scan_results.npz")
sc = res["scalar"]
names = [str(x) for x in res["scalar_names"]]
ix = {n: i for i, n in enumerate(names)}
N = sc.shape[0]

iflag = sc[:, ix["iflag"]].astype(np.int64)
n = np.where(np.isnan(sc[:, ix["n"]]), 0, sc[:, ix["n"]]).astype(np.int64)
first_pos = sc[:, ix["first_pos"]].astype(np.int64)

# topology string: strictly alternating -> determined by (first sign, n)
topo = np.empty(N, dtype=object)
for i in range(N):
    if iflag[i] != 1:
        topo[i] = ""
        continue
    c0 = "+" if first_pos[i] == 1 else "-"
    s = ("+-" if c0 == "+" else "-+") * ((n[i] + 1) // 2)
    topo[i] = s[: n[i]]

# ragged list columns, truncated to each row's n
cross = res["cross_lnp"]
rabs = res["region_abs"]
ps = res["psums"]
crossings, regions, partials = [], [], []
for i in range(N):
    k = n[i]
    if iflag[i] != 1 or k == 0:
        crossings.append([]); regions.append([]); partials.append([])
    else:
        crossings.append(cross[i, : k - 1].tolist())
        regions.append(rabs[i, : k].tolist())
        partials.append(ps[i, : k].tolist())

df = pd.DataFrame({
    "latitude": d["lat"], "longitude": d["lon"],
    "time": d["time"].astype("datetime64[s]"),
    "sst_C": d["sst_C"], "sp_hPa": d["sp_hPa"],
    "iflag": iflag,
    "topology": topo,
    "n_regions": n,
    "crossings_lnp": crossings,
    "region_cape_abs": regions,
    "partial_sums": partials,
    "argmax_partial": sc[:, ix["argmax"]].astype(np.int64),
    "E_top": sc[:, ix["E_top"]], "E_max": sc[:, ix["E_max"]],
    "E_reach_strict": sc[:, ix["E_reach"]], "E_first": sc[:, ix["E_first"]],
    "E_reach_lfc": sc[:, ix["E_reach_lfc"]],
    "LNB_top_hPa": sc[:, ix["LNB_top"]], "LNB_max_hPa": sc[:, ix["LNB_max"]],
    "LNB_reach_strict_hPa": sc[:, ix["LNB_reach"]],
    "LNB_reach_lfc_hPa": sc[:, ix["LNB_reach_lfc"]],
    "LNB_first_hPa": sc[:, ix["LNB_first"]],
    "clipped_top": sc[:, ix["clipped"]].astype(np.float64),
    "clamped_E_top": sc[:, ix["clamped"]].astype(np.float64),
    "lcl_hPa": sc[:, ix["lcl_hPa"]],
    "b_lowest_K": sc[:, ix["b_low"]], "b_top_K": sc[:, ix["b_top"]],
    "subsurface_launch": d["sp_hPa"] < 1000.0,
})
df.to_parquet(f"{SCRATCH}/buoyancy_topology_4000h.parquet", index=False)
print(f"wrote buoyancy_topology_4000h.parquet ({len(df)} rows)\n")

# ---------------- statistics ----------------
ok = iflag == 1
print(f"iflag: ok={ok.sum()}  improper-parcel={(iflag==0).sum()}  "
      f"entropy-nonconv={(iflag==2).sum()}")
print(f"subsurface launch (sp<1000): {(d['sp_hPa']<1000).sum()} "
      f"({100*(d['sp_hPa']<1000).mean():.1f}%)\n")

v = df[ok]
print("=== topology distribution (n = number of sign regions) ===")
tc = v.groupby("n_regions").size()
for k, c in tc.items():
    print(f"  n={k:2d}: {c:7d}  ({100*c/len(v):6.2f}%)")
print(f"  multi-crossing (n>=3): {(v.n_regions>=3).sum()} "
      f"({100*(v.n_regions>=3).mean():.2f}%)")
print("  most common topologies:")
print(v.topology.value_counts().head(8).to_string())

print("\n=== convention disagreements (J/kg) ===")
for a, b in [("E_top", "E_max"), ("E_reach_lfc", "E_max"), ("E_first", "E_max"),
             ("E_top", "E_reach_lfc"), ("E_reach_strict", "E_max")]:
    dd = (v[a] - v[b]).abs()
    print(f"  |{a}-{b}|: >0.01: {(dd>0.01).sum():6d} ({100*(dd>0.01).mean():5.2f}%)  "
          f">10: {(dd>10).sum():6d} ({100*(dd>10).mean():5.2f}%)  "
          f"max={dd.max():9.2f}  mean={dd.mean():7.3f}")

pos = v[v.E_max > 0]
print(f"\nprofiles with positive max-work CAPE: {len(pos)} ({100*len(pos)/len(v):.1f}%)")
dd = (pos.E_top - pos.E_max).abs()
rel = dd / pos.E_max
print(f"  among those, E_top disagrees >1%: {(rel>0.01).sum()} ({100*(rel>0.01).mean():.2f}%), "
      f">50%: {(rel>0.5).sum()} ({100*(rel>0.5).mean():.2f}%)")
print(f"  E_top clamped to 0 while E_max>0: {((pos.E_top==0)).sum()} "
      f"({100*(pos.E_top==0).mean():.2f}%)  (the issue-77 configuration)")
lnbd = (pos.LNB_top_hPa - pos.LNB_max_hPa).abs()
print(f"  LNB_top vs LNB_max differ >50 hPa: {(lnbd>50).sum()} ({100*(lnbd>50).mean():.2f}%)")

print(f"\nclamped E_top (raw signed integral < 0): {(v.clamped_E_top==1).sum()} "
      f"({100*(v.clamped_E_top==1).mean():.2f}%)")
print(f"clipped at profile top (b>0 at {70:.0f} hPa): {(v.clipped_top==1).sum()} "
      f"({100*(v.clipped_top==1).mean():.2f}%)")
print(f"argmax=-1 (all partial sums <=0): {(v.argmax_partial==-1).sum()} "
      f"({100*(v.argmax_partial==-1).mean():.2f}%)")

oc = v[~v.subsurface_launch]
print(f"\n=== same stats, ocean-valid launches only (sp>=1000; {len(oc)} rows) ===")
print(f"  multi-crossing (n>=3): {100*(oc.n_regions>=3).mean():.2f}%")
p2 = oc[oc.E_max > 0]
print(f"  E_max>0: {100*len(p2)/len(oc):.1f}%;  E_top==0 among them: "
      f"{100*(p2.E_top==0).mean():.2f}%;  |E_top-E_max|>10 J/kg: "
      f"{100*((p2.E_top-p2.E_max).abs()>10).mean():.2f}%")
