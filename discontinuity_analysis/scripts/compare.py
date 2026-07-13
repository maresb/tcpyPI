"""Compare branch (pcmin CAPE + rescue) vs branch_fix (max-W CAPE, no rescue)."""

import numpy as np
import xarray as xr

a = np.load("branch_pi.npz")      # baseline: branch as merged
b = np.load("branch_fix_pi.npz")  # max-W CAPE, rescue deleted

both1 = (a["IFL"] == 1) & (b["IFL"] == 1)
n_both = int(both1.sum())

# flag agreement
flag_diff = np.argwhere(a["IFL"] != b["IFL"])
print(f"columns with IFL=1 in both: {n_both}")
print(f"columns where IFL differs: {len(flag_diff)}")
for idx in flag_diff[:20]:
    m, j, i = idx
    print(f"  (m={m},j={j},i={i}): IFL {a['IFL'][m,j,i]} -> {b['IFL'][m,j,i]}, "
          f"VMAX {a['VMAX'][m,j,i]:.4f} -> {b['VMAX'][m,j,i]:.4f}")

for var in ["VMAX", "PMIN", "TO", "OTL"]:
    x, y = a[var], b[var]
    d = np.abs(x - y)[both1]
    ident = int((d == 0).sum())
    print(f"{var}: bit-identical {ident}/{n_both} "
          f"({100*ident/n_both:.2f}%), max|d|={np.nanmax(d):.6g}, "
          f"n(|d|>1e-8)={int((d>1e-8).sum())}, n(|d|>0.1)={int((d>0.1).sum())}")

# which columns changed materially, and were they rescue cases?
d_vmax = np.abs(a["VMAX"] - b["VMAX"])
chg = np.argwhere(both1 & (d_vmax > 1e-8))
print(f"\ncolumns with VMAX change > 1e-8: {len(chg)}")
for idx in chg[:30]:
    m, j, i = idx
    print(f"  (m={m},j={j},i={i}): VMAX {a['VMAX'][m,j,i]:.6f}->{b['VMAX'][m,j,i]:.6f} "
          f"TO {a['TO'][m,j,i]:.3f}->{b['TO'][m,j,i]:.3f} "
          f"OTL {a['OTL'][m,j,i]:.2f}->{b['OTL'][m,j,i]:.2f}")

# MATLAB reference comparison for both
ml = xr.open_dataset("matlab_ref.nc")
print("\nMATLAB ref vars:", list(ml.data_vars))
mv = ml["Vmax"].values
mp = ml["Pmin"].values
mfl = ml["PI_flag"].values if "PI_flag" in ml else None
print("MATLAB shapes:", mv.shape)

for name, r in [("branch", a), ("fix", b)]:
    ok = (r["IFL"] == 1) & np.isfinite(mv) & (mfl == 1 if mfl is not None else True)
    dv = np.abs(r["VMAX"] - mv)[ok]
    dp = np.abs(r["PMIN"] - mp)[ok]
    print(f"{name}: vs MATLAB over {int(ok.sum())} cols: "
          f"VMAX max|d|={np.nanmax(dv):.6f} PMIN max|d|={np.nanmax(dp):.6f}")
