"""Fine scan of g(PM) to locate and classify discontinuities."""

import numpy as np
from gmap import gmap, CAPEA, P_FULL

# coarse-to-fine: scan a wide window first
pm_grid = np.arange(945.0, 958.0, 0.002)
rows = []
for pm in pm_grid:
    o = gmap(pm)
    rows.append((pm, o["PNEW"], o["CAPEM"], o["CAPEMS"], o["INB_M"], o["INB_S"],
                 o["TO"], o["CAT_raw"], o["TVRDIF_M"][18], o["TVRDIF_M"][10],
                 o["TVRDIF_M"][11], o["PLCL_M"]))
arr = np.array(rows)

pm, g, capem, capems, inbm, inbs = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
to, catraw, tv18, tv10, tv11, plclm = arr[:, 6], arr[:, 7], arr[:, 8], arr[:, 9], arr[:, 10], arr[:, 11]

# find jumps in g
dg = np.abs(np.diff(g))
jump_idx = np.where(dg > 10 * np.median(dg) + 1e-6)[0]
print("=== discontinuities of g on [945, 958] (grid step 0.002) ===")
for i in jump_idx:
    print(
        f"PM in ({pm[i]:.4f}, {pm[i+1]:.4f}): g jumps {g[i]:.6f} -> {g[i+1]:.6f} "
        f"(|dg|={dg[i]:.4f});  INB_M {int(inbm[i])}->{int(inbm[i+1])}, "
        f"INB_S {int(inbs[i])}->{int(inbs[i+1])}, "
        f"CAPEM {capem[i]:.4f}->{capem[i+1]:.4f}, "
        f"TVRDIF_M[18] {tv18[i]:+.2e}->{tv18[i+1]:+.2e}, "
        f"PLCL {plclm[i]:.4f}->{plclm[i+1]:.4f}"
    )

print()
print("=== slope of g away from jumps ===")
mask = np.ones(len(pm), bool)
for i in jump_idx:
    mask[max(0, i - 2) : i + 3] = False
sl = np.diff(g)[mask[:-1] & mask[1:]] / 0.002
print(f"median slope {np.median(sl):.6f}, max |slope| {np.max(np.abs(sl)):.6f}")

# where does the diagonal fall?
print()
print("=== fixed-point analysis ===")
F = g - pm
sign_changes = np.where(np.diff(np.sign(F)) != 0)[0]
for i in sign_changes:
    kind = "JUMP (no root)" if i in jump_idx else "continuous crossing (root)"
    print(f"F=g(PM)-PM changes sign in ({pm[i]:.4f},{pm[i+1]:.4f}): F {F[i]:+.4f}->{F[i+1]:+.4f}  [{kind}]")

# raw signed integral in the INB=18 phase: is the clamp active?
o = gmap(950.6533454438)
print()
print("=== at cycle point A=950.6533 (INB_M=18 phase) ===")
print(f"PA={o['PA_M']:.4f} NA={o['NA_M']:.4f} PAT={o['PAT_M']:.4f} "
      f"raw signed integral={o['PA_M'] + o['PAT_M'] - o['NA_M']:.4f} -> CAPEM={o['CAPEM']:.4f}")
print("TVRDIF_M profile (levels 8..20):")
for j in range(8, 21):
    print(f"  j={j:2d} P={P_FULL[j]:6.1f} hPa  TVRDIF={o['TVRDIF_M'][j]:+.6f} K")
o2 = gmap(951.2790079839)
print()
print("=== at cycle point B=951.2790 (INB_M=10 phase) ===")
print(f"PA={o2['PA_M']:.4f} NA={o2['NA_M']:.4f} PAT={o2['PAT_M']:.4f} CAPEM={o2['CAPEM']:.4f}")
print(f"TVRDIF_M[18] = {o2['TVRDIF_M'][18]:+.6f} K")
