"""Regenerate the 'healthy column' case with a representative interior-LNB
column (m=0, j=20, i=1). Saves healthy2.npz."""

import os
import sys

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRATCH, "branch", "src"))
sys.path.insert(0, SCRATCH)

import numpy as np
import xarray as xr
from tcpyPI import utilities

from datagen import g_scan, iterate_raw, W_curve

M, J, I = 0, 20, 1

ds = xr.open_dataset(os.path.join(SCRATCH, "sample_data.nc"))
p_s = ds["p"].values.astype(float)
sst = float(ds["sst"].values[M, J, I])
msl = float(ds["msl"].values[M, J, I])
T = utilities.T_Ctok(ds["t"].values[M, :, J, I].astype(float))
R = ds["r"].values[M, :, J, I].astype(float) * 0.001
R[np.isnan(R)] = 0.0
lat = float(ds["lat"].values[J])
lon = float(ds["lon"].values[I])
month = int(ds["month"].values[M]) if "month" in ds else M + 1
print(f"column: month={month} lat={lat} lon={lon} SST={sst:.2f}")

SSTK = utilities.T_Ctok(sst)
ES0 = utilities.es_cc(sst)

save = {}
pms = np.arange(850.0, 1005.0, 0.02)
sc = g_scan(pms, SSTK, msl, p_s, T, R, ES0)
save["c_pm"] = pms
save["c_g"] = sc[:, 0]
st, xs = iterate_raw(SSTK, msl, p_s, T, R, ES0, nmax=60)
save["c_iter"] = xs
print(f"iteration: status={st}, fixed point ~ {xs[-1]:.4f}, n={len(xs)}")

pmc = xs[-1]
PP = min(pmc, 1000.0)
pts, Ws, b, Pt = W_curve(SSTK, utilities.rv(ES0, PP), PP, T, R, p_s)
save["c_W_p"] = pts
save["c_W_W"] = Ws
save["c_b"] = b
save["c_blevels"] = p_s[: int(np.count_nonzero(p_s > 50))]
kmax = int(np.argmax(Ws))
print(f"argmax LNB: {pts[kmax]:.2f} hPa, CAPE={Ws[kmax]:.1f}; "
      f"b sign change: b[{save['c_blevels'][-2]:.0f}]={b[-2]:+.2f}, "
      f"b[{save['c_blevels'][-1]:.0f}]={b[-1]:+.2f}")
np.savez(os.path.join(SCRATCH, "healthy2.npz"), **save)
print("saved healthy2.npz")
