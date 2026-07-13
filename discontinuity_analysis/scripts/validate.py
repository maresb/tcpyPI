"""Full-sample validation of max-W CAPE (rescue removed) vs committed outputs.

Usage: python validate.py <tree>   where <tree> is 'branch' or 'branch_fix'.
Runs pi() over every column/month of the committed 2024 sample and writes
<tree>_pi.npz.
"""

import os
import sys

SCRATCH = os.path.dirname(os.path.abspath(__file__))
tree = sys.argv[1]
sys.path.insert(0, os.path.join(SCRATCH, tree, "src"))

import numpy as np
import xarray as xr
from tcpyPI import pi

ds = xr.open_dataset(os.path.join(SCRATCH, "sample_data.nc"))
p = ds["p"].values.astype(float)

sst = ds["sst"].values  # (month, lat, lon) or similar
msl = ds["msl"].values
t = ds["t"].values
r = ds["r"].values

print("dims:", ds["sst"].dims, sst.shape, "t:", ds["t"].dims, t.shape)

nm, ny, nx = sst.shape
VMAX = np.full((nm, ny, nx), np.nan)
PMIN = np.full((nm, ny, nx), np.nan)
IFL = np.zeros((nm, ny, nx), dtype=np.int64)
TO = np.full((nm, ny, nx), np.nan)
OTL = np.full((nm, ny, nx), np.nan)

for m in range(nm):
    for j in range(ny):
        for i in range(nx):
            if np.isnan(sst[m, j, i]) or np.isnan(msl[m, j, i]):
                IFL[m, j, i] = 3
                continue
            out = pi(
                sst[m, j, i], msl[m, j, i], p,
                t[m, :, j, i].astype(float), r[m, :, j, i].astype(float),
                CKCD=0.9, ascent_flag=0, diss_flag=1, V_reduc=0.8,
                ptop=50, miss_handle=1,
            )
            VMAX[m, j, i], PMIN[m, j, i], IFL[m, j, i], TO[m, j, i], OTL[m, j, i] = out

np.savez(os.path.join(SCRATCH, f"{tree}_pi.npz"),
         VMAX=VMAX, PMIN=PMIN, IFL=IFL, TO=TO, OTL=OTL)
print(f"saved {tree}_pi.npz; IFL counts:",
      {int(k): int(v) for k, v in zip(*np.unique(IFL, return_counts=True))})
