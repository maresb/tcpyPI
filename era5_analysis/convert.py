"""parquet -> npz: units and ordering for the tcpyPI-convention scanner.

Input arrays are top-down on the 37 standard ERA5 pressure levels.
Output: bottom-up (decreasing pressure) float64 arrays, T in Celsius,
mixing ratio r = q/(1-q) in g/kg, MSL-ish surface pressure in hPa.
"""

import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("ERA5_SCRATCH", str(_Path(__file__).resolve().parent / "work"))
_SRC = str(_Path(__file__).resolve().parents[1] / "src")

import numpy as np
import pandas as pd



# standard ERA5 37 pressure levels, top-down (hPa)
PLEV_TD = np.array(
    [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
     300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
     875, 900, 925, 950, 975, 1000], dtype=np.float64)

df = pd.read_parquet(f"{SCRATCH}/profiles.parquet")
n = len(df)
print("profiles:", n)

t = np.stack(df["temperature_K"].to_numpy()).astype(np.float64)   # (n, 37) top-down
q = np.stack(df["specific_humidity_kgkg"].to_numpy()).astype(np.float64)

# sanity: top-down ordering means q tiny at index 0
assert np.nanmedian(q[:, 0]) < 1e-4 < np.nanmedian(q[:, -1]), "ordering assumption violated"

TC = t[:, ::-1] - 273.15                    # bottom-up, Celsius
r = q[:, ::-1]
r = r / (1.0 - r) * 1000.0                  # specific humidity -> mixing ratio, g/kg
P = PLEV_TD[::-1].copy()                    # bottom-up: 1000 ... 1

np.savez(
    f"{SCRATCH}/profiles_converted.npz",
    P=P,
    TC=np.ascontiguousarray(TC),
    R=np.ascontiguousarray(r),
    sst_C=df["sst_K"].to_numpy().astype(np.float64) - 273.15,
    sp_hPa=df["sp_Pa"].to_numpy().astype(np.float64) / 100.0,
    lat=df["latitude"].to_numpy().astype(np.float64),
    lon=df["longitude"].to_numpy().astype(np.float64),
    time=df["time"].to_numpy().astype("datetime64[s]").astype(np.int64),
)
print("saved profiles_converted.npz")
print("NaN rows (any T NaN):", int(np.isnan(TC).any(axis=1).sum()))
print("sp_hPa range:", df["sp_Pa"].min() / 100, df["sp_Pa"].max() / 100)
