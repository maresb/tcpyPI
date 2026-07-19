#!/usr/bin/env python3
"""Assemble generic tcpyPI input samples from ARCO-ERA5.

What it does
------------
For N random hours in [1980, 2025), it reads four ERA5 fields from the public
Analysis-Ready Cloud-Optimized ERA5 store on GCS (no credentials needed):

    sea_surface_temperature, surface_pressure   (single level)
    temperature, specific_humidity              (37 pressure levels)

For each hour it draws M cosine-weighted (equal-area) random lat/lon points,
snaps them to the 0.25 deg grid, discards land points (NaN SST), and emits one
record per surviving ocean point containing latitude, longitude, time, the two
surface scalars, and the two 37-level vertical profiles. Output is a single
zstd-compressed parquet file whose schema metadata carries the pressure levels,
units, the exact tcpyPI conversion recipe, the sampling parameters, and known
data caveats.

Provenance / vendoring
----------------------
The ARCO-ERA5 URL and the variable set are taken from Climate Central's
attribution pipeline (attribution/hurricanes/aggregate.py, which uses the same
store for hurricane potential-intensity climatology) but are inlined here so
this script stands entirely on its own.

Requirements (all pip-installable, no attribution package):
    pip install numpy pandas pyarrow xarray zarr gcsfs

Reading the ARCO store is anonymous/public; writing a local parquet needs no
credentials. (Uploading the result to S3 is out of scope for this script.)

Reproduce the two published samples
-----------------------------------
    python build_tcpypi_era5_sample.py --n-hours 1000 --seed 20260718 \
        --threads 8 --out tcpypi_inputs_1000h_seed20260718.parquet
    python build_tcpypi_era5_sample.py --n-hours 4000 --seed 20260719 \
        --threads 8 --out tcpypi_inputs_4000h_seed20260719.parquet

Convert a stored record to tcpyPI inputs
----------------------------------------
    SST_C = sst_K - 273.15
    MSL_hPa = sp_Pa / 100.0            # over ocean surface_pressure ~= MSL
    P_hPa   = pressure_level_hPa       # from file metadata (ascending index)
    T_C     = temperature_K - 273.15
    R_gkg   = 1000 * q / (1 - q)       # q = specific_humidity_kgkg
"""

import argparse
import json
import time as _time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

# --- vendored constants (see attribution/hurricanes/aggregate.py) -------------
ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SURFACE_VARS = ("sea_surface_temperature", "surface_pressure")
PROFILE_VARS = ("temperature", "specific_humidity")

# ERA5 0.25 deg grid geometry: latitude[i] = 90 - 0.25*i (i in 0..720),
# longitude[j] = 0.25*j (j in 0..1439).
LAT0, DLAT, NLAT = 90.0, 0.25, 721
DLON, NLON = 0.25, 1440
TIME_START, TIME_STOP = "1980-01-01", "2025-01-01"  # [start, stop)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-hours", type=int, default=1000,
                   help="number of distinct random hours to sample")
    p.add_argument("--n-points", type=int, default=200,
                   help="random lat/lon points drawn per hour (before land filter)")
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--threads", type=int, default=8,
                   help="concurrent hours (I/O bound; each holds ~0.3 GB)")
    p.add_argument("--out", type=str, required=True, help="output .parquet path")
    return p.parse_args()


def open_arco():
    """Open the public ARCO-ERA5 store lazily (anonymous access)."""
    return xr.open_zarr(ARCO_URL, chunks=None, storage_options=dict(token="anon"))


def sample_grid_indices(rng, n):
    """Draw n cosine-weighted (equal-area) points; return nearest grid indices.

    Latitude density proportional to cos(lat) is achieved by lat = arcsin(U),
    U ~ Uniform(-1, 1); longitude is uniform on [0, 360).
    """
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    lon = rng.uniform(0.0, 360.0, size=n)
    lat_idx = np.rint((LAT0 - lat) / DLAT).astype(int).clip(0, NLAT - 1)
    lon_idx = np.rint(lon / DLON).astype(int) % NLON
    return lat_idx, lon_idx


def process_hour(ds, it, timestamp, rng, n_points):
    """Read one hour's global fields, sample points, keep ocean points."""
    t_arr = ds.temperature.isel(time=it).values         # (37, 721, 1440) K
    q_arr = ds.specific_humidity.isel(time=it).values   # (37, 721, 1440) kg/kg
    sp_arr = ds.surface_pressure.isel(time=it).values   # (721, 1440) Pa
    sst_arr = ds.sea_surface_temperature.isel(time=it).values  # (721, 1440) K

    lat_idx, lon_idx = sample_grid_indices(rng, n_points)
    sst_pt = sst_arr[lat_idx, lon_idx]
    keep = np.isfinite(sst_pt)
    lat_idx, lon_idx = lat_idx[keep], lon_idx[keep]
    if lat_idx.size == 0:
        return None

    return dict(
        latitude=(LAT0 - DLAT * lat_idx).astype("float32"),
        longitude=(((DLON * lon_idx + 180.0) % 360.0) - 180.0).astype("float32"),
        time=np.repeat(np.datetime64(timestamp, "ns"), lat_idx.size),
        sst_K=sst_pt[keep].astype("float32"),
        sp_Pa=sp_arr[lat_idx, lon_idx].astype("float32"),
        temperature_K=t_arr[:, lat_idx, lon_idx].T.astype("float32"),        # (n, 37)
        specific_humidity_kgkg=q_arr[:, lat_idx, lon_idx].T.astype("float32"),
    )


def build_metadata(levels, args):
    return {
        b"source": ARCO_URL.encode(),
        b"pressure_level_hPa": json.dumps(levels.tolist()).encode(),
        b"units": json.dumps({
            "sst_K": "K", "sp_Pa": "Pa", "temperature_K": "K",
            "specific_humidity_kgkg": "kg/kg", "pressure_level": "hPa",
            "latitude": "degrees_north", "longitude": "degrees_east (-180..180)",
        }).encode(),
        b"profile_order": (
            b"ascending pressure_level index "
            b"(level[0]=1 hPa ... level[-1]=1000 hPa)"
        ),
        b"tcpyPI_conversion": json.dumps({
            "SST_C": "sst_K - 273.15",
            "MSL_hPa": "sp_Pa / 100.0  (over ocean surface_pressure ~= MSL)",
            "P_hPa": "pressure_level_hPa",
            "T_C": "temperature_K - 273.15",
            "R_gkg": "1000 * q/(1-q)  with q = specific_humidity_kgkg",
        }).encode(),
        b"sampling": json.dumps({
            "n_hours": args.n_hours, "n_points_per_hour": args.n_points,
            "seed": args.seed, "time_window": f"[{TIME_START}, {TIME_STOP})",
            "lat_weighting": "cosine (equal-area via arcsin)",
            "land_filter": "dropped points with NaN sea_surface_temperature",
        }).encode(),
        b"known_caveats": json.dumps({
            "inland_lakes": (
                "ERA5 sea_surface_temperature is defined over large/high-elevation "
                "inland lakes (Victoria ~885 hPa, Titicaca ~633 hPa, Urmia, etc.), so "
                "~0.1% of records are lakes, not open ocean. Filter on sp_Pa "
                "(e.g. >= 95000) if only open ocean is desired."
            ),
            "negative_specific_humidity": (
                "A tiny fraction of stratospheric q values are slightly negative "
                "(~-4e-6 kg/kg), an ERA5 interpolation artifact; negligible for tcpyPI."
            ),
        }).encode(),
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ds = open_arco()
    levels = ds.level.values.astype("int64")  # hPa

    times = pd.DatetimeIndex(ds.time.values)
    valid = np.where((times >= TIME_START) & (times < TIME_STOP))[0]
    chosen = np.sort(rng.choice(valid, size=args.n_hours, replace=False))
    print(f"sampling {args.n_hours} hours x {args.n_points} pts; "
          f"valid hour pool={valid.size}; threads={args.threads}", flush=True)

    # Independent per-hour RNGs => result is identical regardless of thread order.
    hour_rngs = [np.random.default_rng([args.seed, int(it)]) for it in chosen]

    def _run(k):
        it = int(chosen[k])
        t0 = _time.time()
        rec = process_hour(ds, it, times[it], hour_rngs[k], args.n_points)
        kept = 0 if rec is None else rec["latitude"].size
        print(f"  hour {k + 1}/{args.n_hours} {times[it]} "
              f"kept={kept}/{args.n_points} ({_time.time() - t0:.1f}s)", flush=True)
        return rec

    t_start = _time.time()
    if args.threads > 1:
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            recs = list(ex.map(_run, range(args.n_hours)))
    else:
        recs = [_run(k) for k in range(args.n_hours)]
    recs = [r for r in recs if r is not None]

    cat = {key: np.concatenate([r[key] for r in recs]) for key in recs[0]}
    elapsed = _time.time() - t_start
    print(f"total ocean records: {cat['latitude'].size}  ({elapsed:.1f}s, "
          f"{elapsed / args.n_hours:.2f}s/hour)", flush=True)

    table = pa.table({
        "latitude": pa.array(cat["latitude"]),
        "longitude": pa.array(cat["longitude"]),
        "time": pa.array(cat["time"]),
        "sst_K": pa.array(cat["sst_K"]),
        "sp_Pa": pa.array(cat["sp_Pa"]),
        "temperature_K": pa.array(
            list(cat["temperature_K"]), type=pa.list_(pa.float32())
        ),
        "specific_humidity_kgkg": pa.array(
            list(cat["specific_humidity_kgkg"]), type=pa.list_(pa.float32())
        ),
    }).replace_schema_metadata(build_metadata(levels, args))
    pq.write_table(table, args.out, compression="zstd")
    print(f"wrote {args.out}  rows={table.num_rows}", flush=True)


if __name__ == "__main__":
    main()
