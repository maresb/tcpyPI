# ERA5 buoyancy-topology analysis

Population study of lifted-parcel buoyancy topologies and LNB-convention
sensitivity over 569,478 random ERA5 ocean columns (1980–2024), for the three
parcels used by the tcpyPI potential-intensity calculation. Companion to
`discontinuity_analysis/` (the issue #77 root-cause study): this directory
measures, at climatological scale, how often the multi-crossing buoyancy
structures behind that failure occur and how much the LNB definition matters.

## Data

Input sample (public ARCO-ERA5, no credentials needed to rebuild):

    s3://gridded-data-dev/bmares/2026-07-18-era5-tcpypi-sample/
        tcpypi_inputs_1000h_seed20260718.parquet   (~40 MB,  ~142k columns)
        tcpypi_inputs_4000h_seed20260719.parquet   (~154 MB, 569,478 columns)

Rebuild from scratch with `build_tcpypi_era5_sample.py` (see its docstring for
the exact commands and the conversion recipe; schema metadata inside the
parquet carries units, levels, sampling parameters, and caveats).

## Pipeline

All scripts read/write a work directory: `export ERA5_SCRATCH=...` (default:
`./work`). They import tcpyPI from this repository's `src/` (numba required;
scientific results in the writeups used the max-work cape() of the
`lnb-max-work` branch — the scan scripts implement both conventions
internally, so the checked-out kernel only matters for `validate_scan.py`
and `pi_conv.py` cross-checks).

1. `convert.py` — parquet -> `profiles_converted.npz` (units, bottom-up
   level order, r = q/(1-q)); expects the parquet at
   `$ERA5_SCRATCH/profiles.parquet`.
2. `buoyancy_scan.py` — environmental-parcel topology/crossings/partial-sums
   scan under four LNB conventions -> `buoyancy_scan_results.npz`.
3. `validate_scan.py` — cross-check E_top/E_max against real legacy/max-work
   `cape()` on random profiles.
4. `export_stats.py` — per-profile results parquet + population statistics.
5. `pi_conv.py` — full pi() under four LNB conventions (2.28M solves)
   -> `pi_conv_results.npz` (+ flat parquet); includes the wild-population
   legacy non-convergence census.
6. `scan_final.py` — the definitive three-parcel scan (A environmental,
   B eyewall, C saturated core at each column's converged max-work P_M),
   perturbation-convention topology labels, candidate-based conventions
   -> `parcel_topology_final.npz`.
7. `scan_pi_parcels.py` — earlier two-parcel scan + hybrid-vs-ln(p)
   quadrature comparison (kept for the quadrature numbers).
8. `gen_curves.py` + `plot_curves.py` — spaghetti figures per topology class
   (`figures/curves_parcel_{A,B,C}.png`).
9. `prep_artifact_data.py` + `sounding_scope_template.html` — the interactive
   "Sounding Scope" flipbook (inject `artifact_data.json` into the template's
   `__DATA__` placeholder to obtain a single self-contained HTML file).

## Headline results (TC-relevant subset: SST >= 26 C, sp >= 1000 hPa)

- Topology frequencies (perturbation convention: leading sign taken
  infinitesimally above launch): parcel A is a rich mixture (`+-+-` 32%,
  `-+-` 17%, `+-+-+-` 16%, `+-` 14%, ...); parcel B is 84% `+-` with a 16%
  multi-hump tail; parcel C is 98.4% `+-` plus 1.24% `+` (buoyant at the
  70 hPa retained-column top).
- LNB-convention sensitivity is inversely proportional to parcel energy:
  E_top clamps to 0 while E_max > 0 on 15.4% (A), 1.75% (B), 0.00% (C) of
  columns; |E_top - E_max| > 1 J/kg on 13.1% / 5.7% / 0.0%.
- Full-pi() convention comparison over 476k columns (SST > 5 C):
  legacy topmost-level LNB fails to converge on 8 columns (~1 in 60,000;
  median max-work VMAX 64 m/s — real storm environments returned missing);
  max-work fails on 0; first-crossing and lifted-ballistic conventions fail
  ~20x more often than legacy. Where PI matters (SST >= 26), top vs max
  agree to p99 = 0.53 m/s.
- Column ceiling: the topology chain (`scan_final.py` onward) retains
  levels down to 30 hPa (ptop=20) -- the minimal ceiling observing every
  buoyancy crossing in the sample. Highest crossing: 47.6 hPa (saturated
  core parcel, interpolated between the 50 and 30 hPa nodes; parcel B max
  65.0, parcel A max 87.5). At this ceiling zero profiles clip AND zero
  entropy solves fail (retaining the 30-20 hPa layer caused 21 failures).
  tcpyPI's own ptop=50 convention (used by the pi()/CAPE scans above)
  truncates at 70 hPa and clips 1.24% of TC-relevant core parcels, whose
  true crossings sit at median 68.6 hPa. The solver's `ENEW > P-1` guard
  prevents extension beyond ~10-20 hPa without modification. Buoyancy-plot
  x-ranges are set by the deepest interior dip (the most negative value a
  curve reaches before its last positive level) plus 10% — the terminal
  stratospheric plunge never sets the range.
- Strict from-rest ballistic CAPE is degenerate for surface-launched
  environmental parcels (b(launch) = 0 by construction and surface CIN or
  neutral layering is universal); the lifted-to-LFC variant is the
  meaningful ballistic convention.

Outputs referenced above (results parquets, npz files) are regenerated by
the pipeline; the two published to date live next to the input sample on S3.
