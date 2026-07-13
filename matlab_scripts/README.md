# MATLAB scripts

The original Bister & Emanuel (2002) potential-intensity algorithm that pyPI
ports to Python, kept here for reference and validation.

| File | Description |
| --- | --- |
| `pc_min.m` | Kerry Emanuel's `pcmin` algorithm: maximum wind speed and minimum central pressure from a sounding and an SST. Contains the `cape` subfunction. |
| `reference_calculations.m` | Drives `pc_min.m` over the MERRA2 sample data and writes `data/sample_data.nc` (the reference the Python tests compare against). |
| `clock_matpi.m` | Timing/benchmark helper. |
| `run_pc_min.m` | Small driver that runs `pc_min.m` on a few soundings and checks the outputs against the reference values in `data/sample_data.nc`. |

## Running `pc_min.m` with GNU Octave

`pc_min.m` runs unmodified under [GNU Octave](https://octave.org/), so no MATLAB
license is needed to exercise the reference algorithm.

The repository's [pixi](https://pixi.sh) configuration provides an `octave`
environment and a task that runs and validates the code:

```bash
pixi run -e octave pc-min-octave
```

This installs Octave from conda-forge and runs `run_pc_min.m`, which prints the
`PMIN`/`VMAX`/`TO`/`LNB`/`IFL` outputs next to the reference values and exits
non-zero if any of them drift beyond a small tolerance.

To run it directly against an Octave already on your `PATH`:

```bash
octave-cli --no-gui matlab_scripts/run_pc_min.m
```

> Octave is not packaged for Windows (`win-64`) on conda-forge, so the `octave`
> pixi environment is limited to Linux and macOS.
