"""Cross-validate the scanner against the real cape() implementations:
E_top vs legacy (modernize_eg) cape(), E_max vs max-work (lnb-max-work) cape().
"""

import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("ERA5_SCRATCH", str(_Path(__file__).resolve().parent / "work"))
_SRC = str(_Path(__file__).resolve().parents[1] / "src")

import importlib.util
import sys

import numpy as np


OLD_SCRATCH = _os.environ.get("LEGACY_TREE", SCRATCH)


def load_pkg(name, src_root):
    spec = importlib.util.spec_from_file_location(
        name, f"{src_root}/tcpyPI/__init__.py",
        submodule_search_locations=[f"{src_root}/tcpyPI"])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # make relative imports resolve under the alias
    sys.path.insert(0, src_root)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


d = np.load(f"{SCRATCH}/profiles_converted.npz")
res = np.load(f"{SCRATCH}/buoyancy_scan_results.npz")
sc = res["scalar"]
names = list(res["scalar_names"])
ix = {n: i for i, n in enumerate(names)}

P = d["P"]
rng = np.random.default_rng(7)
ok_rows = np.where(sc[:, ix["iflag"]] == 1)[0]
sel = rng.choice(ok_rows, size=400, replace=False)
# ensure multi-region rows are represented
multi = ok_rows[sc[ok_rows, ix["n"]] >= 4]
if len(multi):
    sel = np.concatenate([sel, rng.choice(multi, size=min(200, len(multi)), replace=False)])

# import the NEW (max-work) cape from the lnb-max-work checkout
sys.path.insert(0, _SRC)
from tcpyPI.pi import cape as cape_new  # noqa: E402

# legacy cape via the archived modernize_eg tree (pure-python instrumented copy)
sys.path.insert(0, OLD_SCRATCH)
from gmap import cape_instr as cape_legacy  # noqa: E402  (pcmin convention)

bad_top = bad_max = 0
worst_top = worst_max = 0.0
for i in sel:
    T = d["TC"][i] + 273.15
    R = d["R"][i] * 0.001
    out_new = cape_new(T[0], R[0], P[0], T, R, P, 0, 50, 1)
    o_leg = cape_legacy(T[0], R[0], P[0], T, R, P, 0, 50)
    e_max_scan = sc[i, ix["E_max"]]
    e_top_scan = sc[i, ix["E_top"]]
    dmax = abs(out_new[0] - e_max_scan)
    dtop = abs(o_leg["CAPED"] - e_top_scan)
    worst_max = max(worst_max, dmax)
    worst_top = max(worst_top, dtop)
    if dmax > 1e-8 * max(1.0, e_max_scan):
        bad_max += 1
    if dtop > 1e-8 * max(1.0, e_top_scan):
        bad_top += 1

print(f"checked {len(sel)} profiles "
      f"({int((sc[sel, ix['n']] >= 4).sum())} with n>=4 regions)")
print(f"E_max vs new cape():   mismatches {bad_max}, worst |d| = {worst_max:.3e}")
print(f"E_top vs legacy cape(): mismatches {bad_top}, worst |d| = {worst_top:.3e}")
