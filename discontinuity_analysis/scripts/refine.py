"""Discretization/recovery experiment on the PR#77 column.

Reconstruct the environment (T, R as functions of log p) with a monotone cubic
(PCHIP / Fritsch-Carlson) interpolant, resample at 5 hPa from 1000 to 55 hPa
(plus the original coarse levels above), and re-run:
  - the buoyancy profile of the environmental parcel at cycle point A
  - the fixed-point map g on the refined profile (scan + iteration)

Also: the same with plain linear-in-logp resampling (the PWL 'reconstruction'
the trapezoid scheme implicitly commits to), for contrast.
"""

import os
import sys

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRATCH, "branch", "src"))
sys.path.insert(0, SCRATCH)

import numpy as np
from tcpyPI import constants, utilities

from datagen import (P77, TC77, R77, SST77, MSL77, prep, g_scan, iterate_raw,
                     g_eval)
from gmap import cape_instr

EPS = constants.EPS


def pchip_slopes(x, y):
    """Fritsch-Carlson monotone cubic slopes."""
    h = np.diff(x)
    d = np.diff(y) / h
    n = len(x)
    m = np.zeros(n)
    m[0] = d[0]
    m[-1] = d[-1]
    for k in range(1, n - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    return m


def pchip_eval(x, y, xq):
    m = pchip_slopes(x, y)
    idx = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (xq - x[idx]) / h
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t)
    h11 = t * t * (t - 1)
    return (h00 * y[idx] + h10 * h * m[idx] + h01 * y[idx + 1] + h11 * h * m[idx + 1])


def build_refined(P, TC, R, method="pchip", dp=5.0, pmin_fine=55.0):
    """Refined profile: fine levels from P[0] down to pmin_fine, then original."""
    x = np.log(P[::-1])  # increasing in log p ... P descending -> reverse
    tcv = TC[::-1]
    rv_ = R[::-1]
    fine = np.arange(P[0], pmin_fine - 1e-9, -dp)
    fine = np.unique(np.concatenate([fine, P[P >= pmin_fine]]))[::-1]
    keep_above = P[P < pmin_fine]
    Pn = np.concatenate([fine, keep_above])
    xq = np.log(Pn[::-1])
    if method == "pchip":
        tq = pchip_eval(x, tcv, xq)[::-1]
        rq = pchip_eval(x, rv_, xq)[::-1]
    else:
        tq = np.interp(xq, x, tcv)[::-1]
        rq = np.interp(xq, x, rv_)[::-1]
    return Pn, tq, rq


save = {}
for method in ("pchip", "linear"):
    Pn, TCn, Rn = build_refined(P77, TC77, R77, method=method)
    S = prep(SST77, MSL77, TCn, Rn)
    pms = np.arange(946.0, 958.0, 0.005)
    sc = g_scan(pms, S[0], S[1], Pn, S[2], S[3], S[4])
    st, xs = iterate_raw(S[0], S[1], Pn, S[2], S[3], S[4], nmax=200)
    save[f"{method}_pm"] = pms
    save[f"{method}_g"] = sc[:, 0]
    save[f"{method}_capem"] = sc[:, 1]
    save[f"{method}_status"] = st
    save[f"{method}_iters"] = xs
    # env-parcel buoyancy at cycle point A (native cycle value)
    pmA = 950.6533454438
    PP = min(pmA, 1000.0)
    RP = EPS * S[3][0] * S[1] / (PP * (EPS + S[3][0]) - S[3][0] * S[1])
    o = cape_instr(S[2][0], RP, PP, S[2], S[3], Pn, 0, 50)
    N = int(np.count_nonzero(Pn > 50))
    save[f"{method}_b"] = o["TVRDIF"]
    save[f"{method}_blev"] = Pn[:N]
    save[f"{method}_capemA"] = o["CAPED"]
    print(f"{method}: status={st} n_it={len(xs)} final={xs[-1]:.4f} "
          f"CAPEM(A)={o['CAPED']:.3f} nlev={len(Pn)}")

# native for reference
S = prep(SST77, MSL77, TC77, R77)
pmA = 950.6533454438
PP = min(pmA, 1000.0)
RP = EPS * S[3][0] * S[1] / (PP * (EPS + S[3][0]) - S[3][0] * S[1])
o = cape_instr(S[2][0], RP, PP, S[2], S[3], P77, 0, 50)
N = int(np.count_nonzero(P77 > 50))
save["native_b"] = o["TVRDIF"]
save["native_blev"] = P77[:N]
print(f"native: CAPEM(A)={o['CAPED']:.3f}")

np.savez(os.path.join(SCRATCH, "refinedata.npz"), **save)
print("saved refinedata.npz")
