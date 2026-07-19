"""Topology scan for the two PI parcels (eyewall, saturated core) at each
profile's converged max-work P_M, plus a quadrature-scheme comparison
(current hybrid vs self-consistent linear-in-ln(p)) for the env parcel.
"""

import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("ERA5_SCRATCH", str(_Path(__file__).resolve().parent / "work"))
_SRC = str(_Path(__file__).resolve().parents[1] / "src")

import sys
import time

sys.path.insert(0, _SRC)

import numpy as np
from numba import njit, prange

from tcpyPI import constants, utilities
from tcpyPI.pi import solve_temperature_from_entropy


PTOP = 50.0
RD = constants.RD
EPS = constants.EPS
CKCD = 0.9


@njit(cache=True)
def buoyancy(TP, RP, PP, T, R, P, nlvl, b):
    """Fill b; return 1 ok / 0 improper / 2 entropy no-converge."""
    if (RP < 1e-6) or (TP < 200.0):
        return 0
    TPC = utilities.T_ktoC(TP)
    RH = min(utilities.ev(RP, PP) / utilities.es_cc(TPC), 1.0)
    S = utilities.entropy_S(TP, RP, PP)
    PLCL = utilities.e_pLCL(TP, RH, PP)
    for j in range(nlvl):
        if P[j] >= PLCL:
            TG = TP * (P[j] / PP) ** (constants.RD / constants.CPD)
            b[j] = utilities.Trho(TG, RP, RP) - utilities.Trho(T[j], R[j], R[j])
        else:
            TG, RG, IFLAG = solve_temperature_from_entropy(S, P[j], RP, T[j])
            if IFLAG == 2:
                return 2
            b[j] = utilities.Trho(TG, RP, RG) - utilities.Trho(T[j], R[j], R[j])
    return 1


@njit(cache=True)
def summarize(b, P, PP, nlvl, lnp_mode):
    """Topology summary for one buoyancy profile.
    lnp_mode: 0 = current hybrid quadrature, 1 = linear-in-ln(p) consistent.
    Returns (n, n_plus, E_top, E_max, clamped, clipped, first_pos)."""
    if lnp_mode == 0:
        W = RD * (PP - P[0]) / (PP + P[0]) * b[0]
    else:
        W = RD * np.log(PP / P[0]) * 0.5 * (b[0] + b[0])
    ps = np.empty(nlvl + 1)
    k = 0
    for j in range(1, nlvl):
        b0 = b[j - 1]
        b1 = b[j]
        if (b0 > 0.0) != (b1 > 0.0):
            if lnp_mode == 0:
                if b0 * b1 < 0.0:
                    PC = (P[j] * b0 - P[j - 1] * b1) / (b0 - b1)
                else:
                    PC = P[j] if b1 == 0.0 else P[j - 1]
                W += RD * b0 * (P[j - 1] - PC) / (P[j - 1] + PC)
                ps[k] = W
                W += RD * b1 * (PC - P[j]) / (PC + P[j])
            else:
                l0 = np.log(P[j - 1])
                l1 = np.log(P[j])
                if b0 * b1 < 0.0:
                    lc = (l1 * b0 - l0 * b1) / (b0 - b1)
                else:
                    lc = l1 if b1 == 0.0 else l0
                W += RD * 0.5 * b0 * (l0 - lc)
                ps[k] = W
                W += RD * 0.5 * b1 * (lc - l1)
            k += 1
        else:
            if lnp_mode == 0:
                W += RD * (b0 + b1) * (P[j - 1] - P[j]) / (P[j] + P[j - 1])
            else:
                W += RD * 0.5 * (b0 + b1) * np.log(P[j - 1] / P[j])
    ps[k] = W
    n = k + 1
    first_pos = 1 if b[0] > 0.0 else 0
    top_pos = b[nlvl - 1] > 0.0
    n_plus = (n + (1 if first_pos == 1 else -1)) // 2 + (1 if (first_pos == 0 and top_pos) else 0)
    # count '+' regions robustly: alternating from first_pos over n regions
    n_plus = 0
    for kk in range(n):
        pos = (kk % 2 == 0) == (first_pos == 1)
        if pos:
            n_plus += 1
    best = 0.0
    for kk in range(n):
        if ps[kk] > best:
            best = ps[kk]
    E_max = best
    clamped = 0
    if top_pos:
        raw = ps[n - 1]
    elif n >= 2:
        raw = ps[n - 2]
    else:
        raw = 0.0
    has_pos_above = False
    for j in range(1, nlvl):
        if b[j] > 0.0:
            has_pos_above = True
            break
    if not has_pos_above:
        E_top = 0.0
    elif raw < 0.0:
        E_top = 0.0
        clamped = 1
    else:
        E_top = raw
    clipped = 1 if top_pos else 0
    return n, n_plus, E_top, E_max, clamped, clipped, first_pos


@njit(parallel=True, cache=True)
def run(SST, MSL, PMs, TCs, Rs, P, nlvl, out_eye, out_core, out_env_cmp):
    nprof = TCs.shape[0]
    for i in prange(nprof):
        T = TCs[i] + 273.15
        R = Rs[i] * 0.001
        b = np.empty(nlvl)
        SSTK = SST[i] + 273.15
        # env-parcel quadrature comparison (hybrid vs lnp-consistent)
        f = buoyancy(T[0], R[0], P[0], T, R, P, nlvl, b)
        if f == 1:
            n0, np0, Et0, Em0, cl0, _, _ = summarize(b, P, P[0], nlvl, 0)
            n1, np1, Et1, Em1, cl1, _, _ = summarize(b, P, P[0], nlvl, 1)
            out_env_cmp[i, 0] = Em0
            out_env_cmp[i, 1] = Em1
            out_env_cmp[i, 2] = Et0
            out_env_cmp[i, 3] = Et1
        if not np.isfinite(PMs[i]):
            out_eye[i, 0] = -1.0
            out_core[i, 0] = -1.0
            continue
        PP = min(PMs[i], 1000.0)
        # eyewall parcel: enriched boundary-layer air at PP
        RP = EPS * R[0] * MSL[i] / (PP * (EPS + R[0]) - R[0] * MSL[i])
        f = buoyancy(T[0], RP, PP, T, R, P, nlvl, b)
        if f == 1:
            n, npl, Et, Em, cl, cp, fp = summarize(b, P, PP, nlvl, 0)
            out_eye[i, 0] = n
            out_eye[i, 1] = npl
            out_eye[i, 2] = Et
            out_eye[i, 3] = Em
            out_eye[i, 4] = cl
            out_eye[i, 5] = cp
            out_eye[i, 6] = fp
        else:
            out_eye[i, 0] = -float(f + 1)
        # saturated core parcel
        ES0 = utilities.es_cc(SST[i])
        RPS = utilities.rv(ES0, PP)
        f = buoyancy(SSTK, RPS, PP, T, R, P, nlvl, b)
        if f == 1:
            n, npl, Et, Em, cl, cp, fp = summarize(b, P, PP, nlvl, 0)
            out_core[i, 0] = n
            out_core[i, 1] = npl
            out_core[i, 2] = Et
            out_core[i, 3] = Em
            out_core[i, 4] = cl
            out_core[i, 5] = cp
            out_core[i, 6] = fp
        else:
            out_core[i, 0] = -float(f + 1)


def main():
    d = np.load(f"{SCRATCH}/profiles_converted.npz")
    r = np.load(f"{SCRATCH}/pi_conv_results.npz")
    P_full = d["P"]
    nlvl = int((P_full > PTOP).sum())
    P = P_full[:nlvl].copy()
    TCs = d["TC"][:, :nlvl].copy()
    Rs = d["R"][:, :nlvl].copy()
    # converged eye pressure under max-work as the P_M proxy; PMIN is the eye
    # pressure -- close enough to RMW pressure for a population scan, and
    # available per profile. Only rows converged under max (IFL==1).
    PMs = np.where(r["IFL"][:, 1] == 1, r["PMIN"][:, 1], np.nan)

    nprof = TCs.shape[0]
    out_eye = np.full((nprof, 7), np.nan)
    out_core = np.full((nprof, 7), np.nan)
    out_env_cmp = np.full((nprof, 4), np.nan)

    t0 = time.time()
    run(d["sst_C"][:50], d["sp_hPa"][:50], PMs[:50], TCs[:50], Rs[:50], P, nlvl,
        out_eye[:50], out_core[:50], out_env_cmp[:50])
    run(d["sst_C"], d["sp_hPa"], PMs, TCs, Rs, P, nlvl, out_eye, out_core, out_env_cmp)
    print(f"scan: {time.time()-t0:.1f}s")

    np.savez_compressed(f"{SCRATCH}/pi_parcel_topology.npz",
                        eye=out_eye, core=out_core, env_cmp=out_env_cmp)

    # ---- quadrature comparison ----
    ok = np.isfinite(out_env_cmp[:, 0])
    dEm = np.abs(out_env_cmp[ok, 0] - out_env_cmp[ok, 1])
    base = np.maximum(out_env_cmp[ok, 0], 50.0)
    print("\n=== hybrid vs linear-in-ln(p) quadrature (env parcel, E_max) ===")
    print(f"  |dE_max|: median {np.median(dEm):.3f}  p99 {np.percentile(dEm,99):.2f}  "
          f"max {dEm.max():.2f} J/kg;  rel max {np.max(dEm/base)*100:.2f}%")

    # ---- PI-parcel topology stats (TC-relevant subset) ----
    tc = (d["sst_C"] >= 26) & (d["sp_hPa"] >= 1000) & (r["IFL"][:, 1] == 1)
    for name, out in [("eyewall", out_eye), ("core(sat)", out_core)]:
        v = out[tc]
        okv = v[:, 0] > 0
        n = v[okv, 0].astype(np.int64)
        npl = v[okv, 1].astype(np.int64)
        print(f"\n=== {name} parcel at converged P_M (TC-relevant, n={okv.sum()}) ===")
        print(f"  launches buoyant (b_surface>0): {100*v[okv,6].mean():.1f}%")
        print(f"  candidate-LNB count (+ regions): 1: {100*(npl==1).mean():.1f}%  "
              f"2: {100*(npl==2).mean():.1f}%  >=3: {100*(npl>=3).mean():.1f}%  0: {100*(npl==0).mean():.2f}%")
        print(f"  E_top clamped to 0 while E_max>0: "
              f"{100*((v[okv,4]==1)&(v[okv,3]>0)).mean():.2f}%")
        dd = np.abs(v[okv, 2] - v[okv, 3])
        print(f"  |E_top-E_max|: >1 J/kg {100*(dd>1).mean():.2f}%  >10 {100*(dd>10).mean():.2f}%  "
              f"max {dd.max():.1f} J/kg")
        print(f"  clipped at profile top (b>0 at 70 hPa): {100*v[okv,5].mean():.2f}%")


if __name__ == "__main__":
    main()
