"""Potential intensity under four LNB conventions, over all ERA5 profiles.

Conventions (terminal-point rule for every CAPE evaluation inside pi()):
  0 = top       legacy pcmin: signed W at the crossing above the topmost
                positive grid level (or profile top), clamped at 0
  1 = max       max-W: maximum of W over all region-end candidates (>=0)
  2 = first     signed W at the first down-crossing, clamped at 0
  3 = reach_lfc lifted-ballistic: best W in the component reachable from the
                first positive region with floor = W at its entry

Quadrature: running signed work with trapezoids split at every crossing
(<=1% from cape()'s unsplit-interval arithmetic; validated separately).
Outputs pi_conv_results.npz: (nprof, 4) arrays VMAX, PMIN, IFL, TO, OTL, NIT.
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
V_REDUC = 0.8


@njit(cache=True)
def cape_conv(TP, RP, PP, T, R, P, nlvl, conv):
    """CAPE/TOB/LNB for one parcel under one convention.
    Returns (E, TOB, LNB, flag): flag 1 ok, 0 improper parcel, 2 no-converge."""
    if (RP < 1e-6) or (TP < 200.0):
        return 0.0, T[0], 0.0, 0

    TPC = utilities.T_ktoC(TP)
    ESP = utilities.es_cc(TPC)
    EVP = utilities.ev(RP, PP)
    RH = min(EVP / ESP, 1.0)
    S = utilities.entropy_S(TP, RP, PP)
    PLCL = utilities.e_pLCL(TP, RH, PP)

    b = np.empty(nlvl)
    for j in range(nlvl):
        if P[j] >= PLCL:
            TG = TP * (P[j] / PP) ** (constants.RD / constants.CPD)
            TLVR = utilities.Trho(TG, RP, RP)
            b[j] = TLVR - utilities.Trho(T[j], R[j], R[j])
        else:
            TG, RG, IFLAG = solve_temperature_from_entropy(S, P[j], RP, T[j])
            if IFLAG == 2:
                return 0.0, T[0], P[0], 2
            TLVR = utilities.Trho(TG, RP, RG)
            b[j] = TLVR - utilities.Trho(T[j], R[j], R[j])

    # running signed work with per-crossing splits; region-end candidates
    W = RD * (PP - P[0]) / (PP + P[0]) * b[0]   # surface partial term
    maxreg = nlvl + 1
    ps = np.empty(maxreg)       # W at region ends
    cp = np.empty(maxreg)       # region-end pressures (crossing or top)
    ct = np.empty(maxreg)       # temperature at region ends (interp / top)
    k = 0
    for j in range(1, nlvl):
        b0 = b[j - 1]
        b1 = b[j]
        if (b0 > 0.0) != (b1 > 0.0):
            if b0 * b1 < 0.0:
                PC = (P[j] * b0 - P[j - 1] * b1) / (b0 - b1)
            else:
                PC = P[j] if b1 == 0.0 else P[j - 1]
            W += RD * b0 * (P[j - 1] - PC) / (P[j - 1] + PC)
            ps[k] = W
            cp[k] = PC
            ct[k] = (T[j - 1] * (PC - P[j]) + T[j] * (P[j - 1] - PC)) / (P[j - 1] - P[j])
            k += 1
            W += RD * b1 * (PC - P[j]) / (PC + P[j])
        else:
            W += RD * (b0 + b1) * (P[j - 1] - P[j]) / (P[j] + P[j - 1])
    ps[k] = W
    cp[k] = P[nlvl - 1]
    ct[k] = T[nlvl - 1]
    n = k + 1
    first_pos = b[0] > 0.0
    top_pos = b[nlvl - 1] > 0.0

    E = 0.0
    TOB = T[0]
    LNB = 0.0
    if conv == 0:
        # legacy: topmost positive grid level among j>=1 (pcmin's INB loop
        # excludes the launch node)
        has_pos = False
        for j in range(1, nlvl):
            if b[j] > 0.0:
                has_pos = True
                break
        if has_pos:
            kk = n - 1 if top_pos else n - 2
            if kk >= 0 and ps[kk] > 0.0:
                E = ps[kk]
                TOB = ct[kk]
                LNB = cp[kk]
    elif conv == 1:
        best = 0.0
        for kk in range(n):
            if ps[kk] > best:
                best = ps[kk]
                TOB = ct[kk]
                LNB = cp[kk]
        E = best
        if E == 0.0:
            TOB = T[0]
            LNB = 0.0
    elif conv == 2:
        r1 = 0 if first_pos else 1
        if r1 < n and ps[r1] > 0.0:
            E = ps[r1]
            TOB = ct[r1]
            LNB = cp[r1]
    else:
        r1 = 0 if first_pos else 1
        if r1 < n:
            floor = ps[r1 - 1] if r1 >= 1 else 0.0
            best = -np.inf
            bi = -1
            for kk in range(r1, n):
                if ps[kk] > best:
                    best = ps[kk]
                    bi = kk
                if ps[kk] <= floor and kk > r1:
                    break
            if best > 0.0:
                E = best
                TOB = ct[bi]
                LNB = cp[bi]
    return E, TOB, LNB, 1


@njit(cache=True)
def pi_conv(SSTC, MSL, T, R, P, nlvl, conv):
    """pi() minimum-pressure loop under one CAPE convention.
    Returns (VMAX, PMIN, IFL, TO, OTL, NP)."""
    SSTK = utilities.T_Ctok(SSTC)
    if SSTC <= 5.0 or SSTC > 100.0:
        return np.nan, np.nan, 0, np.nan, np.nan, 0
    if np.min(T) <= 100.0 or np.max(T) - 273.15 > 100.0:
        return np.nan, np.nan, 0, np.nan, np.nan, 0
    ES0 = utilities.es_cc(SSTC)

    IFL = 1
    CAPEA, _, _, fA = cape_conv(T[0], R[0], P[0], T, R, P, nlvl, conv)
    if fA != 1:
        IFL = fA

    NP = 0
    PM = 970.0
    PMOLD = PM
    PNEW = 0.0
    TO = np.nan
    OTL = np.nan
    CAPEM = 0.0
    CAPEMS = 0.0
    RAT = 1.0
    TVAV = 300.0
    while np.abs(PNEW - PMOLD) > 0.5:
        PP = min(PM, 1000.0)
        RP = EPS * R[0] * MSL / (PP * (EPS + R[0]) - R[0] * MSL)
        CAPEM, _, _, fM = cape_conv(T[0], RP, PP, T, R, P, nlvl, conv)
        if fM != 1:
            IFL = fM
        RPS = utilities.rv(ES0, PP)
        CAPEMS, TOMS, LNBS, fS = cape_conv(SSTK, RPS, PP, T, R, P, nlvl, conv)
        if fS != 1:
            IFL = fS
        TO = TOMS
        OTL = LNBS
        RAT = SSTK / TO
        TV0 = utilities.Trho(T[0], R[0], R[0])
        TVSST = utilities.Trho(SSTK, RPS, RPS)
        TVAV = 0.5 * (TV0 + TVSST)
        CAT = max((CAPEM - CAPEA) + 0.5 * CKCD * RAT * (CAPEMS - CAPEM), 0.0)
        PNEW = MSL * np.exp(-CAT / (RD * TVAV))
        PMOLD = PM
        PM = PNEW
        NP += 1
        if (NP > 200) or (PM < 400.0):
            return np.nan, np.nan, 2, np.nan, np.nan, NP

    CATFAC = 0.5 * (1.0 + 1.0 / constants.b)
    CAT = max((CAPEM - CAPEA) + CKCD * RAT * CATFAC * (CAPEMS - CAPEM), 0.0)
    PMIN = MSL * np.exp(-CAT / (RD * TVAV))
    FAC = max(0.0, CAPEMS - CAPEM)
    VMAX = V_REDUC * np.sqrt(CKCD * RAT * FAC)
    return VMAX, PMIN, IFL, TO, OTL, NP


@njit(parallel=True, cache=True)
def run_all(SST, MSL, TCs, Rs, P, nlvl, VMAX, PMIN, IFL, TO, OTL, NIT):
    nprof = TCs.shape[0]
    for i in prange(nprof):
        T = TCs[i] + 273.15
        R = Rs[i] * 0.001
        for conv in range(4):
            v, p, f, t, o, np_ = pi_conv(SST[i], MSL[i], T, R, P, nlvl, conv)
            VMAX[i, conv] = v
            PMIN[i, conv] = p
            IFL[i, conv] = f
            TO[i, conv] = t
            OTL[i, conv] = o
            NIT[i, conv] = np_


def main():
    d = np.load(f"{SCRATCH}/profiles_converted.npz")
    P_full = d["P"]
    nlvl = int((P_full > PTOP).sum())
    P = P_full[:nlvl].copy()
    TCs = d["TC"][:, :nlvl].copy()
    Rs = d["R"][:, :nlvl].copy()
    SST = d["sst_C"]
    MSL = d["sp_hPa"]
    nprof = TCs.shape[0]

    VMAX = np.full((nprof, 4), np.nan)
    PMIN = np.full((nprof, 4), np.nan)
    IFL = np.zeros((nprof, 4), dtype=np.int64)
    TO = np.full((nprof, 4), np.nan)
    OTL = np.full((nprof, 4), np.nan)
    NIT = np.zeros((nprof, 4), dtype=np.int64)

    t0 = time.time()
    run_all(SST[:50], MSL[:50], TCs[:50], Rs[:50], P, nlvl,
            VMAX[:50], PMIN[:50], IFL[:50], TO[:50], OTL[:50], NIT[:50])
    print(f"warmup: {time.time()-t0:.1f}s")
    t0 = time.time()
    run_all(SST, MSL, TCs, Rs, P, nlvl, VMAX, PMIN, IFL, TO, OTL, NIT)
    print(f"full run ({nprof} x 4 conventions): {time.time()-t0:.1f}s")

    np.savez_compressed(f"{SCRATCH}/pi_conv_results.npz",
                        VMAX=VMAX, PMIN=PMIN, IFL=IFL, TO=TO, OTL=OTL, NIT=NIT,
                        conv_names=np.array(["top", "max", "first", "reach_lfc"]))
    print("saved pi_conv_results.npz")


if __name__ == "__main__":
    main()
