"""Final topology/convention scan for all three PI parcels.

Topology convention: leading sign = sign of buoyancy infinitesimally above
launch (= sign of first nonzero b for the PWL interpolant); no zero-width
leading regions. Conventions are computed candidate-based (down-crossings of
the running work W), so they are unaffected by the labeling change:
  E_top   : W at the last down-crossing / top (legacy; requires a positive
            grid level above the launch node), clamped at 0
  E_max   : max of W over all candidates (>= 0)
  E_first : W at the first down-crossing, clamped at 0
  E_reach : lifted-ballistic (floor = W entering the first positive region)

Parcel A at ambient; parcels B, C at the converged max-work P_M.
Output: parcel_topology_final.npz
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

PTOP = 20.0  # retain levels with P > 20 hPa -> top 30 hPa: the minimal
# ceiling observing every buoyancy crossing in the 569k-column sample (highest:
# 47.6 hPa, saturated core parcel, interpolated between the 50 and 30 hPa
# nodes). At this ceiling zero profiles clip and zero entropy solves fail (the
# 30-20 hPa layer caused 21 failures when retained).
RD = constants.RD
EPS = constants.EPS
CKCD = 0.9


@njit(cache=True)
def buoyancy(TP, RP, PP, T, R, P, nlvl, b):
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
def summarize(b, P, PP, nlvl, out):
    """out = [first_sign(+1/-1/0), n, E_top, E_max, E_first, E_reach,
              clamped, clipped, LNB_top, LNB_max, LNB_first, LNB_reach]
    (LNB pressures in hPa; 0 where the convention returns zero CAPE)"""
    # ---- topology under the perturbation convention ----
    # effective sign sequence: skip zero values for the leading sign; a zero
    # value elsewhere adopts the '-' side as before (b>0 is '+')
    j0 = 0
    while j0 < nlvl and b[j0] == 0.0:
        j0 += 1
    if j0 == nlvl:
        out[0] = 0.0
        out[1] = 1.0
        return
    first_pos = b[j0] > 0.0
    n = 1
    prev = first_pos
    for j in range(j0 + 1, nlvl):
        cur = b[j] > 0.0
        if cur != prev:
            n += 1
            prev = cur
    out[0] = 1.0 if first_pos else -1.0
    out[1] = n

    # ---- running work; candidates at down-crossings (+ -> -) and top ----
    W = RD * (PP - P[0]) / (PP + P[0]) * b[0]
    ncand = 0
    cand_W = np.empty(nlvl + 1)
    cand_P = np.empty(nlvl + 1)
    up_W = 0.0          # W entering the first positive region
    seen_pos = b[0] > 0.0
    if seen_pos:
        up_W = 0.0      # launch is inside a positive region (parcels B, C)
    have_up = seen_pos
    for j in range(1, nlvl):
        b0 = b[j - 1]
        b1 = b[j]
        if (b0 > 0.0) != (b1 > 0.0):
            if b0 * b1 < 0.0:
                PC = (P[j] * b0 - P[j - 1] * b1) / (b0 - b1)
            else:
                PC = P[j] if b1 == 0.0 else P[j - 1]
            W += RD * b0 * (P[j - 1] - PC) / (P[j - 1] + PC)
            if b0 > 0.0:
                cand_W[ncand] = W          # down-crossing candidate
                cand_P[ncand] = PC
                ncand += 1
            else:
                if not have_up:
                    up_W = W               # first entry into a + region
                    have_up = True
            W += RD * b1 * (PC - P[j]) / (PC + P[j])
        else:
            W += RD * (b0 + b1) * (P[j - 1] - P[j]) / (P[j] + P[j - 1])
    top_pos = b[nlvl - 1] > 0.0
    if top_pos:
        cand_W[ncand] = W                  # profile-top candidate (clipped)
        cand_P[ncand] = P[nlvl - 1]
        ncand += 1
    out[7] = 1.0 if top_pos else 0.0

    if ncand == 0:
        return

    # E_max over all candidates (launch, W=0, implicit)
    best = 0.0
    for kk in range(ncand):
        if cand_W[kk] > best:
            best = cand_W[kk]
            out[9] = cand_P[kk]            # LNB_max
    out[3] = best

    # E_top: legacy requires a positive grid level above the launch node
    has_pos_above = False
    for j in range(1, nlvl):
        if b[j] > 0.0:
            has_pos_above = True
            break
    if has_pos_above:
        raw = cand_W[ncand - 1]
        if raw < 0.0:
            out[6] = 1.0
        else:
            out[2] = raw
            out[8] = cand_P[ncand - 1]     # LNB_top

    # E_first
    if cand_W[0] > 0.0:
        out[4] = cand_W[0]
        out[10] = cand_P[0]                # LNB_first

    # E_reach (lifted-ballistic): walk candidates; between candidate k and
    # k+1 the parcel traverses a negative region whose minimum W is the entry
    # of the next + region; we approximate the stall check with the next
    # up-crossing W (monotone within regions makes this exact).
    # Reconstruct up-crossing Ws in a second pass for correctness.
    floor = up_W
    bestr = 0.0
    W2 = RD * (PP - P[0]) / (PP + P[0]) * b[0]
    entered = b[0] > 0.0
    stalled = False
    for j in range(1, nlvl):
        b0 = b[j - 1]
        b1 = b[j]
        if (b0 > 0.0) != (b1 > 0.0):
            if b0 * b1 < 0.0:
                PC = (P[j] * b0 - P[j - 1] * b1) / (b0 - b1)
            else:
                PC = P[j] if b1 == 0.0 else P[j - 1]
            W2 += RD * b0 * (P[j - 1] - PC) / (P[j - 1] + PC)
            if b0 > 0.0:
                if entered and not stalled and W2 > bestr:
                    bestr = W2
                    out[11] = PC           # LNB_reach
            else:
                if not entered:
                    entered = True
                elif entered and W2 <= floor:
                    stalled = True
            W2 += RD * b1 * (PC - P[j]) / (PC + P[j])
        else:
            W2 += RD * (b0 + b1) * (P[j - 1] - P[j]) / (P[j] + P[j - 1])
    if top_pos and entered and not stalled and W2 > bestr:
        bestr = W2
        out[11] = P[nlvl - 1]
    if bestr > 0.0:
        out[5] = bestr
    else:
        out[5] = 0.0
        out[11] = 0.0


@njit(cache=True)
def pm_converged(SSTC, MSL, T, R, P, nlvl):
    """Max-work pi loop; returns converged PM (RMW-level pressure) or NaN."""
    if SSTC <= 5.0 or SSTC > 100.0:
        return np.nan
    SSTK = utilities.T_Ctok(SSTC)
    ES0 = utilities.es_cc(SSTC)
    b = np.empty(nlvl)
    tmp = np.empty(12)

    # E_max-only cape via summarize (out[3])
    def _capemax(TP, RP, PP):
        for q in range(12):
            tmp[q] = 0.0
        f = buoyancy(TP, RP, PP, T, R, P, nlvl, b)
        if f != 1:
            return 0.0, T[0], -1.0
        summarize(b, P, PP, nlvl, tmp)
        return tmp[3], 0.0, 1.0

    CAPEA, _, _ = _capemax(T[0], R[0], P[0])
    NP = 0
    PM = 970.0
    PMOLD = PM
    PNEW = 0.0
    while np.abs(PNEW - PMOLD) > 0.5:
        PP = min(PM, 1000.0)
        RP = EPS * R[0] * MSL / (PP * (EPS + R[0]) - R[0] * MSL)
        CAPEM, _, _ = _capemax(T[0], RP, PP)
        RPS = utilities.rv(ES0, PP)
        # saturated call: need TOB -> compute buoyancy then find argmax temp:
        f = buoyancy(SSTK, RPS, PP, T, R, P, nlvl, b)
        if f != 1:
            return np.nan
        for q in range(12):
            tmp[q] = 0.0
        summarize(b, P, PP, nlvl, tmp)
        CAPEMS = tmp[3]
        # outflow temperature: T at coldest buoyancy-weighted... use simple
        # proxy: T at the level nearest the LNB is not tracked here; RAT via
        # tropopause-min temperature of the environment as in BE02 spirit.
        # For PM convergence only the CAT magnitude matters; use env min T.
        TOmin = T[0]
        for j in range(nlvl):
            if T[j] < TOmin:
                TOmin = T[j]
        RAT = SSTK / TOmin
        TV0 = utilities.Trho(T[0], R[0], R[0])
        TVSST = utilities.Trho(SSTK, RPS, RPS)
        TVAV = 0.5 * (TV0 + TVSST)
        CAT = max((CAPEM - CAPEA) + 0.5 * CKCD * RAT * (CAPEMS - CAPEM), 0.0)
        PNEW = MSL * np.exp(-CAT / (RD * TVAV))
        PMOLD = PM
        PM = PNEW
        NP += 1
        if (NP > 200) or (PM < 400.0):
            return np.nan
    return PM


@njit(parallel=True, cache=True)
def run(SST, MSL, TCs, Rs, P, nlvl, PMs, outA, outB, outC):
    nprof = TCs.shape[0]
    for i in prange(nprof):
        T = TCs[i] + 273.15
        R = Rs[i] * 0.001
        b = np.empty(nlvl)
        # parcel A at ambient
        f = buoyancy(T[0], R[0], P[0], T, R, P, nlvl, b)
        if f == 1:
            summarize(b, P, P[0], nlvl, outA[i])
        else:
            outA[i, 0] = np.nan
        # converged PM (max-work)
        PM = pm_converged(SST[i], MSL[i], T, R, P, nlvl)
        PMs[i] = PM
        if not np.isfinite(PM):
            outB[i, 0] = np.nan
            outC[i, 0] = np.nan
            continue
        PP = min(PM, 1000.0)
        RP = EPS * R[0] * MSL[i] / (PP * (EPS + R[0]) - R[0] * MSL[i])
        f = buoyancy(T[0], RP, PP, T, R, P, nlvl, b)
        if f == 1:
            summarize(b, P, PP, nlvl, outB[i])
        else:
            outB[i, 0] = np.nan
        SSTK = SST[i] + 273.15
        ES0 = utilities.es_cc(SST[i])
        RPS = utilities.rv(ES0, PP)
        f = buoyancy(SSTK, RPS, PP, T, R, P, nlvl, b)
        if f == 1:
            summarize(b, P, PP, nlvl, outC[i])
        else:
            outC[i, 0] = np.nan


def main():
    d = np.load(f"{SCRATCH}/profiles_converted.npz")
    P_full = d["P"]
    nlvl = int((P_full > PTOP).sum())
    P = P_full[:nlvl].copy()
    TCs = d["TC"][:, :nlvl].copy()
    Rs = d["R"][:, :nlvl].copy()
    nprof = TCs.shape[0]
    PMs = np.full(nprof, np.nan)
    outA = np.zeros((nprof, 12))
    outB = np.zeros((nprof, 12))
    outC = np.zeros((nprof, 12))

    t0 = time.time()
    run(d["sst_C"][:50], d["sp_hPa"][:50], TCs[:50], Rs[:50], P, nlvl,
        PMs[:50], outA[:50], outB[:50], outC[:50])
    run(d["sst_C"], d["sp_hPa"], TCs, Rs, P, nlvl, PMs, outA, outB, outC)
    print(f"scan: {time.time()-t0:.1f}s")
    np.savez_compressed(f"{SCRATCH}/parcel_topology_final.npz",
                        A=outA, B=outB, C=outC, PM=PMs,
                        cols=np.array(["first_sign", "n", "E_top", "E_max",
                                       "E_first", "E_reach", "clamped", "clipped",
                                       "LNB_top", "LNB_max", "LNB_first", "LNB_reach"]))
    print("saved parcel_topology_final.npz")


if __name__ == "__main__":
    main()
