"""Buoyancy-topology scan over ERA5 profiles (tcpyPI thermodynamic conventions).

For each profile, lift the lowest-level parcel (cape()'s environmental-parcel
semantics, ascent_flag=0, ptop=50 -> 28 retained levels 1000..70 hPa), compute
the buoyancy (density-temperature difference) at each level, and extract:
sign topology, interpolated crossings (log-pressure), per-region unsigned CAPE
contributions, running signed partial sums, argmax, and CAPE under the four
LNB conventions (legacy topmost / max-work / ballistic-reachable / first-LNB).

Outputs padded arrays in buoyancy_scan_results.npz (MAXR regions).
"""

import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("ERA5_SCRATCH", str(_Path(__file__).resolve().parent / "work"))
_SRC = str(_Path(__file__).resolve().parents[1] / "src")

import os
import sys
import time

sys.path.insert(0, _SRC)

import numpy as np
from numba import njit, prange

from tcpyPI import constants, utilities
from tcpyPI.pi import solve_temperature_from_entropy


MAXR = 28  # = retained level count; regions can never exceed this
PTOP = 50.0

RD = constants.RD


@njit(cache=True)
def scan_one(TC, Rgkg, P, nlvl, srow,
             region_abs, region_peak, psums, cross_lnp):
    """Scan one profile, filling srow = [iflag, n, first_pos, argmax, clipped,
    clamped, E_top, E_max, E_reach, E_first, LNB_top, LNB_max, LNB_reach,
    LNB_first, lcl, b_low, b_top] and the padded per-region rows."""
    T = TC + 273.15
    R = Rgkg * 0.001

    TP = T[0]
    RP = R[0]
    PP = P[0]
    if (RP < 1e-6) or (TP < 200.0):
        srow[0] = 0.0
        srow[3] = -1.0
        return

    TPC = utilities.T_ktoC(TP)
    ESP = utilities.es_cc(TPC)
    EVP = utilities.ev(RP, PP)
    RH = min(EVP / ESP, 1.0)
    S = utilities.entropy_S(TP, RP, PP)
    PLCL = utilities.e_pLCL(TP, RH, PP)

    # buoyancy profile (identical formulas to cape())
    b = np.empty(nlvl)
    for j in range(nlvl):
        if P[j] >= PLCL:
            TG = TP * (P[j] / PP) ** (constants.RD / constants.CPD)
            RG = RP
            TLVR = utilities.Trho(TG, RG, RG)
            TVENV = utilities.Trho(T[j], R[j], R[j])
            b[j] = TLVR - TVENV
        else:
            TG, RG, IFLAG = solve_temperature_from_entropy(S, P[j], RP, T[j])
            if IFLAG == 2:
                srow[0] = 2.0
                srow[3] = -1.0
                srow[14] = PLCL
                return
            TLVR = utilities.Trho(TG, RP, RG)  # reversible ascent (ascent_flag=0)
            TENV = utilities.Trho(T[j], R[j], R[j])
            b[j] = TLVR - TENV

    # --- segmentation into alternating sign regions (b > 0 is '+') ---
    first_pos = 1 if b[0] > 0.0 else 0
    k = 0                      # current region index
    W = 0.0                    # running signed work
    region_abs[0] = 0.0
    region_peak[0] = abs(b[0])
    cross_p = np.empty(MAXR)   # crossing pressures (region ends), local
    for j in range(1, nlvl):
        b0 = b[j - 1]
        b1 = b[j]
        pos0 = b0 > 0.0
        pos1 = b1 > 0.0
        if pos0 != pos1:
            # region boundary: interpolated crossing (node itself if b hits 0)
            if b0 * b1 < 0.0:
                PC = (P[j] * b0 - P[j - 1] * b1) / (b0 - b1)
            else:
                PC = P[j] if b1 == 0.0 else P[j - 1]
            piece0 = RD * b0 * (P[j - 1] - PC) / (P[j - 1] + PC)
            piece1 = RD * b1 * (PC - P[j]) / (PC + P[j])
            region_abs[k] += abs(piece0)
            cross_p[k] = PC
            cross_lnp[k] = np.log(PC)
            W += piece0
            psums[k] = W
            k += 1
            region_abs[k] = abs(piece1)
            region_peak[k] = abs(b1)
            W += piece1
        else:
            piece = RD * (b0 + b1) * (P[j - 1] - P[j]) / (P[j] + P[j - 1])
            region_abs[k] += abs(piece)
            if abs(b1) > region_peak[k]:
                region_peak[k] = abs(b1)
            W += piece
    n = k + 1
    psums[k] = W
    cross_p[k] = P[nlvl - 1]   # last region "ends" at the profile top
    nstore = n

    # --- diagnostics from the partial sums ---
    # argmax of partial sums (-1 if all <= 0)
    best = -np.inf
    aidx = -1
    for kk in range(nstore):
        if psums[kk] > best:
            best = psums[kk]
            aidx = kk
    if best <= 0.0:
        aidx = -1

    # sign of region kk: first_pos alternating
    top_sign_pos = ((n - 1) % 2 == 0) == (first_pos == 1)
    clipped = 1 if top_sign_pos else 0

    # E_max (max-work) and its LNB
    E_max = best if best > 0.0 else 0.0
    LNB_max = cross_p[aidx] if aidx >= 0 else 0.0

    # E_top (legacy pcmin: signed integral to the topmost positive level,
    # clamped at zero)
    clamped = 0
    if n == 1 and first_pos == 0:
        E_top = 0.0
        LNB_top = 0.0
    else:
        if top_sign_pos:
            raw = psums[nstore - 1]
            LNB_top = cross_p[nstore - 1]
        else:
            raw = psums[nstore - 2]
            LNB_top = cross_p[nstore - 2]
        if raw < 0.0:
            clamped = 1
            E_top = 0.0
        else:
            E_top = raw

    # E_reach_strict (ballistic from rest at launch: max W over the first
    # connected component of {W>0}; W monotone within regions, so region-end
    # checks suffice). NOTE: b[0]=0 by construction and surface CIN is nearly
    # universal, so this is ~always 0 for launch-level parcels.
    E_reach = 0.0
    LNB_reach = 0.0
    for kk in range(nstore):
        if psums[kk] > E_reach:
            E_reach = psums[kk]
            LNB_reach = cross_p[kk]
        if psums[kk] <= 0.0:
            break

    # index of the first positive region (the parcel's LFC-to-LNB layer)
    r1 = 0 if first_pos == 1 else 1

    # E_first (first-LNB convention: signed integral through any leading CIN
    # up to the FIRST down-crossing), clamped at zero
    if r1 < nstore:
        E_first = psums[r1] if psums[r1] > 0.0 else 0.0
        LNB_first = cross_p[r1]
    else:
        E_first = 0.0
        LNB_first = 0.0

    # E_reach_lfc (lifted-parcel ballistic: externally lifted to the LFC with
    # zero kinetic energy, then coasting; kinetic energy at p is W(p) - floor,
    # floor = W at LFC entry; stalls where W returns to the floor)
    E_reach_lfc = 0.0
    LNB_reach_lfc = 0.0
    if r1 < nstore:
        floor = psums[r1 - 1] if r1 >= 1 else 0.0
        best = -np.inf
        for kk in range(r1, nstore):
            if psums[kk] > best:
                best = psums[kk]
                LNB_reach_lfc = cross_p[kk]
            if psums[kk] <= floor and kk > r1:
                break
        if best > 0.0:
            E_reach_lfc = best
        else:
            LNB_reach_lfc = 0.0

    srow[0] = 1.0
    srow[1] = n
    srow[2] = first_pos
    srow[3] = aidx
    srow[4] = clipped
    srow[5] = clamped
    srow[6] = E_top
    srow[7] = E_max
    srow[8] = E_reach
    srow[9] = E_first
    srow[10] = LNB_top
    srow[11] = LNB_max
    srow[12] = LNB_reach
    srow[13] = LNB_first
    srow[14] = PLCL
    srow[15] = b[0]
    srow[16] = b[nlvl - 1]
    srow[17] = E_reach_lfc
    srow[18] = LNB_reach_lfc
    return


@njit(parallel=True, cache=True)
def scan_all(TCs, Rs, P, nlvl, out_scalar, region_abs, region_peak, psums, cross_lnp):
    nprof = TCs.shape[0]
    for i in prange(nprof):
        scan_one(TCs[i], Rs[i], P, nlvl, out_scalar[i],
                 region_abs[i], region_peak[i], psums[i], cross_lnp[i])


def main():
    d = np.load(f"{SCRATCH}/profiles_converted.npz")
    P_full = d["P"]
    nlvl = int((P_full > PTOP).sum())
    P = P_full[:nlvl].copy()
    TCs = d["TC"][:, :nlvl].copy()
    Rs = d["R"][:, :nlvl].copy()
    nprof = TCs.shape[0]
    print(f"{nprof} profiles, {nlvl} retained levels ({P[0]:.0f}..{P[-1]:.0f} hPa)")

    out_scalar = np.full((nprof, 19), np.nan)
    region_abs = np.full((nprof, MAXR), np.nan)
    region_peak = np.full((nprof, MAXR), np.nan)
    psums = np.full((nprof, MAXR), np.nan)
    cross_lnp = np.full((nprof, MAXR - 1), np.nan)

    t0 = time.time()
    scan_all(TCs[:100], Rs[:100], P, nlvl, out_scalar[:100],
             region_abs[:100], region_peak[:100], psums[:100], cross_lnp[:100])
    print(f"compile+warmup: {time.time()-t0:.1f}s")
    t0 = time.time()
    scan_all(TCs, Rs, P, nlvl, out_scalar, region_abs, region_peak, psums, cross_lnp)
    print(f"full scan: {time.time()-t0:.1f}s")

    np.savez_compressed(
        f"{SCRATCH}/buoyancy_scan_results.npz",
        scalar=out_scalar, region_abs=region_abs, region_peak=region_peak,
        psums=psums, cross_lnp=cross_lnp,
        scalar_names=np.array(
            ["iflag", "n", "first_pos", "argmax", "clipped", "clamped",
             "E_top", "E_max", "E_reach", "E_first",
             "LNB_top", "LNB_max", "LNB_reach", "LNB_first",
             "lcl_hPa", "b_low", "b_top", "E_reach_lfc", "LNB_reach_lfc"]),
    )
    print("saved buoyancy_scan_results.npz")


if __name__ == "__main__":
    main()
