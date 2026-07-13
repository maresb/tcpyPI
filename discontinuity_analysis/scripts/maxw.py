"""max-W CAPE: CAPE = max over outflow level p_t of the running signed
buoyancy integral W(p_t), evaluated on the same piecewise-linear interpolant
(trapezoid + interpolated-crossing partial areas) the current code uses.

Candidates for the max: every grid node above jmin, and every interpolated
sign-change crossing. Continuous in the profile/parcel parameters; equals the
pcmin value exactly for single-crossing (typical) profiles.
"""

import numpy as np
from tcpyPI import constants, utilities

from gmap import (CAPEA as _unused, P_FULL, T, R, MSL, SSTK, ES0, NK, CKCD,
                  ASCENT_FLAG, PTOP, cape_instr, gmap)


def cape_maxw(TP, RP, PP, T, R, P, ascent_flag=0, ptop=50):
    """cape() with the max-W integration; reuses cape_instr for TVRDIF."""
    base = cape_instr(TP, RP, PP, T, R, P, ascent_flag, ptop)
    if base["IFLAG"] != 1 and base["INB"] == -1:
        return base  # improper parcel / non-convergence: unchanged
    N = int(np.count_nonzero(P > ptop))
    Pt, Tt = P[:N], T[:N]
    TVRDIF = base["TVRDIF"]
    nlvl = N
    jmin = 0

    RD = constants.RD
    W = RD * (PP - Pt[jmin]) / (PP + Pt[jmin]) * TVRDIF[jmin]
    BESTW, BESTP, BESTT = 0.0, 0.0, Tt[0]
    if W > BESTW:
        BESTW, BESTP, BESTT = W, Pt[jmin], Tt[jmin]
    for j in range(jmin + 1, nlvl):
        b0, b1 = TVRDIF[j - 1], TVRDIF[j]
        if b0 * b1 < 0.0:
            # interpolated crossing between j-1 and j (same formula as PINB/PAT)
            Pc = (Pt[j] * b0 - Pt[j - 1] * b1) / (b0 - b1)
            Wc = W + RD * b0 * (Pt[j - 1] - Pc) / (Pt[j - 1] + Pc)
            if Wc > BESTW:
                Tc = (Tt[j - 1] * (Pc - Pt[j]) + Tt[j] * (Pt[j - 1] - Pc)) / (
                    Pt[j - 1] - Pt[j]
                )
                BESTW, BESTP, BESTT = Wc, Pc, Tc
        W += RD * (b1 + b0) * (Pt[j - 1] - Pt[j]) / (Pt[j] + Pt[j - 1])
        if W > BESTW:
            BESTW, BESTP, BESTT = W, Pt[j], Tt[j]

    if BESTW <= 0.0:
        return dict(CAPED=0.0, TOB=Tt[0], LNB=0.0, IFLAG=1, INB=0,
                    PLCL=base["PLCL"], TVRDIF=TVRDIF, PA=0.0, NA=0.0, PAT=0.0)
    return dict(CAPED=BESTW, TOB=BESTT, LNB=BESTP, IFLAG=1, INB=-99,
                PLCL=base["PLCL"], TVRDIF=TVRDIF, PA=np.nan, NA=np.nan,
                PAT=np.nan)


def gmap_fix(PM, diss_flag=1):
    PP = min(PM, 1000.0)
    RP_M = constants.EPS * R[NK] * MSL / (PP * (constants.EPS + R[NK]) - R[NK] * MSL)
    m = cape_maxw(T[NK], RP_M, PP, T, R, P_FULL, ASCENT_FLAG, PTOP)
    RP_S = utilities.rv(ES0, PP)
    s = cape_maxw(SSTK, RP_S, PP, T, R, P_FULL, ASCENT_FLAG, PTOP)
    env = cape_maxw(T[NK], R[NK], P_FULL[NK], T, R, P_FULL, ASCENT_FLAG, PTOP)
    TO = s["TOB"]
    RAT = SSTK / TO if diss_flag else 1.0
    TV0 = utilities.Trho(T[NK], R[NK], R[NK])
    TVSST = utilities.Trho(SSTK, RP_S, RP_S)
    TVAV = 0.5 * (TV0 + TVSST)
    CAT = max((m["CAPED"] - env["CAPED"]) + 0.5 * CKCD * RAT * (s["CAPED"] - m["CAPED"]), 0.0)
    PNEW = MSL * np.exp(-CAT / (constants.RD * TVAV))
    return dict(PNEW=PNEW, CAPEM=m["CAPED"], CAPEMS=s["CAPED"], TO=TO, RAT=RAT,
                CAPEA=env["CAPED"], TVAV=TVAV)


if __name__ == "__main__":
    print("=== fixed-point iteration with max-W CAPE, x0=970 ===")
    PM, PMOLD, PNEW, NP = 970.0, 970.0, 0.0, 0
    while abs(PNEW - PMOLD) > 0.5 and NP <= 200:
        o = gmap_fix(PM)
        PNEW = o["PNEW"]
        print(f"n={NP:3d} PM={PM:.10f} -> PNEW={PNEW:.10f}  CAPEM={o['CAPEM']:9.4f} TO={o['TO']:8.3f}")
        PMOLD, PM = PM, PNEW
        NP += 1
    print(f"converged: NP={NP}, PM={PM:.8f}")

    # final outputs (mirror _pi_numba post-loop)
    o = gmap_fix(PM)
    CATFAC = 0.5 * (1.0 + 1 / constants.b)
    CAT = max((o["CAPEM"] - o["CAPEA"]) + CKCD * o["RAT"] * CATFAC * (o["CAPEMS"] - o["CAPEM"]), 0.0)
    PMIN = MSL * np.exp(-CAT / (constants.RD * o["TVAV"]))
    FAC = max(0.0, o["CAPEMS"] - o["CAPEM"])
    VMAX = 0.8 * np.sqrt(CKCD * o["RAT"] * FAC)
    print(f"VMAX={VMAX:.6f} m/s  PMIN={PMIN:.6f} hPa  TO={o['TO']:.4f} K")

    # continuity check: rescan the old jump neighborhood
    print()
    print("=== g_fix(PM) over the old jump (950.7..950.95, step 2e-3) ===")
    pms = np.arange(950.70, 950.95, 0.002)
    gs = np.array([gmap_fix(x)["PNEW"] for x in pms])
    dg = np.abs(np.diff(gs))
    print(f"max |step-to-step change| = {dg.max():.3e} hPa (was 0.837 at the jump)")

    # fixed-point location vs the rescue's midpoint answer
    print()
    print("comparison: branch rescue returns midpoint-ish ~950.966; PR cycle was (950.653, 951.279)")
