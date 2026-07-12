"""Instrumented reconstruction of the pcmin fixed-point map g(PM) for PR #77.

Replicates _pi_numba's iteration body (branch upstream/modernize_eg) exactly,
but exposes every internal quantity so discontinuities can be located and
classified. Pure-python (numba disabled) for introspection.
"""

import os
import sys

os.environ["TCPYPI_DISABLE_NUMBA"] = "1"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRATCH, "branch", "src"))

import numpy as np
from tcpyPI import constants, utilities
from tcpyPI.pi import solve_temperature_from_entropy

# ---- PR #77 profile ----
SSTC = 28.20263671875
MSL = 1014.9654541015625
P_FULL = np.array(
    [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750]
    + [700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225]
    + [200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7]
    + [5, 3, 2, 1],
    dtype=float,
)
TC_FULL = np.array(
    [25.260956, 23.078949, 20.881561, 18.652649, 16.615143]
    + [14.63147, 12.794586, 11.79306, 11.01236, 10.847565]
    + [10.354492, 7.807007, 5.473297, 2.5278625, -1.532135]
    + [-7.3142395, -13.635345, -20.613602, -28.588928, -37.270096]
    + [-45.40825, -50.047455, -54.84575, -59.173737, -62.878662]
    + [-65.78009, -69.30669, -65.512024, -60.76361, -55.100464]
    + [-51.38333, -43.95462, -39.709717, -33.193268, -21.576843]
    + [-15.402496, -13.186676]
)
R_FULL = np.array(
    [1.0783693e01, 1.0704287e01, 1.0680210e01, 1.0617845e01]
    + [1.0320683e01, 9.8811483e00, 9.1884289e00, 7.1884680e00]
    + [5.6963191e00, 3.5568204e00, 1.5912720e00, 1.0433695e00]
    + [5.9723043e-01, 4.3974420e-01, 4.8722979e-01, 5.8590513e-01]
    + [4.5599860e-01, 3.1293562e-01, 1.9222400e-01, 9.7611703e-02]
    + [3.3851895e-02, 2.4188591e-02, 1.9636340e-02, 1.3214327e-02]
    + [7.2453087e-03, 4.3027173e-03, 3.7014042e-03, 2.9414182e-03]
    + [2.7806845e-03, 2.8306348e-03, 2.9053832e-03, 2.9848625e-03]
    + [3.0779461e-03, 3.1315640e-03, 3.2939769e-03, 3.3872949e-03]
    + [3.7360021e-03]
)

CKCD = 0.9
ASCENT_FLAG = 0
PTOP = 50

SSTK = utilities.T_Ctok(SSTC)
T = utilities.T_Ctok(TC_FULL)
R = R_FULL * 0.001
ES0 = utilities.es_cc(SSTC)
NK = 0


def cape_instr(TP, RP, PP, T, R, P, ascent_flag=0, ptop=50):
    """Copy of branch cape() with full instrumentation. Returns dict."""
    N = int(np.count_nonzero(P > ptop))
    P = P[:N]
    T = T[:N]
    R = R[:N]
    nlvl = len(P)
    TVRDIF = np.zeros((nlvl,))

    if (RP < 1e-6) or (TP < 200):
        return dict(CAPED=0.0, TOB=np.nan, LNB=np.nan, IFLAG=0, INB=-1,
                    PLCL=np.nan, TVRDIF=TVRDIF, PA=0.0, NA=0.0, PAT=0.0)

    TPC = utilities.T_ktoC(TP)
    ESP = utilities.es_cc(TPC)
    EVP = utilities.ev(RP, PP)
    RH = min(EVP / ESP, 1.0)
    S = utilities.entropy_S(TP, RP, PP)
    PLCL = utilities.e_pLCL(TP, RH, PP)

    CAPED = 0
    TOB = T[0]
    IFLAG = 1
    jmin = int(1e6)
    ncsteps = []

    for j in range(nlvl):
        jmin = int(min(jmin, j))
        if P[j] >= PLCL:
            TG = TP * (P[j] / PP) ** (constants.RD / constants.CPD)
            RG = RP
            TLVR = utilities.Trho(TG, RG, RG)
            TVENV = utilities.Trho(T[j], R[j], R[j])
            TVRDIF[j] = TLVR - TVENV
        else:
            TG, RG, IFLAG_N = solve_temperature_from_entropy(
                S=S, P=P[j], RP=RP, T_initial=T[j]
            )
            if IFLAG_N == 2:
                return dict(CAPED=0.0, TOB=T[0], LNB=P[0], IFLAG=2, INB=-1,
                            PLCL=PLCL, TVRDIF=TVRDIF, PA=0.0, NA=0.0, PAT=0.0)
            RMEAN = ascent_flag * RG + (1 - ascent_flag) * RP
            TLVR = utilities.Trho(TG, RMEAN, RG)
            TENV = utilities.Trho(T[j], R[j], R[j])
            TVRDIF[j] = TLVR - TENV

    NA = 0.0
    PA = 0.0
    INB = 0
    for j in range(nlvl - 1, jmin, -1):
        if TVRDIF[j] > 0:
            INB = max(INB, j)

    if INB == 0:
        return dict(CAPED=0.0, TOB=T[0], LNB=0.0, IFLAG=IFLAG, INB=0,
                    PLCL=PLCL, TVRDIF=TVRDIF, PA=0.0, NA=0.0, PAT=0.0)

    for j in range(jmin + 1, INB + 1, 1):
        PFAC = (
            constants.RD * (TVRDIF[j] + TVRDIF[j - 1]) * (P[j - 1] - P[j])
            / (P[j] + P[j - 1])
        )
        PA = PA + max(PFAC, 0.0)
        NA = NA - min(PFAC, 0.0)

    PMA = PP + P[jmin]
    PFAC = constants.RD * (PP - P[jmin]) / PMA
    PA = PA + PFAC * max(TVRDIF[jmin], 0.0)
    NA = NA - PFAC * min(TVRDIF[jmin], 0.0)

    PAT = 0.0
    TOB = T[INB]
    LNB = P[INB]
    if INB < nlvl - 1:
        PINB = (P[INB + 1] * TVRDIF[INB] - P[INB] * TVRDIF[INB + 1]) / (
            TVRDIF[INB] - TVRDIF[INB + 1]
        )
        LNB = PINB
        PAT = constants.RD * TVRDIF[INB] * (P[INB] - PINB) / (P[INB] + PINB)
        TOB = (T[INB] * (PINB - P[INB + 1]) + T[INB + 1] * (P[INB] - PINB)) / (
            P[INB] - P[INB + 1]
        )

    CAPED = max(PA + PAT - NA, 0.0)
    return dict(CAPED=CAPED, TOB=TOB, LNB=LNB, IFLAG=1, INB=INB,
                PLCL=PLCL, TVRDIF=TVRDIF, PA=PA, NA=NA, PAT=PAT)


# Environmental CAPE (PM-independent)
_env = cape_instr(T[NK], R[NK], P_FULL[NK], T, R, P_FULL, ASCENT_FLAG, PTOP)
CAPEA = _env["CAPED"]


def gmap(PM, diss_flag=1):
    """One iteration body: PM -> PNEW, with internals."""
    PP = min(PM, 1000.0)
    RP_M = constants.EPS * R[NK] * MSL / (
        PP * (constants.EPS + R[NK]) - R[NK] * MSL
    )
    m = cape_instr(T[NK], RP_M, PP, T, R, P_FULL, ASCENT_FLAG, PTOP)

    RP_S = utilities.rv(ES0, PP)
    s = cape_instr(SSTK, RP_S, PP, T, R, P_FULL, ASCENT_FLAG, PTOP)

    TO = s["TOB"]
    RAT = SSTK / TO if diss_flag else 1.0

    TV0 = utilities.Trho(T[NK], R[NK], R[NK])
    TVSST = utilities.Trho(SSTK, RP_S, RP_S)
    TVAV = 0.5 * (TV0 + TVSST)
    CAT_raw = (m["CAPED"] - CAPEA) + 0.5 * CKCD * RAT * (s["CAPED"] - m["CAPED"])
    CAT = max(CAT_raw, 0.0)
    PNEW = MSL * np.exp(-CAT / (constants.RD * TVAV))
    return dict(PNEW=PNEW, CAPEM=m["CAPED"], CAPEMS=s["CAPED"], TO=TO,
                RAT=RAT, CAT_raw=CAT_raw, CAT=CAT,
                INB_M=m["INB"], INB_S=s["INB"],
                PLCL_M=m["PLCL"], PLCL_S=s["PLCL"],
                LNB_M=m["LNB"], LNB_S=s["LNB"],
                PA_M=m["PA"], NA_M=m["NA"], PAT_M=m["PAT"],
                PA_S=s["PA"], NA_S=s["NA"], PAT_S=s["PAT"],
                TVRDIF_M=m["TVRDIF"], TVRDIF_S=s["TVRDIF"])


if __name__ == "__main__":
    # 1) Reproduce the raw (no-rescue) iteration to find the 2-cycle
    print("=== Raw fixed-point iteration (no rescue), x0=970 ===")
    PM, PMOLD, PNEW, NP = 970.0, 970.0, 0.0, 0
    hist = []
    while abs(PNEW - PMOLD) > 0.5 and NP <= 30:
        out = gmap(PM)
        PNEW = out["PNEW"]
        hist.append((NP, PM, PNEW, out["CAPEM"], out["CAPEMS"], out["TO"],
                     out["INB_M"], out["INB_S"], out["CAT_raw"]))
        PMOLD, PM = PM, PNEW
        NP += 1
    for h in hist[:30]:
        print(
            f"n={h[0]:3d} PM={h[1]:.10f} -> PNEW={h[2]:.10f} "
            f"CAPEM={h[3]:9.4f} CAPEMS={h[4]:9.4f} TO={h[5]:8.3f} "
            f"INB_M={h[6]:2d} INB_S={h[7]:2d} CAT={h[8]:8.4f}"
        )
