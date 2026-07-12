"""g_fix(PM) scan for PR77 (max-W CAPE) + iterates; append-style save."""

import os
import sys

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)

import numpy as np
from maxw import gmap_fix

pms = np.arange(946.0, 958.0, 0.01)
gs = np.array([gmap_fix(x)["PNEW"] for x in pms])

# iterates
PM, PMOLD, PNEW, NP = 970.0, 970.0, 0.0, 0
xs = []
while abs(PNEW - PMOLD) > 0.5 and NP <= 60:
    PNEW = gmap_fix(PM)["PNEW"]
    xs.append(PM)
    PMOLD, PM = PM, PNEW
    NP += 1
xs.append(PM)

np.savez(os.path.join(SCRATCH, "fixdata.npz"), pm=pms, g=gs, iters=np.array(xs))
print(f"g_fix scan saved; converged in {NP} iterations to {PM:.6f}")

# sanity: humpdata cape_pc continuity check
h = np.load(os.path.join(SCRATCH, "humpdata.npz"))
dpc = np.max(np.abs(np.diff(h["cape_pc"])))
dmw = np.max(np.abs(np.diff(h["cape_mw"])))
dlnb = np.nanmax(np.abs(np.diff(h["lnb_mw"])))
print(f"hump sweep: max step cape_pcmin={dpc:.3f} cape_maxW={dmw:.3f} lnb_maxW={dlnb:.1f}")
