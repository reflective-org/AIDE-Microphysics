# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""tomas40 reference physics: TOMAS-JAX `fast` (gpu-fast branch) on per-box arrays, with the
production settings of the AIDE-SAI-link coupled model (src/microphysics/tomas_fast.py and the
physics subset of tomas_emulator.py).

    from aide_micro.schemes.tomas40 import physics as PH
    Nk2, Mk2, H2, S2 = PH.step(Nk, Mk, H, S, T, P, RH, OH)     # one 6 h step

Same per-box conventions as the emulator (Nk, Mk (n, 40) per box, H/S kg per box, box 1e6 cm3),
except OH: (60, n) [molec/cm3], one value per 360 s inner step (or (n,) constant).

Inner step 360 s x 60; adaptive coagulation (cap 256 substeps, c_max 0.05) and condensation
(cap 40); stiffness-sorted chunks; SO2 + OH chemistry interleaved by run_fast itself.
Cells are padded with empty boxes to a multiple of `pad` so the jit cache is reused
(adaptive substepping makes empty rows nearly free).

The coupled model's EMU_PHYS_EXACT / EMU_COAG_PACK coagulation kernels are speed-only and
bit-identical to the stock kernel used here.
"""
import numpy as np

from .bins import XK, DT_STEP, BOX, N_BINS

FAST_DT, COAG_CAP, COND_CAP, COAG_CMAX, ALPHA_COND, FN_SCALE = 360.0, 256, 40, 0.05, 1.0, 1.0
N_INNER = int(round(DT_STEP / FAST_DT))     # 60


def step(Nk, Mk, H, S, T, P, RH, OH, pad=4096, n_cell_chunks=1, steps_per_segment=10):
    """One 6 h step of TOMAS-JAX fast on n boxes (numpy in, numpy out). Returns Nk2, Mk2, H2, S2 and
    stores diagnostics in step.last (cap hits, substep counts)."""
    from tomas_jax.fast import FastState, run_fast
    from tomas_jax.fast.config import NBINS, SRTSO4, GH2SO4, GSO2
    assert NBINS == N_BINS
    f = lambda a: np.asarray(a, dtype=np.float64)
    k = f(Nk).shape[0]; kp = -(-k // pad) * pad if pad else k
    Nk_np = np.zeros((kp, N_BINS)); Nk_np[:k] = f(Nk)
    Mk3 = np.zeros((kp, N_BINS, 2)); Mk3[:k, :, SRTSO4] = f(Mk)
    Gc = np.zeros((kp, 2)); Gc[:k, GH2SO4] = f(H); Gc[:k, GSO2] = f(S)
    T_np = np.full(kp, 230.0); T_np[:k] = f(T)
    P_np = np.full(kp, 5000.0); P_np[:k] = f(P)
    RH_np = np.full(kp, 0.3); RH_np[:k] = f(RH)
    OH = f(OH)
    if OH.ndim == 1:
        OH = np.repeat(OH[None], N_INNER, 0)
    OH_np = np.zeros((OH.shape[0], kp)); OH_np[:, :k] = OH
    BV_np = np.full(kp, BOX)
    state = FastState.create(Nk=Nk_np, Mk=Mk3, Gc=Gc, xk=np.asarray(XK), temp=T_np, pres=P_np,
                             boxvol=BV_np, rh=RH_np)
    state2, diags = run_fast(state, N_INNER, dt=FAST_DT, oh_conc=OH_np, n_cell_chunks=n_cell_chunks,
                             sort_by_coag_cost=True, steps_per_segment=steps_per_segment,
                             alpha=ALPHA_COND, fn_scale=FN_SCALE,
                             coag_sub_cap=COAG_CAP, cond_sub_cap=COND_CAP, coag_c_max=COAG_CMAX)
    step.last = {"cells": k, "padded": kp,
                 "coag_cap_hit": int(np.asarray(diags["coag_cap_hit"]).sum()),
                 "cond_cap_hit": int(np.asarray(diags["cond_cap_hit"]).sum())}
    return (np.asarray(state2.Nk)[:k], np.asarray(state2.Mk[:, :, SRTSO4])[:k],
            np.asarray(state2.Gc[:, GH2SO4])[:k], np.asarray(state2.Gc[:, GSO2])[:k])


step.last = {}
