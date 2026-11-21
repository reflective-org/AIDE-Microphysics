# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""One stratospheric box with a 10 ppbv SO2 pulse, 48 h, through a tomas40 emulator.

    python examples/tomas40_box.py [--model bintf_d384|bintf_d256] [--hours 48] [--cpu]

Uses project='chem' (sulfur-conserving, as the coupled model) when TOMAS-JAX is installed; otherwise
'clip', which does not conserve sulfur: the printed mass and SO2 then do not balance (docs/MODELS.md).
"""
import argparse
import time
from math import erf

import numpy as np
import torch

from aide_micro.schemes.tomas40 import emulator
from aide_micro.schemes.tomas40.bins import BOX, D_EDGE_M, D_MID_UM, M_MID, N_END

KB, AV = 1.380649e-23, 6.02214076e23


def lognormal_bins(n_cm3, dg_nm, gsd):
    z = (np.log(D_EDGE_M) - np.log(dg_nm * 1e-9)) / (np.sqrt(2) * np.log(gsd))
    return np.maximum(n_cm3 * BOX * np.diff(0.5 * (1 + np.vectorize(erf)(z))), 1e-20)


def gas_kg(vmr, P, T, mw):
    return vmr * (P / (KB * T) * 1e-6) * BOX * (mw / 1000.0) / AV


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bintf_d384"); ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    try:
        import tomas_jax  # noqa: F401
        proj = "chem"
    except ImportError:
        proj = "clip"
        print("TOMAS-JAX not installed: project='clip' (sulfur NOT conserved; install .[tomas40] for 'chem')")
    emu = emulator.load(a.model, device=dev, dtype="fp16" if dev == "cuda" else "fp32",
                        compile_model=False, batch=1, project=proj)
    T, P, RH, oh_peak = 205.0, 5500.0, 0.05, 3e6
    Nk = lognormal_bins(300, 80, 1.6)[None]; Mk = Nk * M_MID
    H = np.array([gas_kg(0.05e-12, P, T, 98.0)]); S = np.array([gas_kg(20e-12 + 10e-9, P, T, 64.0)])
    print(f"{a.model} on {dev}, project={proj}")
    print(f"{'t [h]':>6} {'N [cm-3]':>10} {'mass [ug m-3]':>14} {'SO2 [ug/box]':>13}")
    t0 = time.time()
    for k in range(a.hours // 6 + 1):
        print(f"{6 * k:6d} {Nk.sum() / BOX:10.3e} {Mk.sum() * 1e9:14.3f} {S[0] * 1e9:13.3f}")
        if k == a.hours // 6:
            break
        t_h = 6 * k + np.arange(N_END) * 0.5
        OH13 = (oh_peak * np.maximum(0, np.cos(2 * np.pi * t_h / 24.0)))[None]
        Nk, Mk, H, S = emu.step(Nk, Mk, H, S, np.array([T]), np.array([P]), np.array([RH]), OH13)
    print(f"{a.hours} h in {(time.time() - t0) * 1e3:.0f} ms; most populated bin: "
          f"d = {D_MID_UM[int(np.argmax(Nk[0]))] * 1e3:.0f} nm")
