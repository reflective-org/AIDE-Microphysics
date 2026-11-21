# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""TOMAS 40-bin layout, floors, log transform and the emulators' normalization.

State vector (82), natural logs of per-box quantities, box volume 1e6 cm3 (1 m3):
    [0:40]   Nk    particle number per bin   [particles / box]   floor 1e-20
    [40:80]  Mk    sulfate mass per bin      [kg SO4 / box]      floor 1e-30
    [80]     H2SO4 gas                       [kg / box]          floor 1e-30
    [81]     SO2 gas                         [kg / box]          floor 1e-17

Forcing on the 30-min grid of one 6 h step (13 points, both ends), 6 channels:
    0 log P [Pa] · 1 T [K] · 2 RH [0..1] · 3 log OH [molec/cm3] (floor 100)
    4 log SO2 source [kg/box/30 min] · 5 log dilution [1/s]   (log 1e-30 = none)

Floors are load-bearing (found in coupled runs): a zero SO2 logged at 1e-30 is
7 sigma outside the training data and produced 1660x the physics' mass; a zero
night-time OH is 17 sigma out and produced spurious 1e8 cm-3 nucleation bursts.

Bin k spans dry mass XK0 * 2**k .. 2**(k+1), XK0 = pi/6 (1.7 nm)^3 * 1770 kg/m3.
"""
import os

import numpy as np
import torch

N_BINS, N_COND, N_STATE, N_END = 40, 6, 82, 13
BOX = 1e6                                   # cm3 per box (the models were trained at 1 m3)
NK_FLOOR, MK_FLOOR, GAS_FLOOR, SO2_FLOOR = 1e-20, 1e-30, 1e-30, 1e-17
OH_FLOOR, RH_MIN, LOG_ZERO = 100.0, 0.01, 1e-30
NEPS = 1e-5                                 # below this many particles per box a bin is empty
DT_STEP = 21600.0                           # every model is a 6 h operator
RHO_SO4 = 1770.0
XK0 = np.pi / 6.0 * (1.7e-9) ** 3 * RHO_SO4
# (41,) bin-edge dry masses [kg]: XK0 * 2**k, stored bit-exactly as TOMAS-JAX computes them on the GPU
# (jnp.power there rounds 5 of the 41 edges one ulp below numpy's XK0 * 2.0**k). The coupled model uses
# these values; with numpy's, outputs differ from it in the last bit.
XK = np.array([float.fromhex(h) for h in (
    "0x1.6049d1385be99p-78", "0x1.6049d1385be99p-77", "0x1.6049d1385be99p-76", "0x1.6049d1385be99p-75",
    "0x1.6049d1385be99p-74", "0x1.6049d1385be99p-73", "0x1.6049d1385be99p-72", "0x1.6049d1385be99p-71",
    "0x1.6049d1385be99p-70", "0x1.6049d1385be99p-69", "0x1.6049d1385be99p-68", "0x1.6049d1385be98p-67",
    "0x1.6049d1385be99p-66", "0x1.6049d1385be99p-65", "0x1.6049d1385be99p-64", "0x1.6049d1385be99p-63",
    "0x1.6049d1385be99p-62", "0x1.6049d1385be99p-61", "0x1.6049d1385be99p-60", "0x1.6049d1385be99p-59",
    "0x1.6049d1385be99p-58", "0x1.6049d1385be98p-57", "0x1.6049d1385be99p-56", "0x1.6049d1385be99p-55",
    "0x1.6049d1385be99p-54", "0x1.6049d1385be99p-53", "0x1.6049d1385be99p-52", "0x1.6049d1385be99p-51",
    "0x1.6049d1385be99p-50", "0x1.6049d1385be98p-49", "0x1.6049d1385be99p-48", "0x1.6049d1385be98p-47",
    "0x1.6049d1385be99p-46", "0x1.6049d1385be99p-45", "0x1.6049d1385be99p-44", "0x1.6049d1385be99p-43",
    "0x1.6049d1385be99p-42", "0x1.6049d1385be99p-41", "0x1.6049d1385be99p-40", "0x1.6049d1385be98p-39",
    "0x1.6049d1385be99p-38")])
M_MID = np.sqrt(XK[:-1] * XK[1:])           # (40,) bin-centre mass [kg]
D_EDGE_M = 1.7e-9 * 2.0 ** (np.arange(N_BINS + 1) / 3.0)
D_MID_UM = 1.7e-3 * 2.0 ** ((np.arange(N_BINS) + 0.5) / 3.0)

NORM_STATS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "norm_stats.npz")


def state_to_log(Nk, Mk, h2so4_kg, so2_kg):
    """Per-box linear quantities (numpy) -> (..., 82) float32 log-state, floors applied."""
    Nk = np.maximum(np.asarray(Nk, dtype=np.float64), NK_FLOOR)
    Mk = np.maximum(np.asarray(Mk, dtype=np.float64), MK_FLOOR)
    g4 = np.maximum(np.asarray(h2so4_kg, dtype=np.float64), GAS_FLOOR)[..., None]
    g2 = np.maximum(np.asarray(so2_kg, dtype=np.float64), SO2_FLOOR)[..., None]
    return np.log(np.concatenate([Nk, Mk, g4, g2], -1)).astype(np.float32)


def log_to_state(s):
    """(..., 82) log-state -> dict of linear per-box quantities (exp clamped at e^60)."""
    s = np.minimum(np.asarray(s, dtype=np.float64), 60.0)
    return {"Nk": np.exp(s[..., :N_BINS]), "Mk": np.exp(s[..., N_BINS:2 * N_BINS]),
            "h2so4_kg": np.exp(s[..., 80]), "so2_kg": np.exp(s[..., 81])}


def cond_to_log(pres_pa, temp_k, rh, oh_cm3, so2_src_kg_per_30min=0.0, kdil_per_s=0.0):
    """Forcing arrays (..., 13) [scalars broadcast] -> (..., 13, 6) float32 in the training convention."""
    pres_pa, temp_k, rh, oh_cm3, src, kdil = np.broadcast_arrays(
        *[np.asarray(a, dtype=np.float64) for a in (pres_pa, temp_k, rh, oh_cm3, so2_src_kg_per_30min, kdil_per_s)])
    c = np.stack([np.log(pres_pa), temp_k, np.clip(rh, 1e-3, 1.0),
                  np.log(np.maximum(oh_cm3, OH_FLOOR)),
                  np.log(np.maximum(src, LOG_ZERO)), np.log(np.maximum(kdil, LOG_ZERO))], -1)
    return c.astype(np.float32)


class NormStats:
    """Per-channel mean / std of the log-state and forcing over the training corpus.
    All shipped tomas40 models use this one file. Never recompute it: the weights are tied to it."""

    def __init__(self, sm, ss, cm, cs):
        self.state_mean = torch.as_tensor(sm, dtype=torch.float32)
        self.state_std = torch.as_tensor(ss, dtype=torch.float32)
        self.cond_mean = torch.as_tensor(cm, dtype=torch.float32)
        self.cond_std = torch.as_tensor(cs, dtype=torch.float32)

    @classmethod
    def load(cls, path=NORM_STATS):
        d = np.load(path)
        return cls(d["sm"], d["ss"], d["cm"], d["cs"])

    def norm_state(self, s):
        return (s - self.state_mean.to(s.device)) / self.state_std.to(s.device)

    def denorm_state(self, s):
        return s * self.state_std.to(s.device) + self.state_mean.to(s.device)

    def norm_cond(self, c):
        return (c - self.cond_mean.to(c.device)) / self.cond_std.to(c.device)
