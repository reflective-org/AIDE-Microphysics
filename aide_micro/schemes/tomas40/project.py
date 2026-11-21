# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""Output projection applied after every 6 h emulator step (torch float64).

The network satisfies two constraints of the box model only approximately:
  (a) two-moment consistency, Mk in [Nk * xk_lo, Nk * xk_hi]  (TOMAS's mnfix);
  (b) sulfur conservation between SO2, H2SO4 and particulate SO4.

`clip` enforces (a). `chem` (the production setting of the coupled model since
2026-10-13) enforces (a) and replaces the network's gas answer with the exact
chemistry: SO2 decays as exp(-k1 OH dt); H2SO4 relaxes to production / sink with
the physics' own Fuchs-Sutugin condensation sink; the aerosol gains exactly the
sulfur the gas lost, deposited by the sink fraction; then the bounds are restored
without creating or destroying sulfur. Keeping the network's own H2SO4 instead
raised the active-cell mass p90 error from 5.9 to 9.9 %.

`chem` needs tomas_jax (the rate constant and the condensation sink are the
physics' own functions). Ported from AIDE-SAI-link src/microphysics/tomas_emulator.py
(project, project_chem, cond_sink; EMU_CHEM_NCLIP=1, EMU_LOSTFIX=0).
"""
import numpy as np
import torch

from .bins import XK as _XK_NP, DT_STEP, RHO_SO4, BOX

_MW_SO2, _MW_H2SO4, _MW_SO4 = 64.066, 98.0, 96.0
ALPHA_COND = 1.0          # H2SO4 accommodation coefficient (the coupled model's SAI value)


class _Consts:
    """Device-resident bin constants, built once per device."""
    _cache = {}

    @classmethod
    def get(cls, device):
        k = str(device)
        if k not in cls._cache:
            xk = torch.tensor(np.asarray(_XK_NP), dtype=torch.float64, device=device)
            mmid = torch.sqrt(xk[:-1] * xk[1:])
            dp = (6.0 * mmid / (np.pi * RHO_SO4)) ** (1.0 / 3.0)
            cls._cache[k] = (xk, mmid, dp ** 2)
        return cls._cache[k]


def clip(Nk2, Mk2):
    xk, _, _ = _Consts.get(Nk2.device)
    return torch.minimum(torch.maximum(Mk2, Nk2 * xk[:-1]), Nk2 * xk[1:])


# ---- jax pieces (lazy: only `chem` needs tomas_jax) ---------------------------
_JX = {}


def _jax():
    if not _JX:
        import jax
        import jax.numpy as jnp
        from tomas_jax.fast.condensation import _condensation_sink_cell
        from tomas_jax.fast.config import SRTSO4
        from tomas_jax.physics.so2_chemistry import calc_k1_so2_oh
        xk = jnp.asarray(_XK_NP)
        _JX.update(jax=jax, jnp=jnp, SRTSO4=SRTSO4, k1=calc_k1_so2_oh,
                   cs=jax.jit(jax.vmap(lambda n, m, t, p: _condensation_sink_cell(n, m, t, p, BOX, xk, ALPHA_COND))))
    return _JX


def _to_jax(t):
    return _jax()["jnp"].asarray(t) if not torch.is_tensor(t) else _jax()["jax"].dlpack.from_dlpack(t.contiguous())


def _to_torch(a, device):
    if isinstance(a, np.ndarray) or not hasattr(a, "__dlpack__"):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)).to(device)
    return torch.from_dlpack(a).to(device, torch.float64)


def calc_k1(T, P, RH):
    """SO2 + OH rate constant [cm3 molec-1 s-1] (tomas_jax.physics.so2_chemistry)."""
    return _to_torch(_jax()["k1"](_to_jax(T.contiguous()), _to_jax(P.contiguous()), _to_jax(RH.contiguous())), T.device)


def cond_sink(Nk, Mk, T, P, pad_to=0):
    """The physics' H2SO4 condensation sink per cell: (CS (n,) 1/s, sink fraction (n, 40)).
    pad_to > n pads with copies of row 0 to one fixed jit shape (the per-cell math is independent)."""
    J = _jax(); n = Nk.shape[0]; dev = Nk.device
    Mk3 = torch.zeros((n, Nk.shape[1], 2), dtype=torch.float64, device=dev)
    Mk3[:, :, J["SRTSO4"]] = Mk
    if pad_to and n < pad_to:
        p = pad_to - n
        Nk, T, P = (torch.cat([x, x[:1].expand(p, *x.shape[1:])], 0) for x in (Nk, T, P))
        Mk3 = torch.cat([Mk3, Mk3[:1].expand(p, *Mk3.shape[1:])], 0)
    cs, sf = J["cs"](_to_jax(Nk.contiguous()), _to_jax(Mk3), _to_jax(T.contiguous()), _to_jax(P.contiguous()))
    return _to_torch(cs, dev)[:n], _to_torch(sf, dev)[:n]


def project_chem(Mk, H, S, Nk2, Mk2, H2, k1, OHbar, T, P, dt=DT_STEP, cs_pad=0):
    """Mk, H, S: input state; Nk2, Mk2, H2: the network's (clipped) answer; k1, OHbar, T, P per cell.
    Returns Nk2, Mk2, H2, S2 and a diagnostics dict. All torch float64, one device."""
    xk, _, dp2 = _Consts.get(Mk.device)
    S2 = S * torch.exp(-k1 * OHbar * dt)
    produced = (S - S2) * (_MW_H2SO4 / _MW_SO2)                     # kg H2SO4 made in the step
    avail = H + produced
    CS, sf = cond_sink(Nk2, Mk2, T, P, cs_pad)
    hss = (produced / dt) / CS.clamp(min=1e-30)
    H2 = torch.where(CS > 0.0, hss + (H - hss) * torch.exp(-CS * dt), avail)
    H2 = torch.minimum(H2, avail)
    dM_target = (avail - H2) * (_MW_SO4 / _MW_H2SO4)                # kg SO4 the aerosol must gain (>= 0)
    resid = dM_target - (Mk2.sum(1) - Mk.sum(1))
    area = Nk2 * dp2
    wgt = torch.where(sf.sum(1, keepdim=True) > 0.0, sf, area / area.sum(1, keepdim=True).clamp(min=1e-300))
    # a negative residual (network added more mass than the budget allows) is removed multiplicatively
    target_tot = Mk.sum(1) + dM_target
    scale = (target_tot / Mk2.sum(1).clamp(min=1e-300)).clamp(min=0.0)
    Mk2 = torch.where((resid > 0.0)[:, None], Mk2 + resid.clamp(min=0.0)[:, None] * wgt, Mk2 * scale[:, None]).clamp(min=0.0)
    # conserving bounds: number down where mass is short, excess mass redistributed into bins with room
    Nk2 = torch.where(Mk2 < Nk2 * xk[:-1], Mk2 / xk[:-1], Nk2)
    hi = Nk2 * xk[1:]
    excess = (Mk2 - hi).clamp(min=0.0); Mk2 = Mk2 - excess; extra = excess.sum(1)
    for _ in range(2):
        room = (hi - Mk2).clamp(min=0.0); tot = room.sum(1)
        add = torch.where(tot > 0.0, extra / tot.clamp(min=1e-300), torch.zeros_like(tot))[:, None] * room
        add = torch.minimum(add, room); Mk2 = Mk2 + add; extra = (extra - add.sum(1)).clamp(min=0.0)
    lost = extra
    rel = resid.abs() / Mk.sum(1).clamp(min=1e-30)
    diag = dict(chem_resid_rel_p50=float(rel.median()), chem_resid_rel_p99=float(torch.quantile(rel, 0.99)),
                chem_gain_ratio=float((Mk2.sum(1) - Mk.sum(1)).sum() / dM_target.sum().clamp(min=1e-300)),
                chem_lost=float(lost.sum() / Mk.sum().clamp(min=1e-300)))
    return Nk2, Mk2, H2, S2, diag
