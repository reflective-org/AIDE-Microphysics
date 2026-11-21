# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""tomas40 emulators: load a published model and run one 6 h step on per-box arrays.

    from aide_micro.schemes.tomas40 import emulator as E
    emu = E.load("bintf_d384")                       # weights from the Hugging Face Hub (cached)
    Nk2, Mk2, H2, S2 = emu.step(Nk, Mk, H, S, T, P, RH, OH13)

Inputs per box (torch float64 on the device, or numpy): Nk, Mk (n, 40) [#/box, kg SO4/box],
H, S (n,) H2SO4 / SO2 [kg/box], T [K], P [Pa], RH [0-1] (n,), OH13 (n, 13) [molec/cm3] on the
30-min grid of the step (both ends). Box volume 1e6 cm3. Returns the state after 6 h.

Which model to use: docs/MODELS.md. In short: bintf_d384 standalone (default), bintf_d256
standalone and 2.6x cheaper, cs_b only in the coupled model with TOMAS physics on the top cells.

Precision: fp16 autocast + torch.compile(reduce-overhead) is the deployment recipe (lossless vs
fp32 at every horizon measured); bf16 is refused (14-day tails widen). Batches are padded to a
fixed size so the compiled graph is reused.
"""
import json
import os

import numpy as np
import torch

from . import bins as B
from . import models as M
from . import project as PJ

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")


def manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def weights_path(name, local_dir=None):
    """Local path of a model's weights: `local_dir/<file>` if given, else downloaded from the Hub (sha256 checked)."""
    m = manifest()
    if name not in m["models"]:
        raise ValueError(f"unknown tomas40 model {name!r}; available: {list(m['models'])}")
    e = m["models"][name]
    if local_dir:
        p = os.path.join(local_dir, e["file"])
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(m["hf_repo"], e["file"], revision=m.get("hf_revision"))
    _check_sha(p, e["sha256"])
    return p


def _check_sha(p, want):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != want:
        raise RuntimeError(f"{p}: sha256 {h.hexdigest()} != manifest {want}")


class Emulator:
    def __init__(self, name, step6h, norm, device, batch, project="chem"):
        if project not in ("chem", "clip", "none"):
            raise ValueError("project must be chem | clip | none")
        self.name, self.step6h, self.norm, self.device, self.batch, self.project = name, step6h, norm, device, batch, project
        self.last = {}

    def _t(self, a):
        if torch.is_tensor(a):
            return a.to(self.device, torch.float64)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)).to(self.device)

    @torch.no_grad()
    def forward_norm(self, s_all, c_all):
        """Network only: normalized (n, 82), (n, 13, 6) -> normalized (n, 82), fixed-size padded batches."""
        n = s_all.shape[0]; bs = self.batch
        out = torch.empty((n, B.N_STATE), dtype=torch.float32, device=self.device)
        for a in range(0, n, bs):
            b = min(a + bs, n); k = b - a
            s = s_all[a:b]; c = c_all[a:b]
            if k < bs:
                s = torch.cat([s, s[:1].expand(bs - k, -1)], 0)
                c = torch.cat([c, c[:1].expand(bs - k, -1, -1)], 0)
            out[a:b] = self.step6h(s, c)[:k]
        return out

    @torch.no_grad()
    def step(self, Nk, Mk, H, S, T, P, RH, OH13, OHbar=None, k1=None):
        """One 6 h step (see module docstring). OHbar: step-mean OH (default: trapezoid of OH13);
        k1: SO2+OH rate constant per cell (default: computed). Both only used by project='chem'."""
        was_np = not torch.is_tensor(Nk)
        Nk, Mk, H, S, T, P, RH, OH13 = (self._t(a) for a in (Nk, Mk, H, S, T, P, RH, OH13))
        n = Nk.shape[0]
        lin = torch.empty((n, B.N_STATE), dtype=torch.float32, device=self.device)
        lin[:, 0:40] = Nk.clamp(min=B.NK_FLOOR); lin[:, 40:80] = Mk.clamp(min=B.MK_FLOOR)
        lin[:, 80] = H.clamp(min=B.MK_FLOOR); lin[:, 81] = S.clamp(min=B.SO2_FLOOR)
        s_all = self.norm.norm_state(torch.log(lin)); del lin
        c_all = torch.empty((n, B.N_END, B.N_COND), dtype=torch.float32, device=self.device)
        c_all[:, :, 0] = torch.log(P).float()[:, None]; c_all[:, :, 1] = T.float()[:, None]
        c_all[:, :, 2] = RH.clamp(B.RH_MIN, 1.0).float()[:, None]
        c_all[:, :, 3] = torch.log(OH13.clamp(min=B.OH_FLOOR)).float()
        c_all[:, :, 4] = float(np.log(B.LOG_ZERO)); c_all[:, :, 5] = float(np.log(B.LOG_ZERO))
        c_all = self.norm.norm_cond(c_all)
        out = self.forward_norm(s_all, c_all)
        E = torch.exp(torch.clamp(self.norm.denorm_state(out), max=60.0)).double()
        empty = E[:, 0:40] < B.NEPS
        Nk2 = torch.where(empty, 0.0, E[:, 0:40]); Mk2 = torch.where(empty, 0.0, E[:, 40:80])
        H2, S2 = E[:, 80], E[:, 81]
        self.last = {}
        if self.project != "none":
            Mk2 = PJ.clip(Nk2, Mk2)
        if self.project == "chem":
            if OHbar is None:
                OHbar = (0.5 * (OH13[:, 0] + OH13[:, -1]) + OH13[:, 1:-1].sum(1)) / (B.N_END - 1)
            else:
                OHbar = self._t(OHbar)
            k1 = PJ.calc_k1(T, P, RH) if k1 is None else self._t(k1)
            Nk2, Mk2, H2, S2, self.last = PJ.project_chem(Mk, H, S, Nk2, Mk2, H2, k1, OHbar, T, P, cs_pad=n)
        if was_np:
            return tuple(x.cpu().numpy() for x in (Nk2, Mk2, H2, S2))
        return Nk2, Mk2, H2, S2


def load(name="bintf_d384", device="cuda", dtype="fp16", compile_model=True, batch=16384,
         project="chem", local_dir=None):
    """Load a published tomas40 model. dtype fp16 | fp32 (bf16 refused). compile_model uses
    torch.compile(mode='reduce-overhead') on CUDA. local_dir: read weights from a directory
    instead of the Hub."""
    if dtype not in ("fp16", "fp32"):
        raise ValueError("dtype must be fp16 or fp32 (bf16 widens the 14-day tails; docs/MODELS.md)")
    e = manifest()["models"][name] if name in manifest()["models"] else None
    path = weights_path(name, local_dir)
    ck = torch.load(path, map_location=device, weights_only=True)
    net, core, last = M.build(e["arch"], e["config"])
    net.load_state_dict(ck["model"])
    core = core.to(device).eval()
    dt = torch.float16 if (dtype == "fp16" and device == "cuda") else torch.float32
    fwd = torch.compile(core, mode="reduce-overhead") if (compile_model and device == "cuda") else core

    @torch.no_grad()
    def step6h(s, c):
        with torch.autocast(device, dtype=dt, enabled=(dt != torch.float32)):
            out = fwd(s, c)
            if last:
                out = out[:, -1]
        return torch.nan_to_num(out.float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50, 50)

    return Emulator(name, step6h, B.NormStats.load(), device, batch, project)
