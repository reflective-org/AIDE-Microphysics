# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""Light checks (CPU). The bit-for-bit comparison against AIDE-SAI-link needs that tree and a GPU; it is
described in README > Provenance. Run: pytest tests/  (set AIDE_MICRO_WEIGHTS=<dir> to test local weights
instead of downloading)."""
import json
import os

import numpy as np
import pytest
import torch

from aide_micro.schemes.tomas40 import bins, models

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = json.load(open(os.path.join(HERE, "aide_micro/schemes/tomas40/models.json")))


def test_bin_edges_are_mass_doubling():
    ref = bins.XK0 * 2.0 ** np.arange(bins.N_BINS + 1)
    assert np.max(np.abs(bins.XK / ref - 1)) <= 2.3e-16       # within one ulp of XK0 * 2**k


@pytest.mark.parametrize("name", list(MAN["models"]))
def test_architecture_matches_manifest(name):
    e = MAN["models"][name]
    net, core, last = models.build(e["arch"], e["config"])
    n = sum(p.numel() for p in net.parameters()) / 1e6
    assert abs(n - float(e["params"].rstrip("M"))) < 0.05


@pytest.mark.parametrize("name", list(MAN["models"]))
def test_step_runs_and_bounds_hold(name):
    pytest.importorskip("huggingface_hub")
    from aide_micro.schemes.tomas40 import emulator
    emu = emulator.load(name, device="cpu", dtype="fp32", compile_model=False, batch=8, project="clip",
                        local_dir=os.environ.get("AIDE_MICRO_WEIGHTS"))
    r = np.random.default_rng(0); n = 5
    Nk = np.exp(r.uniform(0, 20, (n, 40))); Mk = Nk * np.sqrt(bins.XK[:-1] * bins.XK[1:])
    out = emu.step(Nk, Mk, np.full(n, 1e-12), np.full(n, 1e-9), np.full(n, 210.0), np.full(n, 5000.0),
                   np.full(n, 0.1), np.full((n, 13), 1e6))
    Nk2, Mk2 = out[0], out[1]
    assert all(np.isfinite(o).all() for o in out)
    assert (Mk2 >= Nk2 * bins.XK[:-1] * (1 - 1e-12)).all() and (Mk2 <= Nk2 * bins.XK[1:] * (1 + 1e-12)).all()
