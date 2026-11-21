# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""The contract every aerosol scheme in AIDE-Microphysics provides.

A *scheme* is an aerosol representation (TOMAS 40 sectional bins, MAM4 4 modes, ...). Each scheme
has up to two *engines* with the same per-box call signature:

    physics   the reference process model (TOMAS-JAX, MAM4-JAX), JAX, float64
    emulator  learned 6 h operators trained on that physics (torch)

so a host model can swap engines, or compare them on the same state, without other changes.
Everything is per box (one row per grid cell) and one step is 6 hours; transport, emissions,
settling and radiation belong to the host model.

Scheme attributes (see schemes/<name>/__init__.py):
    name       short id ("tomas40", "mam4")
    nrows      rows per moment in the host's tracer stack (40 bins, 4 modes)
    units      dict describing the state arrays the engines take and return
    physics    module with step(...) -> next state
    emulator   module with load(model_name) -> object with .step(...), or None
    to_bins    function mapping the scheme's state to the common TOMAS 40-bin view (Nk, Mk), used
               for optics and size diagnostics so every scheme is compared through one code path
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Scheme:
    name: str
    nrows: int
    units: dict
    physics: Callable[[], Any]                  # lazy: returns the physics module (heavy imports)
    emulator: Optional[Callable[[], Any]]       # lazy: returns the emulator module, or None
    to_bins: Callable[[], Callable]             # lazy: returns the to-bins function
    description: str = ""
