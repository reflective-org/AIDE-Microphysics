# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""Look up schemes by name: aide_micro.get("tomas40"), aide_micro.available()."""
from .schemes import mam4, tomas40

_SCHEMES = {s.name: s for s in (tomas40.SCHEME, mam4.SCHEME)}


def available():
    return sorted(_SCHEMES)


def get(name):
    if name not in _SCHEMES:
        raise KeyError(f"unknown scheme {name!r}; available: {available()}")
    return _SCHEMES[name]
