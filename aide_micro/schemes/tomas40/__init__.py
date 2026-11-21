# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""tomas40: TOMAS sectional sulfate microphysics, 40 mass-doubling bins (1.7 nm .. ~10 um).

    physics   TOMAS-JAX `fast` (nucleation Dunne 2016, condensation, coagulation, SO2+OH)  -> physics.step
    emulator  bintf_d384 (default), bintf_d256, cs_b                                         -> emulator.load
"""
import importlib

from ...interface import Scheme


def _identity_bins():
    return lambda Nk, Mk: (Nk, Mk)


SCHEME = Scheme(
    name="tomas40",
    nrows=40,
    units={"Nk": "(n, 40) particles per box", "Mk": "(n, 40) kg SO4 per box",
           "H2SO4": "(n,) kg per box", "SO2": "(n,) kg per box", "box": "1e6 cm3",
           "T": "K", "P": "Pa", "RH": "0-1", "OH": "molec/cm3"},
    physics=lambda: importlib.import_module(".physics", __name__),
    emulator=lambda: importlib.import_module(".emulator", __name__),
    to_bins=_identity_bins,
    description="TOMAS 40-bin sectional sulfate aerosol",
)
