"""mam4: MAM4 modal sulfate microphysics (accumulation, Aitken, coarse, primary carbon).

    physics   MAM4-JAX core + Dunne nucleation + Tabazadeh water (the AIDE-SAI "run 3" physics) -> physics.step
    emulator  none yet. A MAM4 emulator goes in schemes/mam4/emulator.py with the same load()/step()
              shape as tomas40 (docs/ADDING_A_SCHEME.md).
"""
import importlib

from ...interface import Scheme


def _to_bins():
    from .bins import modes_to_bins

    def f(num, so4):
        """(n, 4) modal mixing ratios -> (n, 40) binned number / mass, same units."""
        nb, mb, _, _ = modes_to_bins(num.T, so4.T)
        return nb.T, mb.T
    return f


SCHEME = Scheme(
    name="mam4",
    nrows=4,
    units={"num": "(n, 4) # per kg air", "so4": "(n, 4) kg SO4(96) per kg air",
           "h2so4": "(n,) kg/kg", "so2": "(n,) kg/kg", "T": "K", "P": "Pa", "RH": "0-1",
           "OH": "molec/cm3"},
    physics=lambda: importlib.import_module(".physics", __name__),
    emulator=None,
    to_bins=_to_bins,
    description="MAM4 4-mode modal sulfate aerosol (MAM4-JAX)",
)
