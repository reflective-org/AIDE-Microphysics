"""AIDE-Microphysics: aerosol microphysics schemes (reference physics + learned emulators) behind one
per-box interface. See interface.py for the contract and docs/MODELS.md for which model to use."""
from .interface import Scheme
from .registry import available, get

__all__ = ["Scheme", "available", "get"]
__version__ = "0.1.0"
