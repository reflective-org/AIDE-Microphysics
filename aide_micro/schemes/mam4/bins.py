# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Reflective
"""MAM4 modes <-> the TOMAS 40-bin grid (for bin-based consumers: optics, settling, and comparisons
with tomas40). Mass is binned with the mass-weighted lognormal, so both moments round-trip exactly per
mode. Ported from AIDE-SAI-link src/microphysics/mam4_adapter.py (numpy versions)."""
import numpy as np
import jax
import jax.numpy as jnp

from .physics import C, D, NMODE, SIGMA_G, ALNSG, RHO_SO4

_LNXK = np.log(C.XK_NP)                      # (41,)


def _phi(z):
    # standard-normal CDF, as coupling._phi (the /sqrt(2) was missing in the
    # first version of this file: number was conserved but placed in bins drawn
    # with width s/sqrt(2), so num*MMID recovered only 47 % of the mode mass)
    return 0.5 * (1.0 + np.asarray(jax.scipy.special.erf(jnp.asarray(z) / np.sqrt(2.0))))


def mode_bin_fractions(num_m, so4_m, sigma_g):
    """Lognormal fractions of one mode's number and mass per TOMAS bin.
    num_m, so4_m: (G,). Returns (fn, fm), each (NBINS, G), zero where invalid."""
    G = num_m.shape[0]
    s = 3.0 * np.log(sigma_g)
    valid = (num_m > 0) & (so4_m > 0)
    amm = np.where(valid, so4_m / np.maximum(num_m, 1e-300), 1.0)
    lnmg = np.log(amm) - 0.5 * s * s             # number-median mass
    lnmg_mass = lnmg + s * s                     # mass-median mass
    fn = np.diff(_phi((_LNXK[:, None] - lnmg[None, :]) / s), axis=0)
    fm = np.diff(_phi((_LNXK[:, None] - lnmg_mass[None, :]) / s), axis=0)
    fn[:, ~valid] = 0.0; fm[:, ~valid] = 0.0
    return fn, fm


def modes_to_bins(num_m, so4_m):
    """(4, ...) modal -> (NBINS, ...) binned num/mas plus the per-(bin, mode)
    fractions needed to refit after settling. Returns
    num_bin, mas_bin, fn (NBINS, 4, G), fm (NBINS, 4, G) with G = prod(...)."""
    shape = num_m.shape[1:]
    G = int(np.prod(shape))
    nm = np.asarray(num_m, np.float64).reshape(NMODE, G)
    sm = np.asarray(so4_m, np.float64).reshape(NMODE, G)
    fn = np.zeros((C.NBINS, NMODE, G)); fm = np.zeros((C.NBINS, NMODE, G))
    for m in range(NMODE):
        fn[:, m], fm[:, m] = mode_bin_fractions(nm[m], sm[m], SIGMA_G[m])
    num_bin = (fn * nm[None]).sum(1).reshape(C.NBINS, *shape)
    mas_bin = (fm * sm[None]).sum(1).reshape(C.NBINS, *shape)
    return num_bin, mas_bin, fn, fm


def bins_to_modes(num_bin, mas_bin, fn, fm, num_m, so4_m):
    """Refit the modes from (settled) bins: each bin's number/mass is handed
    back to the modes in the proportion they contributed to that bin before
    settling (fn, fm from modes_to_bins), conserving total number and mass per
    cell exactly. On a static state this is the identity up to the lognormal
    tails that fall outside the 40-bin grid (measured in STATUS.md)."""
    shape = num_bin.shape[1:]
    G = int(np.prod(shape))
    nb = np.asarray(num_bin, np.float64).reshape(C.NBINS, G)
    mb = np.asarray(mas_bin, np.float64).reshape(C.NBINS, G)
    nm = np.asarray(num_m, np.float64).reshape(NMODE, G)
    sm = np.asarray(so4_m, np.float64).reshape(NMODE, G)
    wn = fn * nm[None]; wm = fm * sm[None]                  # (NBINS, 4, G)
    sn = wn.sum(1, keepdims=True); smm = wm.sum(1, keepdims=True)
    # share of bin k owned by mode m (bins nobody filled keep their content
    # with the mode of the nearest filled bin's dominant owner is overkill:
    # such bins are empty before settling, so give them to the mode whose
    # size range contains the bin -- see _BIN_HOME_MODE)
    share_n = np.where(sn > 0, wn / np.where(sn > 0, sn, 1.0), _BIN_HOME[:, :, None])
    share_m = np.where(smm > 0, wm / np.where(smm > 0, smm, 1.0), _BIN_HOME[:, :, None])
    num_new = (share_n * nb[:, None, :]).sum(0).reshape(NMODE, *shape)
    so4_new = (share_m * mb[:, None, :]).sum(0).reshape(NMODE, *shape)
    return num_new, so4_new


def _bin_home_mode():
    """Fallback owner of a bin that was empty before settling: the sulfate
    mode whose [dgnumlo, dgnumhi] range is nearest the bin's dry diameter (in
    log space). Primary carbon never owns a bin."""
    dp = C.DP_BIN * 1e-9
    home = np.zeros((C.NBINS, NMODE))
    lo = np.asarray(D.DGNUMLO_AMODE); hi = np.asarray(D.DGNUMHI_AMODE)
    for k in range(C.NBINS):
        best, bestd = 0, np.inf
        for m in range(NMODE - 1):            # modes 0..2 only
            d = 0.0 if lo[m] <= dp[k] <= hi[m] else min(abs(np.log(dp[k] / lo[m])), abs(np.log(dp[k] / hi[m])))
            if d < bestd:
                best, bestd = m, d
        home[k, best] = 1.0
    return home


_BIN_HOME = _bin_home_mode()


# =============================================================================
# per-mode size diagnostics (TOMAS size convention: SO4(96) at 1770 kg/m3)
# =============================================================================
def mode_dgnum(num_m, so4_m):
    """Dry number-median diameter per mode [m]; 0 where the mode is empty."""
    valid = (num_m > 0) & (so4_m > 0)
    amm = np.where(valid, so4_m / np.where(valid, num_m, 1.0), 0.0)
    shp = (NMODE,) + (1,) * (np.ndim(num_m) - 1)
    return np.where(valid, np.cbrt(6.0 * amm / (np.pi * RHO_SO4)
                                   / np.exp(4.5 * ALNSG.reshape(shp) ** 2)), 0.0)


def mode_reff(num_m, so4_m):
    """Effective radius over the modes [m]: 0.5 * sum N dg^3 e^{4.5 s^2} /
    sum N dg^2 e^{2 s^2} (the lognormal moments), summed over modes."""
    dg = mode_dgnum(num_m, so4_m)
    shp = (NMODE,) + (1,) * (np.ndim(num_m) - 1)
    s2 = ALNSG.reshape(shp) ** 2
    m3 = (num_m * dg ** 3 * np.exp(4.5 * s2)).sum(0)
    m2 = (num_m * dg ** 2 * np.exp(2.0 * s2)).sum(0)
    return 0.5 * m3 / np.maximum(m2, 1e-300)


