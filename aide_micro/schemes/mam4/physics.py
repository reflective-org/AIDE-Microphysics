# SPDX-License-Identifier: GPL-3.0-only
# Copyright 2026 Reflective
# Contains code adapted from MAM4-JAX (mam4_jax.physics.newnuc, GPL-3.0), so this file is GPL-3.0;
# see aide_micro/schemes/mam4/LICENSE. The rest of the package is Apache-2.0.
"""MAM4 (4-mode modal) reference physics on per-box arrays: MAM4-JAX condensation, coagulation and
calcsize/rename, with Dunne 2016 nucleation and Tabazadeh 1997 water uptake swapped in (the TOMAS-JAX
functions), and SO2 + OH chemistry outside the core. This is the "run 3" physics of the AIDE-SAI
campaign, ported from AIDE-SAI-link src/microphysics/mam4_adapter.py (Apache-2.0) with the coupling
module's constants inlined; the per-box math is unchanged.

    from aide_micro.schemes.mam4 import physics as PH
    num2, so4_2, h2so4_2, so2_2 = PH.step(num, so4, h2so4, so2, T, P, RH, OH)   # one 6 h step

State per cell, MIXING RATIOS (MAM4's own units):
    num (n, 4) [#/kg], so4 (n, 4) [kg SO4(96)/kg], h2so4, so2 (n,) [kg/kg]
Mode order = MAM4-JAX MODE_NAMES: accum, aitken, coarse, primary_carbon (always 0 here: sulfate only).
OH: (n_inner, n) [molec/cm3] per inner step, or (n,) constant. Inner step 1800 s x 12.

phys='swap' (default) = Dunne + Tabazadeh; phys='stock' = MAM4-JAX as shipped (Vehkamaki + Koehler).

Units: MAM4 carries so4_a at MW 115.107; the state here keeps TOMAS's SO4(96) meaning and converts at
the core boundary (SO4_TO_Q), so total sulfate is comparable with tomas40. The chemistry works on
per-box kg at a 1 m3 box (mixing ratio x rho_air), as TOMAS-JAX fast does.

IMPORTING THIS MODULE CONFIGURES MAM4-JAX PROCESS-WIDE: condensation backend 'substep' (4) and the
core's built-in gas production set to zero (chemistry is done here).
"""
import numpy as np
import jax
import jax.numpy as jnp

import mam4_jax
from mam4_jax.coupling import amicphys as _amic
from mam4_jax.physics import newnuc as _nn
from mam4_jax.physics.calcsize import calcsize as _calcsize
from mam4_jax.physics.wateruptake import wateruptake as _wateruptake_stock
from mam4_jax.core import data as D
from mam4_jax.core.constants import RHOH2O
from tomas_jax.fast.nucleation import dunne_binary_neutral_rate
from tomas_jax.fast.water import h2so4_equilibrium_wt
from tomas_jax.physics.nucleation import _MNUC
from tomas_jax.physics.so2_chemistry import so2_oxidation_step
from tomas_jax.core.config import (N_GAS_SPECIES, SRTSO4 as _G_H2SO4,
                                   SRTSO2 as _G_SO2, MW_SO4 as _MW_SO4_TOMAS)
from tomas_jax.core.config import xk_boundaries as _xk_boundaries


class _C:
    """The AIDE-SAI-link coupling constants this physics uses (values as in coupling.py)."""
    DT_MICRO = 21600.0          # one 6 h step [s]
    BOXVOL = 1.0e6              # cm3 (1 m3)
    RD = 287.05                 # J/kg/K
    RHO_AER = 1770.0            # sulfate density for mass <-> size [kg/m3]
    XK_NP = np.asarray(_xk_boundaries())
    NBINS = XK_NP.shape[0] - 1
    MMID = np.sqrt(XK_NP[:-1] * XK_NP[1:])
    DP_BIN = np.cbrt(MMID / RHO_AER * 6.0 / np.pi) * 1e9    # (40,) bin diameter [nm]

    @staticmethod
    def qsat_ice(T, p):
        """Saturation specific humidity over ice [kg/kg] (Goff-Gratch ice, as CAM)."""
        t0 = 273.16
        es = 10.0 ** (-9.09718 * (t0 / T - 1.0) - 3.56654 * np.log10(t0 / T)
                      + 0.876793 * (1.0 - T / t0) + np.log10(6.1071)) * 100.0
        return 0.622 * es / np.maximum(p - 0.378 * es, 1.0)


C = _C
MAM4_DT = 1800.0
MAM4_PHYS = 'swap'
N_INNER = int(round(C.DT_MICRO / MAM4_DT))

# the core's process-global configuration, set once at import
_amic.configure_condensation(backend='substep', n_substeps=4)   # exact H2SO4 closed form
mam4_jax.configure_gas_netprod(h2so4=0.0, soa=0.0)             # chemistry is ours

# ---- layout ------------------------------------------------------------------
NMODE      = D.NTOT_AMODE                                   # 4
PCNST      = D.PCNST                                        # 35
NUM_SLOTS  = np.asarray(D.LMAP_NUM, dtype=np.int32)         # [17, 22, 30, 34]
_SO4_ROW   = np.asarray(D.LMAP_AER[:, D.AMICPHYS_IAER_SO4]) # [10, 18, 25, -1]
SO4_VALID  = _SO4_ROW >= 0                                  # pcarbon has no so4 slot
SO4_SLOTS  = np.where(SO4_VALID, _SO4_ROW, 0).astype(np.int32)
H2SO4_SLOT = int(D.LMAP_GAS[1])                             # 6
H2OMMR_SLOT = 0
SIGMA_G    = np.asarray(D.SIGMAG_AMODE)                     # (1.8, 1.6, 1.8, 1.6)
ALNSG      = np.log(SIGMA_G)
DGNUM0     = np.asarray(D.DGNUM_AMODE)                      # calcsize's seed dgnum
RHO_SO4    = float(C.RHO_AER)                               # 1770, both engines
MW_SO4A_Q  = float(D.ADV_MASS[SO4_SLOTS[0] - 5])            # 115.107 (vmr factor of the so4 slots)
MW_H2SO4   = 98.0784
SO4_TO_Q   = MW_SO4A_Q / _MW_SO4_TOMAS                      # kg SO4(96) -> MAM4 so4_a mass
Q_TO_SO4   = 1.0 / SO4_TO_Q
ZMID_STRAT = 20000.0   # [m] any value above pblh=0: the PBL nucleation branch is off
                       # (and gone entirely under MAM4_PHYS=swap)


# =============================================================================
# the two physics swaps
# =============================================================================
# Dunne cluster expressed as a dry sulfate sphere at MAM4's sulfate density, so
# the KK2002 size correction starts from the same cluster TOMAS nucleates at
# (_MNUC = 3.47e-24 kg, Kulmala 2013 cluster; TOMAS's own cluster density is
# 1350 kg/m3 -> r 0.85 nm; at 1770 kg/m3 the same mass is d = 1.55 nm).
DCLUS_M = (6.0 * _MNUC / (np.pi * RHO_SO4)) ** (1.0 / 3.0)


def dunne_nuc_mosaic_1box(dtnuc, temp, rh, press, zm, pblh,
                          qh2so4_cur, qh2so4_avg, h2so4_uptkrate,
                          dplom_sect, dphim_sect, newnuc_method_flagaa=11):
    """Drop-in for mam4_jax.physics.newnuc.mer07_veh02_nuc_mosaic_1box with the
    Dunne 2016 binary-neutral rate in place of Vehkamaki 2002 (+ PBL overlay).

    Same signature and the same 8-tuple return, so the amicphys glue
    (_mam_newnuc_1subarea: Aitken-mode deposition, the size constraints and
    the H2SO4 cap) runs unchanged. Stage by stage, against the stock leaf:

      rate      J_bn = dunne_binary_neutral_rate(T, [H2SO4]) on the
                gasaerexch time-mean H2SO4 (qh2so4_avg, MAM4's
                newnuc_h2so4_conc_optaa=2 convention). NO clamps on T, RH or
                [H2SO4] (stock clamps T >= 230.15 K -- i.e. stock is pinned to
                its lower T bound everywhere in the stratosphere -- and
                1e4 <= [H2SO4] <= 1e11 cm-3). MAM4's exit gate J > 1e-6 cm-3
                s-1 (rateloge > -13.82) is kept: it is part of MAM4's handling,
                not of the rate.
      PBL       none. Dunne binary-neutral only (as in tomas_jax.fast).
      cluster   fixed: the Dunne/Kulmala cluster mass _MNUC (TOMAS nucleates
                every particle at this mass) as a d = 1.55 nm sulfate sphere.
                Stock derives the cluster from Vehkamaki's molecule count.
      KK2002    kept, with dnuc = that 1.55 nm cluster (stock: 2*r_cluster,
                floored at 1 nm). The new particles are grown to the Aitken
                lower edge dplom = exp(0.67 ln dgnumlo + 0.33 ln dgnum) =
                12.4 nm and the surviving rate is J * exp(nu/dfin - nu/dnuc),
                with the uptake rate h2so4_uptkrate the glue passes (MAM4-JAX
                passes a 1e-3 1/s placeholder there, upstream shortcut -- same
                for stock and swap, flagged in STATUS.md).
      mass      each new particle carries the mass of a dplom sphere; the gas
                removed is that mass (capped at 0.9999 of the available
                H2SO4). This is how MAM4 books nucleation: the growth from the
                cluster to 12 nm is charged to the gas here, not to
                condensation. TOMAS instead deposits the cluster mass itself
                (1 particle per 3.47e-24 kg) in bin 0 and lets condensation
                grow it -- so for the SAME J the two engines create very
                different NUMBERS of particles per unit sulfur (factor
                mass(dplom)/_MNUC ~ 500). That is the modal scheme's
                nucleation handling, and the comparison measures it.
    """
    del zm, pblh, newnuc_method_flagaa          # PBL overlay: not in Dunne
    rgas_local = _nn._RGAS_J_PER_K_PER_KMOL / 1.0e3   # J/K/mol
    avogad = 6.02214e26 / 1.0e3                       # 1/mol
    pi = jnp.pi
    onethird = 1.0 / 3.0

    cair = press / (temp * rgas_local)                # mol/m3
    so4vol_in = qh2so4_avg * cair * avogad * 1.0e-6   # molec/cm3

    # ---- the Dunne rate, unclamped --------------------------------------
    ratenuclt = dunne_binary_neutral_rate(temp, so4vol_in)        # cm-3 s-1
    rateloge = jnp.log(jnp.maximum(ratenuclt, 1.0e-38))
    rateloge = rateloge + jnp.log(jnp.maximum(1.0e-38, _nn._ADJUST_FACTOR_BIN_TERN_RATENUCL))
    rate_ok = rateloge > -13.82
    ratenuclt = jnp.exp(rateloge)
    ratenuclt_bb = ratenuclt * 1.0e6                              # #/m3/s
    dnclusterdt = jnp.where(rate_ok, ratenuclt_bb, 0.0)

    # ---- from here on: stock MAM4 stages 5-10, cluster size fixed --------
    rh_clamp = jnp.maximum(0.10, jnp.minimum(0.95, rh))
    wetvol_dryvol = 1.0 - 0.56 / jnp.log(rh_clamp)

    dpdry_clus = jnp.full_like(ratenuclt, DCLUS_M)
    isize_nuc = jnp.ones_like(jnp.asarray(so4vol_in), dtype=jnp.int32)
    igrow = (dpdry_clus <= dplom_sect).astype(jnp.float64)
    dpdry_part = jnp.where(
        dpdry_clus <= dplom_sect, dplom_sect,
        jnp.where(dpdry_clus >= dphim_sect, dphim_sect, dpdry_clus))
    voldry_part = (pi / 6.0) * (dpdry_part ** 3)

    dens_part = _nn._DENS_SULFACID
    dens_nh4so4a = jnp.full_like(rateloge, dens_part)
    mass_part = voldry_part * dens_part
    kgaero_per_moleso4a = 1.0e-3 * _nn._MW_SULFACID * (D.MW_SO4A_HOST / _nn._MW_SO4A)
    wet_volfrac_so4a = 1.0 / wetvol_dryvol

    # Kerminen-Kulmala 2002, as stock
    tmp_spd = 14.7 * jnp.sqrt(temp)
    gr_kk = 3.0e-9 * tmp_spd * _nn._MW_SULFACID * so4vol_in / (dens_part * wet_volfrac_so4a)
    dfin_kk = 1.0e9 * dpdry_part * (wetvol_dryvol ** onethird)
    dnuc_kk = jnp.maximum(1.0e9 * dpdry_clus, 1.0)
    gamma_kk = (0.23 * dnuc_kk ** 0.2 * (dfin_kk / 3.0) ** 0.075
                * (dens_part * 1.0e-3) ** (-0.33) * (temp / 293.0) ** (-0.75))
    tmpa_kk = jnp.maximum(h2so4_uptkrate * 3600.0, 0.0)
    tmpb_kk = 6.7037e-6 * (temp ** 0.75) / cair * 3600.0
    cs_prime_kk = tmpa_kk / (4.0 * pi * tmpb_kk * _nn._ACCOM_COEF_H2SO4)
    nu_kk = gamma_kk * cs_prime_kk / gr_kk
    factor_kk_grow = jnp.exp(nu_kk / dfin_kk - nu_kk / dnuc_kk)
    factor_kk = jnp.where(igrow > 0, factor_kk_grow, 1.0)
    ratenuclt_kk = ratenuclt_bb * factor_kk

    qmolso4a_del_max = jnp.maximum(0.0, ratenuclt_kk * dtnuc * mass_part) / (
        kgaero_per_moleso4a * cair)
    freduce = jnp.where(qmolso4a_del_max > qh2so4_cur,
                        qh2so4_cur / jnp.maximum(qmolso4a_del_max, 1.0e-30), 1.0)
    final_ok = rate_ok & (freduce * ratenuclt_kk > 1.0e-12)
    qh2so4_del_pos = jnp.minimum(0.9999 * qh2so4_cur, freduce * qmolso4a_del_max)
    qh2so4_del = jnp.where(final_ok, -qh2so4_del_pos, 0.0)
    qnh3_del = jnp.zeros_like(qh2so4_del)
    qso4a_del = -qh2so4_del
    qnh4a_del = jnp.zeros_like(qh2so4_del)
    qnuma_del_raw = 1.0e-3 * (qso4a_del * _nn._MW_SO4A) / jnp.maximum(mass_part, 1.0e-30)
    qnuma_del = jnp.where(final_ok, qnuma_del_raw, 0.0)
    return (isize_nuc, qnuma_del, qso4a_del, qnh4a_del,
            qh2so4_del, qnh3_del, dens_nh4so4a, dnclusterdt)


def wateruptake_tabazadeh(state):
    """Drop-in for mam4_jax.physics.wateruptake.wateruptake, sulfate only.

    Water on each mode from the Tabazadeh 1997 H2SO4/H2O equilibrium -- the
    exact tomas_jax.fast.water relations: wt% = h2so4_equilibrium_wt(T, RH%)
    and water = acid * (100/wt - 1) with acid = moles(so4) * 98 (wt% is defined
    on H2SO4 mass). RH comes from state['relhum'] (the coupling's 0-1 field,
    as for TOMAS), not from q[0]/qsat_water as stock does, and there is no
    crystallisation/deliquescence hysteresis: a stratospheric sulfate particle
    is a solution droplet at any RH.

    Outputs follow MAM4's conventions so the core's coagulation and
    condensation see what they expect: wet volume = dry volume + water/RHOH2O
    (volume-additive, as stock), dgncur_awet = dgncur_a * (wet/dry)^(1/3),
    wetdens = (dry mass + water)/wet volume with the dry mass in MAM4's
    115-mass convention (that is what q carries inside the core), qaerwat in
    kg water / kg air. Modes with no sulfate (primary carbon) get no water.
    """
    q = jnp.asarray(state['q'], dtype=jnp.float64)
    dgncur_a = jnp.asarray(state['dgncur_a'], dtype=jnp.float64)
    t = jnp.asarray(state['t'], dtype=jnp.float64)
    rh = jnp.asarray(state['relhum'], dtype=jnp.float64)

    so4_q = q[..., SO4_SLOTS] * jnp.asarray(SO4_VALID, dtype=jnp.float64)  # (..., 4) MAM4 mass
    moles = so4_q / MW_SO4A_Q                                 # kmol SO4 / kg air
    acid = moles * MW_H2SO4                                   # kg H2SO4 / kg air
    dryvolmr = so4_q / RHO_SO4                                # m3 / kg air

    wt = h2so4_equilibrium_wt(t, rh * 100.0)                  # (...,) in [10, 80]
    water = acid * (100.0 / wt[..., None] - 1.0)              # kg water / kg air
    wtrvolmr = water / RHOH2O

    v2ncur_a = 1.0 / ((jnp.pi / 6.0) * dgncur_a ** 3 * jnp.exp(4.5 * ALNSG ** 2))
    has_dry = dryvolmr > 1.0e-30
    ratio = jnp.where(has_dry, 1.0 + wtrvolmr / jnp.where(has_dry, dryvolmr, 1.0), 1.0)
    dgncur_awet = dgncur_a * ratio ** (1.0 / 3.0)
    wetdens = jnp.where(has_dry,
                        (so4_q + water) / jnp.where(has_dry, dryvolmr + wtrvolmr, 1.0),
                        RHO_SO4)
    del v2ncur_a   # not needed: qaerwat == RHOH2O * naer * wtrvol_per_particle == water
    return {**state, 'dgncur_awet': dgncur_awet, 'qaerwat': water, 'wetdens': wetdens}


# =============================================================================
# one core step for one cell
# =============================================================================
_PARAMS = mam4_jax.AmicphysParams()


def core_step(state, phys=MAM4_PHYS):
    """calcsize -> water -> amicphys for ONE cell (mam4_jax.driver.run_step's
    sequence, re-composed here because run_step binds the stock water
    uptake at import and is jitted). Under phys='swap' the Dunne leaf is
    installed into the newnuc module for the duration of the TRACE: amicphys
    looks the name up when traced, so the swap is baked into the compiled
    graph and the module is restored before anything else can see it.
    """
    state = _calcsize(state)
    if phys == 'swap':
        state = wateruptake_tabazadeh(state)
        saved = _nn.mer07_veh02_nuc_mosaic_1box
        _nn.mer07_veh02_nuc_mosaic_1box = dunne_nuc_mosaic_1box
        try:
            state = _amic.amicphys(state, _PARAMS)
        finally:
            _nn.mer07_veh02_nuc_mosaic_1box = saved
    else:
        state = _wateruptake_stock(state)
        state = _amic.amicphys(state, _PARAMS)
    return state


def _make_run_cells(phys, n_inner, dt_inner):
    """Build the jitted (cells x inner steps) integrator for one physics choice.

    Arguments of the returned function (all leading axis = cell, float64):
      q0 (ncell, 35) core tracer array; so2_kg (ncell,) per-box SO2;
      T, P, RH, rho (ncell,); OH (n_inner, ncell) [molec/cm3].
    Returns (q, so2_kg) after n_inner steps of: so2_oxidation_step on the
    per-box gases -> q[H2SO4] <- the new H2SO4 -> core_step.
    """
    dt = float(dt_inner)
    boxvol = float(C.BOXVOL)

    def chem_cell(h2so4_kg, so2_kg, T, P, RH, oh):
        Gc = jnp.zeros((N_GAS_SPECIES,), dtype=jnp.float64)
        Gc = Gc.at[_G_H2SO4].set(h2so4_kg).at[_G_SO2].set(so2_kg)
        Gc = so2_oxidation_step(Gc, T, P, boxvol, dt, oh, RH)
        return Gc[_G_H2SO4], Gc[_G_SO2]

    def cell_state(q, T, P, RH):
        return {
            'q': q, 'qqcw': jnp.zeros_like(q),
            'dgncur_a': jnp.asarray(DGNUM0),         # calcsize recomputes it from q
            't': T, 'pmid': P, 'cldn': jnp.zeros_like(T),
            'zmid': jnp.full_like(T, ZMID_STRAT), 'pblh': jnp.zeros_like(T),
            'relhum': RH, 'deltat': jnp.asarray(dt),
        }

    def step_cell(q, T, P, RH):
        return core_step(cell_state(q, T, P, RH), phys)['q']

    v_chem = jax.vmap(chem_cell)
    v_step = jax.vmap(step_cell, in_axes=(0, 0, 0, 0))

    @jax.jit
    def run(q0, so2_kg0, T, P, RH, rho, OH):
        def body(carry, oh):
            q, so2_kg = carry
            h2so4_kg, so2_kg = v_chem(q[:, H2SO4_SLOT] * rho, so2_kg, T, P, RH, oh)
            q = q.at[:, H2SO4_SLOT].set(h2so4_kg / rho)
            q = v_step(q, T, P, RH)
            return (q, so2_kg), None
        (q, so2_kg), _ = jax.lax.scan(body, (q0, so2_kg0), OH, length=n_inner)
        return q, so2_kg
    return run


_RUN_CACHE = {}


def get_run_cells(phys=MAM4_PHYS, n_inner=N_INNER, dt_inner=MAM4_DT):
    key = (phys, int(n_inner), float(dt_inner))
    if key not in _RUN_CACHE:
        _RUN_CACHE[key] = _make_run_cells(*key)
    return _RUN_CACHE[key]


# =============================================================================
# modal state <-> core q array
# =============================================================================
def pack_q(num_m, so4_m, h2so4, T, P, RH):
    """(ncell,4) num [#/kg], (ncell,4) so4 [kg SO4/kg], (ncell,) h2so4 [kg/kg]
    -> (ncell, 35) core tracer array. so4 converted to MAM4's 115-mass; water
    vapour slot = RH * qsat_ice (coupling's own formula, so the stock water
    path sees a consistent humidity)."""
    ncell = num_m.shape[0]
    q = np.zeros((ncell, PCNST), dtype=np.float64)
    q[:, NUM_SLOTS] = np.maximum(num_m, 0.0)
    q[:, SO4_SLOTS[SO4_VALID]] = np.maximum(so4_m[:, SO4_VALID], 0.0) * SO4_TO_Q
    q[:, H2SO4_SLOT] = np.maximum(h2so4, 0.0)
    q[:, H2OMMR_SLOT] = np.asarray(RH) * C.qsat_ice(np.asarray(T, np.float64), np.asarray(P, np.float64))
    return q


def unpack_q(q):
    """(ncell,35) -> (num (ncell,4), so4 (ncell,4) [kg SO4(96)/kg], h2so4 (ncell,))"""
    q = np.asarray(q)
    num_m = q[:, NUM_SLOTS]
    so4_m = np.zeros((q.shape[0], NMODE), dtype=np.float64)
    so4_m[:, SO4_VALID] = q[:, SO4_SLOTS[SO4_VALID]] * Q_TO_SO4
    return num_m, so4_m, q[:, H2SO4_SLOT]



# =============================================================================
# per-box API
# =============================================================================
def step(num, so4, h2so4, so2, T, P, RH, OH, phys=MAM4_PHYS, cell_cap=250000):
    """One 6 h step on n cells (numpy in/out; see module docstring for units). Returns
    num2 (n,4), so4_2 (n,4), h2so4_2 (n,), so2_2 (n,), all mixing ratios."""
    T_f = np.asarray(T, np.float64); P_f = np.asarray(P, np.float64); RH_f = np.asarray(RH, np.float64)
    n = T_f.shape[0]
    rho_f = P_f / (C.RD * T_f)
    OH = np.asarray(OH, np.float64)
    OH_f = OH if OH.ndim == 2 else np.repeat(OH.reshape(1, n), N_INNER, 0)
    q0 = pack_q(np.asarray(num, np.float64), np.asarray(so4, np.float64), np.asarray(h2so4, np.float64), T_f, P_f, RH_f)
    so2_kg = np.maximum(np.asarray(so2, np.float64), 0.0) * rho_f
    run = get_run_cells(phys)
    q_out = np.empty_like(q0); so2_out = np.empty_like(so2_kg)
    for a in range(0, n, cell_cap):
        b = min(a + cell_cap, n)
        qo, so = run(jnp.asarray(q0[a:b]), jnp.asarray(so2_kg[a:b]),
                     jnp.asarray(T_f[a:b]), jnp.asarray(P_f[a:b]), jnp.asarray(RH_f[a:b]),
                     jnp.asarray(rho_f[a:b]), jnp.asarray(OH_f[:, a:b]))
        q_out[a:b] = np.asarray(qo); so2_out[a:b] = np.asarray(so)
    num2, so4_2, h2so4_2 = unpack_q(q_out)
    return num2, so4_2, h2so4_2, so2_out / rho_f
