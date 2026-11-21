# Training data (tomas40 emulators)

TOMAS-JAX box-model runs (6 h to 14 days, 30-min saves) with the environment (T, P, RH, OH)
evolving through each run. Five populations:

| set | initial states | size |
|---|---|---|
| `v8_carma` (6 h, global) / `v8_carma_2wk` (14 days, stratosphere) | Real grid cells of a CESM2.2 WACCM-CARMA run for June 1991 (1°, L70, half-hourly; branched from a CARMA free run at 1991-06-01). The month includes the **Pinatubo eruption** (SO2 injected 1991-06-15, 10:00–12:00 UTC); plume cells, only 0.02–0.15 % of stratospheric cells, are oversampled. | — |
| `v9_sai` (pulse injection) | Stratospheric cells from the same CARMA run plus a sampled SO2 spike of 1 ppbv – 1000 ppmv and a sampled H2SO4-vapour split | 214,280 runs |
| `v9_sai_cont` (continuous injection) | Same, injection lasting ~1.7 h – 102 h (p10 – p90) | 107,842 runs |
| `v10_ideal` | **No CARMA.** Synthetic 1–3-mode lognormal size distributions and synthetic environments (3–150 hPa, 185–265 K, RH 1e-3–0.6, diurnal or constant OH), with and without injection and dilution | 96,328 runs |

Why `v10_ideal` exists: CARMA-derived initial states carry the remapped-sectional shape of the
CARMA bins and June-1991 conditions. An earlier emulator trained only on them was ~30x worse on
clean lognormal initial states. The synthetic set removes that dependence and stretches T beyond
June's range.

`cs_b` starts from a model trained on the sets above and was further trained on states captured
from the coupled AIDE-SAI model's own runs, so it has also seen that model's conditions.

**Summary for citing.** Most training runs start from real grid cells of a CESM2.2 WACCM-CARMA
simulation of June 1991, which includes the Pinatubo eruption (SO2 injected 1991-06-15) with plume
cells oversampled. Stratospheric-injection runs add sampled SO2 pulses or continuous injection on
those states. A separate set uses synthetic lognormal size distributions and environments, so the
models do not depend on CARMA's size-distribution shapes or June-1991 conditions. Accuracy outside
these conditions (other seasons, tropospheric pollution, non-sulfate aerosol) has not been tested.

Generators: TOMAS-JAX `generate_training_data_v8_carma.py` (`--inject` for v9),
`generate_training_data_v10_ideal.py`; details in the TOMAS-JAX repository.
