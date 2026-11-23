---
license: apache-2.0
library_name: pytorch
tags:
  - climate
  - aerosol
  - microphysics
  - emulator
  - stratospheric-aerosol-injection
---

# AIDE-Microphysics weights

Learned 6-hour operators for **TOMAS** sectional sulfate aerosol microphysics (40 mass-doubling
bins, 1.7 nm – ~10 µm), used by the AIDE climate-intervention emulator stack. Code, interface and
documentation: **https://github.com/reflective-org/AIDE-Microphysics**.

```python
from aide_micro.schemes.tomas40 import emulator
emu = emulator.load("bintf_d384")     # downloads tomas40/bintf_d384/model.pt from this repo, checks sha256
Nk2, Mk2, H2SO4_2, SO2_2 = emu.step(Nk, Mk, H2SO4, SO2, T, P, RH, OH13)
```

| file | model | params | use it for |
|---|---|---|---|
| `tomas40/bintf_d384/model.pt` | bin-token transformer d384 x 6 | 7.16 M | **default standalone**: 6 h total N / M error 3.1 % / 2.6 % (median), stable over 14-day chains, 2.9 µs/cell (fp16, H200) |
| `tomas40/bintf_d256/model.pt` | bin-token transformer d256 x 4 | 2.15 M | standalone when speed matters: 1.1 µs/cell, ~1.5–2 points worse at 6 h |
| `tomas40/cs_b/model.pt` | PackTF student d256 x 4 | 2.1 M | the coupled AIDE-SAI model **with TOMAS physics on the top 0.3 % of cells** (15 days vs full physics: N −0.8 %, AOD +0.2 %, H2SO4 +1.4 %); not validated standalone |
| `tomas40/norm_stats.npz` | input / output normalization shared by all three | | |

Each `model.pt` is `{"model": state_dict}`; the architecture is in `aide_micro/schemes/tomas40/models.py`.
Inputs: per-box log-state (40 bins number + mass, H2SO4, SO2; 1 m³ box) and the 6 h forcing on a
30-min grid (P, T, RH, OH). Always apply the shipped output projection (`project="chem"`), which
enforces two-moment consistency and sulfur conservation. Use fp16 or fp32 (bf16 widens the 14-day
error tails). Full guidance: `docs/MODELS.md` in the code repository.

## Training data

TOMAS-JAX box-model runs (6 h – 14 days). Most start from real grid cells of a CESM2.2 WACCM-CARMA
simulation of June 1991, which includes the **Pinatubo eruption** (SO2 injected 1991-06-15) with
plume cells oversampled. Stratospheric-injection runs add sampled SO2 pulses (1 ppbv – 1000 ppmv) or
continuous injection on those states. A separate set uses synthetic lognormal size distributions and
environments, so the models do not depend on CARMA's size-distribution shapes or June-1991
conditions. `cs_b` was further trained on states captured from the coupled AIDE-SAI model.
Accuracy outside these conditions (other seasons, tropospheric pollution, non-sulfate aerosol) has
not been tested.

## Limitations

- Sulfate only; 6 h steps only; trained for the stratosphere and upper troposphere.
- Error grows in multi-day chains (day-14 N error p90 ~70 %); for long coupled runs the AIDE-SAI
  model runs TOMAS physics on the cells with the largest expected change.
- Reported numbers come from each model's own evaluation protocol and are not comparable across rows.
