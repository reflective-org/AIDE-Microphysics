# Which model to use

All tomas40 emulators are 6-hour, 40-bin, sulfate-only operators with a 1 m³ box. Accuracy
numbers below are each model's own evaluation protocol and are **not comparable across rows**;
see the notes under each entry.

| model | params | cost | use it for |
|---|---|---|---|
| `bintf_d384` (default) | 7.16 M | 2.9 µs/cell, fp16 compiled, H200 | standalone emulation, offline studies, other host models |
| `bintf_d256` | 2.15 M | 1.1 µs/cell | standalone when speed matters: sweeps, ensembles, quick experiments |
| `cs_b` | 2.1 M | fastest; see below | the AIDE-SAI coupled model **with TOMAS physics on the top cells** |

## `bintf_d384` — default standalone model

- Bin-token transformer: one token per bin plus two gas tokens; the 12 half-hour steps of a
  6 h block live in the channel dimension.
- 6 h total number / mass error on held-out SAI injection burst cells: 3.1 % / 2.6 % (p50),
  16 % / 13 % (p90).
- Chained 14-day rollouts stay bounded: day-14 injection N 20 % / 68 % (p50 / p90), M 10 % / 36 %.
- 4.3 s for 1.5 M cells x one 6 h step on one H200 (TOMAS-JAX fast: 24–74 s).

## `bintf_d256` — standalone, faster

- Same architecture, smaller. 2.6x cheaper than d384, about 1.5–2 points worse at 6 h
  (N / M 4.7 % / 4.9 % p50), about the same at 14 days (day-14 N 17 % / 71 %, M 10 % / 46 %).
- 1.7 s for 1.5 M cells x one 6 h step on one H200.
- Run in fp16 or fp32 only. bf16 widens its day-14 N p90 from 71 % to 84 %, so `load` refuses it.

## `cs_b` — coupled model, hybrid with physics

- PackTF student (4 bins per token, d=256, 4 layers), distilled from the BinTF models and then
  trained on states captured from the coupled AIDE-SAI model, with an extra loss on the
  condensation sink of small particles in the upper stratosphere (2–7 hPa).
- Validated only as the coupled model runs it: the emulator on every cell and TOMAS-JAX physics
  on the top 0.3 % of cells by expected change. Coupled 15 days vs. full physics: N −0.8 %,
  M 0.0 %, r_eff −0.1 %, AOD +0.2 %, H2SO4 burden +1.4 % (the previous student: +11.6 %).
- In the coupled model the network costs 0.43 s per 6 h step for the full grid (0.26 s with the
  packed-attention kernel of the AIDE-SAI-link speed work).
- **Not validated standalone** (without physics on the top cells). Use `bintf_d384` there.

## Settings that apply to all of them

- **Projection.** `emulator.load(..., project="chem")` (default) applies, after every step, the
  two-moment bounds and the exact gas chemistry with sulfur conservation (`project.py`). This
  is how the coupled model runs every model above; without it sulfur is not conserved.
  `project="clip"` applies only the bounds; `"none"` returns the raw network answer.
- **Input floors** (SO2 1e-17 kg/box, OH 100 cm⁻³, RH ≥ 0.01) keep inputs inside the training
  data; zeros are many sigma outside it and produce nonsense (see `bins.py`). `step()` applies them.
- **Precision.** fp16 autocast + `torch.compile(mode="reduce-overhead")` reproduces fp32 within
  rounding at every horizon measured and is 2x faster. bf16 is refused.
- **Domain.** Trained on stratospheric and upper-tropospheric sulfate (docs/DATA.md). Other
  seasons, tropospheric pollution and non-sulfate aerosol are untested.

## Older generations (git tags only, not recommended for new work)

| tag | model | note |
|---|---|---|
| `legacy/p7` | conservation-projected multi-head MLP, 60 s steps | 12.5 % / 10.7 % per-bin median at 6 h |
| `legacy/axial-aug5` | axial trajectory operator, 29.5 M | non-autoregressive over one 6 h window; ~467 µs/cell |
| `legacy/axial-cons2` | axial, long-range fine-tune | sulfur-monotone, better past 24 h |
| `legacy/dt-champion` | Δt-conditioned (time-skip) | 9.0 % per-bin median at 6 h |
| `legacy/bintf` | the bin-token-transformer release before this layout | same weights as `bintf_*` here |

Sources: `FRONTIER_RESULTS.md` / `EMULATOR_HANDOFF.md` of the emulator campaign (BinTF numbers,
H200), the `h2so4_netonly_20261112` campaign notes (`cs_b`), and the branch READMEs (legacy).
