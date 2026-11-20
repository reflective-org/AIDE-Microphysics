# AIDE-Microphysics

> **Status: active development.** APIs, shipped models and reported numbers may change.

Aerosol microphysics for the AIDE climate-intervention emulator stack, as interchangeable
**schemes** behind one per-box interface. Each scheme has a reference **physics** engine and,
where trained, learned **emulators** with the same call signature, so a host model can swap
engines or run both on the same state.

| scheme | representation | physics engine | emulators |
|---|---|---|---|
| `tomas40` | TOMAS, 40 mass-doubling sulfate bins (1.7 nm – ~10 µm) | [TOMAS-JAX](https://github.com/reflective-org/TOMAS-JAX) `fast` | `bintf_d384` (default), `bintf_d256`, `cs_b` |
| `mam4` | MAM4, 4 lognormal modes (sulfate only) | [MAM4-JAX](https://github.com/reflective-org/MAM4-JAX) + Dunne nucleation + Tabazadeh water | none yet |

Every engine advances one **6-hour step** on per-box arrays (one row per grid cell). Transport,
emissions, settling and radiation belong to the host model
([AIDE-SAI-link](https://github.com/reflective-org/AIDE-SAI-link) is the coupled model these come from).

## Which emulator?

- **`bintf_d384`**: default for standalone use. 6 h total N / M error 3.1 % / 2.6 % (median, held-out
  injection cells), stable over 14-day chains, 2.9 µs/cell.
- **`bintf_d256`**: standalone when speed matters, 2.6x cheaper (1.1 µs/cell), ~1.5–2 points worse at 6 h.
- **`cs_b`**: the fastest; for the coupled AIDE-SAI model with TOMAS physics on the top 0.3 % of
  cells, where it is validated (15 days: N −0.8 %, AOD +0.2 %, H2SO4 +1.4 % vs full physics).
  Not validated standalone.

Details, caveats and older generations: [docs/MODELS.md](docs/MODELS.md). Training data (CESM2.2
WACCM-CARMA June 1991 incl. Pinatubo, SAI injection scenarios, synthetic lognormal states):
[docs/DATA.md](docs/DATA.md).

## Install

```bash
pip install -e ".[tomas40-emulator]"     # torch + huggingface_hub: emulators (projection='clip')
pip install -e ".[tomas40]"              # + TOMAS-JAX: physics, and the default projection='chem'
pip install -e ".[mam4]"                 # + MAM4-JAX: MAM4 physics
```

TOMAS-JAX `fast` lives on its `gpu-fast` branch. Weights are downloaded from the Hugging Face Hub
([reflective-org/AIDE-Microphysics](https://huggingface.co/reflective-org/AIDE-Microphysics)) on
first use and checked against the sha256 in `aide_micro/schemes/tomas40/models.json`.

## Use

```python
import aide_micro
from aide_micro.schemes.tomas40 import emulator, physics

emu = emulator.load("bintf_d384")                         # fp16 + torch.compile on CUDA
Nk2, Mk2, H2, S2 = emu.step(Nk, Mk, H, S, T, P, RH, OH13) # per box: Nk, Mk (n, 40); OH13 (n, 13)
Nk2p, Mk2p, H2p, S2p = physics.step(Nk, Mk, H, S, T, P, RH, OH60)   # the reference, same state

aide_micro.available()            # ['mam4', 'tomas40']
aide_micro.get("mam4").to_bins()  # modes -> the common 40-bin view
```

Units and floors: `aide_micro/schemes/tomas40/bins.py`. A runnable box example:
`examples/tomas40_box.py`.

## Layout

```
aide_micro/
  interface.py            the Scheme contract
  registry.py             aide_micro.get(name), available()
  schemes/tomas40/        bins.py (layout, floors, normalization) · models.py (networks)
                          emulator.py (load, step) · project.py (conservation step) · physics.py
  schemes/mam4/           physics.py (MAM4-JAX run-3 physics) · bins.py (modes <-> 40 bins)
docs/                     MODELS.md · DATA.md · ADDING_A_SCHEME.md
scripts/upload_hf.py      publish weights to the Hub
```

Adding a scheme or the first MAM4 emulator: [docs/ADDING_A_SCHEME.md](docs/ADDING_A_SCHEME.md).

## Provenance

The tomas40 emulator, projection and physics wrappers and the MAM4 physics were ported from
AIDE-SAI-link (`src/microphysics/`) and checked **bit-for-bit** against that code on real cells of a
coupled run (all outputs identical). Earlier emulator generations remain available as git tags
`legacy/*`.

## License

MIT (this repository). TOMAS-JAX (LGPL-3.0) and MAM4-JAX (GPL-3.0) are separate projects, imported as
optional dependencies and not distributed here.
