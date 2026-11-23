# Adding a scheme or an emulator

## A new scheme (e.g. a different aerosol representation)

1. `aide_micro/schemes/<name>/` with
   - `physics.py`: `step(<state>, T, P, RH, OH, ...) -> <next state>` for one 6 h step on per-box
     arrays (one row per cell). Wrap the reference model; inline any host-model constants with
     their values and source, so the scheme does not import the host.
   - `bins.py`: map the scheme's state to the TOMAS 40-bin view (`Nk`, `Mk`). Optics and size
     diagnostics are computed from that view for every scheme, so schemes are compared through one
     code path.
   - `__init__.py`: a `Scheme(...)` (see `aide_micro/interface.py`) with `nrows`, `units`, and
     lazy loaders, so importing `aide_micro` never pulls in a scheme's heavy dependencies.
2. Register it in `aide_micro/registry.py`.
3. Add an optional dependency group in `pyproject.toml`.
4. Name physics variants explicitly (e.g. MAM4 `swap` = Dunne nucleation + Tabazadeh water vs
   `stock`): an emulator is trained on one variant and must record which.

## A new emulator for an existing scheme (e.g. the first MAM4 emulator)

1. `schemes/<name>/emulator.py` with `load(model_name, ...) -> obj` and `obj.step(...)` taking and
   returning exactly what `physics.step` takes and returns. Then a host swaps engines without
   other changes and can run both on the same state.
2. Weights go to the Hugging Face Hub repo (`scripts/upload_hf.py`), never into git. Add an entry
   to `schemes/<name>/models.json`: file, sha256, architecture config, physics variant.
3. Before publishing, check:
   - conservation (sulfur) and non-negativity after the step (with the projection the model ships with);
   - a 6 h error table against `physics.step` on held-out cells: N, M, H2SO4 and, through the bins
     view, r_eff and AOD;
   - if it is ported from other code, bit-for-bit equality with that code on real inputs.
4. Write its "use it for" line in `docs/MODELS.md`, including the conditions it has *not* been
   validated in.
