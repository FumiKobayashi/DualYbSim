# Parameter reference

All parameters live on `NoiseModelParameters` and are named after the symbols in the paper's noise-model appendix. See `docs/channel_reference.md` for the notation-to-identifier convention and for what each channel means.

## Layout

Per-encoding parameters are stored flat with a trailing **encoding tag**:

| Tag | Encoding |
|---|---|
| `_c` | 174Yb optical clock qubit (`isotope="174"`, `qubit_type="gm"`) |
| `_g` | 171Yb ground-manifold nuclear-spin qubit (`"171"`, `"g"`) |
| `_m` | 171Yb metastable-manifold nuclear-spin qubit (`"171"`, `"m"`) |

Use `for_qubit()` to read them grouped, with the tag dropped:

```python
params = NoiseModelParameters()
v = params.for_qubit("171", "m")
v.p_1  # == params.p_1_m
v.gate_time  # == params.gate_time_m
```

The view is built on demand, so it always reflects the current values. A field that reads `0.0` means the channel does not fire for that encoding — either it is physically absent (`p_flip_m` on the 174Yb clock qubit) or it has been calibrated away — so the usual `if p > 0:` guard works unchanged.

Rates that are properties of the atom rather than of the encoding are stored once, as shared constants.

## Presets

| Constructor | Meaning |
|---|---|
| `NoiseModelParameters()` | the paper's tabulated values |
| `NoiseModelParameters.paper_defaults()` | identical to the above, explicit |
| `NoiseModelParameters.legacy_defaults()` | the legacy parameter set that predates this library |

`legacy_defaults()` exists so that numbers produced before this library existed can be reproduced. It restores *parameters* only; formula-level corrections stay in place, in particular the factor of two in `t_move = 2 sqrt(l / a)`.

## Shared physical constants

| Parameter | Paper | Default | Meaning |
|---|---|---|---|
| `gamma_Ryd` | `Γ_Ryd` | `2.0e4` s⁻¹ | total Rydberg decay rate, `1/50 µs` |
| `ryd_branching` | — | `{LOSS_r: 0.51, DECAY_rg: 0.42, DECAY_rm: 0.07}` | branching of `Γ_Ryd` |
| `gamma_gL` | `Γ_gL` | `1/30` s⁻¹ | ground-manifold trap loss |
| `gamma_mL` | `Γ_mL` | `1/30` s⁻¹ | metastable-manifold trap loss |
| `gamma_Z_gm` | `1/T_2^(gm)` | `1/5` s⁻¹ | 171Yb clock-transition coherence. **Currently inert** — see below |

`Γ_mg` is *not* shared: it is per encoding, because the effective metastable lifetime depends on the trap depth the encoding operates in. See the coherent-control table below.

`gamma_Z_gm` is set to its tabulated value but no circuit path reads it, so `ZERR_gm` never reaches a Stim circuit. The channel exists in the Kraus layer (`Kraus1Q_171m.ZERR_gm` and friends) and can be used directly from there. What is unsettled is when it should fire in a circuit: an idling 171Yb qubit sits wholly in one manifold, so there is no ground–metastable superposition to dephase, and the windows where such a superposition does exist are the clock pulses, which already carry `DEP1_gm`.

## Transport geometry and schedule

| Parameter | Paper | Default | Meaning |
|---|---|---|---|
| `l_site` | `l_site` | `3e-6` m | site spacing |
| `l_zone` | `l_zone` | `100e-6` m | computation-to-readout zone separation |
| `t_hand` | `t_hand` | `200e-6` s | handover time |
| `a` | `a` | `5500` m s⁻² | transport acceleration |

Transport time is `t_move(l) = 2 sqrt(l / a)`, and the one-way readout distance is `l = d · l_site + l_zone` for code distance `d`.

## Per-encoding parameters

`—` marks a parameter the encoding does not define. Where the legacy column is blank the two presets agree.

### Coherent-control

| Parameter | Paper | `_c` | `_g` | `_m` | legacy |
|---|---|---|---|---|---|
| `p_1_*` | `p_1^(c/g/m)` | `1e-4` | `1e-4` | `1e-4` | `2e-3` / `1e-3` / `1e-3` |
| `p_2_*` | `p_2^(c/m)` | `1e-3` | `1e-3` | `1e-3` | `2e-3` (all) |
| `p_2_dual` | `p_2^(dual)` | `1e-3` (cross-encoding) | | | `2e-3` |
| `p_1_gm` | `p_1^(gm)` | — | `1e-4` (shared with `_m`) | `1e-4` | `2e-3` |
| `p_m_g_gate` | `p_(m→g)^(gate)` | — | `1e-3` (shared) | `1e-3` | |
| `gamma_mg_*` | `Γ_mg` | `1/20` | `1` | `1` | `_c` was `1` |

`p_1_gm` and `p_m_g_gate` are 171Yb clock-transition channels stored once and exposed on both 171Yb encodings, because the g qubit reaches the Rydberg state through the same clock pulse. `DEP1_gm` reaches a g qubit only while that pulse is driving it into the metastable manifold, so it is emitted with the metastable rates for both encodings.

`gamma_mg_*` is per encoding because the effective metastable lifetime depends on the trap depth. The clock qubit idles in a shallow trap and so uses the natural <sup>3</sup>P<sub>0</sub> lifetime of 20 s; the 171Yb channels are reached only through the clock pulse and the readout shelving step, both of which run in the deep imaging trap, and keep the scattering-dominated `1` s⁻¹. The clock qubit's value must also satisfy `T_2 ≤ 2 T_1`, since amplitude damping at `T_1` alone already decoheres at `2 T_1`; this is the one parameter `legacy_defaults()` does not restore, because the legacy `1` s⁻¹ against `T_2^(c) = 5 s` does not.

### Measurement

| Parameter | Paper | `_c` | `_g` | `_m` | legacy |
|---|---|---|---|---|---|
| `p_meas_*` | `p_meas` | `1e-4` | `1e-4` | `1e-4` | `1e-3` / `1e-3` / `1/300` |
| `q_BB_*` | `q_BB` | `0.5` | `0.5` | `0.5` | |
| `p_g_L_meas_*` | `p_(g→L)^(meas)` | `1e-3` | `1e-3` | `1e-3` | |
| `p_flip_g_*` | `p_flip^(g)` | — | `1e-3` | `1e-3` | |
| `p_depol_meas_idling_*` | — | — | `0.0` | `0.0` | |

`q_BB_*` is the probability of assigning the ambiguous bright–bright readout record to `|0>`. The twirled `p_Z` of the MERR channel scales as `(2 q_BB − 1)²`, so it vanishes at the default even split.

`p_depol_meas_idling_*` is not part of the paper's channel set. It depolarises an idle 171Yb qubit while 174Yb is measured in place, and is held at zero.

### Reset

| Parameter | Paper | `_c` | `_g` | `_m` | legacy |
|---|---|---|---|---|---|
| `p_g_L_reset_*` | `p_(g→L)^(reset)` | `1e-3` | `1e-3` | — | `_g` was `0.0` |
| `p_m_L_reset_m` | `p_(m→L)^(reset)` | — | — | `6e-3` | `1e-3` |
| `p_flip_m_m` | `p_flip^(m)` | — | — | `1e-3` | |

### Idling

| Parameter | Paper | `_c` | `_g` | `_m` | legacy |
|---|---|---|---|---|---|
| `gamma_Z_*` | `1/T_2^(c/g/m)` | `1/5` | `1/10` | `1/10` | |
| `gamma_X_*` | `1/T_1^(g/m)` | — | `1/200` | `1/200` | `0.0` |

Note the dephasing convention for the 171Yb encodings. `ZERR_g` and `ZERR_m` are emitted as `Z_ERROR(1 − exp(−t/T_2))`, following the paper's channel definition `K_1 = √(p_Z) Z`. A Pauli-Z channel of probability `p` maps the transverse Bloch component by `1 − 2p`, so the effective coherence time is about `T_2/2` rather than `T_2`: at `T_2^(m) = 10 s` and `t = 1 ms` the emitted channel implies `T_2,eff = 5.0 s`. Set `gamma_Z_g` / `gamma_Z_m` accordingly if you want the nominal value to be the effective one.

`ZERR_c` is not affected. It acts on the same two levels as `DECAY_mg`, so the two are twirled together into a single `PAULI_CHANNEL_1` whose `p_Z = p_2(t)/2 − p_1(t)/4` already carries the factor of a half, reproducing `exp(−t/T_2^(c))` exactly.

### Transportation

| Parameter | Paper | `_c` | `_g` | `_m` |
|---|---|---|---|---|
| `p_hand_*` | `p_(g→L)^(hand)`, `p_(m→L)^(hand)` | `1e-3` | `1e-3` | `1e-3` |

The paper splits handover loss by manifold; this library splits it by encoding, which is the finer granularity.

### Operation times

Seconds. Keys of the `gate_time_c` / `gate_time_g` / `gate_time_m` dicts.

| Key | Paper | `_c` | `_g` | `_m` | legacy |
|---|---|---|---|---|---|
| `t_1Q` | `t_1Q^(c/g/m)` | `100e-6` | `1e-6` | `1e-6` | `100e-6` for `_g` / `_m` |
| `t_1Q_gm` | `t_1Q^(gm)` | — | `10e-6` | — | `100e-6` |
| `t_2Q` | `t_2Q` | `300e-9` | `300e-9` | `300e-9` | |
| `t_reset` | `t_reset` | `2e-3` | `2e-3` | `2e-3` | |
| `t_read` | `t_read` | `1e-3` | `1e-3` | `1e-3` | |

## Reading and writing parameter files

`save_noise_params_to_file` / `load_noise_params_from_file` use a flat `name: value` text format, with slash notation for the nested dicts:

```
# 174Yb clock qubit
p_1_c: 0.0001
gate_time_c/t_1Q: 0.0001
```

`set_parameters` accepts the same slash notation and rejects unknown names:

```python
params.set_parameters(p_1_c=3e-4, gamma_mg=1.5)
params.set_parameters(**{"gate_time_c/t_1Q": 200e-6})
```

`rescale_error_params(scale)` multiplies every parameter whose name starts with `p_` or `gamma_`, leaving ratios (`q_BB_*`), geometry (`l_site`, `l_zone`, `a`) and times (`t_hand`, `gate_time_*`) untouched.

## Mapping from the legacy parameter names

| Legacy | Here |
|---|---|
| `p_gm_174` | `p_1_c` |
| `p_g_171` | `p_1_g` |
| `p_m_171` | `p_1_m` |
| `p_gm_171` | `p_1_gm` |
| `p_2_dep` | `p_2_c`, `p_2_g`, `p_2_m`, `p_2_dual` |
| `p_MG_gate_171m` | `p_m_g_gate` |
| `p_MERR_174` | `p_meas_c` |
| `p_MERR` | `p_meas_g` |
| `p_MERR_171m` | `p_meas_m` |
| `q_MERR_174`, `q_MERR_171m` | `q_BB_c`, `q_BB_m` |
| `p_GLOSSR_174/171g/171m` | `p_g_L_meas_c/_g/_m` |
| `p_GLOSSP_174/171g` | `p_g_L_reset_c/_g` |
| `p_MLOSSP_171` | `p_m_L_reset_m` |
| `p_X_g_171` | `p_flip_g_g` |
| `p_X_meas_171m` | `p_flip_g_m` |
| `p_X_motional_171` | `p_flip_m_m` |
| `p_handover_loss_174/171g/171m` | `p_hand_c/_g/_m` |
| `gamma_Zidl_174/171g/171m` | `gamma_Z_c/_g/_m` |
| `gamma_MG_174` | `gamma_mg_c` |
| `gamma_MG_idl_171m` | `gamma_mg_g`, `gamma_mg_m` |
| `gamma_RL` | `gamma_Ryd` |
| `RL_ratio` | `ryd_branching` |
| `site_separation`, `zone_separation` | `l_site`, `l_zone` |
| `handover_time`, `acceleration` | `t_hand`, `a` |
| `gate_time_174/171g/171m` | `gate_time_c/_g/_m` |
| `gate_1q`, `gate_1q_clock`, `gate_2q`, `reset`, `measure` | `t_1Q`, `t_1Q_gm`, `t_2Q`, `t_reset`, `t_read` |
| `p_X_mr_174`, `p_X_mr_171` | removed with the `XERR_mr` channel |
| `alpha` | removed; MERR `p_Z` uses `((2 q_BB − 1)²/16) p_meas²` |
| `T_2_174` | removed; nothing read it |
