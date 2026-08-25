# Channel ordering

The order channels are composed in, and where each one is emitted in the code. Channels act right to left, matching the paper's notation; in the generated Stim circuit that is the same as top to bottom, since a Stim instruction acts after everything above it.

The general rule is: the ideal operation first, then the channels specific to that operation, then the time-dependent channels, then the decay block.

## The decay block

Whenever the decay block is inserted, its internal order is fixed by decreasing damping rate:

```
D = LOSS_m . LOSS_g . DECAY_mg . DECAY_rm . DECAY_rg . LOSS_r
```

Emitted top to bottom as `LOSS_r`, `DECAY_rg`, `DECAY_rm`, `DECAY_mg`, `LOSS_g`, `LOSS_m`.

The Rydberg branches carry a renormalisation, because factorising one aggregate channel into three sequential ones means each later branch acts on a population the earlier ones have already depleted. With `p̂[r→i] = b_i (1 − exp(−Γ_Ryd t))`:

```
p[r→L] = p̂[r→L]
p[r→g] = p̂[r→g] / (1 − p̂[r→L])
p[r→m] = p̂[r→m] / ((1 − p̂[r→L])(1 − p̂[r→g]))
```

`NoiseModelParameters.rydberg_branch_rates` computes these.

The Rydberg branches only fire while an atom is actually driven to `|r>`, which happens inside a two-qubit gate, so they are emitted there rather than in the shared time-dependent path. `LOSS_g` / `LOSS_m` and `DECAY_mg` accumulate whenever time passes.

## Time-dependent channels

`YbCircuit._apply_time_dependent_noise` emits the idling block followed by the non-Rydberg part of the decay block, and is shared by `idling`, `transport_with_time` and `handover`:

| Order | 174Yb clock | 171Yb-g | 171Yb-m |
|---|---|---|---|
| 1 | — | `ZERR_g` | `ZERR_m` |
| 2 | — | `XERR_g` | `XERR_m` |
| 3 | `ZERR_c` + `DECAY_mg`, twirled together | — | `DECAY_mg` (loss) |
| 4 | `LOSS_g` / `LOSS_m` | `LOSS_g` | `LOSS_m` |

Three encoding-dependent choices are worth spelling out.

`DECAY_mg` acts only on encodings that occupy the metastable manifold. For the clock qubit `m → g` stays inside the computational subspace, so it is amplitude damping; for the 171Yb-m qubit it leaves the subspace, so it is a loss channel. A 171Yb-g qubit leaves the metastable manifold empty while it waits, so the channel does not act on it at all.

For the clock qubit, `ZERR_c` and `DECAY_mg` act on the same two levels, so they are twirled into a single channel rather than emitted separately:

```
p_X = p_Y = p_1(t)/4,    p_Z = p_2(t)/2 − p_1(t)/4
```

with `p_1 = 1 − exp(−t/T_1)` and `p_2 = 1 − exp(−t/T_2)`, taking `T_1` from `gamma_mg_c` and `T_2` from `gamma_Z_c`. The factor of a half on `p_2` is what makes the channel reproduce the transverse decay `exp(−t/T_2^(c))`: a Pauli channel maps that component by `1 − 2p_Y − 2p_Z`.

`Γ_mg` is per encoding, because the effective metastable lifetime depends on the trap depth the encoding operates in. The clock qubit idles in a shallow trap and so uses the natural <sup>3</sup>P<sub>0</sub> lifetime, `gamma_mg_c = 1/20` s⁻¹; the 171Yb channels are reached only through the clock pulse and the readout shelving step, both of which run in the deep imaging trap, and keep the scattering-dominated `1` s⁻¹. The clock qubit's value also has to satisfy `T_2 ≤ 2 T_1` — amplitude damping at `T_1` alone already decoheres at `2 T_1` — which `T_2^(c) = 5 s` against `T_1 = 20 s` does and against `T_1 = 1 s` would not.

The idling channels are disabled during coherent-control operations and reset, because their contribution is already inside those operations' own error channels. In the code this is implicit: those paths call `twirled_amplitude_damping` without a `T_2_inv`, which selects the damping limit and returns `p_Z = 0`.

## Per-operation ordering

### Coherent-control

```
U~(2Q, dual) = D . DEP2_dual . U(2Q)
U~(1Q, m)    = D . DECAY_mg_gate . DEP1_m . U(1Q)
```

Two-qubit gates take their depolarising rate from the pattern:

| Pattern | Parameter | Emitted by |
|---|---|---|
| 174–174 | `p_2_c` | `_apply_same_isotope_noisy_cz` |
| 171m–171m | `p_2_m` | `_apply_same_isotope_noisy_cz` |
| 171g–171g | `p_2_g` | `_apply_same_isotope_noisy_cz` |
| 171m–174 | `p_2_dual` | `_apply_dual_isotope_noisy_cz` |

A 171Yb-g two-qubit gate is bracketed by clock pulses, because the g qubit has to be driven into the metastable manifold to reach the Rydberg state. So the sequence is clock pulse, `CZ` with its noise, clock pulse — and the g qubit sees all the same Rydberg decay channels an m qubit would.

### Measurement

Readout is a sequence of state-selective steps, with the classical discrimination error `MERR` applied once at the end of the whole quantum sequence. `D_c` denotes an ideal clock de-excitation.

```
M~(171, g) = MERR . (D . LOSS_g_meas . FLIP_g)          x2
M~(171, m) = MERR . (D . LOSS_g_meas . FLIP_g)
                  . (D . DEP1_gm . D_c)                  x2 alternating
M~(174)    = MERR . (D . LOSS_g_meas)
                  . (D . DEP1_c . D_c) . (D . LOSS_g_meas)
```

For the 171Yb-m qubit the library does not emit these factors one by one. The whole eight-step sequence is twirled as a unit into a single loss plus Pauli channel, because twirling each factor separately loses the correlations between them. `NoiseModelParameters.get_twirled_171m_measurement_error_rates` evaluates the resulting closed form.

### Reset

```
R~(171, g) = D . LOSS_g_reset . FLIP_g   . R(171, g)
R~(171, m) = D . LOSS_m_reset . FLIP_m   . R(171, m)
R~(174)    = D . LOSS_g_reset            . R(174)
```

The 171Yb-m qubit has two reset patterns. Pattern `b`, the default, is a direct metastable reset and emits `LOSS_m_reset`. Pattern `a` goes via the ground manifold and so emits `FLIP_m` plus the clock-transition channels.

### Idling

```
I~ = XERR . ZERR . I
```

### Transportation

```
I~(trans) = LOSS_hand . D . I~ . LOSS_hand . I(trans)
```

Two handover events, one into the movable trap and one back out, with the shuttling noise between them. Each uses the loss probability of its own encoding. The transport time is `t_move(l) = 2 sqrt(l / a)` over `l = d · l_site + l_zone`.
