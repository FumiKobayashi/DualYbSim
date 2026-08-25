# Noise channel reference

Every channel in this library is named after the corresponding channel in the noise-model appendix of the dual-Yb surface-code paper. This page is the mapping between the paper's notation and the Python identifiers.

## Naming convention

| Paper notation | Identifier | Example |
|---|---|---|
| subscript | joined with `_` | `LOSS_g`, `ZERR_gm`, `DECAY_rg` |
| superscript `(...)` | appended with `_` | `LOSS_g^(meas)` → `LOSS_g_meas`, `DECAY_mg^(gate)` → `DECAY_mg_gate` |
| probability `p_n^(x)` | `p_n_x` | `p_1^(gm)` → `p_1_gm`, `p_2^(dual)` → `p_2_dual` |
| transition probability `p_(a→b)^(x)` | `p_a_b_x` | `p_(g→L)^(meas)` → `p_g_L_meas` |
| rate `Γ_ab` | `gamma_ab` | `Γ_rg` → `gamma_rg`, `Γ_gL` → `gamma_gL` |
| time `T_n^(x)`, `t_x` | `T_n_x`, `t_x` | `T_2^(gm)` → `T_2_gm`, `t_1Q^(c)` → `t_1Q_c` |

Channel names appear as keys of the `noise_channels` dict on each Kraus class in `dualybsim.kraus.yb171` / `dualybsim.kraus.yb174`, and as the `channel` argument of `YbNoiseChannel.get_kraus_operators` and `CPTP`.

## Channels by category

The paper groups channels into six categories. Loss and decay channels are realised in Stim as `HERALDED_ERASE`; Pauli channels as `PAULI_CHANNEL_1`, `DEPOLARIZE1`, `DEPOLARIZE2`, `X_ERROR` or `Z_ERROR`.

### Coherent-control

| Channel | Isotope | Meaning |
|---|---|---|
| `DEP1_c` | 174 | single-qubit depolarising error on the clock transition |
| `DEP1_g` | 171 | single-qubit depolarising error inside the ground manifold |
| `DEP1_m` | 171 | single-qubit depolarising error inside the metastable manifold |
| `DEP1_gm` | 171 | depolarising error during clock-transition excitation |
| `DECAY_mg_gate` | 171 | metastable→ground damping driven by the nuclear-spin control laser |
| `DEP2_c` | 174 | two-qubit depolarising error, clock–clock |
| `DEP2_m` | 171 | two-qubit depolarising error, m–m |
| `DEP2_dual` | 171, 174 | two-qubit depolarising error, 171-m with 174-clock |

### Measurement

| Channel | Isotope | Meaning |
|---|---|---|
| `LOSS_g_meas` | 171, 174 | atom loss from the ground manifold during fluorescence imaging |
| `FLIP_g` | 171 | in-manifold bit flip induced by the imaging lasers |
| `MERR` | 171, 174 | classical discrimination error of the loss-aware readout |

### Reset

| Channel | Isotope | Meaning |
|---|---|---|
| `LOSS_g_reset` | 171, 174 | loss from the ground manifold during preparation or motional reset |
| `LOSS_m_reset` | 171 | loss from the metastable manifold during spin reset |
| `FLIP_g` | 171 | nuclear-spin flip in the ground manifold during reset |
| `FLIP_m` | 171 | nuclear-spin flip in the metastable manifold during reset |

### Idling

Probabilities follow `p(t) = 1 − exp(−t/T)`.

| Channel | Isotope | Timescale |
|---|---|---|
| `ZERR_c` | 174 | `T_2^(c)` |
| `ZERR_g` | 171 | `T_2^(g)` |
| `ZERR_m` | 171 | `T_2^(m)` |
| `ZERR_gm` | 171 | `T_2^(gm)`, clock-transition coherence. Available in the Kraus layer but not injected into circuits — see [`parameter_reference.md`](parameter_reference.md) |
| `XERR_g` | 171 | `T_1^(g)` |
| `XERR_m` | 171 | `T_1^(m)` |

### Decay

Probabilities follow `p(t) = 1 − exp(−Γt)`. The Rydberg branches share a total rate `Γ_Ryd` split 0.51 / 0.42 / 0.07 into `LOSS_r` / `DECAY_rg` / `DECAY_rm`.

| Channel | Isotope | Rate |
|---|---|---|
| `LOSS_r` | 171, 174 | `Γ_rL`, Rydberg decay to untrapped or dark states |
| `DECAY_rg` | 171, 174 | `Γ_rg`, Rydberg decay to the ground manifold |
| `DECAY_rm` | 171, 174 | `Γ_rm`, Rydberg decay to the metastable manifold |
| `DECAY_mg` | 171, 174 | `Γ_mg`, metastable→ground decay |
| `LOSS_g` | 171, 174 | `Γ_gL`, ground-manifold loss from the finite trap lifetime |
| `LOSS_m` | 171, 174 | `Γ_mL`, metastable-manifold loss from the finite trap lifetime |

### Transportation

| Channel | Isotope | Meaning |
|---|---|---|
| `LOSS_g_hand` | 171, 174 | ground-manifold loss during trap handover |
| `LOSS_m_hand` | 171, 174 | metastable-manifold loss during trap handover |

## Which channels each Kraus class provides

| Class | Channels |
|---|---|
| `Kraus1Q_174` | `DEP1_c`, `ZERR_c`, `LOSS_g`, `LOSS_m`, `DECAY_mg`, `LOSS_r`, `DECAY_rg`, `DECAY_rm` |
| `Kraus2Q_174174` | `DEP2_c`, `ZERR_c`, `LOSS_g`, `LOSS_m`, `DECAY_mg`, `LOSS_r`, `DECAY_rg`, `DECAY_rm` |
| `KrausRESET_174` | `LOSS_g_reset` |
| `KrausMEASURE_DISC_174` | `MERR` |
| `KrausMEASURE_174` | `LOSS_g_meas`, `LOSS_m`, `DECAY_mg`, `LOSS_r`, `DECAY_rg`, `DECAY_rm` |
| `Kraus1QClock_171m` | `DEP1_gm`, `LOSS_g`, `LOSS_m`, `DECAY_mg`, `LOSS_r`, `DECAY_rg`, `DECAY_rm` |
| `Kraus1Q_171m` | `DEP1_m`, `DECAY_mg_gate`, `ZERR_g`, `XERR_g`, `ZERR_m`, `XERR_m`, `ZERR_gm`, `LOSS_g`, `LOSS_m`, `DECAY_mg`, `LOSS_r`, `DECAY_rg`, `DECAY_rm` |
| `Kraus2Q_171m171m` | `DEP2_m`, `ZERR_g`, `XERR_g`, `ZERR_m`, `XERR_m`, `ZERR_gm`, `LOSS_g`, `LOSS_m`, `DECAY_mg`, `LOSS_r`, `DECAY_rg`, `DECAY_rm` |
| `KrausRESET_171m` | `LOSS_m_reset`, `FLIP_m` |
| `KrausMEASURE_DISC_171m` | `MERR` |
| `KrausMEASURE_171m` | `FLIP_g`, `LOSS_g_meas`, `ZERR_m`, `XERR_m`, `ZERR_gm`, `LOSS_m`, `DECAY_mg`, `LOSS_r`, `DECAY_rg`, `DECAY_rm` |

## Mapping from the legacy channel names

For anyone cross-reading against results produced before this library existed. The paper's conventions section explicitly avoids *leakage* as a way of classifying errors, which is why every `*LEAK` name became either `LOSS_*` or `DECAY_*`.

| Legacy name | Here |
|---|---|
| `GLEAK` | `LOSS_g` |
| `MLEAK` | `LOSS_m` |
| `MGLEAK` | `DECAY_mg` |
| `MGLEAK_gate` | `DECAY_mg_gate` |
| `RLEAK` | `LOSS_r` |
| `RGLEAK` | `DECAY_rg` |
| `RMLEAK` | `DECAY_rm` |
| `GZERR` | `ZERR_g` |
| `MZERR` | `ZERR_m` |
| `GMZERR` | `ZERR_gm` |
| `GXERR` (T1 idling) | `XERR_g` |
| `MXERR` (T1 idling) | `XERR_m` |
| `GXERR_read` | `FLIP_g` |
| `MXERR_prep` | `FLIP_m` |
| `GLOSS_prep` | `LOSS_g_reset` |
| `MLOSS_prep` | `LOSS_m_reset` |
| `GLOSS_meas`, `GLOSS_read` | `LOSS_g_meas` |
| `DEP1_gm` (174 class) | `DEP1_c` |
| `DEP2_gm` (174 class) | `DEP2_c` |
| `ZERR_gm` (174 class) | `ZERR_c` |
| `XERR_mr` | removed — no such channel in the paper, and the legacy default was 0 |

Three legacy keys were ambiguous because the same string named different physical channels in different classes; the paper names disambiguate them:

- `MXERR` was the `T_1^(m)` idling channel in `Kraus1Q_171m` but the reset spin flip in `KrausRESET_171m` → now `XERR_m` and `FLIP_m`.
- `GXERR` was the `T_1^(g)` idling channel but the imaging bit flip in `KrausMEASURE_171m` → now `XERR_g` and `FLIP_g`.
- `GLOSS` was the reset loss in `KrausRESET_174` but the readout loss in `KrausMEASURE_174` / `KrausMEASURE_171m` → now `LOSS_g_reset` and `LOSS_g_meas`.

The twirled measurement helpers on `NoiseModelParameters` return the loss probability under the key `p_loss` (legacy: `p_leak`), matching the paper's `E_loss` channel.
