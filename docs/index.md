# DualYbSim

A noise model for dual-isotope ytterbium neutral-atom qubits, packaged as a
wrapper around [Stim](https://github.com/quantumlib/Stim). Give it an ideal
Clifford circuit and a description of which atom each qubit is; get back the
same circuit with physically-motivated noise inserted, ready for detector
sampling and decoding.

These pages are the reference documentation. For installation, usage and the
test suite, see the
[README](https://github.com/FumiKobayashi/DualYbSim#readme).

## The processor model

The model covers a processor built from 171Yb and 174Yb:

- **174Yb** as an optical clock qubit, `|0> = |g>` (1S0), `|1> = |m>` (3P0)
- **171Yb-g**, a nuclear-spin qubit in the 1S0 ground manifold
- **171Yb-m**, a nuclear-spin qubit in the 3P0 metastable manifold

Noise is defined as Kraus operators on the full multi-level Hilbert space —
including the Rydberg state `|r>` and an effective loss state `|L>` — and then
reduced to Pauli plus loss channels by the generalised Pauli twirling
approximation, so it can be simulated as a stabiliser circuit.

## Reference

- [Channel reference](https://fumikobayashi.github.io/DualYbSim/channel_reference)
  — every noise channel, what it models, and which Kraus class provides it.
- [Parameter reference](https://fumikobayashi.github.io/DualYbSim/parameter_reference)
  — every parameter, both presets' values, and the file format.
- [Channel ordering](https://fumikobayashi.github.io/DualYbSim/channel_ordering)
  — the order channels are composed in, per operation.

Alongside these,
[examples/noise_channel_tour.ipynb](https://github.com/FumiKobayashi/DualYbSim/blob/main/examples/noise_channel_tour.ipynb)
reads the same rules off real circuits: every operation, for every encoding,
with the emitted channels annotated line by line.

## Status

Pre-1.0, and the accompanying paper is still in preparation, so both the public
API and the tabulated parameter values may change. Released under the MIT
licence; see
[CONTRIBUTING.md](https://github.com/FumiKobayashi/DualYbSim/blob/main/CONTRIBUTING.md)
if you would like to help.
