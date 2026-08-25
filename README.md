# DualYbSim

[![CI](https://github.com/FumiKobayashi/DualYbSim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FumiKobayashi/DualYbSim/actions/workflows/ci.yml)

This library is a noise model replicating ytterbium atom qubits on Clifford circuits.

A noise model for dual-isotope ytterbium neutral-atom qubits, packaged as a wrapper around [Stim](https://github.com/quantumlib/Stim). Give it an ideal Clifford circuit and a description of which atom each qubit is; get back the same circuit with physically-motivated noise inserted, ready for detector sampling and decoding.

The model covers a processor built from 171Yb and 174Yb:

- **174Yb** as an optical clock qubit, `|0> = |g>` (1S0), `|1> = |m>` (3P0)
- **171Yb-g**, a nuclear-spin qubit in the 1S0 ground manifold
- **171Yb-m**, a nuclear-spin qubit in the 3P0 metastable manifold

Noise is defined as Kraus operators on the full multi-level Hilbert space — including the Rydberg state `|r>` and an effective loss state `|L>` — and then reduced to Pauli plus loss channels by the generalised Pauli twirling approximation, so it can be simulated as a stabiliser circuit.

## Install

[uv](https://docs.astral.sh/uv/) is the recommended way to install the library and the only supported way to develop it.

Using it from another project — not on PyPI yet, so from git:

```bash
uv add git+https://github.com/FumiKobayashi/DualYbSim
```

Working on the library itself:

```bash
uv sync                    # .venv from uv.lock: the library, editable, plus the dev tooling
uv run pre-commit install  # ruff (lint, format and docstrings), mypy and bandit on commit
uv run pytest
```

Nothing needs activating: every `uv run` uses that environment. `uv sync` does not need an interpreter on the machine either — it reads the `3.13` pin in `.python-version` and fetches that toolchain if it is missing.

Three dependency groups are defined. `dev` is what a plain `uv sync` installs; `test` is pytest alone, as CI's version matrix installs it; `notebook` adds the IPython kernel that [examples/noise_channel_tour.ipynb](examples/noise_channel_tour.ipynb) needs to re-run.

```bash
uv sync --group notebook
uv sync --locked --no-default-groups --group test
```

After changing a dependency, `uv lock` regenerates `uv.lock`; commit it, because CI installs with `uv sync --locked` and fails if the two have drifted.

Requires Python 3.10 or newer; the only runtime dependencies are `numpy` and `stim`. Without uv, `pip install -e .` still works, and pip 25.1 or newer reads the same groups with `pip install --group dev -e .` — neither path is locked, so versions can drift from what CI tests.

## Usage

```python
import stim
from dualybsim import NoiseModelParameters, QubitManager, YbNoiseModel

# Say which atom each qubit is.
qubits = QubitManager()
for q in (0, 2, 4):
    qubits.add_qubit(q, isotope="171", qubit_type="m", role="data")
for q in (1, 3):
    qubits.add_qubit(q, isotope="174", qubit_type="gm", role="ancilla")

ideal = stim.Circuit("""
    R 0 1 2 3 4
    TICK
    H 1 3
    TICK
    CZ 0 1 2 3
    TICK
    CZ 1 2 3 4
    TICK
    H 1 3
    TICK
    M 1 3
    TICK
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    OBSERVABLE_INCLUDE(0) rec[-1]
""")

noisy = YbNoiseModel(NoiseModelParameters()).noisy_circuit(ideal, qubits)

dem = noisy.detector_error_model(
    decompose_errors=False,
    allow_gauge_detectors=True,
    approximate_disjoint_errors=True,  # required: the model emits HERALDED_ERASE
)
samples = noisy.compile_detector_sampler().sample(shots=10_000)
```

`examples/minimal_repetition_code.py` is an example of usage for QEC, and in [examples/noise_channel_tour.ipynb](examples/noise_channel_tour.ipynb) the noise channels attached to every operation are described and printed.

The common two-qubit gate for Rydberg blockade is `CZ`, so `CZ` is the only two-qubit gate this library accepts; `CX` is not supported. Supported instructions are the single-qubit Cliffords, `CZ`, `M`, `R`, `I`, `TICK`, `QUBIT_COORDS`, `SHIFT_COORDS`, `DETECTOR` and `OBSERVABLE_INCLUDE`.

Close every moment with a `TICK`, the final one included. The `TICK` is what tells the model a moment is over and which qubits idled through it, so the last moment of a circuit that ends without one gets no idling noise, and a `readout_protocol` cannot tell which qubits sat out the measurement. The library warns when it has to drop idling for that reason.

Loss is represented as `HERALDED_ERASE`, following the paper: the qubit is replaced by the maximally mixed state and a herald is recorded but deliberately not given to the decoder, since detecting it would need leakage-detection operations the device model does not include.

### Readout protocols

Mid-circuit measurement can be modelled three ways, each with its own noise:

```python
model = YbNoiseModel(NoiseModelParameters())

model.noisy_circuit(ideal, qubits)  # in place
model.noisy_circuit(ideal, qubits, readout_protocol="in_place_direct")
model.noisy_circuit(ideal, qubits, readout_protocol="transport", code_distance=5)
model.noisy_circuit(ideal, qubits, readout_protocol="shelving")  # 171Yb-g only
```

`transport` shuttles the measured qubits to a readout zone and back, accumulating dephasing, decay and handover loss over the trip; `shelving` hides the unmeasured qubits in the metastable manifold across a clock transition.

### Parameters

Every parameter is named after the symbol used in the paper, with a trailing tag for the encoding it belongs to. Read them grouped:

```python
params = NoiseModelParameters()

params.p_1_m  # DEP1_m depolarising probability
params.for_qubit("171", "m").p_1  # the same value, tag dropped
params.for_qubit("174", "gm").gamma_Z  # 1/T_2 of the clock qubit

params.set_parameters(p_meas_c=5e-4)
params.set_parameters(**{"gate_time_c/t_1Q": 200e-6})
# Halves every probability and rate; returns a copy unless inplace=True.
halved = params.rescale_error_params(0.5)
```

Two presets are available. `NoiseModelParameters()` carries the paper's tabulated values; `NoiseModelParameters.legacy_defaults()` reproduces an earlier, legacy parameter set, for comparison against numbers produced before this library existed.

### Building circuits directly

`YbCircuit` subclasses `stim.Circuit` and adds noise as you go, which is useful for constructions Stim's text format cannot express, such as transport:

```python
from dualybsim import YbCircuit

c = YbCircuit(qubits, NoiseModelParameters())
c.reset_qubit([0, 1, 2, 3, 4], pattern="b")
c.single_qubit_gate("H", [1, 3])
c.two_qubit_gate("CZ", [0, 1])
c.idling([2, 3, 4], duration=3e-4)
c.transport([1, 3], distance=1e-4)
c.measurement([1, 3])
```

## Documentation

|                                                                          |                                                                                                                                                                                             |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [docs/channel_reference.md](docs/channel_reference.md)                 | every noise channel, what it models, and which Kraus class provides it                                                                                                                      |
| [docs/parameter_reference.md](docs/parameter_reference.md)             | every parameter, both presets' values, and the file format                                                                                                                                  |
| [docs/channel_ordering.md](docs/channel_ordering.md)                   | the order channels are composed in, per operation                                                                                                                                           |
| [examples/noise_channel_tour.ipynb](examples/noise_channel_tour.ipynb) | the same rules read off real circuits: every operation, for every encoding, with the emitted channels annotated line by line (committed with outputs; `uv sync --group notebook` to re-run) |

## Tests

```bash
uv run pytest                # fast suite
uv run pytest --run-slow     # adds the Kraus-level twirling checks
```

`tests/test_paper_consistency.py` transcribes the paper's tables and pins the library to them, so a drift in either direction fails the build.

## Licence

MIT. See [LICENSE](LICENSE).
