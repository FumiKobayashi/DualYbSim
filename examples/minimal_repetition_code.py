"""Smallest end-to-end example: ideal circuit in, noisy circuit and samples out.

Run with:  python examples/minimal_repetition_code.py
"""

from collections import Counter

import stim

from dualybsim import NoiseModelParameters, QubitManager, YbNoiseModel

# --- Which atom is each qubit? -------------------------------------------
# Data qubits are 171Yb encoded in the metastable manifold; ancillas are 174Yb
# optical clock qubits. This is the dual-isotope layout the model is built for:
# the ancillas can be read out in place because the imaging light is off-resonant
# for the other isotope.
qubits = QubitManager()
for q in (0, 2, 4):
    qubits.add_qubit(q, isotope="171", qubit_type="m", role="data")
for q in (1, 3):
    qubits.add_qubit(q, isotope="174", qubit_type="gm", role="ancilla")

# --- An ideal Z-check round ----------------------------------------------
# Gates are Rydberg-mediated, so the two-qubit gate is CZ rather than CX.
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

# --- Add the noise -------------------------------------------------------
params = NoiseModelParameters()  # the paper's tabulated values
noisy = YbNoiseModel(params).noisy_circuit(ideal, qubits)

print("=== noisy circuit ===")
print(noisy)

print()
print("=== instruction counts ===")
for name, count in sorted(Counter(i.name for i in noisy.flattened()).items()):
    print(f"  {name:20s} {count}")

# --- Decode-ready output -------------------------------------------------
# approximate_disjoint_errors is required because loss is modelled as
# HERALDED_ERASE, whose four Pauli branches are mutually exclusive.
dem = noisy.detector_error_model(
    decompose_errors=False,
    allow_gauge_detectors=True,
    approximate_disjoint_errors=True,
)
print()
print(f"=== detector error model: {dem.num_errors} error mechanisms ===")

samples = noisy.compile_detector_sampler().sample(shots=100_000)
print(f"detector fire rate over {samples.shape[0]} shots: {samples.mean():.5f}")

# --- Sweeping a parameter ------------------------------------------------
print()
print("=== detector fire rate against the two-qubit gate error ===")
for p_2 in (1e-4, 1e-3, 1e-2):
    swept = NoiseModelParameters()
    swept.p_2_c = swept.p_2_m = swept.p_2_dual = p_2
    circuit = YbNoiseModel(swept).noisy_circuit(ideal, qubits)
    rate = circuit.compile_detector_sampler().sample(shots=100_000).mean()
    print(f"  p_2 = {p_2:.0e}  ->  {rate:.5f}")
