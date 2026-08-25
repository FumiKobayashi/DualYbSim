"""Tests for YbCircuit and YbNoiseModel: what gets injected, and where."""

import warnings
from collections import Counter

import pytest
import stim

from dualybsim import NoiseModelParameters, QubitManager, YbCircuit, YbNoiseModel

Z_CHECK = stim.Circuit("""
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


def qubits(data_isotope: str, data_type: str) -> QubitManager:
    qm = QubitManager()
    for q in (0, 2, 4):
        qm.add_qubit(q, isotope=data_isotope, qubit_type=data_type, role="data")
    for q in (1, 3):
        qm.add_qubit(q, isotope="174", qubit_type="gm", role="ancilla")
    return qm


ENCODINGS = [("171", "m"), ("171", "g"), ("174", "gm")]


def instruction_names(circuit: stim.Circuit) -> Counter:
    return Counter(inst.name for inst in circuit.flattened())


class TestNoiseInjection:
    @pytest.mark.parametrize(("isotope", "qubit_type"), ENCODINGS)
    def test_noise_is_added(self, isotope: str, qubit_type: str):
        qm = qubits(isotope, qubit_type)
        model = YbNoiseModel(NoiseModelParameters())
        ideal = model.noiseless_circuit(Z_CHECK, qm)
        noisy = model.noisy_circuit(Z_CHECK, qm)
        assert len(noisy.flattened()) > len(ideal.flattened())

    @pytest.mark.parametrize(("isotope", "qubit_type"), ENCODINGS)
    def test_ideal_gates_are_preserved(self, isotope: str, qubit_type: str):
        """Noise insertion must not disturb the logical operations."""
        qm = qubits(isotope, qubit_type)
        noisy = YbNoiseModel(NoiseModelParameters()).noisy_circuit(Z_CHECK, qm)
        counts = instruction_names(noisy)
        assert counts["CZ"] == 4
        assert counts["H"] == 2
        assert counts["R"] >= 1
        assert counts["M"] >= 1

    @pytest.mark.parametrize(("isotope", "qubit_type"), ENCODINGS)
    def test_detectors_and_observables_survive(self, isotope: str, qubit_type: str):
        qm = qubits(isotope, qubit_type)
        noisy = YbNoiseModel(NoiseModelParameters()).noisy_circuit(Z_CHECK, qm)
        assert noisy.num_detectors == Z_CHECK.num_detectors
        assert noisy.num_observables == Z_CHECK.num_observables

    def test_noise_disabled_matches_the_input(self):
        qm = qubits("171", "m")
        ideal = YbNoiseModel(NoiseModelParameters()).noiseless_circuit(Z_CHECK, qm)
        names = set(instruction_names(ideal))
        assert not names & {
            "DEPOLARIZE1",
            "DEPOLARIZE2",
            "PAULI_CHANNEL_1",
            "HERALDED_ERASE",
            "X_ERROR",
            "Z_ERROR",
        }

    @pytest.mark.parametrize(("isotope", "qubit_type"), ENCODINGS)
    def test_trap_loss_is_present(self, isotope: str, qubit_type: str):
        """LOSS_g / LOSS_m must fire; with the rate zeroed they must not."""
        qm = qubits(isotope, qubit_type)
        with_loss = YbNoiseModel(NoiseModelParameters()).noisy_circuit(Z_CHECK, qm)

        p = NoiseModelParameters()
        p.gamma_gL = 0.0
        p.gamma_mL = 0.0
        without = YbNoiseModel(p).noisy_circuit(Z_CHECK, qubits(isotope, qubit_type))

        erase = "HERALDED_ERASE"
        assert instruction_names(with_loss)[erase] > instruction_names(without)[erase]

    def test_xerr_fires_on_idling_for_171(self):
        """XERR_g / XERR_m accumulate over idling windows."""
        qm = qubits("171", "m")
        with_x = YbNoiseModel(NoiseModelParameters()).noisy_circuit(Z_CHECK, qm)

        p = NoiseModelParameters()
        p.gamma_X_g = 0.0
        p.gamma_X_m = 0.0
        without = YbNoiseModel(p).noisy_circuit(Z_CHECK, qubits("171", "m"))

        assert (
            instruction_names(with_x)["X_ERROR"] > instruction_names(without)["X_ERROR"]
        )

    def test_174_idling_uses_one_combined_channel(self):
        """ZERR_c and DECAY_mg are twirled together, not emitted separately."""
        qm = QubitManager()
        qm.add_qubit(0, isotope="174", qubit_type="gm", role="data")
        p = NoiseModelParameters()
        c = YbCircuit(qm, p)
        c.idling([0], 1e-3)

        emitted = [
            (inst.name, tuple(inst.gate_args_copy()))
            for inst in c.flattened()
            if inst.name not in ("I", "TICK")
        ]
        # One PAULI_CHANNEL_1 carrying p_X, p_Y and p_Z together, plus the trap
        # loss. No separate Z_ERROR.
        assert [name for name, _ in emitted] == ["PAULI_CHANNEL_1", "HERALDED_ERASE"]
        p_X, p_Y, p_Z = emitted[0][1]
        assert p_X == p_Y > 0
        assert p_Z > 0

    def test_174_combined_channel_reproduces_t2(self):
        """The emitted channel must decay coherence at exactly 1/T_2^(c)."""
        import numpy as np

        qm = QubitManager()
        qm.add_qubit(0, isotope="174", qubit_type="gm", role="data")
        p = NoiseModelParameters()
        duration = 1e-3
        c = YbCircuit(qm, p)
        c.idling([0], duration)

        args = next(
            inst.gate_args_copy()
            for inst in c.flattened()
            if inst.name == "PAULI_CHANNEL_1"
        )
        _p_X, p_Y, p_Z = args
        transverse = 1 - 2 * p_Y - 2 * p_Z
        assert transverse == pytest.approx(np.exp(-duration * p.gamma_Z_c), rel=1e-12)


class TestCzPatterns:
    """Each CZ pattern draws its depolarising rate from its own parameter."""

    @pytest.mark.parametrize(
        ("isotope", "qubit_type", "attr"),
        [("174", "gm", "p_2_c"), ("171", "m", "p_2_m"), ("171", "g", "p_2_g")],
    )
    def test_same_isotope_pattern(self, isotope: str, qubit_type: str, attr: str):
        qm = QubitManager()
        for q in (0, 1):
            qm.add_qubit(q, isotope=isotope, qubit_type=qubit_type, role="data")
        p = NoiseModelParameters()
        setattr(p, attr, 0.0123)
        c = YbCircuit(qm, p)
        c.two_qubit_gate("CZ", [0, 1])
        rates = [
            inst.gate_args_copy()[0]
            for inst in c.flattened()
            if inst.name == "DEPOLARIZE2"
        ]
        assert 0.0123 in rates

    def test_dual_pattern(self):
        qm = QubitManager()
        qm.add_qubit(0, isotope="171", qubit_type="m", role="data")
        qm.add_qubit(1, isotope="174", qubit_type="gm", role="ancilla")
        p = NoiseModelParameters()
        p.p_2_dual = 0.0456
        c = YbCircuit(qm, p)
        c.two_qubit_gate("CZ", [0, 1])
        rates = [
            inst.gate_args_copy()[0]
            for inst in c.flattened()
            if inst.name == "DEPOLARIZE2"
        ]
        assert 0.0456 in rates


class TestReadoutProtocols:
    PROTOCOLS = ["in_place_direct", "transport", "shelving"]

    @pytest.mark.parametrize("protocol", PROTOCOLS)
    def test_builds_and_compiles(self, protocol: str):
        # Shelving is only implemented for the 171Yb g encoding.
        isotope, qubit_type = ("171", "g") if protocol == "shelving" else ("171", "m")
        qm = qubits(isotope, qubit_type)
        noisy = YbNoiseModel(NoiseModelParameters()).noisy_circuit(
            Z_CHECK, qm, readout_protocol=protocol, code_distance=3
        )
        dem = noisy.detector_error_model(
            decompose_errors=False,
            allow_gauge_detectors=True,
            approximate_disjoint_errors=True,
        )
        assert dem.num_errors > 0
        assert noisy.compile_detector_sampler().sample(shots=16).shape[0] == 16

    def test_transport_needs_a_code_distance(self):
        qm = qubits("171", "m")
        with pytest.raises(ValueError, match="code_distance"):
            YbNoiseModel(NoiseModelParameters()).noisy_circuit(
                Z_CHECK, qm, readout_protocol="transport"
            )

    def test_transport_scales_with_code_distance(self):
        """A larger code means a longer shuttle, so more accumulated noise."""
        model = YbNoiseModel(NoiseModelParameters())
        rates = []
        for d in (3, 11):
            noisy = model.noisy_circuit(
                Z_CHECK,
                qubits("171", "m"),
                readout_protocol="transport",
                code_distance=d,
            )
            rates.append(
                sum(
                    inst.gate_args_copy()[0]
                    for inst in noisy.flattened()
                    if inst.name == "HERALDED_ERASE"
                )
            )
        assert rates[1] > rates[0]


class TestBuilderApi:
    def test_operations_chain(self):
        qm = qubits("171", "g")
        c = YbCircuit(qm, NoiseModelParameters())
        (
            c.reset_qubit([0, 1, 2, 3, 4], pattern="b")
            .single_qubit_gate("H", [1, 3])
            .two_qubit_gate("CZ", [0, 1])
            .idling([4], 3e-4)
            .transport([0, 2, 4], 1e-4)
            .handover([1, 3])
            .shelve([0])
            .unshelve([0])
            .measurement([1, 3])
        )
        assert c.num_measurements > 0
        assert len(c.get_operation_log()) > 0

    def test_with_noise_replays_the_log(self):
        qm = qubits("171", "m")
        params = NoiseModelParameters()
        ideal = YbCircuit(qm, params, noise_enabled=False, track_operations=True)
        ideal.reset_qubit([0, 1]).single_qubit_gate("H", [1]).measurement([1])
        noisy = ideal.with_noise(params)
        assert len(noisy.flattened()) > len(ideal.flattened())

    def test_with_noise_needs_a_log(self):
        qm = qubits("171", "m")
        c = YbCircuit(qm, NoiseModelParameters(), track_operations=False)
        c.reset_qubit([0])
        with pytest.raises(ValueError, match="Operation log is empty"):
            c.with_noise()

    def test_shelving_unsupported_for_m(self):
        qm = qubits("171", "m")
        c = YbCircuit(qm, NoiseModelParameters())
        with pytest.raises(NotImplementedError, match="shelve"):
            c.shelve([0])

    def test_unsupported_gate(self):
        qm = qubits("171", "m")
        c = YbCircuit(qm, NoiseModelParameters())
        with pytest.raises(ValueError, match="Unsupported gate"):
            c.single_qubit_gate("T", [0])

    def test_reset_pattern_must_be_a_or_b(self):
        qm = qubits("171", "m")
        c = YbCircuit(qm, NoiseModelParameters())
        with pytest.raises(ValueError, match="Invalid pattern"):
            c.reset_qubit([0], pattern="z")


class TestFromStim:
    def test_rejects_unsupported_instructions(self):
        """Only CZ-based circuits are modelled; CX has no Rydberg counterpart."""
        qm = qubits("171", "m")
        with pytest.raises(ValueError, match="Unsupported Stim instruction"):
            YbNoiseModel(NoiseModelParameters()).noisy_circuit(
                stim.Circuit("R 0 1\nTICK\nCX 0 1\nTICK\nM 1"), qm
            )

    def test_empty_program_is_rejected(self):
        qm = qubits("171", "m")
        with pytest.raises(ValueError, match="no replayable operations"):
            YbNoiseModel(NoiseModelParameters()).noisy_circuit(stim.Circuit(), qm)

    def test_repeat_blocks_are_expanded(self):
        qm = qubits("171", "m")
        program = stim.Circuit("""
            R 0 1
            TICK
            REPEAT 3 {
                H 1
                TICK
                CZ 0 1
                TICK
            }
            M 1
            TICK
        """)
        noisy = YbNoiseModel(NoiseModelParameters()).noisy_circuit(program, qm)
        assert instruction_names(noisy)["CZ"] == 3

    def test_a_missing_final_tick_warns_that_idling_was_dropped(self):
        """The TICK is what buys the spectators their idling noise."""
        qm = qubits("171", "m")
        model = YbNoiseModel(NoiseModelParameters())
        without_tick = stim.Circuit("R 0 1 2 3 4\nTICK\nM 1")
        with_tick = stim.Circuit("R 0 1 2 3 4\nTICK\nM 1\nTICK")

        with pytest.warns(UserWarning, match="not terminated by a TICK"):
            bare = model.noisy_circuit(without_tick, qm)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ticked = model.noisy_circuit(with_tick, qm)

        assert sum(instruction_names(ticked).values()) > sum(
            instruction_names(bare).values()
        )
