"""Tests for Generalized Twirling Approximation."""

import numpy as np
import pytest

from dualybsim.kraus import YbNoiseChannelFactory
from dualybsim.twirling import GeneralizedTwirlingApproximation


class TestGeneralizedTwirlingApproximation:
    """Test cases for Generalized Twirling Approximation."""

    def test_initialization(self) -> None:
        """Test GTA initialization."""
        # Create a noise channel
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        gta = GeneralizedTwirlingApproximation(noise_channel)

        assert gta.noise_channel == noise_channel
        assert len(gta.channels) > 0
        assert "I" in gta.pauli_operators
        assert "X" in gta.pauli_operators
        assert "Y" in gta.pauli_operators
        assert "Z" in gta.pauli_operators

    def test_cptp_validation(self) -> None:
        """Test CPTP validation."""
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        gta = GeneralizedTwirlingApproximation(noise_channel)

        # CPTP conditions should be satisfied
        assert gta.validate_cptp()

    def test_kraus_decomposition(self) -> None:
        """Test Kraus operator decomposition."""
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        gta = GeneralizedTwirlingApproximation(noise_channel)

        blocks = gta.decompose_kraus_operators()

        # Should have some blocks
        for channel_name, channel_blocks in blocks.items():
            assert channel_name in noise_channel.get_available_channels()
            assert len(channel_blocks) > 0

            # Check block structure
            for key, block_list in channel_blocks.items():
                assert "->" in key  # Format: "initial->final"
                assert len(block_list) > 0

                for block in block_list:
                    assert isinstance(block, np.ndarray)
                    assert block.shape[0] == block.shape[1]  # Square matrix

    def test_kraus_decomposition_concatenated(self) -> None:
        """Test Kraus operator decomposition for concatenated channel."""
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        noise_channel.concatenate_channels()
        gta = GeneralizedTwirlingApproximation(noise_channel, subchannel="concatenated")

        blocks = gta.decompose_kraus_operators()

        # Should have some blocks
        assert len(blocks) > 0

        # Check block structure
        for key, block_list in blocks.items():
            assert "->" in key  # Format: "initial->final"
            assert len(block_list) > 0

            for block in block_list:
                assert isinstance(block, np.ndarray)
                assert block.shape[0] == block.shape[1]  # Square matrix

    def test_pauli_decomposition_174Yb(self) -> None:
        """Test Pauli decomposition for 174Yb."""
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        gta = GeneralizedTwirlingApproximation(noise_channel)

        # Create a test computational block (4x4 for 174Yb)
        test_block = np.eye(4) * 0.25  # Identity matrix

        coeffs = gta.compute_pauli_decomposition(test_block)

        # Should have coefficients for all Pauli operators
        assert "I" in coeffs
        assert "X" in coeffs
        assert "Y" in coeffs
        assert "Z" in coeffs

        # Coefficients should be real
        for coeff in coeffs.values():
            assert np.isreal(coeff)

    def test_pauli_decomposition_171Yb(self) -> None:
        """Test Pauli decomposition for 171Yb."""
        noise_channel = YbNoiseChannelFactory.create_171Yb_1Q_channel()
        gta = GeneralizedTwirlingApproximation(noise_channel)

        # Create a test computational block (6x6 for 171Yb)
        test_block = np.eye(6) * 0.16666666666666666  # Identity matrix

        coeffs = gta.compute_pauli_decomposition(test_block)

        # Should have coefficients for single qubit Pauli operators
        assert "I" in coeffs
        assert "X" in coeffs
        assert "Y" in coeffs
        assert "Z" in coeffs

        # Coefficients should be real
        for coeff in coeffs.values():
            assert np.isreal(coeff)

    def test_concatenate_channels(self) -> None:
        """Test concatenate channels."""
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        noise_channel.concatenate_channels()
        assert "concatenated" in noise_channel.get_available_channels()
        available_channels = noise_channel.get_available_channels()
        num_ops = []
        for channel in available_channels:
            if channel == "concatenated":
                continue
            num_ops.append(len(noise_channel.get_kraus_operators(channel)))
        assert len(noise_channel.get_kraus_operators("concatenated")) == np.prod(
            num_ops
        )

    def test_conditional_probabilities(self) -> None:
        """Test conditional probability derivation."""
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        # concatenate channels before applying GTA
        noise_channel.concatenate_channels()
        gta = GeneralizedTwirlingApproximation(noise_channel, "concatenated")

        probs = gta.derive_conditional_probabilities()

        # Should have some probabilities
        assert len(probs) > 0

        # Check probability format
        for key, prob in probs.items():
            assert "->" in key
            assert ":" in key  # Should have Pauli error specification
            assert 0 <= prob <= 1

    def test_174_measure_disc_q_sweep_probabilities(self) -> None:
        """Test GTA outputs for 174Yb measurement discrimination across q values."""
        for q in (0.0, 0.5, 1.0):
            noise_channel = YbNoiseChannelFactory.create_174Yb_MEASURE_DISC_channel(
                p_meas=0.002, q=q
            )
            gta = GeneralizedTwirlingApproximation(noise_channel, subchannel="MERR")

            assert gta.validate_cptp()

            probs = gta.derive_conditional_probabilities()
            c_to_c_total = sum(probs.get(f"c->c:{pauli}", 0.0) for pauli in "IXYZ")

            assert np.isclose(c_to_c_total, 1.0, atol=1e-8)
            assert 0.0 <= probs.get("c->L:I", 0.0) <= 1.0

    def test_174Yb_conversion(self) -> None:
        """Test 174Yb computational to Yb density matrix conversion."""
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()

        # Test computational density matrix
        computational_rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|

        # Convert to Yb density matrix
        yb_rho = noise_channel.convert_computational_to_yb_density_matrix(
            computational_rho
        )

        # Check dimensions (4x4 for 174Yb)
        assert yb_rho.shape == (4, 4)

        # Check that computational states are mapped correctly
        assert np.isclose(yb_rho[0, 0], 1.0)  # |g⟩⟨g| population
        assert np.isclose(yb_rho[1, 1], 0.0)  # |e⟩⟨e| population
        assert np.isclose(yb_rho[2, 2], 0.0)  # |r⟩⟨r| population
        assert np.isclose(yb_rho[3, 3], 0.0)  # |L⟩⟨L| population

        # Check trace preservation
        assert np.isclose(np.trace(yb_rho), 1.0)

    def test_171Yb_ground_conversion(self) -> None:
        """Test 171Yb ground state qubit conversion."""
        noise_channel = YbNoiseChannelFactory.create_171Yb_1Q_channel()

        # Test computational density matrix
        computational_rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|

        # Convert to Yb density matrix (ground state qubit)
        yb_rho = noise_channel.convert_computational_to_yb_density_matrix(
            computational_rho, qubit_type="ground"
        )

        # Check dimensions (6x6 for 171Yb)
        assert yb_rho.shape == (6, 6)

        # Check that computational states are mapped to ground state subspace
        assert np.isclose(yb_rho[0, 0], 1.0)  # |g0⟩⟨g0| population
        assert np.isclose(yb_rho[1, 1], 0.0)  # |g1⟩⟨g1| population
        assert np.isclose(yb_rho[2, 2], 0.0)  # |m0⟩⟨m0| population
        assert np.isclose(yb_rho[3, 3], 0.0)  # |m1⟩⟨m1| population
        assert np.isclose(yb_rho[4, 4], 0.0)  # |r⟩⟨r| population
        assert np.isclose(yb_rho[5, 5], 0.0)  # |L⟩⟨L| population

        # Check trace preservation
        assert np.isclose(np.trace(yb_rho), 1.0)

    def test_171Yb_metastable_conversion(self) -> None:
        """Test 171Yb metastable state qubit conversion."""
        noise_channel = YbNoiseChannelFactory.create_171Yb_1Q_channel()

        # Test computational density matrix
        computational_rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|

        # Convert to Yb density matrix (metastable state qubit)
        yb_rho = noise_channel.convert_computational_to_yb_density_matrix(
            computational_rho, qubit_type="metastable"
        )

        # Check dimensions (6x6 for 171Yb)
        assert yb_rho.shape == (6, 6)

        # Check that computational states are mapped to metastable state subspace
        assert np.isclose(yb_rho[0, 0], 0.0)  # |g0⟩⟨g0| population
        assert np.isclose(yb_rho[1, 1], 0.0)  # |g1⟩⟨g1| population
        assert np.isclose(yb_rho[2, 2], 1.0)  # |m0⟩⟨m0| population
        assert np.isclose(yb_rho[3, 3], 0.0)  # |m1⟩⟨m1| population
        assert np.isclose(yb_rho[4, 4], 0.0)  # |r⟩⟨r| population
        assert np.isclose(yb_rho[5, 5], 0.0)  # |L⟩⟨L| population

        # Check trace preservation
        assert np.isclose(np.trace(yb_rho), 1.0)

    def test_171Yb_invalid_qubit_type(self) -> None:
        """Test that invalid qubit_type raises appropriate error for 171Yb."""
        noise_channel = YbNoiseChannelFactory.create_171Yb_1Q_channel()

        computational_rho = np.array([[1, 0], [0, 0]], dtype=complex)

        with pytest.raises(ValueError, match="Invalid qubit_type"):
            noise_channel.convert_computational_to_yb_density_matrix(
                computational_rho, qubit_type="invalid"
            )

    def test_apply_to_computational_state(self) -> None:
        """Test apply_to_computational_state method."""
        # Test 174Yb
        noise_174 = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        computational_rho = np.array([[1, 0], [0, 0]], dtype=complex)

        result_174 = noise_174.apply_to_computational_state(computational_rho)
        assert result_174.shape == (4, 4)
        assert np.isclose(np.trace(result_174), 1.0)

        # Test 171Yb ground state qubit
        noise_171 = YbNoiseChannelFactory.create_171Yb_1Q_channel()
        result_171_ground = noise_171.apply_to_computational_state(
            computational_rho, qubit_type="ground"
        )
        assert result_171_ground.shape == (6, 6)
        assert np.isclose(np.trace(result_171_ground), 1.0)

        # Test 171Yb metastable state qubit
        result_171_metastable = noise_171.apply_to_computational_state(
            computational_rho, qubit_type="metastable"
        )
        assert result_171_metastable.shape == (6, 6)
        assert np.isclose(np.trace(result_171_metastable), 1.0)

    def test_kraus_operator_normalization(self) -> None:
        """Test that Kraus operators are properly normalized to satisfy CPTP."""
        # Test 174Yb
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        gta = GeneralizedTwirlingApproximation(noise_channel)

        # Check CPTP condition: sum_i K_i† K_i = I
        for channel in gta.channels.values():
            dim = channel[0].shape[0]
            if channel[0].ndim == 3:
                dim = channel[0].shape[1]
            completeness = np.zeros((dim, dim), dtype=complex)
            for K in channel:
                completeness += K.conj().T @ K

            trace_comp = np.trace(completeness).real
            assert np.isclose(trace_comp, dim, atol=1e-8), (
                f"CPTP condition violated: Tr(sum K†K) = {trace_comp}, expected {dim}"
            )

        # Test 171Yb
        noise_channel = YbNoiseChannelFactory.create_171Yb_1Q_channel()
        gta = GeneralizedTwirlingApproximation(noise_channel)

        for channel in gta.channels.values():
            dim = channel[0].shape[0]
            completeness = np.zeros((dim, dim), dtype=complex)
            for K in channel:
                completeness += K.conj().T @ K

            trace_comp = np.trace(completeness).real
            assert np.isclose(trace_comp, dim, atol=1e-8), (
                f"CPTP condition violated: Tr(sum K†K) = {trace_comp}, expected {dim}"
            )

    def test_transition_probability_normalization(self) -> None:
        """Test that transition probabilities are properly normalized."""
        # Test 174Yb
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        noise_channel.concatenate_channels()
        gta = GeneralizedTwirlingApproximation(noise_channel, subchannel="concatenated")

        probs = gta.derive_conditional_probabilities()

        # Check that computational subspace transitions sum to 1.0
        c_probs = {k: v for k, v in probs.items() if not k.startswith("c->L")}
        total_c = sum(c_probs.values())

        assert np.isclose(total_c, 1.0, atol=1e-8), (
            f"Computational subspace probabilities don't sum to 1.0: {total_c}"
        )

    def test_pauli_decomposition_normalization(self) -> None:
        """Test that Pauli decomposition uses correct normalization factor."""
        # Test 174Yb
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel()
        gta = GeneralizedTwirlingApproximation(noise_channel, subchannel="concatenated")

        # Create a simple test block (4x4 for 174Yb, but only computational part matters)
        test_block = np.eye(4) * 0.25  # 4x4 identity matrix

        coeffs = gta.compute_pauli_decomposition(test_block)

        # For 2x2 computational subspace, Pauli decomposition should be: M = (1/2) * sum_μ Tr(σ_μ M) σ_μ
        # So the coefficients should reconstruct the computational part
        comp_block = test_block[:2, :2]  # Extract 2x2 computational part
        reconstructed = np.zeros((2, 2), dtype=complex)
        for pauli_name, coeff in coeffs.items():
            pauli_op = gta.pauli_operators[pauli_name]
            reconstructed += coeff * pauli_op

        # Check that reconstruction is close to computational part
        assert np.allclose(reconstructed, comp_block, atol=1e-10), (
            "Pauli decomposition reconstruction failed"
        )

        # Check that coefficients sum to reasonable values
        total_weight = sum(abs(c) ** 2 for c in coeffs.values())
        # For 4x4 matrix with 0.25 on diagonal, computational part is 2x2 with 0.25
        # Pauli decomposition gives: I coefficient = 0.25, others = 0
        # Total weight = 0.25^2 = 0.0625
        expected_weight = 0.0625
        assert np.isclose(total_weight, expected_weight, atol=1e-10), (
            f"Total Pauli weight should be {expected_weight}, got {total_weight}"
        )

    def test_depolarizing_channel_accuracy(self) -> None:
        """Test that depolarizing channel gives accurate error rates."""
        # Create a channel with minimal other errors (mostly depolarization)
        p_dep = 0.01
        noise_channel = YbNoiseChannelFactory.create_174Yb_1Q_channel(
            p_dep1=p_dep,
            T2=1e10,  # Very large to minimize phase errors
            gate_time=10e-6,
            lifetime_gs=1e10,  # Very large to minimize leakage
            lifetime_es=1e10,
            leaktime_eg=1e10,
            lifetime_ryd=1e10,
            leaktime_ryd_gs=1e10,
            leaktime_ryd_es=1e10,
        )

        gta = GeneralizedTwirlingApproximation(noise_channel, subchannel="concatenated")
        probs = gta.derive_conditional_probabilities()

        # Get computational subspace transitions
        c_probs = {k: v for k, v in probs.items() if k.startswith("c->c")}

        # Check that identity probability is close to expected
        identity_prob = c_probs.get("c->c:I", 0)
        expected_identity = 1 - p_dep

        assert np.isclose(identity_prob, expected_identity, atol=0.01), (
            f"Identity probability {identity_prob} doesn't match expected {expected_identity}"
        )

        # Check that total Pauli error is reasonable
        pauli_x = c_probs.get("c->c:X", 0)
        pauli_y = c_probs.get("c->c:Y", 0)
        pauli_z = c_probs.get("c->c:Z", 0)
        total_pauli = pauli_x + pauli_y + pauli_z

        # Total Pauli error should be approximately p_dep
        assert np.isclose(total_pauli, p_dep, atol=0.01), (
            f"Total Pauli error {total_pauli} doesn't match expected {p_dep}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
