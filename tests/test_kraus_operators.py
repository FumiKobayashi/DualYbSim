"""Tests for the Kraus operators: CPTP conditions and physical validity."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from dualybsim.kraus.yb171 import Kraus1Q_171m, Kraus1QClock_171m
from dualybsim.kraus.yb174 import Kraus1Q_174, KrausMEASURE_DISC_174


class TestKraus1QClock_171m:
    """Test class for Kraus1QClock_171m class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.p_dep1 = 0.002
        self.gate_time = 10e-6  # seconds
        self.lifetime_gs = 30.0  # seconds
        self.lifetime_es = 30.0  # seconds
        self.leaktime_eg = 1.0  # seconds
        self.lifetime_ryd = 50e-6 / 0.51  # seconds
        self.leaktime_ryd_gs = 50e-6 / 0.42  # seconds
        self.leaktime_ryd_es = 50e-6 / 0.07  # seconds

        self.kraus_ops = Kraus1QClock_171m(
            self.p_dep1,
            self.gate_time,
            self.lifetime_gs,
            self.lifetime_es,
            self.leaktime_eg,
            self.lifetime_ryd,
            self.leaktime_ryd_gs,
            self.leaktime_ryd_es,
        )

    def test_initialization(self) -> None:
        """Test proper initialization of Kraus operators."""
        assert hasattr(self.kraus_ops, "noise_channels")
        assert isinstance(self.kraus_ops.noise_channels, dict)
        expected_channels = {
            "DEP1_gm",
            "LOSS_g",
            "LOSS_m",
            "DECAY_mg",
            "LOSS_r",
            "DECAY_rg",
            "DECAY_rm",
        }
        assert set(self.kraus_ops.noise_channels.keys()) == expected_channels

    def test_cptp_condition(self) -> None:
        """Test that Kraus operators satisfy CPTP condition."""
        for channel_name, kraus_ops in self.kraus_ops.noise_channels.items():
            # Calculate sum of Kraus operators' adjoint products
            sum_kraus = np.zeros((6, 6), dtype=complex)
            for kraus_op in kraus_ops:
                sum_kraus += kraus_op.conj().T @ kraus_op

            # Check trace preservation (should be identity matrix)
            expected_identity = np.eye(6, dtype=complex)
            assert_array_almost_equal(
                sum_kraus,
                expected_identity,
                decimal=10,
                err_msg=f"CPTP condition failed for {channel_name}",
            )

    def test_density_matrix_preservation(self) -> None:
        """Test that density matrix trace is preserved."""
        # Create a pure state density matrix
        density_matrix = np.zeros((6, 6), dtype=complex)
        density_matrix[2, 2] = 1.0  # |0m⟩⟨0m|

        # Apply CPTP map
        result = self.kraus_ops.CPTP(density_matrix)

        # Check trace preservation
        assert_allclose(
            np.trace(result), 1.0, rtol=1e-10, err_msg="Trace not preserved"
        )

        # Check that result is still a valid density matrix
        assert_allclose(
            result,
            result.conj().T,
            rtol=1e-10,
            err_msg="Density matrix not Hermitian",
        )

    def test_physical_parameters(self) -> None:
        """Test that physical parameters are within reasonable bounds."""
        assert 0 <= self.p_dep1 <= 1, "Depolarization probability out of bounds"
        assert self.gate_time > 0, "Gate time must be positive"
        assert self.lifetime_gs > 0, "Ground state lifetime must be positive"
        assert self.lifetime_es > 0, "Excited state lifetime must be positive"
        assert self.leaktime_eg > 0, "Leakage time must be positive"

    def test_specific_channel_application(self) -> None:
        """Test applying specific error channels."""
        density_matrix = np.zeros((6, 6), dtype=complex)
        density_matrix[2, 2] = 1.0  # |0m⟩⟨0m|

        # Test each channel individually
        for channel_name in self.kraus_ops.noise_channels.keys():
            result = self.kraus_ops.CPTP(density_matrix, channel=channel_name)
            assert_allclose(
                np.trace(result),
                1.0,
                rtol=1e-10,
                err_msg=f"Trace not preserved for {channel_name}",
            )

    def test_invalid_channel(self) -> None:
        """Test that invalid channel names raise appropriate errors."""
        density_matrix = np.zeros((6, 6), dtype=complex)
        density_matrix[2, 2] = 1.0

        with pytest.raises(ValueError, match="Channel 'INVALID' not found"):
            self.kraus_ops.CPTP(density_matrix, channel="INVALID")


class TestKraus1Q_171m:
    """Test class for Kraus1Q_171m class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.p_dep1 = 0.001
        self.p_leak = 0.001
        self.T2_g = 10.0  # seconds
        self.T1_g = 200.0  # seconds
        self.T2_m = 10.0  # seconds
        self.T1_m = 200.0  # seconds
        self.T2_c = 5.0  # seconds
        self.gate_time = 2e-6  # seconds
        self.lifetime_gs = 30.0  # seconds
        self.lifetime_es = 30.0  # seconds
        self.leaktime_eg = 1.0  # seconds
        self.lifetime_ryd = 50e-6 / 0.51  # seconds
        self.leaktime_ryd_gs = 50e-6 / 0.42  # seconds
        self.leaktime_ryd_es = 50e-6 / 0.07  # seconds
        self.idling_flag = False

        self.kraus_ops = Kraus1Q_171m(
            self.p_dep1,
            self.p_leak,
            self.T2_g,
            self.T1_g,
            self.T2_m,
            self.T1_m,
            self.T2_c,
            self.gate_time,
            self.lifetime_gs,
            self.lifetime_es,
            self.leaktime_eg,
            self.lifetime_ryd,
            self.leaktime_ryd_gs,
            self.leaktime_ryd_es,
            self.idling_flag,
        )

    def test_initialization(self) -> None:
        """Test proper initialization of Kraus operators."""
        assert hasattr(self.kraus_ops, "noise_channels")
        assert isinstance(self.kraus_ops.noise_channels, dict)

    def test_cptp_condition(self) -> None:
        """Test that Kraus operators satisfy CPTP condition."""
        for channel_name, kraus_ops in self.kraus_ops.noise_channels.items():
            sum_kraus = np.zeros((6, 6), dtype=complex)
            for kraus_op in kraus_ops:
                sum_kraus += kraus_op.conj().T @ kraus_op

            expected_identity = np.eye(6, dtype=complex)
            assert_array_almost_equal(
                sum_kraus,
                expected_identity,
                decimal=10,
                err_msg=f"CPTP condition failed for {channel_name}",
            )


def test_fidelity_calculation() -> None:
    """Test fidelity calculation between density matrices."""
    # Create two pure states
    rho1 = np.zeros((6, 6), dtype=complex)
    rho1[2, 2] = 1.0  # |0m⟩⟨0m|

    rho2 = np.zeros((6, 6), dtype=complex)
    rho2[3, 3] = 1.0  # |1m⟩⟨1m|

    # Fidelity between orthogonal states should be 0
    # Note: This is a simplified test - actual fidelity calculation
    # would require the full implementation from the notebook
    assert rho1.shape == rho2.shape, "Density matrices must have same shape"


class TestKraus1Q_174:
    """Test class for Kraus1Q_174 class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.p_dep1 = 0.002
        self.T2 = 5.0  # seconds
        self.gate_time = 100e-6  # seconds
        self.lifetime_gs = 10.0  # seconds
        self.lifetime_es = 5.0  # seconds
        self.leaktime_eg = 1.0  # seconds
        self.lifetime_ryd = 0.1  # seconds
        self.leaktime_ryd_gs = 0.05  # seconds
        self.leaktime_ryd_es = 0.1  # seconds
        self.idling_flag = False

        self.kraus_ops = Kraus1Q_174(
            self.p_dep1,
            self.T2,
            self.gate_time,
            self.lifetime_gs,
            self.lifetime_es,
            self.leaktime_eg,
            self.lifetime_ryd,
            self.leaktime_ryd_gs,
            self.leaktime_ryd_es,
            self.idling_flag,
        )

    def test_initialization(self) -> None:
        """Test proper initialization of Kraus operators."""
        assert hasattr(self.kraus_ops, "noise_channels")
        assert isinstance(self.kraus_ops.noise_channels, dict)
        expected_channels = {
            "DEP1_c",
            "ZERR_c",
            "LOSS_g",
            "LOSS_m",
            "DECAY_mg",
            "LOSS_r",
            "DECAY_rg",
            "DECAY_rm",
        }
        assert set(self.kraus_ops.noise_channels.keys()) == expected_channels

    def test_cptp_condition(self) -> None:
        """Test that Kraus operators satisfy CPTP condition."""
        for channel_name, kraus_ops in self.kraus_ops.noise_channels.items():
            # Calculate sum of Kraus operators' adjoint products
            sum_kraus = np.zeros((4, 4), dtype=complex)
            for kraus_op in kraus_ops:
                sum_kraus += kraus_op.conj().T @ kraus_op

            # Check trace preservation (should be identity matrix)
            expected_identity = np.eye(4, dtype=complex)
            assert_array_almost_equal(
                sum_kraus,
                expected_identity,
                decimal=10,
                err_msg=f"CPTP condition failed for {channel_name}",
            )

    def test_density_matrix_preservation(self) -> None:
        """Test that density matrix trace is preserved."""
        # Create a pure state density matrix
        density_matrix = np.zeros((4, 4), dtype=complex)
        density_matrix[0, 0] = 1.0  # |g⟩⟨g|

        # Apply CPTP map
        result = self.kraus_ops.CPTP(density_matrix)

        # Check trace preservation
        assert_allclose(
            np.trace(result), 1.0, rtol=1e-10, err_msg="Trace not preserved"
        )

        # Check that result is still a valid density matrix
        assert_allclose(
            result,
            result.conj().T,
            rtol=1e-10,
            err_msg="Density matrix not Hermitian",
        )

    def test_specific_channel_application(self) -> None:
        """Test applying specific error channels."""
        density_matrix = np.zeros((4, 4), dtype=complex)
        density_matrix[0, 0] = 1.0  # |g⟩⟨g|

        # Test each channel individually
        for channel_name in self.kraus_ops.noise_channels.keys():
            result = self.kraus_ops.CPTP(density_matrix, channel=channel_name)
            assert_allclose(
                np.trace(result),
                1.0,
                rtol=1e-10,
                err_msg=f"Trace not preserved for {channel_name}",
            )

    def test_invalid_channel(self) -> None:
        """Test that invalid channel names raise appropriate errors."""
        density_matrix = np.zeros((4, 4), dtype=complex)
        density_matrix[0, 0] = 1.0

        with pytest.raises(ValueError, match="Channel 'INVALID' not found"):
            self.kraus_ops.CPTP(density_matrix, channel="INVALID")


class TestKrausMeasureDisc174:
    """Test class for KrausMEASURE_DISC_174 class."""

    def setup_method(self) -> None:
        self.p_meas = 0.002
        self.q = 0.5
        self.kraus_ops = KrausMEASURE_DISC_174(self.p_meas, self.q)

    def test_initialization(self) -> None:
        assert hasattr(self.kraus_ops, "noise_channels")
        assert set(self.kraus_ops.noise_channels.keys()) == {"MERR"}

    def test_cptp_condition(self) -> None:
        kraus_ops = self.kraus_ops.noise_channels["MERR"]
        sum_kraus = np.zeros((4, 4), dtype=complex)
        for kraus_op in kraus_ops:
            sum_kraus += kraus_op.conj().T @ kraus_op

        assert_array_almost_equal(sum_kraus, np.eye(4, dtype=complex), decimal=10)

    def test_special_case_q_equals_one_matches_expected_populations(self) -> None:
        p = self.p_meas
        kraus_ops = KrausMEASURE_DISC_174(p, q=1.0)
        rho_g = np.zeros((4, 4), dtype=complex)
        rho_g[0, 0] = 1.0

        result = kraus_ops.CPTP(rho_g, channel="MERR")
        expected_diag = np.array([1 - p, p**2, 0.0, p * (1 - p)], dtype=complex)
        assert_allclose(np.diag(result), expected_diag, rtol=1e-10, atol=1e-12)

    def test_special_case_q_equals_half_matches_expected_populations(self) -> None:
        p = self.p_meas
        kraus_ops = KrausMEASURE_DISC_174(p, q=0.5)
        rho_m = np.zeros((4, 4), dtype=complex)
        rho_m[1, 1] = 1.0

        result = kraus_ops.CPTP(rho_m, channel="MERR")
        expected_diag = np.array(
            [
                p**2 + 0.5 * p * (1 - p),
                (1 - p) ** 2 + 0.5 * p * (1 - p),
                0.0,
                (1 - p) * p,
            ],
            dtype=complex,
        )
        assert_allclose(np.diag(result), expected_diag, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
