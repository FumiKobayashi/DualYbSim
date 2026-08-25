"""Yb noise channel implementation.

This module provides noise channel implementations for Yb quantum devices,
wrapping the Kraus operator classes in :mod:`dualybsim.kraus.yb171` and
:mod:`dualybsim.kraus.yb174`.

Key Concepts:
- MEASURE_DISC: Models measurement discrimination errors (reading |0⟩ as |1⟩, etc.)
- MEASURE: Models physical errors during fluorescence imaging readout
  (atomic loss, leakage, etc.)
- For complete measurement simulation, typically apply both channels in sequence
"""

import logging
from typing import Any

import numpy as np

from .yb171 import (
    Kraus1Q_171m,
    Kraus1QClock_171m,
    Kraus2Q_171m171m,
    KrausMEASURE_171m,
    KrausMEASURE_DISC_171m,
    KrausRESET_171m,
)
from .yb174 import (
    Kraus1Q_174,
    Kraus2Q_174174,
    KrausMEASURE_174,
    KrausMEASURE_DISC_174,
    KrausRESET_174,
)

logger = logging.getLogger(__name__)

# The eleven Kraus classes share no base class, so the union is the only
# honest type for whichever one `YbNoiseChannel` selects.
KrausModel = (
    Kraus1Q_174
    | Kraus2Q_174174
    | KrausRESET_174
    | KrausMEASURE_174
    | KrausMEASURE_DISC_174
    | Kraus1Q_171m
    | Kraus1QClock_171m
    | Kraus2Q_171m171m
    | KrausRESET_171m
    | KrausMEASURE_171m
    | KrausMEASURE_DISC_171m
)


class YbNoiseChannel:
    """Ybデバイスのノイズチャンネル

    This class provides a unified interface for different types of noise channels:

    Gate Operations:
    - 1Q: Single-qubit gate errors (depolarization, leakage, etc.)
    - 2Q: Two-qubit gate errors (depolarization, cross-talk, leakage, etc.)

    Measurement Operations:
    - MEASURE_DISC: Measurement discrimination errors (reading wrong state)
    - MEASURE: Physical errors during fluorescence imaging (atomic loss, leakage)

    State Preparation:
    - RESET: Errors during state preparation/reset operations
    """

    def __init__(
        self,
        atom_species: str = "174Yb",
        gate_type: str = "1Q",
        qubit_type: str = "ground",
        **kwargs: Any,
    ) -> None:
        """Initialize Yb noise channel.

        Args:
            atom_species: "174Yb" or "171Yb"
            gate_type: "1Q", "2Q", "RESET", "MEASURE", or "MEASURE_DISC"
            qubit_type: For 171Yb, "ground" or "metastable". For 174Yb, ignored.
            **kwargs: Parameters for the specific Kraus operator class
        """
        self.atom_species = atom_species
        self.gate_type = gate_type
        self.qubit_type = qubit_type
        self.kwargs = kwargs
        self.kraus_class: KrausModel

        # Initialize the appropriate noise channel provider class
        self._initialize_kraus_operators()

    def _initialize_kraus_operators(self) -> None:
        """Initialize noise channels based on atom species and gate type.

        MEASURE_DISC vs MEASURE distinction:
        - MEASURE_DISC: Uses KrausMEASURE_DISC_* classes that model measurement
          discrimination errors (e.g., reading |g⟩ as |m⟩, |l⟩, or |r⟩)
        - MEASURE: Uses KrausMEASURE_* classes that model physical errors during
          fluorescence imaging (atomic loss, leakage between states, etc.)
        """
        if self.atom_species == "174Yb":
            if self.gate_type == "1Q":
                self.kraus_class = Kraus1Q_174(**self.kwargs)
            elif self.gate_type == "2Q":
                self.kraus_class = Kraus2Q_174174(**self.kwargs)
            elif self.gate_type == "RESET":
                self.kraus_class = KrausRESET_174(**self.kwargs)
            elif self.gate_type == "MEASURE":
                self.kraus_class = KrausMEASURE_174(**self.kwargs)
            elif self.gate_type == "MEASURE_DISC":
                self.kraus_class = KrausMEASURE_DISC_174(**self.kwargs)
            else:
                raise ValueError(f"Unknown gate type: {self.gate_type}")
        elif self.atom_species == "171Yb":
            if self.gate_type == "1Q":
                self.kraus_class = Kraus1Q_171m(**self.kwargs)
            elif self.gate_type == "1Q_CLOCK":
                # Clock excitation channel acting on m-qubit (DEP1_gm etc.)
                self.kraus_class = Kraus1QClock_171m(**self.kwargs)
            elif self.gate_type == "2Q":
                self.kraus_class = Kraus2Q_171m171m(**self.kwargs)
            elif self.gate_type == "RESET":
                self.kraus_class = KrausRESET_171m(**self.kwargs)
            elif self.gate_type == "MEASURE":
                self.kraus_class = KrausMEASURE_171m(**self.kwargs)
            elif self.gate_type == "MEASURE_DISC":
                self.kraus_class = KrausMEASURE_DISC_171m(**self.kwargs)
            else:
                raise ValueError(f"Unknown gate type: {self.gate_type}")
        else:
            raise ValueError(f"Unknown atom species: {self.atom_species}")

    def get_kraus_operators(
        self, channel: str | None = None
    ) -> dict[str, list[np.ndarray]] | list[np.ndarray]:
        """Get Kraus operators.

        Args:
            channel: Specific channel name (e.g., 'DEP1_m', 'LOSS_g', 'MERR', etc.).
                     If None, returns a mapping of channel name to its Kraus operators.

        Returns:
            - If channel is None: Dict[str, List[np.ndarray]] mapping channel name to list of 2D Kraus operators
            - If channel is provided: List[np.ndarray] of 2D Kraus operators for that channel
        """
        if channel is None:
            # Return channel-wise mapping of Kraus operators
            per_channel: dict[str, list[np.ndarray]] = {}

            noise_channel_dict = self.kraus_class.noise_channels

            for name, kraus_ops in noise_channel_dict.items():
                ops_list: list[np.ndarray] = []
                if isinstance(kraus_ops, np.ndarray):
                    if kraus_ops.ndim == 3:
                        # 3D tensor: collect each operator
                        for i in range(kraus_ops.shape[0]):
                            ops_list.append(kraus_ops[i])
                    else:
                        # Single operator
                        ops_list.append(kraus_ops)
                else:
                    # Already a list/iterable of operators
                    ops_list.extend(list(kraus_ops))

                per_channel[name] = ops_list

            return per_channel
        elif channel == "concatenated":
            if "concatenated" in self.kraus_class.noise_channels:
                return self.kraus_class.noise_channels["concatenated"]  # type: ignore[return-value]
            else:
                return self.concatenate_channels()
        else:
            # Return specific channel operators
            noise_channel_dict = self.kraus_class.noise_channels
            if channel in noise_channel_dict:
                kraus_ops = noise_channel_dict[channel]
                if isinstance(kraus_ops, np.ndarray):
                    if kraus_ops.ndim == 3:
                        return [kraus_ops[i] for i in range(kraus_ops.shape[0])]
                    else:
                        return [kraus_ops]
                else:
                    return list(kraus_ops)
            else:
                raise ValueError(f"Channel '{channel}' not found")

    def concatenate_channels(
        self, channel_sequence: list[str] | None = None
    ) -> list[np.ndarray]:
        """Concatenate (compose) multiple subchannels into a single Kraus set.

        If channels C1, C2, ..., Cm are applied in this order, the composed
        channel has Kraus operators { K = C_m,β_m ... C_2,β_2 C_1,β_1 } over all
        combinations of indices.

        Args:
            channel_sequence: Optional explicit sequence of subchannel names.
                              If None, uses the insertion order of available
                              noise channels.

        Returns:
            List of 2D Kraus operator matrices for the composed channel.
        """
        # Resolve channel order
        available = self.get_available_channels()
        if channel_sequence is None:
            sequence = available
        else:
            # Validate provided names and preserve given order
            unknown = [c for c in channel_sequence if c not in available]
            if unknown:
                raise ValueError(f"Unknown channels in sequence: {unknown}")
            sequence = channel_sequence

        # Helper to flatten a channel's operators to a list of 2D arrays
        def _flatten_ops(obj: Any) -> list[np.ndarray]:
            if isinstance(obj, np.ndarray):
                if obj.ndim == 3:
                    return [obj[i] for i in range(obj.shape[0])]
                elif obj.ndim == 2:
                    return [obj]
                else:
                    raise ValueError(f"Unsupported operator array with ndim={obj.ndim}")
            # Iterable/list-like
            return list(obj)

        # Gather per-channel lists of 2D ops in the specified order
        per_channel_ops: list[list[np.ndarray]] = []
        for name in sequence:
            ops_obj = self.kraus_class.noise_channels[name]
            ops_list = _flatten_ops(ops_obj)
            if not ops_list:
                # Skip empty channel
                continue
            per_channel_ops.append(ops_list)

        if not per_channel_ops:
            return []

        # Validate consistent dimensions
        dim = per_channel_ops[0][0].shape[0]
        for ops_list in per_channel_ops:
            for K in ops_list:
                if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] != dim:
                    raise ValueError(
                        "Inconsistent Kraus operator dimensions across channels"
                    )

        # Compose: start from first channel's ops, then left-multiply subsequent channels
        composed: list[np.ndarray] = per_channel_ops[0]
        for ops_list in per_channel_ops[1:]:
            new_list: list[np.ndarray] = []
            for K_next in ops_list:
                for K_prev in composed:
                    # Later channel applied after earlier: left multiplication
                    new_list.append(K_next @ K_prev)
            composed = new_list
        self.kraus_class.noise_channels["concatenated"] = composed  # type: ignore[assignment]
        return composed

    def apply(self, state: np.ndarray, channel: str | None = None) -> np.ndarray:
        """Apply noise channel to a density matrix.

        Args:
            state: Input density matrix
            channel: Specific channel to apply (if None, applies all channels)

        Returns:
            Output density matrix after noise application
        """
        return self.kraus_class.CPTP(state, channel)

    def convert_computational_to_yb_density_matrix(
        self, computational_rho: np.ndarray, qubit_type: str = "ground"
    ) -> np.ndarray:
        """Convert 2x2 computational density matrix to Yb density matrix.

        Args:
            computational_rho: 2x2 density matrix in computational basis
            qubit_type: For 171Yb, "ground" or "metastable". For 174Yb, ignored.

        Returns:
            Density matrix for Yb system (4x4 for 174Yb, 6x6 for 171Yb)
        """
        if computational_rho.shape != (2, 2):
            raise ValueError(
                f"Expected 2x2 matrix, got shape {computational_rho.shape}"
            )

        # Ensure the input matrix is complex
        computational_rho = computational_rho.astype(complex)

        if self.atom_species == "174Yb":
            # 174Yb: 4 basis states |g⟩, |e⟩, |r⟩, |L⟩
            # Basis: |0⟩ = |g⟩, |1⟩ = |e⟩, |2⟩ = |r⟩, |3⟩ = |L⟩
            yb_rho = np.zeros((4, 4), dtype=complex)

            # Map computational states to Yb states:
            # |0⟩ → |g⟩ (ground state)
            # |1⟩ → |e⟩ (excited state)
            # |r⟩ and |L⟩ are initially unpopulated

            # Copy computational matrix to ground-excited subspace
            yb_rho[0:2, 0:2] = computational_rho

        elif self.atom_species == "171Yb":
            # 171Yb: 6 basis states |g0⟩, |g1⟩, |m0⟩, |m1⟩, |r⟩, |L⟩
            # Basis: |0⟩ = |g0⟩, |1⟩ = |g1⟩, |2⟩ = |m0⟩, |3⟩ = |m1⟩,
            # |4⟩ = |r⟩, |5⟩ = |L⟩
            yb_rho = np.zeros((6, 6), dtype=complex)

            if qubit_type == "ground":
                # Ground state qubit: |0⟩ → |g0⟩, |1⟩ → |g1⟩
                # Map computational states to ground state subspace
                yb_rho[0:2, 0:2] = computational_rho
                # |m0⟩, |m1⟩, |r⟩, |L⟩ are initially unpopulated

            elif qubit_type == "metastable":
                # Metastable state qubit: |0⟩ → |m0⟩, |1⟩ → |m1⟩
                # Map computational states to metastable state subspace
                yb_rho[2:4, 2:4] = computational_rho
                # |g0⟩, |g1⟩, |r⟩, |L⟩ are initially unpopulated

            else:
                raise ValueError(
                    f"Invalid qubit_type '{qubit_type}' for 171Yb. "
                    "Must be 'ground' or 'metastable'"
                )
        else:
            raise ValueError(f"Unknown atom species: {self.atom_species}")

        return yb_rho

    def apply_to_computational_state(
        self,
        computational_rho: np.ndarray,
        channel: str | None = None,
        qubit_type: str = "ground",
    ) -> np.ndarray:
        """Apply noise channel to a computational density matrix.

        Args:
            computational_rho: 2x2 density matrix in computational basis
            channel: Specific channel to apply (if None, applies all channels)
            qubit_type: For 171Yb, "ground" or "metastable". For 174Yb, ignored.

        Returns:
            Output density matrix after noise application
        """
        # Convert to Yb density matrix
        yb_rho = self.convert_computational_to_yb_density_matrix(
            computational_rho, qubit_type
        )

        # Apply noise channel
        noisy_yb_rho = self.apply(yb_rho, channel)

        return noisy_yb_rho

    def get_available_channels(self) -> list[str]:
        """Get list of available noise channels.

        Returns:
            List of channel names
        """
        return list(self.kraus_class.noise_channels.keys())

    def __repr__(self) -> str:
        return (
            f"YbNoiseChannel(atom_species='{self.atom_species}', "
            f"gate_type='{self.gate_type}')"
        )


class YbNoiseChannelFactory:
    """Factory class for creating Yb noise channels with typical parameters.

    Measurement Channel Types:

    1. MEASURE_DISC (Discrimination Error):
       - Models errors in distinguishing between quantum states during measurement
       - Examples: reading |g⟩ as |m⟩, |l⟩, or |r⟩ in 174Yb
       - Parameter: p_meas (measurement error probability)
       - Use case: Evaluating measurement apparatus performance

    2. MEASURE (Physical Readout Error):
       - Models physical errors during fluorescence imaging
       - Examples: atomic loss, leakage between states, trap loss
       - Parameters: lifetimes, gate time, loss probabilities
       - Use case: Modeling realistic measurement operations

    For complete measurement simulation, typically apply both channels in sequence:
    1. MEASURE_DISC (discrimination errors)
    2. MEASURE (physical errors during readout)
    """

    @staticmethod
    def create_174Yb_1Q_channel(
        p_dep1: float = 0.2e-2,
        T2: float = 5.0,  # sec, clock coherence
        gate_time: float = 100e-6,  # 100 microseconds
        lifetime_gs: float = 30.0,  # second, trap lifetime of ground
        lifetime_es: float = 30.0,  # second, trap lifetime of metastable
        leaktime_eg: float = 1.0,  # second, leakage time from m to g
        lifetime_ryd: float = 50e-6 / 0.51,  # second, rydberg radiative decay to L
        leaktime_ryd_gs: float = 50e-6 / 0.42,  # second, rydberg radiative decay to g
        leaktime_ryd_es: float = 50e-6 / 0.07,  # second, rydberg radiative decay to m
        idling_flag: bool = False,
    ) -> YbNoiseChannel:
        """Create a 174Yb 1Q gate noise channel with typical parameters."""
        return YbNoiseChannel(
            atom_species="174Yb",
            gate_type="1Q",
            p_dep1=p_dep1,
            T2=T2,
            gate_time=gate_time,
            lifetime_gs=lifetime_gs,
            lifetime_es=lifetime_es,
            leaktime_eg=leaktime_eg,
            lifetime_ryd=lifetime_ryd,
            leaktime_ryd_gs=leaktime_ryd_gs,
            leaktime_ryd_es=leaktime_ryd_es,
            idling_flag=idling_flag,
        )

    @staticmethod
    def create_174Yb_2Q_channel(
        p_dep2: float = 0.2e-2,
        T2: float = 5.0,
        gate_time: float = 300e-9,
        lifetime_gs: float = 30.0,
        lifetime_es: float = 30.0,
        leaktime_eg: float = 1.0,
        lifetime_ryd: float = 50e-6 / 0.51,
        leaktime_ryd_gs: float = 50e-6 / 0.42,
        leaktime_ryd_es: float = 50e-6 / 0.07,
        idling_flag: bool = False,
    ) -> YbNoiseChannel:
        """Create a 174Yb 2Q gate noise channel with typical parameters."""
        return YbNoiseChannel(
            atom_species="174Yb",
            gate_type="2Q",
            p_dep2=p_dep2,
            T2=T2,
            gate_time=gate_time,
            lifetime_gs=lifetime_gs,
            lifetime_es=lifetime_es,
            leaktime_eg=leaktime_eg,
            lifetime_ryd=lifetime_ryd,
            leaktime_ryd_gs=leaktime_ryd_gs,
            leaktime_ryd_es=leaktime_ryd_es,
            idling_flag=idling_flag,
        )

    @staticmethod
    def create_171Yb_1Q_channel(
        p_dep1: float = 0.1e-2,
        p_leak: float = 0.1e-2,  # gate induced leakage from m to g
        T2_g: float = 10.0,  # second, ground coherence
        T1_g: float = 200.0,  # second, ground spin relaxation
        T2_m: float = 10.0,  # second, metastable coherence
        T1_m: float = 200.0,  # second, metastable spin relaxation
        T2_c: float = 5.0,  # second, clock coherence
        gate_time: float = 100e-6,
        lifetime_gs: float = 30.0,  # second, trap lifetime of ground
        lifetime_es: float = 30.0,  # second, trap lifetime of metastable
        leaktime_eg: float = 1.0,  # second, leakage time from m to g
        lifetime_ryd: float = 50e-6 / 0.51,  # second, rydberg radiative decay to L
        leaktime_ryd_gs: float = 50e-6 / 0.42,  # second, rydberg radiative decay to g
        leaktime_ryd_es: float = 50e-6 / 0.07,  # second, rydberg radiative decay to m
        idling_flag: bool = False,
        qubit_type: str = "ground",
    ) -> YbNoiseChannel:
        """Create a 171Yb 1Q gate noise channel with typical parameters."""
        return YbNoiseChannel(
            atom_species="171Yb",
            gate_type="1Q",
            qubit_type=qubit_type,
            p_dep1=p_dep1,
            p_leak=p_leak,
            T2_g=T2_g,
            T1_g=T1_g,
            T2_m=T2_m,
            T1_m=T1_m,
            T2_c=T2_c,
            gate_time=gate_time,
            lifetime_gs=lifetime_gs,
            lifetime_es=lifetime_es,
            leaktime_eg=leaktime_eg,
            lifetime_ryd=lifetime_ryd,
            leaktime_ryd_gs=leaktime_ryd_gs,
            leaktime_ryd_es=leaktime_ryd_es,
            idling_flag=idling_flag,
        )

    @staticmethod
    def create_171Yb_2Q_channel(
        p_dep2: float = 0.2e-2,
        T2_g: float = 10.0,
        T1_g: float = 200.0,
        T2_m: float = 10.0,
        T1_m: float = 200.0,
        T2_c: float = 5.0,
        gate_time: float = 300e-9,
        lifetime_gs: float = 30.0,
        lifetime_es: float = 30.0,
        leaktime_eg: float = 1.0,
        lifetime_ryd: float = 50e-6 / 0.51,
        leaktime_ryd_gs: float = 50e-6 / 0.42,
        leaktime_ryd_es: float = 50e-6 / 0.07,
        idling_flag: bool = False,
    ) -> YbNoiseChannel:
        """Create a 171Yb 2Q gate noise channel with typical parameters."""
        return YbNoiseChannel(
            atom_species="171Yb",
            gate_type="2Q",
            p_dep2=p_dep2,
            T2_g=T2_g,
            T1_g=T1_g,
            T2_m=T2_m,
            T1_m=T1_m,
            T2_c=T2_c,
            gate_time=gate_time,
            lifetime_gs=lifetime_gs,
            lifetime_es=lifetime_es,
            leaktime_eg=leaktime_eg,
            lifetime_ryd=lifetime_ryd,
            leaktime_ryd_gs=leaktime_ryd_gs,
            leaktime_ryd_es=leaktime_ryd_es,
            idling_flag=idling_flag,
        )

    @staticmethod
    def create_171Yb_1Q_clock_channel(
        p_dep1: float = 0.2e-2,
        gate_time: float = 10e-6,
        lifetime_gs: float = 30.0,
        lifetime_es: float = 30.0,
        leaktime_eg: float = 1.0,
        lifetime_ryd: float = 50e-6 / 0.51,
        leaktime_ryd_gs: float = 50e-6 / 0.42,
        leaktime_ryd_es: float = 50e-6 / 0.07,
        qubit_type: str = "metastable",
    ) -> YbNoiseChannel:
        """Create a 171Yb 1Q clock-excitation noise channel (DEP1_gm, etc.).

        This channel models the errors during clock excitation that couple
        ground and metastable manifolds. For Pauli twirling on m-qubits,
        set qubit_type to "metastable" (default).
        """
        return YbNoiseChannel(
            atom_species="171Yb",
            gate_type="1Q_CLOCK",
            qubit_type=qubit_type,
            p_dep1=p_dep1,
            gate_time=gate_time,
            lifetime_gs=lifetime_gs,
            lifetime_es=lifetime_es,
            leaktime_eg=leaktime_eg,
            lifetime_ryd=lifetime_ryd,
            leaktime_ryd_gs=leaktime_ryd_gs,
            leaktime_ryd_es=leaktime_ryd_es,
        )

    @staticmethod
    def create_174Yb_RESET_channel(
        p_loss: float = 0.6e-2,
    ) -> YbNoiseChannel:
        """Create a 174Yb RESET (state preparation) noise channel
        with typical parameters.
        """
        return YbNoiseChannel(
            atom_species="174Yb",
            gate_type="RESET",
            p_loss=p_loss,
        )

    @staticmethod
    def create_174Yb_MEASURE_DISC_channel(
        p_meas: float = 0.2e-2,
        q: float = 1.0,
    ) -> YbNoiseChannel:
        """Create a 174Yb measurement discrimination error channel.

        This models the error in discriminating between quantum states
        during measurement.

        MEASURE_DISC vs MEASURE distinction:
        - MEASURE_DISC: Models reading errors (e.g., reading |g⟩ as |m⟩, |l⟩, or |r⟩)
        - MEASURE: Models physical errors during fluorescence imaging

        Parameters:
            p_meas: Measurement error probability (typically 0.1-1% for good apparatus)
            q: Probability of assigning the ambiguous BD outcome to g

        Use case: Evaluating measurement apparatus performance, modeling state
        misidentification during readout.

        Note: For complete measurement simulation, combine with MEASURE channel
        to model both discrimination and physical errors.
        """
        return YbNoiseChannel(
            atom_species="174Yb",
            gate_type="MEASURE_DISC",
            p_meas=p_meas,
            q=q,
        )

    @staticmethod
    def create_174Yb_MEASURE_READ_channel(
        p_loss: float = 0.1e-2,
        gate_time: float = 1e-3,  # second, readout time
        lifetime_es: float = 30.0,
        leaktime_eg: float = 1.0,
        lifetime_ryd: float = 50e-6 / 0.51,
        leaktime_ryd_gs: float = 50e-6 / 0.42,
        leaktime_ryd_es: float = 50e-6 / 0.07,
    ) -> YbNoiseChannel:
        """Create a 174Yb measurement readout error channel.

        This models the physical errors during fluorescence imaging readout.

        MEASURE_DISC vs MEASURE distinction:
        - MEASURE_DISC: Models reading errors (state misidentification)
        - MEASURE: Models physical errors (atomic loss, leakage, trap loss)

        Physical errors modeled:
        - LOSS_g_meas: ground-state loss during measurement (imaging induced loss)
        - LOSS_m: metastable (3P0) trap loss
        - DECAY_mg: decay from 3P0 to 1S0
        - LOSS_r: Rydberg-state decay into untrapped or dark states
        - DECAY_rg: Rydberg decay to the ground manifold
        - DECAY_rm: Rydberg decay to the metastable manifold

        Note: For complete measurement simulation, apply in this sequence:
        1. MEASURE_DISC (discrimination errors)
        2. MEASURE (physical errors during readout)
        3. 1Q gate error (for clock de-excitation if needed)
        """
        return YbNoiseChannel(
            atom_species="174Yb",
            gate_type="MEASURE",
            p_loss=p_loss,
            gate_time=gate_time,
            lifetime_es=lifetime_es,
            leaktime_eg=leaktime_eg,
            lifetime_ryd=lifetime_ryd,
            leaktime_ryd_gs=leaktime_ryd_gs,
            leaktime_ryd_es=leaktime_ryd_es,
        )

    @staticmethod
    def create_171Yb_RESET_channel(
        p_mloss: float = 0.6e-2,
        p_mflip: float = 0.1e-2,
    ) -> YbNoiseChannel:
        """Create a 171Yb RESET (state preparation) noise channel
        with typical parameters.
        """
        return YbNoiseChannel(
            atom_species="171Yb",
            gate_type="RESET",
            p_mloss=p_mloss,
            p_mflip=p_mflip,
        )

    @staticmethod
    def create_171Yb_MEASURE_DISC_channel(
        p_meas: float = 0.2e-2,
        q: float = 1.0,
    ) -> YbNoiseChannel:
        """Create a 171Yb measurement discrimination error channel.

        This models the error in discriminating between quantum states
        during measurement.

        MEASURE_DISC vs MEASURE distinction:
        - MEASURE_DISC: Models reading errors (e.g., reading |g0⟩ as |g1⟩, |m0⟩ as |m1⟩)
        - MEASURE: Models physical errors during fluorescence imaging

        Parameters:
            p_meas: Bright/dark misdiscrimination probability per fluorescence pulse
                (typically 0.1-1% for good apparatus).
            q: Probability of assigning the ambiguous BD outcome (both pulses bright)
                to "0" instead of "1". Defaults to 1.0.

        Use case: Evaluating measurement apparatus performance, modeling state
        misidentification during readout.

        Note: For complete measurement simulation, combine with MEASURE channel
        to model both discrimination and physical errors.
        """
        return YbNoiseChannel(
            atom_species="171Yb",
            gate_type="MEASURE_DISC",
            p_meas=p_meas,
            q=q,
        )

    @staticmethod
    def create_171Yb_MEASURE_READ_channel(
        p_gflip: float = 0.1e-2,
        p_gloss: float = 0.1e-2,
        T2_m: float = 10.0,
        T1_m: float = 200.0,
        T2_c: float = 5.0,
        readout_time: float = 10e-3,
        lifetime_es: float = 30.0,
        leaktime_eg: float = 1.0,
        lifetime_ryd: float = 50e-6 / 0.51,
        leaktime_ryd_gs: float = 50e-6 / 0.42,
        leaktime_ryd_es: float = 50e-6 / 0.07,
    ) -> YbNoiseChannel:
        """Create a 171Yb measurement readout error channel.

        This models the physical errors during fluorescence imaging readout.

        MEASURE_DISC vs MEASURE distinction:
        - MEASURE_DISC: Models reading errors (state misidentification)
        - MEASURE: Models physical errors (atomic loss, leakage, trap loss)

        Physical errors modeled:
        - Ground state flip errors
        - Atomic loss during measurement
        - Decay and dephasing in metastable states
        - Clock state errors
        - Various leakage channels

        Note: For complete measurement simulation, apply in this sequence:
        1. MEASURE_DISC (discrimination errors)
        2. Clock de-excitation (if needed)
        3. MEASURE (physical errors during readout)
        """
        return YbNoiseChannel(
            atom_species="171Yb",
            gate_type="MEASURE",
            p_gflip=p_gflip,
            p_gloss=p_gloss,
            T2_m=T2_m,
            T1_m=T1_m,
            T2_c=T2_c,
            readout_time=readout_time,
            lifetime_es=lifetime_es,
            leaktime_eg=leaktime_eg,
            lifetime_ryd=lifetime_ryd,
            leaktime_ryd_gs=leaktime_ryd_gs,
            leaktime_ryd_es=leaktime_ryd_es,
        )
