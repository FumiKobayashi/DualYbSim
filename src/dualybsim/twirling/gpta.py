"""Generalized Pauli Twirling Approximation implementation.

This module implements the Generalized Pauli Twirling Approximation (GTA)
based on Google AI 2023's Supplementary Information IV.B.3.
"""

import logging

import numpy as np

from ..kraus.channels import YbNoiseChannel
from .pauli_plus import GeneralizedPauliChannel

logger = logging.getLogger(__name__)


class GeneralizedTwirlingApproximation:
    """Generalized Pauli Twirling Approximation"""

    def __init__(
        self, noise_channel: YbNoiseChannel, subchannel: str | None = None
    ) -> None:
        """Initialize Generalized Twirling Approximation.

        Args:
            noise_channel: YbNoiseChannel instance represented by Kraus
                operators on a q^n-dimensional Hilbert space, where q is the
                per-device Hilbert space dimension and n is the number of
                devices.
            subchannel: Optional specific subchannel name to twirl (e.g., 'DEP1').
        """
        self.noise_channel = noise_channel
        self.selected_channel = subchannel

        # Store the noise channel for later use
        # Keep Kraus operators grouped per noise channel
        self.channels: dict[str, list[np.ndarray]] = {}

        # Use only selected channel if provided (e.g., 'DEP1')
        raw_kraus = (
            noise_channel.get_kraus_operators(subchannel)
            if subchannel is not None
            else noise_channel.get_kraus_operators()
        )
        # raw_kraus may be a dict (per-channel) or a list (specific channel)
        flat_ops: list[np.ndarray]
        if isinstance(raw_kraus, dict):
            for channel_name, ops in raw_kraus.items():
                flat_ops = []
                for op in ops:
                    if op.ndim == 3:
                        num_ops, _, _ = op.shape
                        for i in range(num_ops):
                            flat_ops.append(op[i])
                    else:
                        flat_ops.append(op)
                self.channels[channel_name] = flat_ops
        else:
            flat_ops = []
            for op in raw_kraus:
                if op.ndim == 3:
                    num_ops, _, _ = op.shape
                    for i in range(num_ops):
                        flat_ops.append(op[i])
                else:
                    flat_ops.append(op)
            # Use provided subchannel name if any, else fallback to 'ALL'
            channel_key = subchannel if subchannel is not None else "ALL"
            self.channels[channel_key] = flat_ops

        # Normalize per-channel to satisfy CPTP condition approximately
        self.channels = self._normalize_channels()

        # Define Pauli operators
        self.pauli_operators = {
            "I": np.array([[1, 0], [0, 1]]),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
        }

    def _normalize_channels(self) -> dict[str, list[np.ndarray]]:
        """Normalize Kraus operators per channel to approximately satisfy CPTP.

        For each channel C: if sum_i K_i^\u2020 K_i has trace != dim, scale all K_i
        in that channel by a common factor so that trace matches dim. This matches
        the previous global normalization strategy but applied per channel.
        """
        normalized: dict[str, list[np.ndarray]] = {}
        for channel_name, ops in self.channels.items():
            if not ops:
                normalized[channel_name] = ops
                continue
            dim = ops[0].shape[0]
            completeness = np.zeros((dim, dim), dtype=complex)
            for K in ops:
                completeness += K.conj().T @ K
            trace_comp = np.trace(completeness).real
            if np.abs(trace_comp - dim) < 1e-10:
                normalized[channel_name] = ops
                continue
            factor = np.sqrt(dim / trace_comp)
            normalized[channel_name] = [K * factor for K in ops]
        return normalized

    @property
    def kraus_operators(self) -> list[np.ndarray]:
        """Compatibility: flattened list of all Kraus operators across channels."""
        flat: list[np.ndarray] = []
        for ops in self.channels.items():
            # ops is (channel_name, List[np.ndarray])
            flat.extend(ops[1])
        return flat

    def _get_subspaces(self) -> dict[str, str]:
        """Get subspace definitions based on atom species.

        Returns:
            Dictionary mapping subspace keys to descriptions
        """
        if self.noise_channel.atom_species == "174Yb":
            return {
                "c": "computational",  # |g0⟩, |g1⟩ (ground states)
                "r": "rydberg",  # |r⟩ (Rydberg state)
                "L": "leakage",  # |L⟩ (leakage/loss states)
            }
        elif self.noise_channel.atom_species == "171Yb":
            return {
                "c": "computational",  # User-selected computational basis
                "r": "rydberg",  # |r⟩ (Rydberg state)
                "L": "leakage",  # |L⟩ (leakage/loss states)
            }
        else:
            raise ValueError(f"Unknown atom species: {self.noise_channel.atom_species}")

    def decompose_kraus_operators(
        self,
    ) -> dict[str, list[np.ndarray]] | dict[str, dict[str, list[np.ndarray]]]:
        """Decompose Kraus operators into blocks corresponding to subspaces for each channel.

        Returns:
            - If a specific subchannel was requested upstream: Dict[str, List[np.ndarray]]
              mapping (initial_subspace->final_subspace) to block operators
            - Otherwise: Dict[channel_name, Dict[str, List[np.ndarray]]]
              per-noise-channel mapping to the same block structure
        """
        # Build blocks using per-channel operators we hold
        if self.selected_channel is None:
            grouped_blocks: dict[str, dict[str, list[np.ndarray]]] = {}
            for channel_name, ops in self.channels.items():
                blocks: dict[str, list[np.ndarray]] = {}
                for op in ops:
                    if len(op.shape) == 3:
                        # 3D tensor: process each operator in the tensor
                        num_ops, _, _ = op.shape
                        for j in range(num_ops):
                            K = op[j]
                            self._decompose_single_operator(K, blocks)
                    else:
                        # 2D matrix: process directly
                        self._decompose_single_operator(op, blocks)
                grouped_blocks[channel_name] = blocks
            return grouped_blocks
        else:
            # Specific channel path returns a flat mapping
            ops = self.channels.get(self.selected_channel, [])
            blocks_single: dict[str, list[np.ndarray]] = {}
            for op in ops:
                if len(op.shape) == 3:
                    num_ops, _, _ = op.shape
                    for j in range(num_ops):
                        K = op[j]
                        self._decompose_single_operator(K, blocks_single)
                else:
                    self._decompose_single_operator(op, blocks_single)

            return blocks_single

    def _decompose_single_operator(
        self, K: np.ndarray, blocks: dict[str, list[np.ndarray]]
    ) -> None:
        """Decompose a single Kraus operator into subspace blocks."""
        dim = K.shape[0]

        if dim == 4:  # 174Yb: |g0⟩, |g1⟩, |r⟩, |L⟩
            # Define projection operators for 174Yb
            P_c = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            )  # computational (|g0⟩, |g1⟩)
            P_r = np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
            )  # rydberg
            P_L = np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
            )  # leakage

            subspaces = ["c", "r", "L"]
            projections = {"c": P_c, "r": P_r, "L": P_L}

        elif dim == 6:  # 171Yb: |0g⟩, |1g⟩, |0m⟩, |1m⟩, |r⟩, |L⟩
            # Define projection operators for 171Yb
            P_g = np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
            P_m = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
            P_L = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )

            if self.noise_channel.qubit_type == "ground":
                # Computational subspace: |g0⟩, |g1⟩
                P_c = P_g
                P_L = P_L + P_m
            elif self.noise_channel.qubit_type == "metastable":
                # Computational subspace: |m0⟩, |m1⟩
                P_c = P_m
                P_L = P_L + P_g
            else:
                raise ValueError(
                    f"Invalid qubit_type for 171Yb: {self.noise_channel.qubit_type}"
                )

            P_r = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )

            subspaces = ["c", "r", "L"]
            projections = {"c": P_c, "r": P_r, "L": P_L}
        else:
            raise ValueError(f"Unsupported matrix dimension: {dim}")

        # Calculate blocks for each subspace transition
        for i_subspace in subspaces:
            for f_subspace in subspaces:
                P_i = projections[i_subspace]
                P_f = projections[f_subspace]

                # Calculate block: P_f * K * P_i
                block = P_f @ K @ P_i

                if not np.allclose(block, 0):
                    key = f"{i_subspace}->{f_subspace}"
                    if key not in blocks:
                        blocks[key] = []
                    blocks[key].append(block)

    def compute_pauli_decomposition(
        self, computational_block: np.ndarray
    ) -> dict[str, float]:
        """Decompose computational block into Pauli operators.

        Args:
            computational_block: Block operator acting on computational
                               subspace

        Returns:
            Dictionary mapping Pauli operators to coefficients
        """
        # Extract the computational subspace block
        if self.noise_channel.atom_species == "174Yb":
            if computational_block.shape[0] == 4:  # 174Yb
                # For 174Yb, computational space is |g0⟩, |g1⟩ (first 2x2 block)
                comp_block = computational_block[:2, :2]
            else:
                raise ValueError(
                    f"Unexpected block dimension for 174Yb: {computational_block.shape}"
                )
        elif self.noise_channel.atom_species == "171Yb":
            if computational_block.shape[0] == 6:  # 171Yb
                if self.noise_channel.qubit_type == "ground":
                    # For 171Yb ground qubit, computational space is |g0⟩, |g1⟩
                    comp_block = computational_block[:2, :2]
                elif self.noise_channel.qubit_type == "metastable":
                    # For 171Yb metastable qubit, computational space is |m0⟩, |m1⟩
                    comp_block = computational_block[2:4, 2:4]
                else:
                    raise ValueError(
                        f"Invalid qubit_type for 171Yb: {self.noise_channel.qubit_type}"
                    )
            else:
                raise ValueError(
                    f"Unexpected block dimension for 171Yb: {computational_block.shape}"
                )
        else:
            raise ValueError(
                f"Unsupported atom species: {self.noise_channel.atom_species}"
            )

        # Decompose into Pauli operators
        coefficients = {}
        for pauli_name, pauli_op in self.pauli_operators.items():
            # Calculate coefficient: Tr(σ_μ * block) / 2
            # For 2x2 matrices: M = (1/2) * sum_μ Tr(σ_μ M) σ_μ
            coeff = np.trace(pauli_op @ comp_block) / 2
            coefficients[pauli_name] = coeff

        return coefficients

    def derive_conditional_probabilities(self) -> dict[str, float]:
        """Derive conditional probability distribution P(i→f, μ).

        Returns:
            Dictionary mapping (initial_subspace, final_subspace, pauli)
            to probability.
            Note that the transition probabilities from computational subspace are normalized to 1.0, and the probability of leakage transitions are attached to Pauli-I label but meaningless label here.
        """
        blocks = self.decompose_kraus_operators()
        # Calculate total probability for each transition
        transition_probs = {}
        total_prob_from_c = 0
        if self.noise_channel.atom_species == "174Yb":
            dim_subspaces = {"c": 2, "r": 1, "L": 1}
        elif self.noise_channel.atom_species == "171Yb":
            dim_subspaces = {"c": 2, "r": 1, "L": 3}
        else:
            raise ValueError(f"Unknown atom species: {self.noise_channel.atom_species}")

        for key, block_list in blocks.items():
            # Sum over all Kraus operators for this transition
            normalizer = 1 / dim_subspaces[key[0]]
            total_prob = normalizer * sum(
                np.trace(block @ block.conj().T)  # type: ignore[union-attr]
                for block in block_list
            )
            transition_probs[key] = total_prob.real
            if key.startswith("c->"):
                total_prob_from_c += total_prob.__abs__()

        # Normalize probabilities from computational subspace
        if total_prob_from_c <= 0:
            raise ValueError("Total probability from computational subspace is 0")

        transition_probs_from_c = {}
        for key in transition_probs:
            if key.startswith("c->"):
                transition_probs_from_c[key] = transition_probs[key].real

        # For each c->c transition, decompose computational blocks & calculate Pauli probabilities
        pauli_keys = ["I", "X", "Y", "Z"]
        pauli_probabilities = dict.fromkeys(pauli_keys, 0)
        for key, block_list in blocks.items():
            if key != "c->c":
                continue
            # Only decompose if state transition is computational ('c->c')
            for block in block_list:
                pauli_coeffs = self.compute_pauli_decomposition(block)  # type: ignore[arg-type]
                # Calculate conditional probabilities
                for pauli_name, coeff in pauli_coeffs.items():
                    if abs(coeff) > 1e-10:
                        prob_key = f"{pauli_name}"
                        pauli_probabilities[prob_key] += (
                            abs(coeff) ** 2 / transition_probs_from_c["c->c"]
                        )

        probabilities = {}
        for key, prob in transition_probs_from_c.items():
            if key != "c->c":
                new_key = f"{key}:I"
                probabilities[new_key] = prob
            else:
                for pauli_name, pauli_prob in pauli_probabilities.items():
                    new_key = f"{key}:{pauli_name}"
                    probabilities[new_key] = pauli_prob
        return probabilities

    def _get_computational_subspaces(self) -> list:
        """Get list of computational subspaces based on atom species.

        Returns:
            List of computational subspace keys that correspond to the 'c' subspace
        """
        if self.noise_channel.atom_species == "174Yb":
            # For 174Yb, 'c' corresponds to |g0⟩, |g1⟩
            return ["g0", "g1"]
        elif self.noise_channel.atom_species == "171Yb":
            if self.noise_channel.qubit_type == "ground":
                # For 171Yb ground qubit, 'c' corresponds to |g0⟩, |g1⟩
                return ["g0", "g1"]
            elif self.noise_channel.qubit_type == "metastable":
                # For 171Yb metastable qubit, 'c' corresponds to |m0⟩, |m1⟩
                return ["m0", "m1"]
            else:
                raise ValueError(
                    f"Invalid qubit_type for 171Yb: {self.noise_channel.qubit_type}"
                )
        else:
            raise ValueError(f"Unknown atom species: {self.noise_channel.atom_species}")

    def to_generalized_pauli_channel(self) -> GeneralizedPauliChannel:
        """Convert to Generalized Pauli Channel.

        Returns:
            GeneralizedPauliChannel instance
        """
        from .pauli_plus import GeneralizedPauliChannel

        probabilities = self.derive_conditional_probabilities()
        return GeneralizedPauliChannel(
            probabilities,
            atom_species=self.noise_channel.atom_species,
            qubit_type=self.noise_channel.qubit_type,
        )

    def validate_cptp(self) -> bool:
        """Validate that the Kraus operators satisfy CPTP conditions.

        Returns:
            True if CPTP conditions are satisfied
        """
        # Use the original channel's CPTP method to validate
        # Create a test density matrix
        dim = 4 if self.noise_channel.atom_species == "174Yb" else 6
        test_rho = np.eye(dim, dtype=complex) / dim  # Maximally mixed state

        # Apply the channel
        result = self.noise_channel.apply(test_rho)

        # Check if result is a valid density matrix
        trace = np.trace(result).real
        eigenvals = np.linalg.eigvals(result)

        is_valid = abs(trace - 1.0) < 1e-10 and np.all(eigenvals >= -1e-10)

        if not is_valid:
            logger.warning("CPTP condition violated.")
            logger.warning(f"Trace: {trace}, Expected: 1.0")
            logger.warning(f"Eigenvalues: {eigenvals}")

        return bool(is_valid)

    def __repr__(self) -> str:
        return f"GeneralizedTwirlingApproximation(noise_channel={self.noise_channel})"
