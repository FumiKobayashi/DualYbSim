"""Generalized Pauli Channel implementation.

This module implements the Generalized Pauli Channel (GPC) for Pauli+
simulation.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GeneralizedPauliChannel:
    """Generalized Pauli Channel (GPC)
    This class implements the Generalized Pauli Channel (GPC) for Pauli+
    simulation.

    The GPC is a generalization of the Pauli channel that allows for
    arbitrary Pauli errors, but with an important physical constraint:
    Pauli errors only occur when transitioning TO
    computational space ('c'). Transitions between non-computational
    states (r->r, r->L, L->r, L->L) do not have Pauli errors and
    are forced to use 'I' (identity).

    This constraint reflects the physical reality that:
    1. Computational states (|g0⟩, |g1⟩, |m0⟩, |m1⟩) can experience
       Pauli errors (X, Y, Z) during quantum operations
    2. Non-computational states (Rydberg, leakage) do not experience
       Pauli errors - they only have population transfer between levels
    3. When returning to computational space from non-computational
       states, Pauli errors can occur due to imperfect control

    The GPC is defined by a set of transition probabilities between states.

    Attributes:
        transition_probs (Dict[str, float]):
            Dictionary mapping transition keys to probabilities.
            Keys are in format "initial->final:pauli" (e.g., "c->c:X", "r->L:I").
            This stores the raw transition probabilities before normalization.
            Note: For transitions to non-computational states, Pauli errors
            are automatically forced to 'I' regardless of the input.

        atom_species (str):
            The atomic species being simulated. Either "174Yb" or "171Yb".
            This determines the available quantum states and their properties.

        qubit_type (str):
            For 171Yb atoms, specifies whether to use "ground" or "metastable"
            states as computational basis. For 174Yb atoms, this is ignored.

        transition_matrix (Dict[str, List[Dict[str, Any]]]):
            Internal data structure for efficient sampling. Organized by initial state,
            each entry contains a list of possible transitions with:
            - final_state: The destination state ('c', 'r', 'L', etc.)
            - pauli_error: The Pauli error applied ('I', 'X', 'Y', 'Z')
            - probability: Normalized transition probability
            This is built automatically during initialization for fast sampling.
    """

    def __init__(
        self,
        transition_probs: dict[str, float],
        atom_species: str = "174Yb",
        qubit_type: str = "ground",
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize Generalized Pauli Channel.

        Args:
            transition_probs: Dictionary mapping transition keys to
                            probabilities. Keys are in format
                            "initial->final:pauli"
            atom_species: "174Yb" or "171Yb"
            qubit_type: For 171Yb, "ground" or "metastable". For 174Yb, ignored.
            rng: Generator used by :meth:`sample_transition`. Pass a seeded
                ``np.random.default_rng(seed)`` for reproducible sampling;
                omitted, an unseeded generator is created.
        """
        # Store the raw transition probabilities provided by user
        self.transition_probs = transition_probs

        # Store atomic species information for state handling
        self.atom_species = atom_species

        # Store qubit type for 171Yb (ground vs metastable states)
        self.qubit_type = qubit_type

        # Normalize probabilities to ensure they sum to 1 for each initial state
        self._normalize_probabilities()

        # Build internal transition matrix for efficient sampling
        self._build_transition_matrix()

        # Own generator rather than the global RNG, so a caller can seed it.
        self.rng = np.random.default_rng() if rng is None else rng

    def _normalize_probabilities(self) -> None:
        """Normalize transition probabilities."""
        # Group by initial state
        initial_states: dict[str, dict[str, float]] = {}
        for key, prob in self.transition_probs.items():
            if prob > 0:  # Only consider transitions with positive probability
                parts = key.split("->")
                if len(parts) == 2:
                    initial_state = parts[0]
                    # Create a dictionary for this initial state if it doesn't exist
                    if initial_state not in initial_states:
                        initial_states[initial_state] = {}
                    # Store the transition with its probability
                    initial_states[initial_state][key] = prob
                else:
                    # Invalid transition key format
                    raise ValueError(f"Invalid transition key: {key}")

        # Normalize probabilities for each initial state separately
        for _, transitions in initial_states.items():
            # Calculate the total probability for this initial state
            total_prob = sum(transitions.values())
            if total_prob > 0:
                # Scale all probabilities so they sum to 1.0
                for key in transitions:
                    self.transition_probs[key] /= total_prob

    def _build_transition_matrix(self) -> None:
        """Build transition matrix for efficient sampling.

        This method creates an internal data structure that organizes transitions
        by initial state for fast sampling. The transition_matrix attribute is
        structured as:

        {
            'c': [  # Initial state is computational ('c')
                {
                    'final_state': 'c',      # Destination state
                    'pauli_error': 'X',      # Pauli error applied (only for c->c)
                    'probability': 0.3       # Normalized probability
                },
                {
                    'final_state': 'L',      # Another possible destination
                    'pauli_error': 'I',      # No Pauli error
                    'probability': 0.1       # Normalized probability
                }
            ],
            'r': [  # Initial state is Rydberg ('r')
                # ... similar structure for Rydberg state transitions
            ],
            'L': [  # Initial state is leakage ('L')
                # ... similar structure for leakage state transitions
            ]
        }
        """
        # Initialize the transition matrix as an empty dictionary
        # This will store transitions organized by initial state
        self.transition_matrix: dict[str, list[dict[str, Any]]] = {}

        # Group transitions by initial state from the raw transition_probs
        for key, prob in self.transition_probs.items():
            if prob > 0:  # Only consider transitions with non-zero probability
                parts = key.split("->")
                if len(parts) == 2:
                    initial_state = parts[0]

                    # Create entry for this initial state if it doesn't exist
                    if initial_state not in self.transition_matrix:
                        self.transition_matrix[initial_state] = []

                    # Parse the transition part to extract final state and Pauli error
                    transition_part = parts[1]
                    if ":" in transition_part:
                        # Format: "final_state:pauli_error"
                        final_state, pauli_error = transition_part.split(":")
                    else:
                        # Format: "final_state" (no Pauli error)
                        final_state = transition_part
                        pauli_error = "I"  # Identity (no error)

                    # Add this transition to the matrix
                    self.transition_matrix[initial_state].append(
                        {
                            "final_state": final_state,  # Where the state goes
                            "pauli_error": pauli_error,  # What Pauli error is applied
                            "probability": prob,  # Probability of this transition
                        }
                    )

        # Sort transitions by probability (descending) for efficient sampling
        # This allows us to check high-probability transitions first
        for initial_state in self.transition_matrix:
            self.transition_matrix[initial_state].sort(
                key=lambda x: x["probability"], reverse=True
            )

    def sample_transition(self, initial_state: str) -> tuple[str, str]:
        """Sample a transition from the given initial state.

        Args:
            initial_state: Initial state ('c', 'r', 'L')

        Returns:
            Tuple of (final_state, pauli_error)
        """
        if initial_state not in self.transition_matrix:
            # No transitions available, stay in same state
            return initial_state, "I"

        transitions = self.transition_matrix[initial_state]

        # Sample based on probabilities
        r = self.rng.random()
        cumulative_prob = 0.0

        for transition in transitions:
            cumulative_prob += transition["probability"]
            if r <= cumulative_prob:
                final_state = transition["final_state"]
                pauli_error = transition["pauli_error"]

                # Convert 'c' to appropriate specific state
                if final_state == "c":
                    if self.atom_species == "174Yb":
                        # For 174Yb, computational space is |g0⟩, |g1⟩
                        final_state = "c"
                    elif self.atom_species == "171Yb":
                        if self.qubit_type == "ground":
                            # For 171Yb ground qubit, computational space is |g0⟩, |g1⟩
                            final_state = "g"
                        elif self.qubit_type == "metastable":
                            # For 171Yb metastable qubit, computational space is |m0⟩, |m1⟩
                            final_state = "m"
                        else:
                            raise ValueError(
                                f"Invalid qubit_type for 171Yb: {self.qubit_type}"
                            )
                    else:
                        raise ValueError(f"Unknown atom species: {self.atom_species}")

                return final_state, pauli_error

        # Fallback: stay in same state
        return initial_state, "I"

    def apply_pauli_error(self, state: np.ndarray, pauli_type: str) -> np.ndarray:
        """Apply Pauli error to a quantum state.

        Args:
            state: Quantum state (density matrix or state vector)
            pauli_type: Type of Pauli error ('I', 'X', 'Y', 'Z')

        Returns:
            State after applying Pauli error
        """
        # Define Pauli operators
        pauli_operators = {
            "I": np.array([[1, 0], [0, 1]]),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
        }

        if pauli_type not in pauli_operators:
            raise ValueError(f"Unknown Pauli type: {pauli_type}")

        pauli_op = pauli_operators[pauli_type]

        # Apply Pauli operator
        if state.ndim == 1:  # State vector
            return pauli_op @ state
        else:  # Density matrix
            return pauli_op @ state @ pauli_op.conj().T

    def get_transition_probability(
        self, initial_state: str, final_state: str, pauli_error: str = "I"
    ) -> float:
        """Get probability for a specific transition.

        Args:
            initial_state: Initial state
            final_state: Final state
            pauli_error: Pauli error type

        Returns:
            Transition probability
        """
        key = f"{initial_state}->{final_state}:{pauli_error}"
        return self.transition_probs.get(key, 0.0)

    def get_available_transitions(
        self, initial_state: str
    ) -> list[tuple[str, str, float]]:
        """Get all available transitions from an initial state.

        Args:
            initial_state: Initial state

        Returns:
            List of (final_state, pauli_error, probability) tuples
        """
        if initial_state not in self.transition_matrix:
            return []

        return [
            (t["final_state"], t["pauli_error"], t["probability"])
            for t in self.transition_matrix[initial_state]
        ]

    def validate_probabilities(self) -> bool:
        """Validate that probabilities sum to 1 for each initial state.

        Returns:
            True if probabilities are valid
        """
        for initial_state in self.transition_matrix:
            total_prob = sum(
                t["probability"] for t in self.transition_matrix[initial_state]
            )
            if not np.isclose(total_prob, 1.0, atol=1e-10):
                logger.warning(f"Probabilities for {initial_state} sum to {total_prob}")
                return False
        return True

    def __repr__(self) -> str:
        return f"GeneralizedPauliChannel(transitions={len(self.transition_probs)})"


class PauliPlusState:
    """State representation for Pauli+ simulation."""

    def __init__(
        self, n_qubits: int, atom_species: str = "174Yb", qubit_type: str = "ground"
    ) -> None:
        """Initialize Pauli+ state.

        Args:
            n_qubits: Number of qubits
            atom_species: "174Yb" or "171Yb"
            qubit_type: For 171Yb, "ground" or "metastable". For 174Yb, ignored.
        """
        self.n_qubits = n_qubits
        self.atom_species = atom_species
        self.qubit_type = qubit_type

        # Initialize quantum state based on atom species
        if atom_species == "174Yb":
            # For 174Yb: ground states (g0, g1), rydberg (r), leakage (L)
            # Default to ground state (g0)
            self.quantum_state = ["g0"] * n_qubits
        elif atom_species == "171Yb":
            # For 171Yb: ground states (g0, g1), metastable states (m0, m1), rydberg (r), leakage (L)
            if qubit_type == "ground":
                # Default to ground state (g0)
                self.quantum_state = ["g0"] * n_qubits
            elif qubit_type == "metastable":
                # Default to metastable state (m0)
                self.quantum_state = ["m0"] * n_qubits
            else:
                raise ValueError(
                    f"Invalid qubit_type for 171Yb: {qubit_type}. Must be 'ground' or 'metastable'"
                )
        else:
            raise ValueError(f"Unknown atom species: {atom_species}")

        self.stabilizer_tableau = None  # Will be initialized when needed

    def update_quantum_state(self, qubit: int, new_state: str) -> None:
        """Update quantum state for a specific qubit.

        Args:
            qubit: Qubit index
            new_state: New quantum state
        """
        if 0 <= qubit < self.n_qubits:
            # Validate state based on atom species
            valid_states = self._get_valid_states()
            if new_state not in valid_states:
                raise ValueError(
                    f"Invalid state '{new_state}' for {self.atom_species}. "
                    f"Valid states: {valid_states}"
                )
            self.quantum_state[qubit] = new_state
        else:
            raise ValueError(f"Invalid qubit index: {qubit}")

    def _get_valid_states(self) -> list[str]:
        """Get valid states for the current atom species.

        Returns:
            List of valid state names
        """
        if self.atom_species == "174Yb":
            return ["g0", "g1", "r", "L"]
        elif self.atom_species == "171Yb":
            return ["g0", "g1", "m0", "m1", "r", "L"]
        else:
            raise ValueError(f"Unknown atom species: {self.atom_species}")

    def get_computational_qubits(self) -> list[int]:
        """Get list of qubits in computational subspace.

        Returns:
            List of qubit indices in computational subspace
        """
        computational_states = self._get_computational_states()
        return [
            i
            for i, state in enumerate(self.quantum_state)
            if state in computational_states
        ]

    def _get_computational_states(self) -> list[str]:
        """Get computational states for the current atom species.

        Returns:
            List of computational state names
        """
        if self.atom_species == "174Yb":
            return ["g0", "g1"]
        elif self.atom_species == "171Yb":
            return ["g0", "g1", "m0", "m1"]
        else:
            raise ValueError(f"Unknown atom species: {self.atom_species}")

    def get_leakage_qubits(self) -> list[int]:
        """Get list of qubits in leakage states.

        Returns:
            List of qubit indices in leakage states
        """
        computational_states = self._get_computational_states()
        return [
            i
            for i, state in enumerate(self.quantum_state)
            if state not in computational_states
        ]

    def is_computational(self, qubit: int) -> bool:
        """Check if a qubit is in computational subspace.

        Args:
            qubit: Qubit index

        Returns:
            True if qubit is in computational subspace
        """
        computational_states = self._get_computational_states()
        return self.quantum_state[qubit] in computational_states

    def get_rydberg_qubits(self) -> list[int]:
        """Get list of qubits in Rydberg states.

        Returns:
            List of qubit indices in Rydberg states
        """
        return [i for i, state in enumerate(self.quantum_state) if state == "r"]

    def get_loss_qubits(self) -> list[int]:
        """Get list of qubits in loss states.

        Returns:
            List of qubit indices in loss states
        """
        return [i for i, state in enumerate(self.quantum_state) if state == "L"]

    def __repr__(self) -> str:
        return (
            f"PauliPlusState(n_qubits={self.n_qubits}, "
            f"atom_species='{self.atom_species}', "
            f"quantum_state={self.quantum_state})"
        )
