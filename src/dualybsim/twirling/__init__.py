"""Generalised Pauli twirling approximation (GPTA).

Converts the Kraus channels of :mod:`dualybsim.kraus`, which act on the full
multi-level Hilbert space, into a Pauli channel on the computational subspace
composed with a loss channel for transitions out of it. See the supplementary
information of Google Quantum AI (2023) for the construction.
"""

from .gpta import GeneralizedTwirlingApproximation
from .pauli_plus import GeneralizedPauliChannel, PauliPlusState

__all__ = [
    "GeneralizedTwirlingApproximation",
    "GeneralizedPauliChannel",
    "PauliPlusState",
]
