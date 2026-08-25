r"""DualYbSim: a Stim circuit wrapper implementing a dual-isotope Yb noise model.

The library layers physically-motivated noise onto an otherwise ideal Stim
circuit. Kraus operators for 171Yb and 174Yb are defined in
:mod:`dualybsim.kraus`, converted to Pauli + loss channels by the generalised
Pauli twirling approximation in :mod:`dualybsim.twirling`, and injected into a
Stim circuit by :class:`~dualybsim.circuit.YbCircuit`.

The usual entry point is :class:`~dualybsim.model.YbNoiseModel`, which takes an
ideal Stim program plus a :class:`~dualybsim.qubits.QubitManager` describing
which atom each qubit is, and returns the noisy circuit::

    import stim
    from dualybsim import NoiseModelParameters, QubitManager, YbNoiseModel

    qubits = QubitManager()
    qubits.add_qubit(0, isotope="171", qubit_type="m", role="data")
    qubits.add_qubit(1, isotope="174", qubit_type="gm", role="ancilla")

    # Every moment must be closed by a TICK, the last one included: the TICK is
    # what tells the model a moment is over and which qubits idled through it.
    ideal = stim.Circuit("H 0\\nTICK\\nCZ 0 1\\nTICK\\nM 1\\nTICK")
    noisy = YbNoiseModel(NoiseModelParameters()).noisy_circuit(ideal, qubits)
"""

from .circuit import OperationRecord, YbCircuit
from .kraus.channels import YbNoiseChannel, YbNoiseChannelFactory
from .model import YbNoiseModel, YbNoiseModelAdapter, build_yb_noise_model
from .params import ENCODINGS, NoiseModelParameters, QubitNoiseView
from .qubits import QubitManager
from .twirling.gpta import GeneralizedTwirlingApproximation
from .twirling.pauli_plus import GeneralizedPauliChannel

__version__ = "0.1.0"

__all__ = [
    # Top-level API
    "YbNoiseModel",
    "YbNoiseModelAdapter",
    "build_yb_noise_model",
    "YbCircuit",
    "QubitManager",
    "NoiseModelParameters",
    "QubitNoiseView",
    "ENCODINGS",
    "OperationRecord",
    # Kraus / twirling layer
    "YbNoiseChannel",
    "YbNoiseChannelFactory",
    "GeneralizedTwirlingApproximation",
    "GeneralizedPauliChannel",
]
