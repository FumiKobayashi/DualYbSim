"""Adapter that rebuilds noisy Yb circuits from noiseless operation logs.

Replay-style noise insertion: ideal operations are recorded once, and noise is layered
on afterwards by re-executing those operations with noise enabled.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import stim

from .params import NoiseModelParameters
from .qubits import QubitManager

if TYPE_CHECKING:
    from .circuit import OperationRecord, YbCircuit


class YbNoiseModelAdapter:
    """Replays a noiseless operation log to produce a noisy Stim circuit."""

    def __init__(
        self,
        noise_params: NoiseModelParameters,
        qubit_manager: QubitManager,
    ) -> None:
        """Bind the parameter set and qubit metadata used for every replay."""
        self.noise_params = noise_params
        self.qubit_manager = qubit_manager

    def apply(self, operation_log: Iterable[OperationRecord]) -> YbCircuit:
        """Replay *operation_log* with noise enabled and return the result.

        Args:
            operation_log: Ideal operations, as recorded by a tracking
                :class:`~dualybsim.circuit.YbCircuit`.

        Returns:
            A fresh circuit holding the same operations with noise inserted.

        Raises:
            TypeError: If any entry is not an ``OperationRecord``.
        """
        from .circuit import OperationRecord, YbCircuit

        circuit = YbCircuit(
            self.qubit_manager,
            noise_params=self.noise_params,
            noise_enabled=True,
            track_operations=False,
        )
        for record in operation_log:
            if not isinstance(record, OperationRecord):
                raise TypeError("Operation log must contain OperationRecord entries.")
            record.replay(circuit)
        return circuit


def build_yb_noise_model(
    noise_params: NoiseModelParameters, qubit_manager: QubitManager
) -> YbNoiseModelAdapter:
    """Factory helper for constructing a YbNoiseModelAdapter."""
    return YbNoiseModelAdapter(noise_params, qubit_manager)


class YbNoiseModel:
    """High-level entry point for building noisy Yb circuits from Stim programs."""

    def __init__(self, noise_params: NoiseModelParameters | None = None) -> None:
        """Build a model around *noise_params*, or the paper's defaults."""
        self.noise_params = noise_params or NoiseModelParameters()

    def noiseless_circuit(
        self,
        program: str | stim.Circuit,
        qubit_manager: QubitManager,
        *,
        track_operations: bool = True,
    ) -> YbCircuit:
        """Import *program* without inserting any noise.

        Args:
            program: Stim circuit text or a pre-built ``stim.Circuit``.
            qubit_manager: Which atom and encoding each qubit is.
            track_operations: Keep an operation log, so the result can be fed
                back through :meth:`YbNoiseModelAdapter.apply` later.

        Returns:
            The same circuit, validated against the supported instruction set.
        """
        from .circuit import YbCircuit

        return YbCircuit.from_stim(
            program,
            qubit_manager,
            noise_params=self.noise_params,
            noise_enabled=False,
            track_operations=track_operations,
        )

    def noisy_circuit(
        self,
        program: str | stim.Circuit,
        qubit_manager: QubitManager,
        *,
        readout_protocol: str | None = None,
        code_distance: int | None = None,
    ) -> YbCircuit:
        """Insert noise into *program* and return the noisy circuit.

        Close every moment of *program* with a ``TICK``, the final one
        included: idling noise is attributed per moment, and a moment with no
        ``TICK`` to close it is reported as under-noised rather than silently
        accepted.

        Args:
            program: Stim circuit text or a pre-built ``stim.Circuit``.
            qubit_manager: Which atom and encoding each qubit is.
            readout_protocol: How to model mid-circuit measurement. ``None``
                measures in place; otherwise one of ``"in_place_direct"``,
                ``"transport"`` or ``"shelving"`` (the last needs every
                unmeasured qubit to be 171Yb-g).
            code_distance: Readout-zone distance in code sites. Required for
                ``readout_protocol="transport"`` and ignored otherwise.

        Returns:
            A circuit ready for ``detector_error_model`` and sampling.

        Raises:
            ValueError: If *program* yields no replayable operations, if
                *readout_protocol* is not one of the values above, or if
                ``code_distance`` is missing for the transport protocol.
        """
        circuit = self.noiseless_circuit(
            program,
            qubit_manager,
            track_operations=True,
        )
        op_log = circuit.get_operation_log()
        if not op_log:
            raise ValueError(
                "The supplied Stim circuit produced no replayable operations for the Yb noise model."
            )

        if readout_protocol is not None:
            op_log = _replace_measurement_with_selective(
                op_log, readout_protocol, code_distance
            )

        adapter = build_yb_noise_model(self.noise_params, qubit_manager)
        return adapter.apply(op_log)


def _replace_measurement_with_selective(
    op_log: list[OperationRecord],
    protocol: str,
    code_distance: int | None = None,
) -> list[OperationRecord]:
    """Replace ``measurement`` records in *op_log* with ``selective_measurement``.

    The *unmeasured_qubits* for each measurement are inferred from the
    ``idling`` record that immediately follows it in the operation log
    (the ideal circuit always pairs M with an idling on the remaining qubits).
    The paired idling record is removed because ``selective_measurement``
    handles idling internally.

    For the ``"transport"`` protocol, the subsequent ``reset`` record (and its
    paired idling) for the same qubit set is also removed from the log and its
    pattern is forwarded as ``reset_pattern`` so that
    ``_selective_measurement_transport`` can perform the reset *before* the
    return transport — matching the physical sequence where initialisation
    happens in the readout zone.
    """
    from .circuit import OperationRecord

    new_log: list[OperationRecord] = []
    skip_next_idling = False
    # Indices of reset (and paired idling) records to skip for transport.
    skip_indices: set[int] = set()

    # --- First pass (transport only): find resets that pair with measurements ---
    reset_pattern_map: dict[int, str | None] = {}  # measurement index -> pattern
    if protocol == "transport":
        for i, record in enumerate(op_log):
            if record.kind != "measurement":
                continue
            measured_set = frozenset(record.qubits)
            # Scan forward for the next reset on the same qubit set.
            for j in range(i + 1, len(op_log)):
                if (
                    op_log[j].kind == "reset"
                    and frozenset(op_log[j].qubits) == measured_set
                ):
                    reset_pattern_map[i] = op_log[j].params.get("pattern", "b")
                    skip_indices.add(j)
                    # Also skip the idling that immediately follows the reset.
                    for k in range(j + 1, len(op_log)):
                        if op_log[k].kind == "idling":
                            skip_indices.add(k)
                            break
                        elif op_log[k].kind not in ("raw_instruction",):
                            break
                    break
                # Stop searching if we hit another measurement (next round).
                if op_log[j].kind == "measurement":
                    break

    # --- Second pass: build the new log ---
    for i, record in enumerate(op_log):
        if i in skip_indices:
            continue

        if skip_next_idling and record.kind == "idling":
            skip_next_idling = False
            continue

        if record.kind == "measurement":
            # Find the immediately following idling to get unmeasured_qubits
            unmeasured_qubits: list[int] = []
            for j in range(i + 1, len(op_log)):
                if op_log[j].kind == "idling":
                    unmeasured_qubits = list(op_log[j].qubits)
                    skip_next_idling = True
                    break
                elif op_log[j].kind not in ("raw_instruction",):
                    break

            params = {
                **record.params,
                "unmeasured_qubits": unmeasured_qubits,
                "protocol": protocol,
                "code_distance": code_distance,
            }
            if protocol == "transport":
                params["reset_pattern"] = reset_pattern_map.get(i)

            new_record = OperationRecord(
                kind="selective_measurement",
                qubits=record.qubits,
                params=params,
            )
            new_log.append(new_record)
        else:
            new_log.append(record)

    return new_log
