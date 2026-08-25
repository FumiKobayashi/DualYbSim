"""Circuit Builders for Dual Yb Quantum Devices

Implements YbGateBuilder class for constructing quantum circuits with
automatic noise application based on atomic species and qubit types.
"""

import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import stim

from .params import NoiseModelParameters
from .qubits import QubitManager


@dataclass(frozen=True)
class OperationRecord:
    """Keeps track of logical operations for later noise replay."""

    kind: str
    qubits: list[int]
    params: dict[str, Any] = field(default_factory=dict)

    def replay(self, circuit: "YbCircuit") -> None:
        """Re-execute this operation on *circuit*, with its noise settings."""
        circuit._execute_operation_from_record(self)


@dataclass(frozen=True)
class BufferedStimOp:
    """Lightweight description of a Stim instruction within a moment."""

    instruction: stim.CircuitInstruction
    name: str
    qubits: list[int]
    kind: str


@dataclass(frozen=True)
class BufferedMoment:
    """Stim operations grouped between consecutive TICK instructions."""

    operations: list[BufferedStimOp]
    tick_instruction: stim.CircuitInstruction | None = None


class YbCircuit(stim.Circuit):
    """Dual Yb量子デバイス用の回路構築クラス"""

    _SINGLE_QUBIT_GATES = {
        "X",
        "Y",
        "Z",
        "H",
        "S",
        "S_DAG",
        "SQRT_X",
        "SQRT_X_DAG",
        "SQRT_Y",
        "SQRT_Y_DAG",
    }
    # MX, MY, MR, MRX, MRY, MRZ are not supported in YbCircuit
    # because noise insertion is not supported for these gates.
    _MEASUREMENT_GATES = {
        "M",
    }
    _RESET_GATES = {"R"}
    _IDLING_GATES = {
        "I",
    }
    _PASSTHROUGH_OPS = {
        "TICK",
        "SHIFT_COORDS",
        "QUBIT_COORDS",
    }

    def __init__(
        self,
        qubit_manager: QubitManager,
        noise_params: NoiseModelParameters | None = None,
        *,
        noise_enabled: bool = True,
        track_operations: bool = True,
    ):
        """YbGateBuilderの初期化

        Args:
            qubit_manager: QubitManagerインスタンス
            noise_params: ノイズパラメータ (Noneの場合はデフォルト値を使用)
            noise_enabled: Trueの場合は操作ごとにノイズを即時付与
            track_operations: Trueの場合は操作ログを保持しwith_noiseで再利用
        """
        super().__init__()
        self.qubit_manager = qubit_manager
        self.noise_params = noise_params or NoiseModelParameters()
        self.noise_enabled = noise_enabled
        self._track_operations = track_operations
        self._operation_log: list[OperationRecord] = []
        # Measurement bookkeeping used when importing Stim circuits and replaying noise.
        self._next_measurement_id: int = 0
        self._original_index_to_measurement_id: dict[int, int] = {}
        self._measurement_id_to_original_index: dict[int, int] = {}
        self._measurement_id_to_current_rec_index: dict[int, int] = {}
        self._track_measurements = False

    # ------------------------------------------------------------------------
    # Logical measurement bookkeeping utilities
    # ------------------------------------------------------------------------

    def _allocate_measurement_ids(
        self,
        *,
        count: int,
        original_start_index: int,
        qubits: list[int],
        gate: str,
    ) -> list[int]:
        measurement_ids: list[int] = []
        for offset in range(count):
            measurement_id = self._next_measurement_id
            self._next_measurement_id += 1
            original_index = original_start_index + offset
            self._original_index_to_measurement_id[original_index] = measurement_id
            self._measurement_id_to_original_index[measurement_id] = original_index
            measurement_ids.append(measurement_id)
        return measurement_ids

    def _register_measurement_results(
        self,
        measurement_ids: list[int] | None,
        start_rec_index: int,
    ) -> None:
        if not measurement_ids:
            return
        for offset, measurement_id in enumerate(measurement_ids):
            self._measurement_id_to_current_rec_index[measurement_id] = (
                start_rec_index + offset
            )

    @staticmethod
    def _normalize_qubits(qubit: int | list[int]) -> list[int]:
        if isinstance(qubit, int):
            return [qubit]
        return list(qubit)

    def _record_operation(self, kind: str, qubits: list[int], **params: Any) -> None:
        if not self._track_operations:
            return
        self._operation_log.append(OperationRecord(kind, list(qubits), dict(params)))

    @contextmanager
    def _suspend_tracking(self) -> Any:
        prev = self._track_operations
        self._track_operations = False
        try:
            yield
        finally:
            self._track_operations = prev

    def _execute_operation_from_record(self, record: OperationRecord) -> None:
        with self._suspend_tracking():
            kind = record.kind
            params = record.params
            if kind == "idling":
                self.idling(record.qubits, params["duration"])
            elif kind == "transport_with_time":
                self.transport_with_time(record.qubits, params["transport_time"])
            elif kind == "single_qubit_gate":
                self.single_qubit_gate(params["gate"], record.qubits)
            elif kind == "feedforward_gate":
                ff_pairs = params.get("ff_pairs")
                if ff_pairs:
                    # Rebuild the CZ instruction with corrected rec[] indices
                    current_meas = self.num_measurements
                    cz_targets = []
                    for pair in ff_pairs:
                        measurement_id = pair.get("measurement_id")
                        abs_index = pair.get("abs_index")
                        if (
                            measurement_id is not None
                            and measurement_id
                            in self._measurement_id_to_current_rec_index
                        ):
                            rec_idx = self._measurement_id_to_current_rec_index[
                                measurement_id
                            ]
                        elif abs_index is not None:
                            fallback_mid = self._original_index_to_measurement_id.get(
                                abs_index
                            )
                            if (
                                fallback_mid is not None
                                and fallback_mid
                                in self._measurement_id_to_current_rec_index
                            ):
                                rec_idx = self._measurement_id_to_current_rec_index[
                                    fallback_mid
                                ]
                            else:
                                rec_idx = abs_index
                        else:
                            raise ValueError(
                                "Feedforward pair missing measurement reference."
                            )
                        lookback = rec_idx - current_meas
                        cz_targets.append(stim.target_rec(lookback))
                        if pair.get("qubit") is not None:
                            cz_targets.append(pair["qubit"])
                    self.append("CZ", cz_targets)
                else:
                    instruction = params["instruction"]
                    self._append_stim_instruction(instruction)
                if self.noise_enabled:
                    self._apply_single_qubit_noise(record.qubits)
            elif kind == "two_qubit_gate":
                self.two_qubit_gate(params["gate"], record.qubits)
            elif kind == "reset":
                self.reset_qubit(record.qubits, pattern=params.get("pattern", "b"))
            elif kind == "measurement":
                self.measurement(
                    record.qubits,
                    gate=params.get("gate", "M"),
                    measurement_ids=params.get("measurement_ids"),
                )
            elif kind == "detector":
                self._append_detector(
                    params["targets"],
                    coords=params.get("coords"),
                )
            elif kind == "observable":
                self._append_observable(
                    params["targets"],
                    gate_args=params.get("gate_args"),
                )
            elif kind == "handover":
                self.handover(record.qubits)
            elif kind == "shelve":
                self.shelve(record.qubits)
            elif kind == "unshelve":
                self.unshelve(record.qubits)
            elif kind == "selective_measurement":
                self.selective_measurement(
                    record.qubits,
                    params["unmeasured_qubits"],
                    protocol=params["protocol"],
                    code_distance=params.get("code_distance"),
                    gate=params.get("gate", "M"),
                    measurement_ids=params.get("measurement_ids"),
                    reset_pattern=params.get("reset_pattern"),
                )
            elif kind == "raw_instruction":
                instruction = params["instruction"]
                self._append_stim_instruction(instruction)
            else:
                raise ValueError(f"Unsupported operation kind: {kind}")

    def get_operation_log(self) -> list[OperationRecord]:
        """Return a copy of the recorded ideal operations, in order."""
        return list(self._operation_log)

    @classmethod
    def _iter_expanded_stim_instructions(
        cls, source: stim.Circuit
    ) -> Iterator[stim.CircuitInstruction]:
        """Yield Stim instructions with REPEAT blocks fully expanded."""
        for instruction in source:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                yield from cls._expand_stim_repeat_block(instruction)
            else:
                yield instruction

    @classmethod
    def _expand_stim_repeat_block(
        cls, block: stim.CircuitRepeatBlock
    ) -> Iterator[stim.CircuitInstruction]:
        """Recursively expand a Stim repeat block into flat instructions."""
        repeat_count = block.repeat_count
        if repeat_count < 0:
            raise ValueError("Stim repeat count must be non-negative.")
        if repeat_count == 0:
            return

        instructions = list(block.body_copy())
        for _ in range(repeat_count):
            for instruction in instructions:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    yield from cls._expand_stim_repeat_block(instruction)
                else:
                    yield instruction

    def _buffer_instruction(
        self, instruction: stim.CircuitInstruction
    ) -> BufferedStimOp:
        """Convert a Stim instruction into buffered metadata for moment processing."""
        name = instruction.name

        if name in self._SINGLE_QUBIT_GATES:
            qubits = self._extract_qubits(instruction)
            kind = "single_qubit_gate"
        elif name == "CZ":
            if self._is_feedforward_cz(instruction):
                qubits = self._extract_feedforward_qubit_targets(instruction)
                kind = "feedforward_gate"
            else:
                qubits = self._extract_qubits(instruction)
                kind = "two_qubit_gate"
        elif name in self._MEASUREMENT_GATES:
            qubits = self._extract_qubits(instruction)
            kind = "measurement"
        elif name in self._RESET_GATES:
            qubits = self._extract_qubits(instruction)
            kind = "reset"
        elif name in self._IDLING_GATES:
            qubits = self._extract_qubits(instruction)
            kind = "idling"
        else:
            qubits = []
            if name in self._PASSTHROUGH_OPS and name != "TICK":
                kind = "passthrough"
            elif name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
                kind = "annotation"
            else:
                kind = "raw"

        return BufferedStimOp(
            instruction=instruction, name=name, qubits=qubits, kind=kind
        )

    def _buffer_stim_moments(self, source: stim.Circuit) -> Iterator[BufferedMoment]:
        """Group expanded Stim instructions into moments delimited by TICK."""
        current_ops: list[BufferedStimOp] = []

        for instruction in self._iter_expanded_stim_instructions(source):
            if instruction.name == "TICK":
                yield BufferedMoment(
                    operations=list(current_ops), tick_instruction=instruction
                )
                current_ops = []
                continue

            current_ops.append(self._buffer_instruction(instruction))

        if current_ops:
            yield BufferedMoment(operations=list(current_ops))

    def _lookup_gate_time_for_keys(self, qubit: int, keys: list[str]) -> float:
        """Return the first available gate-time entry for the given qubit."""
        try:
            isotope, qubit_type = self.qubit_manager.get_qubit_type(qubit)
        except ValueError:
            return 0.0

        try:
            table = self.noise_params.for_qubit(isotope, qubit_type).gate_time
        except ValueError:
            return 0.0

        for key in keys:
            if key in table:
                return float(table[key])
        return 0.0

    def _estimate_two_qubit_pair_duration(self, control: int, target: int) -> float:
        """Estimate gate time for a two-qubit pair."""
        return max(
            self._lookup_gate_time_for_keys(control, ["t_2Q", "t_1Q_gm"]),
            self._lookup_gate_time_for_keys(target, ["t_2Q", "t_1Q_gm"]),
        )

    def _estimate_operation_duration(self, op: BufferedStimOp) -> float:
        """Estimate the hardware duration associated with a buffered operation."""
        if not op.qubits:
            if op.kind == "idling":
                args = op.instruction.gate_args_copy()
                return float(args[0]) if args else 0.0
            return 0.0

        if op.kind == "single_qubit_gate":
            durations = [
                self._lookup_gate_time_for_keys(q, ["t_1Q", "t_1Q_gm"])
                for q in op.qubits
            ]
            return max(durations, default=0.0)

        if op.kind == "feedforward_gate":
            durations = [
                self._lookup_gate_time_for_keys(q, ["t_1Q", "t_1Q_gm"])
                for q in op.qubits
            ]
            return max(durations, default=0.0)

        if op.kind == "two_qubit_gate":
            pair_durations: list[float] = []
            qubits = op.qubits
            for idx in range(0, len(qubits), 2):
                if idx + 1 >= len(qubits):
                    break
                pair_durations.append(
                    self._estimate_two_qubit_pair_duration(qubits[idx], qubits[idx + 1])
                )
            return max(pair_durations, default=0.0)

        if op.kind == "measurement":
            durations = [
                self._lookup_gate_time_for_keys(q, ["t_read"]) for q in op.qubits
            ]
            return max(durations, default=0.0)

        if op.kind == "reset":
            durations = [
                self._lookup_gate_time_for_keys(q, ["t_reset"]) for q in op.qubits
            ]
            return max(durations, default=0.0)

        if op.kind == "idling":
            args = op.instruction.gate_args_copy()
            return float(args[0]) if args else 0.0

        return 0.0

    def _compute_moment_duration(self, moment: BufferedMoment) -> float:
        """Compute the duration of a Stim moment using noise parameters."""
        durations = [self._estimate_operation_duration(op) for op in moment.operations]
        return max(durations, default=0.0)

    def _collect_active_qubits(self, moment: BufferedMoment) -> set[int]:
        """Collect qubits that participate in operations for the moment."""
        active: set[int] = set()
        for op in moment.operations:
            if op.qubits and op.kind in {
                "single_qubit_gate",
                "feedforward_gate",
                "two_qubit_gate",
                "measurement",
                "reset",
                "idling",
            }:
                active.update(op.qubits)
        return active

    def _replay_buffered_moment(
        self,
        moment: BufferedMoment,
        measurement_counter: int,
        system_qubits: set[int],
    ) -> int:
        """Replay buffered Stim instructions and optional trailing TICK."""
        moment_duration = self._compute_moment_duration(moment)
        active_qubits = self._collect_active_qubits(moment)

        for op in moment.operations:
            measurement_counter = self._ingest_stim_instruction(
                op.instruction, measurement_counter
            )

        # A moment's spectator idling is inserted only when a TICK closes it, since
        # the TICK is what marks the moment as over. `_buffer_stim_moments` can only
        # ever yield one moment without one -- the last -- and dropping its idling
        # would make the circuit optimistically noisy, so say so instead.
        if moment_duration > 0:
            idle_qubits = system_qubits - active_qubits
            if moment.tick_instruction is None:
                if idle_qubits:
                    warnings.warn(
                        "The last moment of the circuit is not terminated by a TICK, "
                        "so no idling noise was inserted on qubits "
                        f"{sorted(idle_qubits)} for the {moment_duration:g} s it "
                        "lasts. Append a TICK after every moment, the final one "
                        "included: without it the circuit is under-noised, and a "
                        "readout_protocol cannot tell which qubits sat out the "
                        "measurement.",
                        UserWarning,
                        stacklevel=2,
                    )
            elif idle_qubits:
                self.idling(sorted(idle_qubits), moment_duration)

        if moment.tick_instruction is not None:
            self._append_raw_instruction(moment.tick_instruction)

        return measurement_counter

    @classmethod
    def from_stim(
        cls,
        stim_circuit: str | stim.Circuit,
        qubit_manager: QubitManager,
        noise_params: NoiseModelParameters | None = None,
        *,
        noise_enabled: bool = False,
        track_operations: bool = True,
    ) -> "YbCircuit":
        """Construct a YbCircuit from a noiseless Stim circuit description.

        Args:
            stim_circuit: Stim Circuit text or a pre-built `stim.Circuit`.
            qubit_manager: Metadata manager for qubit isotopes and roles.
            noise_params: Optional noise configuration to attach to the circuit.
            noise_enabled: Whether to enable noise while replaying operations.
            track_operations: Whether to capture an operation log for later noise injection.
        """
        source = (
            stim_circuit
            if isinstance(stim_circuit, stim.Circuit)
            else stim.Circuit(stim_circuit)
        )
        circuit = cls(
            qubit_manager,
            noise_params=noise_params,
            noise_enabled=noise_enabled,
            track_operations=track_operations,
        )
        circuit._track_measurements = True
        system_qubits = set(qubit_manager.get_all_qubits())
        measurement_counter = 0
        for moment in circuit._buffer_stim_moments(source):
            measurement_counter = circuit._replay_buffered_moment(
                moment, measurement_counter, system_qubits
            )
        return circuit

    def with_noise(
        self,
        noise_params: NoiseModelParameters | None = None,
    ) -> "YbCircuit":
        """Rebuild a noisy circuit from the stored operation log."""
        if not self._operation_log:
            raise ValueError(
                "Operation log is empty. Build the circuit before calling with_noise()."
            )
        params = noise_params or self.noise_params
        from .model import build_yb_noise_model

        adapter = build_yb_noise_model(params, self.qubit_manager)
        return adapter.apply(self.get_operation_log())

    def _ingest_stim_instruction(
        self, instruction: stim.CircuitInstruction, measurement_counter: int
    ) -> int:
        """Replay a Stim instruction as a high-level YbCircuit operation."""
        name = instruction.name
        produced = 0
        if name in self._SINGLE_QUBIT_GATES:
            qubits = self._extract_qubits(instruction)
            self._validate_qubits(qubits, name)
            self.single_qubit_gate(name, qubits)
        elif name == "CZ":
            if self._is_feedforward_cz(instruction):
                qubits = self._extract_feedforward_qubit_targets(instruction)
                self._validate_qubits(qubits, name)
                with self._suspend_tracking():
                    self._append_stim_instruction(instruction)
                if self.noise_enabled:
                    self._apply_single_qubit_noise(qubits)
                # Build a measurement-id–aware description of the
                # feedforward CZ so that replay can reconstruct
                # correct rec[] indices after noise insertion changes
                # the measurement count.
                ff_pairs = self._extract_feedforward_pairs(
                    instruction, measurement_counter
                )
                self._record_operation(
                    "feedforward_gate",
                    qubits,
                    instruction=instruction,
                    ff_pairs=ff_pairs,
                )
            else:
                qubits = self._extract_qubits(instruction)
                if len(qubits) % 2 != 0:
                    raise ValueError(
                        f"Expected an even number of qubit targets for CZ, got {qubits}."
                    )
                for k in range(0, len(qubits), 2):
                    pair = qubits[k : k + 2]
                    self._validate_qubits(pair, name)
                    self.two_qubit_gate("CZ", pair)
        elif name in self._MEASUREMENT_GATES:
            qubits = self._extract_qubits(instruction)
            self._validate_qubits(qubits, name)
            measurement_ids = None
            if self._track_measurements:
                measurement_ids = self._allocate_measurement_ids(
                    count=len(qubits),
                    original_start_index=measurement_counter,
                    qubits=qubits,
                    gate=name,
                )
            self.measurement(
                qubits,
                gate=name,
                measurement_ids=measurement_ids,
            )
            produced = len(qubits)
        elif name in self._RESET_GATES:
            qubits = self._extract_qubits(instruction)
            self._validate_qubits(qubits, name)
            self.reset_qubit(qubits)
        elif name in self._PASSTHROUGH_OPS:
            if name == "I":
                qubits = self._extract_qubits(instruction)
                if qubits:
                    self._validate_qubits(qubits, name)
            self._append_raw_instruction(instruction)
        elif name == "DETECTOR":
            targets = self._parse_detector_targets(instruction, measurement_counter)
            coords = list(instruction.gate_args_copy())
            self._record_operation(
                "detector",
                [],
                targets=targets,
                coords=coords,
            )
            with self._suspend_tracking():
                self.append("DETECTOR", instruction.targets_copy(), coords)
        elif name == "OBSERVABLE_INCLUDE":
            targets = self._parse_detector_targets(instruction, measurement_counter)
            gate_args = list(instruction.gate_args_copy())
            self._record_operation(
                "observable",
                [],
                targets=targets,
                gate_args=gate_args,
            )
            with self._suspend_tracking():
                self.append(
                    "OBSERVABLE_INCLUDE",
                    instruction.targets_copy(),
                    gate_args,
                )
        else:
            raise ValueError(
                f"Unsupported Stim instruction '{name}' in YbCircuit.from_stim."
            )
        return measurement_counter + produced

    def _ingest_stim_repeat_block(
        self, block: stim.CircuitRepeatBlock, measurement_counter: int
    ) -> int:
        """Replay a Stim repeat block by expanding its body."""
        repeat_count = block.repeat_count
        if repeat_count < 0:
            raise ValueError("Stim repeat count must be non-negative.")
        if repeat_count == 0:
            return measurement_counter

        instructions = list(block.body_copy())
        for _ in range(repeat_count):
            for instruction in instructions:
                if isinstance(instruction, stim.CircuitRepeatBlock):
                    measurement_counter = self._ingest_stim_repeat_block(
                        instruction, measurement_counter
                    )
                else:
                    measurement_counter = self._ingest_stim_instruction(
                        instruction, measurement_counter
                    )
        return measurement_counter

    @staticmethod
    def _extract_qubits(instruction: stim.CircuitInstruction) -> list[int]:
        qubits: list[int] = []
        for target in instruction.targets_copy():
            if target.is_combiner:
                continue
            if not target.is_qubit_target:
                raise ValueError(
                    f"Instruction {instruction} references non-qubit targets, which are unsupported."
                )
            qubits.append(target.value)
        return qubits

    @staticmethod
    def _is_feedforward_cz(instruction: stim.CircuitInstruction) -> bool:
        if instruction.name != "CZ":
            return False
        return any(
            target.is_measurement_record_target for target in instruction.targets_copy()
        )

    @staticmethod
    def _extract_feedforward_qubit_targets(
        instruction: stim.CircuitInstruction,
    ) -> list[int]:
        qubits: list[int] = []
        saw_measurement_record = False
        for target in instruction.targets_copy():
            if target.is_combiner:
                continue
            if target.is_measurement_record_target:
                saw_measurement_record = True
                continue
            if not target.is_qubit_target:
                raise ValueError(
                    f"Instruction {instruction} references unsupported non-qubit targets."
                )
            qubits.append(target.value)
        if not saw_measurement_record:
            raise ValueError(
                f"Instruction {instruction} does not contain measurement-record targets."
            )
        return qubits

    def _extract_feedforward_pairs(
        self,
        instruction: stim.CircuitInstruction,
        measurement_counter: int,
    ) -> list[dict[str, Any]]:
        """Extract (measurement_id_or_abs_index, qubit) pairs from a feedforward CZ.

        Each pair maps a ``rec[-k]`` target to its absolute measurement index
        (and, when tracking is active, the corresponding measurement id) plus
        the qubit it is paired with.  This allows the replay path to
        reconstruct the CZ instruction with updated ``rec[]`` indices.
        """
        pairs: list[dict[str, Any]] = []
        targets = instruction.targets_copy()
        i = 0
        while i < len(targets):
            t = targets[i]
            if t.is_measurement_record_target:
                abs_index = measurement_counter + t.value  # rec[-k] → absolute
                measurement_id = self._original_index_to_measurement_id.get(abs_index)
                # Expect the next non-combiner target to be the qubit
                qubit_target = None
                j = i + 1
                while j < len(targets):
                    if targets[j].is_combiner:
                        j += 1
                        continue
                    if targets[j].is_qubit_target:
                        qubit_target = targets[j].value
                    break
                pairs.append(
                    {
                        "abs_index": abs_index,
                        "measurement_id": measurement_id,
                        "qubit": qubit_target,
                    }
                )
                i = j + 1
            else:
                i += 1
        return pairs

    def _validate_qubits(self, qubits: list[int], op_name: str) -> None:
        if not qubits:
            raise ValueError(
                f"Stim instruction '{op_name}' requires at least one qubit target."
            )
        for qubit in qubits:
            self.qubit_manager.get_qubit_type(qubit)

    def _parse_detector_targets(
        self, instruction: stim.CircuitInstruction, measurement_counter: int
    ) -> list[dict[str, int | None]]:
        text = str(instruction)
        matches = list(re.finditer(r"rec\[(\-?\d+)\]", text))
        match_iter = iter(matches)
        targets_meta: list[dict[str, int | None]] = []
        for target in instruction.targets_copy():
            if target.is_combiner:
                continue
            if not target.is_measurement_record_target:
                raise ValueError(
                    "Only measurement record targets are supported in DETECTOR/OBSERVABLE instructions."
                )
            match = next(match_iter, None)
            if match is None:
                raise ValueError(
                    "Failed to parse measurement record reference from Stim instruction."
                )
            offset = int(match.group(1))
            abs_index = measurement_counter + offset if offset < 0 else offset
            if abs_index < 0:
                raise ValueError(
                    f"Detector references measurement index before start of circuit: rec[{offset}]"
                )
            measurement_id = self._original_index_to_measurement_id.get(abs_index)
            targets_meta.append(
                {
                    "measurement_id": measurement_id,
                    "abs_index": None if measurement_id is not None else abs_index,
                }
            )
        return targets_meta

    def _append_detector(
        self,
        targets: list[dict[str, int | None]],
        *,
        coords: list[float] | None = None,
    ) -> None:
        current_measurements = self.num_measurements
        gate_targets = []
        for spec in targets:
            measurement_id = spec.get("measurement_id")
            abs_index = spec.get("abs_index")
            if measurement_id is not None:
                if measurement_id not in self._measurement_id_to_current_rec_index:
                    raise ValueError(
                        f"Detector references measurement {measurement_id} before it is available."
                    )
                rec_index = self._measurement_id_to_current_rec_index[measurement_id]
            elif abs_index is not None:
                fallback_measurement_id = self._original_index_to_measurement_id.get(
                    abs_index
                )
                if fallback_measurement_id is not None and (
                    fallback_measurement_id in self._measurement_id_to_current_rec_index
                ):
                    rec_index = self._measurement_id_to_current_rec_index[
                        fallback_measurement_id
                    ]
                else:
                    rec_index = abs_index
            else:
                raise ValueError("Detector target specification missing indices.")
            lookback = rec_index - current_measurements
            if lookback >= 0:
                raise ValueError(
                    "Detector references a measurement result that is not yet available "
                    f"(measurement index={rec_index}, produced={current_measurements})."
                )
            gate_targets.append(stim.target_rec(lookback))
        args = coords or []
        with self._suspend_tracking():
            self.append("DETECTOR", gate_targets, args)

    def _append_observable(
        self,
        targets: list[dict[str, int | None]],
        gate_args: list[float] | None = None,
    ) -> None:
        current_measurements = self.num_measurements
        gate_targets = []
        for spec in targets:
            measurement_id = spec.get("measurement_id")
            abs_index = spec.get("abs_index")
            if measurement_id is not None:
                if measurement_id not in self._measurement_id_to_current_rec_index:
                    raise ValueError(
                        f"Observable references measurement {measurement_id} before it is available."
                    )
                rec_index = self._measurement_id_to_current_rec_index[measurement_id]
            elif abs_index is not None:
                fallback_measurement_id = self._original_index_to_measurement_id.get(
                    abs_index
                )
                if fallback_measurement_id is not None and (
                    fallback_measurement_id in self._measurement_id_to_current_rec_index
                ):
                    rec_index = self._measurement_id_to_current_rec_index[
                        fallback_measurement_id
                    ]
                else:
                    rec_index = abs_index
            else:
                raise ValueError("Observable target specification missing indices.")
            lookback = rec_index - current_measurements
            if lookback >= 0:
                raise ValueError(
                    "Observable references a measurement result that is not yet available "
                    f"(measurement index={rec_index}, produced={current_measurements})."
                )
            gate_targets.append(stim.target_rec(lookback))
        args = list(gate_args or [])
        with self._suspend_tracking():
            self.append("OBSERVABLE_INCLUDE", gate_targets, args)

    def _append_raw_instruction(self, instruction: stim.CircuitInstruction) -> None:
        """Append Stim instructions that do not have dedicated helpers."""
        self._append_stim_instruction(instruction)
        self._record_operation("raw_instruction", [], instruction=instruction)

    def _append_stim_instruction(self, instruction: stim.CircuitInstruction) -> None:
        tmp = stim.Circuit()
        tmp.append(
            instruction.name, instruction.targets_copy(), instruction.gate_args_copy()
        )
        with self._suspend_tracking():
            self += tmp

    def idling(self, qubit: int | list[int], duration: float) -> "YbCircuit":
        """Idling操作（原子種に応じてノイズを適用）

        Args:
            qubit: Qubit ID or list of qubit IDs
            duration: Idling時間 (秒)

        Returns:
            YbCircuit: Idling回路
        """
        qubits = self._normalize_qubits(qubit)

        # 理想的なidling (何もしない)
        self.append_operation("I", qubits)

        if self.noise_enabled:
            self._apply_time_dependent_noise(qubits, duration)

        self._record_operation("idling", qubits, duration=duration)

        return self

    def _apply_time_dependent_noise(
        self, qubit: int | list[int], duration: float
    ) -> None:
        """Apply every channel whose probability grows with elapsed time.

        This is the idling block followed by the decay block of the paper's
        channel ordering, i.e.

            LOSS_m . LOSS_g . DECAY_mg . XERR . ZERR

        acting left after right. The Rydberg decay channels are excluded on
        purpose: they only fire while an atom is actually driven to ``|r>``, which
        happens inside a two-qubit gate, so they are applied there instead.

        ``DECAY_mg`` is applied only to encodings that occupy the metastable
        manifold -- the 174Yb clock qubit, where it is amplitude damping inside
        the computational subspace, and the 171Yb m qubit, where it is a loss.
        A 171Yb g qubit leaves the metastable manifold empty while it waits, so
        the channel does not act on it.

        Used by :meth:`idling`, :meth:`transport_with_time` and
        :meth:`handover`, which differ only in what else they emit.
        """
        qubits = self._normalize_qubits(qubit)
        if not qubits or not self.noise_enabled or duration <= 0:
            return

        rate = self.noise_params.get_time_dependent_rate
        for (isotope, qubit_type), grouped in self.qubit_manager.group_qubits_by_type(
            qubits
        ).items():
            view = self.noise_params.for_qubit(isotope, qubit_type)

            if isotope == "174":
                # ZERR_c and DECAY_mg act on the same two levels, so they are
                # twirled together into one channel rather than emitted
                # separately: p_X = p_Y = p_1/4, p_Z = p_2/2 - p_1/4, which
                # reproduces the transverse decay exp(-t/T_2^(c)) exactly.
                mg_rates = self.noise_params.twirled_amplitude_damping(
                    duration, T_1_inv=view.gamma_mg, T_2_inv=view.gamma_Z
                )
                if any(p > 0 for p in mg_rates):
                    self.append("PAULI_CHANNEL_1", grouped, mg_rates)
            else:
                # ZERR then XERR, per the paper's idling ordering.
                z_rate = rate(duration, view.gamma_Z)
                if z_rate > 0:
                    self.append("Z_ERROR", grouped, z_rate)
                x_rate = rate(duration, view.gamma_X)
                if x_rate > 0:
                    self.append("X_ERROR", grouped, x_rate)

                # DECAY_mg, only for the encoding that occupies the metastable
                # manifold. Here m -> g leaves the computational subspace, so it
                # is a loss channel.
                if qubit_type == "m":
                    mg_rate = rate(duration, view.gamma_mg)
                    if mg_rate > 0:
                        self.append("HERALDED_ERASE", grouped, mg_rate)

        self._apply_trap_loss(qubits, duration)

    def _apply_trap_loss(self, qubit: int | list[int], duration: float) -> None:
        """Apply LOSS_g / LOSS_m for an exposure of *duration* seconds.

        These are the finite-trap-lifetime loss channels. The paper classifies
        them as decay channels that stay active during every circuit operation,
        so this is called from every operation that advances time, including
        idling and transport.

        Which rate applies follows the manifold the encoding lives in: the 171Yb
        ground qubit sees ``gamma_gL``, the 171Yb metastable qubit sees
        ``gamma_mL``, and the 174Yb clock qubit straddles both manifolds so it
        sees whichever is larger. With the paper's ``gamma_gL == gamma_mL`` that
        distinction does not bite, but it keeps the behaviour defined if the two
        are ever calibrated apart.
        """
        qubits = self._normalize_qubits(qubit)
        if not qubits or not self.noise_enabled or duration <= 0:
            return

        for (isotope, qubit_type), grouped in self.qubit_manager.group_qubits_by_type(
            qubits
        ).items():
            if isotope == "171" and qubit_type == "g":
                gamma = self.noise_params.gamma_gL
            elif isotope == "171" and qubit_type == "m":
                gamma = self.noise_params.gamma_mL
            else:
                gamma = max(self.noise_params.gamma_gL, self.noise_params.gamma_mL)

            p = self.noise_params.get_time_dependent_rate(duration, gamma)
            if p > 0:
                self.append("HERALDED_ERASE", grouped, p)

        return self  # type: ignore[return-value]

    def transport_with_time(
        self, qubit: int | list[int], transport_time: float
    ) -> "YbCircuit":
        """Transport qubits to the target zone with the given transport time.
        This time EXCLUDES the handover time.

        Args:
            qubit: Qubit ID or list of qubit IDs
            transport_time: Transport time (excluding handover time). The unit of transport time is second.

        Returns:
            YbCircuit: YbCircuit instance
        """
        qubits = self._normalize_qubits(qubit)

        self.append("I", qubits)

        if self.noise_enabled:
            # Transport is bracketed by two handover events, one into the movable
            # trap and one back out, so LOSS^(hand) is applied on either side of
            # the shuttling noise.
            self._apply_handover_loss(qubits)
            self._apply_time_dependent_noise(qubits, transport_time)
            self._apply_handover_loss(qubits)

        self._record_operation(
            "transport_with_time", qubits, transport_time=transport_time
        )

        return self

    def transport(self, qubit: int | list[int], distance: float) -> "YbCircuit":
        """Transport qubits to the target zone with the given distance.

        Args:
            qubit: Qubit ID or list of qubit IDs
            distance: Distance to the target zone. The unit of distance is meter.

        Returns:
            YbCircuit: YbCircuit instance
        """
        transport_time = self.noise_params.transportation_time(distance)
        return self.transport_with_time(qubit, transport_time)

    # ------------------------------------------------------------------
    # Selective measurement protocol helpers
    # ------------------------------------------------------------------

    def handover(self, qubit: int | list[int]) -> "YbCircuit":
        """Apply a single handover event (time + loss) to the given qubits.

        The handover time contributes idling noise; the handover loss is
        applied as ``HERALDED_ERASE`` per qubit-type.
        """
        qubits = self._normalize_qubits(qubit)
        handover_time = self.noise_params.t_hand

        self.append("I", qubits)

        if self.noise_enabled and handover_time > 0:
            self._apply_time_dependent_noise(qubits, handover_time)
            self._apply_handover_loss(qubits)

        self._record_operation("handover", qubits)
        return self

    def _apply_handover_loss(self, qubit: int | list[int]) -> None:
        """Apply LOSS^(hand) for one transfer between static and movable traps.

        The loss probability is per encoding, so the 174Yb clock qubit, the
        171Yb g qubit and the 171Yb m qubit each use their own ``p_hand_*``.
        """
        qubits = self._normalize_qubits(qubit)
        if not qubits or not self.noise_enabled:
            return

        for (isotope, qubit_type), grouped in self.qubit_manager.group_qubits_by_type(
            qubits
        ).items():
            p_hand = self.noise_params.for_qubit(isotope, qubit_type).p_hand
            if p_hand > 0:
                self.append("HERALDED_ERASE", grouped, p_hand)

    def shelve(self, qubit: int | list[int]) -> "YbCircuit":
        """Shelve qubits into a hidden state to protect them during readout.

        For 171g this is modelled as a clock transition (ground -> metastable).
        The noise model reuses ``_apply_clock_excitation_noise``.
        For other qubit types a ``NotImplementedError`` is raised for now.
        """
        qubits = self._normalize_qubits(qubit)

        self.append("I", qubits)

        if self.noise_enabled:
            qubit_groups = self.qubit_manager.group_qubits_by_type(qubits)
            for (isotope, qubit_type), grouped in qubit_groups.items():
                if isotope == "171" and qubit_type == "g":
                    self._apply_clock_excitation_noise(grouped, isotope, qubit_type)
                else:
                    raise NotImplementedError(
                        f"shelve is not yet implemented for ({isotope}, {qubit_type})"
                    )

        self._record_operation("shelve", qubits)
        return self

    def unshelve(self, qubit: int | list[int]) -> "YbCircuit":
        """Unshelve qubits back from the hidden state after readout.

        Symmetric to ``shelve`` -- same noise model applies.
        """
        qubits = self._normalize_qubits(qubit)

        self.append("I", qubits)

        if self.noise_enabled:
            qubit_groups = self.qubit_manager.group_qubits_by_type(qubits)
            for (isotope, qubit_type), grouped in qubit_groups.items():
                if isotope == "171" and qubit_type == "g":
                    self._apply_clock_excitation_noise(grouped, isotope, qubit_type)
                else:
                    raise NotImplementedError(
                        f"unshelve is not yet implemented for ({isotope}, {qubit_type})"
                    )

        self._record_operation("unshelve", qubits)
        return self

    def selective_measurement(
        self,
        measured_qubits: int | list[int],
        unmeasured_qubits: int | list[int],
        *,
        protocol: str = "in_place_direct",
        code_distance: int | None = None,
        gate: str = "M",
        measurement_ids: list[int] | None = None,
        reset_pattern: str | None = None,
    ) -> "YbCircuit":
        """Measure *measured_qubits* while protecting *unmeasured_qubits*.

        Args:
            measured_qubits: Qubits to be read out.
            unmeasured_qubits: Qubits that must remain coherent during readout.
            protocol: One of ``"in_place_direct"``, ``"transport"``, ``"shelving"``.
            code_distance: Required for ``"transport"`` to compute geometry.
            gate: Stim measurement gate name.
            measurement_ids: Optional logical measurement IDs for replay bookkeeping.
            reset_pattern: When set, perform a reset in the readout zone before
                the return transport.  The corresponding next-round reset is
                assumed to have been removed from the operation log.
        """
        m_qubits = self._normalize_qubits(measured_qubits)
        u_qubits = self._normalize_qubits(unmeasured_qubits)

        if protocol == "in_place_direct":
            self._selective_measurement_in_place_direct(
                m_qubits, u_qubits, gate=gate, measurement_ids=measurement_ids
            )
        elif protocol == "transport":
            if code_distance is None:
                raise ValueError("code_distance is required for the transport protocol")
            self._selective_measurement_transport(
                m_qubits,
                u_qubits,
                code_distance=code_distance,
                gate=gate,
                measurement_ids=measurement_ids,
                reset_pattern=reset_pattern,
            )
        elif protocol == "shelving":
            self._selective_measurement_shelving(
                m_qubits, u_qubits, gate=gate, measurement_ids=measurement_ids
            )
        else:
            raise ValueError(f"Unknown selective measurement protocol: {protocol}")

        self._record_operation(
            "selective_measurement",
            m_qubits,
            unmeasured_qubits=list(u_qubits),
            protocol=protocol,
            code_distance=code_distance,
            gate=gate,
            measurement_ids=(
                list(measurement_ids) if measurement_ids is not None else None
            ),
            reset_pattern=reset_pattern,
        )
        return self

    def _selective_measurement_in_place_direct(
        self,
        measured_qubits: list[int],
        unmeasured_qubits: list[int],
        *,
        gate: str = "M",
        measurement_ids: list[int] | None = None,
    ) -> None:
        """In-place direct readout: measure + idle unmeasured qubits."""
        with self._suspend_tracking():
            self.measurement(
                measured_qubits, gate=gate, measurement_ids=measurement_ids
            )
            isotope, qtype = self.qubit_manager.get_qubit_type(measured_qubits[0])
            measure_time = self.noise_params.get_gate_time("t_read", isotope, qtype)
            if unmeasured_qubits:
                self.idling(unmeasured_qubits, measure_time)
                if self.noise_enabled:
                    self._apply_measurement_idling_depol(unmeasured_qubits)

    def _apply_measurement_idling_depol(self, unmeasured_qubits: list[int]) -> None:
        """測定光散乱による未測定171Yb量子ビットへの追加depolarizingエラーを適用。

        In-place measurementで174Ybを測定中、測定されない171Yb原子に
        散乱光によるdepolarizingノイズがかかる効果をモデル化する。
        """
        qubit_groups = self.qubit_manager.group_qubits_by_type(unmeasured_qubits)
        for (isotope, qubit_type), grouped_qubits in qubit_groups.items():
            if isotope == "171" and qubit_type == "m":
                p = self.noise_params.p_depol_meas_idling_m
                if p > 0:
                    self.append("DEPOLARIZE1", grouped_qubits, p)
            elif isotope == "171" and qubit_type == "g":
                p = self.noise_params.p_depol_meas_idling_g
                if p > 0:
                    self.append("DEPOLARIZE1", grouped_qubits, p)

    def _selective_measurement_transport(
        self,
        measured_qubits: list[int],
        unmeasured_qubits: list[int],
        *,
        code_distance: int,
        gate: str = "M",
        measurement_ids: list[int] | None = None,
        reset_pattern: str | None = None,
    ) -> None:
        """Transport-based selective readout with round-trip motion.

        Protocol: handover -> move_out -> handover -> measurement
                  -> [reset] -> handover -> move_back -> handover

        When *reset_pattern* is provided, the qubit is reset in the readout
        zone before the return transport so that transport-back noise
        accumulates on the freshly initialised qubit (matching the physical
        sequence).  The corresponding next-round reset should have been
        removed from the operation log by the caller.
        """
        move_time = self.noise_params.readout_transport_one_way_time(code_distance)
        isotope, qtype = self.qubit_manager.get_qubit_type(measured_qubits[0])
        measure_time = self.noise_params.get_gate_time("t_read", isotope, qtype)

        with self._suspend_tracking():
            # 1. handover (pick up from computation zone)
            self.handover(measured_qubits)
            self.idling(unmeasured_qubits, self.noise_params.t_hand)

            # 2. outbound transport
            self.transport_with_time(measured_qubits, move_time)
            self.idling(unmeasured_qubits, move_time)

            # 3. handover (place in readout zone)
            self.handover(measured_qubits)
            self.idling(unmeasured_qubits, self.noise_params.t_hand)

            # 4. measurement
            self.measurement(
                measured_qubits, gate=gate, measurement_ids=measurement_ids
            )
            self.idling(unmeasured_qubits, measure_time)

            # 4.5 reset in readout zone (before return transport)
            if reset_pattern is not None:
                self.reset_qubit(measured_qubits, pattern=reset_pattern)
                reset_time = self.noise_params.get_gate_time("t_reset", isotope, qtype)
                self.idling(unmeasured_qubits, reset_time)

            # 5. handover (pick up from readout zone)
            self.handover(measured_qubits)
            self.idling(unmeasured_qubits, self.noise_params.t_hand)

            # 6. return transport
            self.transport_with_time(measured_qubits, move_time)
            self.idling(unmeasured_qubits, move_time)

            # 7. handover (place back in computation zone)
            self.handover(measured_qubits)
            self.idling(unmeasured_qubits, self.noise_params.t_hand)

    def _selective_measurement_shelving(
        self,
        measured_qubits: list[int],
        unmeasured_qubits: list[int],
        *,
        gate: str = "M",
        measurement_ids: list[int] | None = None,
    ) -> None:
        """Shelving-based selective readout.

        Protocol: shelve(unmeasured) -> idle(measured, clock_time)
                  -> measurement(measured) -> idle(unmeasured, measure_time)
                  -> unshelve(unmeasured) -> idle(measured, clock_time)
        """
        isotope_m, qtype_m = self.qubit_manager.get_qubit_type(measured_qubits[0])
        measure_time = self.noise_params.get_gate_time("t_read", isotope_m, qtype_m)

        clock_time = self.noise_params.gate_time_g.get("t_1Q_gm", 0)

        with self._suspend_tracking():
            # 1. shelve unmeasured qubits
            self.shelve(unmeasured_qubits)
            if clock_time > 0:
                self.idling(measured_qubits, clock_time)

            # 2. measurement
            self.measurement(
                measured_qubits, gate=gate, measurement_ids=measurement_ids
            )
            self.idling(unmeasured_qubits, measure_time)

            # 3. unshelve unmeasured qubits
            self.unshelve(unmeasured_qubits)
            if clock_time > 0:
                self.idling(measured_qubits, clock_time)

    # ------------------------------------------------------------------

    def reset_qubit(
        self,
        qubit: int | list[int],
        pattern: str | None = "b",
    ) -> "YbCircuit":
        """qubit初期化（原子種に応じてノイズを適用）

        Args:
            qubit: Qubit ID or list of qubit IDs
            pattern: 171Yb m-qubitの初期化パターン ('a' or 'b')

        Returns:
            YbCircuit: 初期化回路
        """
        qubits = self._normalize_qubits(qubit)

        # 理想的な初期化
        self.append("R", qubits)

        if self.noise_enabled:
            # qubitを原子種・タイプでグループ化
            qubit_groups = self.qubit_manager.group_qubits_by_type(qubits)

            # 各グループに対してノイズを適用
            for (isotope, qubit_type), group_qubits in qubit_groups.items():
                self._apply_reset_noise(group_qubits, isotope, qubit_type, pattern)

        self._record_operation("reset", qubits, pattern=pattern)

        return self

    def _apply_reset_noise(
        self,
        qubits: list[int],
        isotope: str,
        qubit_type: str,
        pattern: str | None = "b",
    ) -> None:
        """複数qubitにreset noiseを適用

        Args:
            qubits: 対象qubit IDのリスト
            isotope: 原子種 ('171' or '174')
            qubit_type: Qubitタイプ ('m', 'g', 'gm')
            pattern: 171Yb m-qubitの初期化パターン ('a' or 'b')
        """
        if not qubits or not self.noise_enabled:
            return

        if isotope == "174":
            # 174Yb Reset: LOSS_g^(reset) · DECAY_mg (twirled amplitude damping)
            if self.noise_params.p_g_L_reset_c > 0:
                self.append("HERALDED_ERASE", qubits, self.noise_params.p_g_L_reset_c)

            mg_rates = self.noise_params.twirled_amplitude_damping(
                self.noise_params.gate_time_c["t_reset"],
                T_1_inv=self.noise_params.gamma_mg_c,
            )
            if mg_rates[0] > 0:
                self.append("PAULI_CHANNEL_1", qubits, mg_rates)

        elif isotope == "171" and qubit_type == "m":
            # 171Yb m-qubit Reset: パターン選択
            if pattern == "a":
                self._create_171Yb_m_reset_pattern_a(qubits)
            elif pattern == "b":
                self._create_171Yb_m_reset_pattern_b(qubits)
            else:
                raise ValueError(
                    f"Invalid pattern '{pattern}' for 171Yb m-qubit reset. Use 'a' or 'b'."
                )

        elif isotope == "171" and qubit_type == "g":
            # 171Yb g-qubit Reset: LOSS_g^(reset) · FLIP_g
            if self.noise_params.p_g_L_reset_g > 0:
                self.append("HERALDED_ERASE", qubits, self.noise_params.p_g_L_reset_g)

            if self.noise_params.p_flip_g_g > 0:
                self.append("X_ERROR", qubits, self.noise_params.p_flip_g_g)

        # Trap loss keeps accumulating over the reset window.
        self._apply_trap_loss(
            qubits,
            self.noise_params.for_qubit(isotope, qubit_type).gate_time.get(
                "t_reset", 0.0
            ),
        )

    def _create_171Yb_m_reset_pattern_a(self, qubits: list[int]) -> None:
        """171Yb m-qubit Reset (Pattern A): DEP1_gm · ZERR_m · DECAY_mg"""
        if not qubits or not self.noise_enabled:
            raise ValueError("qubits is empty")
        # XERR during motional reset
        if self.noise_params.p_flip_m_m > 0:
            self.append("X_ERROR", qubits, self.noise_params.p_flip_m_m)
        # DEP1_gm after GTA --> DECAY_mg · ZERR_m
        self._clock_excitation_noise(qubits)

    def _create_171Yb_m_reset_pattern_b(self, qubits: list[int]) -> None:
        """171Yb m-qubit Reset (Pattern B): LOSS_m^(reset)"""
        if not qubits or not self.noise_enabled:
            raise ValueError("qubits is empty")
        # MLOSS^PREP → leakage error
        if self.noise_params.p_m_L_reset_m > 0:
            self.append("HERALDED_ERASE", qubits, self.noise_params.p_m_L_reset_m)

    def measurement(
        self,
        qubit: int | list[int],
        *,
        gate: str = "M",
        measurement_ids: list[int] | None = None,
    ) -> "YbCircuit":
        """qubit測定（原子種に応じてノイズを適用）

        Args:
            qubit: Qubit ID or list of qubit IDs
            gate: Stim measurement instruction to emit. Only ``"M"`` is
                supported; the other ``M*`` gates have no noise model here.
            measurement_ids: Logical ids to bind to the emitted records, so
                that ``rec[...]`` references survive noise insertion. Ids are
                allocated automatically when omitted.

        Returns:
            YbCircuit: 測定回路
        """
        qubits = self._normalize_qubits(qubit)

        if self.noise_enabled:
            # qubitを原子種・タイプでグループ化
            qubit_groups = self.qubit_manager.group_qubits_by_type(qubits)

            # 各グループに対してノイズを適用
            for (isotope, qubit_type), group_qubits in qubit_groups.items():
                self._apply_measurement_noise(group_qubits, isotope, qubit_type)

        before_measure = self.num_measurements
        self.append(gate, qubits)
        self._register_measurement_results(measurement_ids, before_measure)
        self._record_operation(
            "measurement",
            qubits,
            gate=gate,
            measurement_ids=(
                list(measurement_ids) if measurement_ids is not None else None
            ),
        )

        return self

    def _apply_measurement_noise(
        self, qubits: list[int], isotope: str, qubit_type: str
    ) -> None:
        """複数qubitにmeasurement noiseを適用

        Args:
            qubits: 対象qubit IDのリスト
            isotope: 原子種 ('171' or '174')
            qubit_type: Qubitタイプ ('m', 'g', 'gm')
        """
        if not qubits or not self.noise_enabled:
            return

        if isotope == "174":
            # 174Yb Measurement: LOSS_g^(meas) · DECAY_mg · DEP1_c · MERR
            if self.noise_params.p_g_L_meas_c > 0:
                self.append("HERALDED_ERASE", qubits, self.noise_params.p_g_L_meas_c)

            mg_rates = self.noise_params.twirled_amplitude_damping(
                self.noise_params.gate_time_c["t_read"],
                T_1_inv=self.noise_params.gamma_mg_c,
            )
            if mg_rates[0] > 0:
                self.append("PAULI_CHANNEL_1", qubits, mg_rates)

            if self.noise_params.p_1_c > 0:
                self.append("DEPOLARIZE1", qubits, self.noise_params.p_1_c)

            # MERR after GTA
            if self.noise_params.p_meas_c > 0:
                merr_rates = self.noise_params.get_twirled_174_measurement_merr_rates()
                if merr_rates["p_loss"] > 0:
                    self.append("HERALDED_ERASE", qubits, merr_rates["p_loss"])
                if (
                    merr_rates["p_X"] > 0
                    or merr_rates["p_Y"] > 0
                    or merr_rates["p_Z"] > 0
                ):
                    self.append(
                        "PAULI_CHANNEL_1",
                        qubits,
                        [
                            merr_rates["p_X"],
                            merr_rates["p_Y"],
                            merr_rates["p_Z"],
                        ],
                    )

        elif isotope == "171" and qubit_type == "m":
            # 171Yb m-qubit Measurement: 6-input twirled closed form covering
            #   DEP1_gm -> ZERR_m -> DECAY_mg -> U_swap
            #            -> FLIP_g -> LOSS_g^(meas) -> MERR -> U_swap.
            # The closed form absorbs all six channels (including loss and the
            # idling-time ZERR_m / DECAY_mg contributions) into a single
            # HERALDED_ERASE + PAULI_CHANNEL_1 pair; no separate p_GLOSSR_171m
            # injection is needed.
            rates = self.noise_params.get_twirled_171m_measurement_error_rates()
            if rates["p_loss"] > 0:
                self.append("HERALDED_ERASE", qubits, rates["p_loss"])
            if rates["p_X"] > 0 or rates["p_Y"] > 0 or rates["p_Z"] > 0:
                self.append(
                    "PAULI_CHANNEL_1",
                    qubits,
                    [rates["p_X"], rates["p_Y"], rates["p_Z"]],
                )

        elif isotope == "171" and qubit_type == "g":
            # 171Yb g-qubit Measurement: FLIP_g · LOSS_g^(meas) · MERR
            if self.noise_params.p_flip_g_g > 0:
                self.append("X_ERROR", qubits, self.noise_params.p_flip_g_g)

            p_loss = self.noise_params.p_g_L_meas_g + self.noise_params.p_meas_g / 2
            if p_loss > 0:
                self.append("HERALDED_ERASE", qubits, p_loss)

            if self.noise_params.p_meas_g > 0:
                p_pauli = self.noise_params.p_meas_g / 4
                p_loss = self.noise_params.p_meas_g / 4
                # MERR p_Z is second order in p_meas and vanishes when the
                # ambiguous bright-bright record is split evenly:
                #   p_Z = ((2 q_BB - 1)^2 / 16) p_meas^2
                q_bb = 2 * self.noise_params.q_BB_g - 1
                p_z = (q_bb**2 / 16) * self.noise_params.p_meas_g**2
                self.append("PAULI_CHANNEL_1", qubits, [p_pauli, p_pauli, p_z])
                self.append("HERALDED_ERASE", qubits, p_loss)

            # State-selective imaging shelves one nuclear-spin population into
            # the metastable manifold. Population that decays back during the
            # readout window lands in either spin state with equal probability,
            # i.e. a bit flip at half the decay probability.
            p_DECAY_mg = self.noise_params.get_time_dependent_rate(
                self.noise_params.gate_time_g["t_read"],
                self.noise_params.gamma_mg_g,
            )
            if p_DECAY_mg > 0:
                self.append("X_ERROR", qubits, p_DECAY_mg / 2)

        # Trap loss keeps accumulating over the readout window.
        self._apply_trap_loss(
            qubits,
            self.noise_params.for_qubit(isotope, qubit_type).gate_time.get(
                "t_read", 0.0
            ),
        )

    def single_qubit_gate(self, gate: str, qubit: int | list[int]) -> "YbCircuit":
        """1量子ビットゲート（原子種に応じてノイズを適用）

        Args:
            gate: ゲート名 ('X', 'Y', 'Z', 'H', 'S', etc.)
            qubit: Qubit ID or list of qubit IDs

        Returns:
            YbCircuit: 1量子ビットゲート回路
        """
        qubits = self._normalize_qubits(qubit)

        # 理想的なゲート
        gate_upper = gate.upper()
        if gate_upper == "X":
            self.append("X", qubits)
        elif gate_upper == "Y":
            self.append("Y", qubits)
        elif gate_upper == "Z":
            self.append("Z", qubits)
        elif gate_upper == "H":
            self.append("H", qubits)
        elif gate_upper == "S":
            self.append("S", qubits)
        elif gate_upper == "S_DAG":
            self.append("S_DAG", qubits)
        elif gate_upper == "SQRT_X":
            self.append("SQRT_X", qubits)
        elif gate_upper == "SQRT_X_DAG":
            self.append("SQRT_X_DAG", qubits)
        elif gate_upper == "SQRT_Y":
            self.append("SQRT_Y", qubits)
        elif gate_upper == "SQRT_Y_DAG":
            self.append("SQRT_Y_DAG", qubits)
        else:
            raise ValueError(f"Unsupported gate: {gate}")

        if self.noise_enabled:
            self._apply_single_qubit_noise(qubits)

        self._record_operation("single_qubit_gate", qubits, gate=gate.upper())

        return self

    def _apply_single_qubit_noise(self, qubits: list[int]) -> None:
        """Apply the standard single-qubit Yb noise channels to target qubits."""
        qubit_groups = self.qubit_manager.group_qubits_by_type(qubits)

        for (isotope, qubit_type), grouped_qubit in qubit_groups.items():
            if isotope == "174":
                if self.noise_params.p_1_c > 0:
                    self.append("DEPOLARIZE1", grouped_qubit, self.noise_params.p_1_c)

                mg_rates = self.noise_params.twirled_amplitude_damping(
                    self.noise_params.gate_time_c["t_1Q"],
                    T_1_inv=self.noise_params.gamma_mg_c,
                )
                if mg_rates[0] > 0:
                    self.append("PAULI_CHANNEL_1", grouped_qubit, mg_rates)

            elif isotope == "171" and qubit_type == "m":
                if self.noise_params.p_1_m > 0:
                    self.append("DEPOLARIZE1", grouped_qubit, self.noise_params.p_1_m)

                z_rate = self.noise_params.get_time_dependent_rate(
                    self.noise_params.gate_time_m["t_1Q"],
                    self.noise_params.gamma_Z_m,
                )
                if z_rate > 0:
                    self.append("Z_ERROR", grouped_qubit, z_rate)

                if self.noise_params.p_m_g_gate > 0:
                    self.append(
                        "HERALDED_ERASE",
                        grouped_qubit,
                        self.noise_params.p_m_g_gate,
                    )

            elif isotope == "171" and qubit_type == "g":
                if self.noise_params.p_1_g > 0:
                    self.append("DEPOLARIZE1", grouped_qubit, self.noise_params.p_1_g)

            # Trap loss keeps accumulating while the gate runs.
            self._apply_trap_loss(
                grouped_qubit,
                self.noise_params.for_qubit(isotope, qubit_type).gate_time.get(
                    "t_1Q", 0.0
                ),
            )

    def _clock_excitation_noise(self, qubits: int | list[int]) -> None:
        """Apply clock excitation noise"""
        if not qubits:
            raise ValueError("qubits is empty")

        if isinstance(qubits, int):
            qubits = [qubits]

        if not self.noise_enabled:
            return

        # DEP1_gm after strict GPTA --> ZERR_m(p) . loss(2p/3); no X/Y survives
        # (Model_171.Kraus1QClock_171m.DEP1_gm twirled via noise/twirling.py GTA;
        # matches the p_dep_gm marginal of the 6-input closed form in params.py).
        if self.noise_params.p_1_gm > 0:
            self.append("Z_ERROR", qubits, self.noise_params.p_1_gm)
            self.append("HERALDED_ERASE", qubits, (2 / 3) * self.noise_params.p_1_gm)

        grouped_qubits = self.qubit_manager.group_qubits_by_type(qubits)
        for (isotope, qubit_type), grouped_qubit in grouped_qubits.items():
            if isotope == "171" and qubit_type == "m":
                p_loss = self.noise_params.get_time_dependent_rate(
                    self.noise_params.gate_time_m["t_reset"],
                    self.noise_params.gamma_mg_m,
                )
                if p_loss > 0:
                    self.append("HERALDED_ERASE", grouped_qubit, p_loss)

    def two_qubit_gate(self, gate: str, qubit: list[int]) -> "YbCircuit":
        """2量子ビットゲート（CZ）- 原子種に応じてノイズを適用

        Args:
            gate: ゲート名 'CZ'
            qubit: ターゲットqubit IDのリスト.
                The order of the target and control qubits should be as follows:
                [control0, target0, control1, target1, ...] such like Stim's
                implementation.

        Returns:
            YbCircuit: 2量子ビットゲート回路
        """
        # ゲート種類のチェック
        gate_upper = gate.upper()
        if gate_upper != "CZ":
            raise ValueError(f"Unsupported two-qubit gate: {gate}")

        if len(qubit) != len(set(qubit)):
            raise ValueError(
                "targets contains duplicate qubit ids.\nThis function does not support this case. Please implement them separately."
            )

        targets = list(qubit)
        grouped_qubits = self._group_pairwise_qubits_by_type(targets)

        handled = False
        for group_type, qubit_properties in grouped_qubits.items():
            # 原子種の組み合わせに応じたノイズ適用
            if group_type in ["174-174", "171m-171m", "171g-171g"]:
                # 同一原子種間のゲート
                self._apply_same_isotope_noisy_cz(
                    targets,
                    qubit_properties["isotopes"][0],  # type: ignore[call-overload]
                    qubit_properties["qubit_types"][0],  # type: ignore[call-overload]
                )
                handled = True
                break

        if not handled:
            # 異なる原子種間のゲート（Dual Yb）
            self._apply_dual_isotope_noisy_cz(
                targets,
                grouped_qubits,  # type: ignore[arg-type]
            )

        self._record_operation("two_qubit_gate", targets, gate=gate_upper)

        return self

    def _group_pairwise_qubits_by_type(
        self, targets: list[int]
    ) -> dict[str, list[int]]:
        """Group pairwise qubits by type"""
        group_types = [
            "174-174",
            "171m-171m",
            "171g-171g",
            "174-171m",
            "174-171g",
            "171m-171g",
        ]
        grouped_qubits = {  # type: ignore[var-annotated]
            k: {
                "isotopes": [],
                "qubit_types": [],
                "targets": [],
            }
            for k in group_types
        }

        for i in range(0, len(targets), 2):
            control = targets[i]
            isotope_c, qubit_type_c = self.qubit_manager.get_qubit_type(control)
            isotope_c += qubit_type_c
            target = targets[i + 1]
            isotope_t, qubit_type_t = self.qubit_manager.get_qubit_type(target)
            isotope_t += qubit_type_t

            if isotope_c == "174gm" and isotope_t == "174gm":
                grouped_qubits["174-174"]["isotopes"].append(isotope_c[:3])
                grouped_qubits["174-174"]["isotopes"].append(isotope_t[:3])
                grouped_qubits["174-174"]["qubit_types"].append(qubit_type_c)
                grouped_qubits["174-174"]["qubit_types"].append(qubit_type_t)
                grouped_qubits["174-174"]["targets"].append(control)
                grouped_qubits["174-174"]["targets"].append(target)
            elif isotope_c == "171m" and isotope_t == "171m":
                grouped_qubits["171m-171m"]["isotopes"].append(isotope_c[:3])
                grouped_qubits["171m-171m"]["isotopes"].append(isotope_t[:3])
                grouped_qubits["171m-171m"]["qubit_types"].append(qubit_type_c)
                grouped_qubits["171m-171m"]["qubit_types"].append(qubit_type_t)
                grouped_qubits["171m-171m"]["targets"].append(control)
                grouped_qubits["171m-171m"]["targets"].append(target)
            elif isotope_c == "171g" and isotope_t == "171g":
                grouped_qubits["171g-171g"]["isotopes"].append(isotope_c[:3])
                grouped_qubits["171g-171g"]["isotopes"].append(isotope_t[:3])
                grouped_qubits["171g-171g"]["qubit_types"].append(qubit_type_c)
                grouped_qubits["171g-171g"]["qubit_types"].append(qubit_type_t)
                grouped_qubits["171g-171g"]["targets"].append(control)
                grouped_qubits["171g-171g"]["targets"].append(target)
            elif (isotope_c == "174gm" and isotope_t == "171m") or (
                isotope_c == "171m" and isotope_t == "174gm"
            ):
                grouped_qubits["174-171m"]["isotopes"].append(isotope_c[:3])
                grouped_qubits["174-171m"]["isotopes"].append(isotope_t[:3])
                grouped_qubits["174-171m"]["qubit_types"].append(qubit_type_c)
                grouped_qubits["174-171m"]["qubit_types"].append(qubit_type_t)
                grouped_qubits["174-171m"]["targets"].append(control)
                grouped_qubits["174-171m"]["targets"].append(target)
            elif (isotope_c == "174gm" and isotope_t == "171g") or (
                isotope_c == "171g" and isotope_t == "174gm"
            ):
                grouped_qubits["174-171g"]["isotopes"].append(isotope_c[:3])
                grouped_qubits["174-171g"]["isotopes"].append(isotope_t[:3])
                grouped_qubits["174-171g"]["qubit_types"].append(qubit_type_c)
                grouped_qubits["174-171g"]["qubit_types"].append(qubit_type_t)
                grouped_qubits["174-171g"]["targets"].append(control)
                grouped_qubits["174-171g"]["targets"].append(target)
            elif (isotope_c == "171m" and isotope_t == "171g") or (
                isotope_c == "171g" and isotope_t == "171m"
            ):
                grouped_qubits["171m-171g"]["isotopes"].append(isotope_c[:3])
                grouped_qubits["171m-171g"]["isotopes"].append(isotope_t[:3])
                grouped_qubits["171m-171g"]["qubit_types"].append(qubit_type_c)
                grouped_qubits["171m-171g"]["qubit_types"].append(qubit_type_t)
                grouped_qubits["171m-171g"]["targets"].append(control)
                grouped_qubits["171m-171g"]["targets"].append(target)
            else:
                raise ValueError(f"Invalid qubit type: {isotope_c} and {isotope_t}")
        grouped_qubits = {k: v for k, v in grouped_qubits.items() if v["targets"] != []}
        return grouped_qubits  # type: ignore[return-value]

    def _apply_same_isotope_noisy_cz(
        self, targets: list[int], isotope: str, qubit_type: str
    ) -> None:
        """同一原子種間のノイズ適用

        Args:
            targets: ターゲットqubit IDのリスト.
                The order of the target and control qubits should be as follows:
                [control0, target0, control1, target1, ...] such like Stim's
                implementation.
            isotope: 原子種 ('174' or '171')
            qubit_type: Qubitタイプ ('mg', 'm' or 'g')
        """
        if len(targets) != len(set(targets)):
            raise ValueError("targets contains duplicate qubit ids")

        if isotope == "174":
            # 174Yb-174Yb
            # ideal CZ gate
            self.append("CZ", targets)
            if self.noise_enabled:
                # DEP2_c
                if self.noise_params.p_2_c > 0:
                    self.append("DEPOLARIZE2", targets, self.noise_params.p_2_c)
                self._apply_174_two_qubit_noise(targets)

        elif isotope == "171" and qubit_type == "m":
            # 171Yb m-qubit - 171Yb m-qubit
            self.append("CZ", targets)
            if self.noise_enabled:
                # DEP2_m
                if self.noise_params.p_2_m > 0:
                    self.append("DEPOLARIZE2", targets, self.noise_params.p_2_m)
                self._apply_171m_two_qubit_noise(targets)

        elif isotope == "171" and qubit_type == "g":
            # 171Yb g-qubit - 171Yb g-qubit
            if self.noise_enabled:
                # Clock transition前
                self._apply_clock_excitation_noise(targets, isotope, qubit_type)
            self.append("CZ", targets)
            if self.noise_enabled:
                # DEP2 for the g encoding; the gate runs in the metastable
                # manifold, reached by the clock pulses either side.
                if self.noise_params.p_2_g > 0:
                    self.append("DEPOLARIZE2", targets, self.noise_params.p_2_g)
                # 171-171 qubitのnoise
                self._apply_171m_two_qubit_noise(targets)
                # Clock transition後
                self._apply_clock_excitation_noise(targets, isotope, qubit_type)

    def _apply_174_two_qubit_noise(self, qubits: int | list[int]) -> None:
        """Noise for 174Yb during two-qubit gate"""
        if qubits == [] or not self.noise_enabled:
            return
        if isinstance(qubits, int):
            qubits = [qubits]

        t_2Q = self.noise_params.gate_time_c["t_2Q"]
        ryd = self.noise_params.rydberg_branch_rates(t_2Q)

        # Applied in the paper's order: LOSS_r, then DECAY_rg, then DECAY_rm,
        # which is the order the branch renormalisation assumes. For the clock
        # qubit, DECAY_rg maps |1> onto |0> (a bit flip) and DECAY_rm returns the
        # population while destroying its phase (a phase flip).
        if ryd["LOSS_r"] > 0:
            self.append("HERALDED_ERASE", qubits, ryd["LOSS_r"])
        if ryd["DECAY_rg"] > 0:
            self.append("X_ERROR", qubits, ryd["DECAY_rg"])
        if ryd["DECAY_rm"] > 0:
            self.append("Z_ERROR", qubits, ryd["DECAY_rm"])

        # DECAY_mg amplitude damping is expected to be negligible for 2-qubit gate
        # since gate time is expected to be short enough
        mg_rates = self.noise_params.twirled_amplitude_damping(
            t_2Q,
            self.noise_params.gamma_mg_c,
        )
        if 10**-6 < mg_rates[0]:
            self.append("PAULI_CHANNEL_1", qubits, mg_rates)

        self._apply_trap_loss(qubits, t_2Q)

    def _apply_171m_two_qubit_noise(self, qubits: int | list[int]) -> None:
        """Noise for 171Yb m-qubit during two-qubit gate"""
        if qubits == [] or not self.noise_enabled:
            return
        if isinstance(qubits, int):
            qubits = [qubits]

        t_2Q = self.noise_params.gate_time_m["t_2Q"]
        ryd = self.noise_params.rydberg_branch_rates(t_2Q)

        # For the m qubit the ground manifold sits outside the computational
        # subspace, so DECAY_rg is a loss and merges with LOSS_r into a single
        # loss channel. DECAY_rm returns the population but destroys its phase.
        loss_rate = ryd["LOSS_r"] + ryd["DECAY_rg"]
        if loss_rate > 0:
            self.append("HERALDED_ERASE", qubits, loss_rate)
        if ryd["DECAY_rm"] > 0:
            self.append("Z_ERROR", qubits, ryd["DECAY_rm"])

        self._apply_trap_loss(qubits, t_2Q)

    def _apply_dual_isotope_noisy_cz(
        self,
        targets: list[int],
        grouped_qubits: dict[str, dict[str, list[int]]],
    ) -> None:
        """異なる原子種間のノイズ適用（Dual Yb）"""
        # targetsのなかに同じqubit idがあればエラーを返す
        if len(targets) != len(set(targets)):
            raise ValueError("targets contains duplicate qubit ids")

        isotopes = [
            self.qubit_manager.get_qubit_type(target)[0]
            + self.qubit_manager.get_qubit_type(target)[1]
            for target in targets
        ]

        if self.noise_enabled and "171g" in isotopes:
            # Clock excitation noise
            # isotopes is built element-wise from targets, so strict=True holds.
            qubits_171g = [
                target
                for target, isotope in zip(targets, isotopes, strict=True)
                if isotope == "171g"
            ]
            idling_qubits = [
                target
                for target, isotope in zip(targets, isotopes, strict=True)
                if isotope != "171g"
            ]
            self._apply_clock_excitation_noise(qubits_171g, "171", "g")
            if idling_qubits:
                with self._suspend_tracking():
                    self.idling(idling_qubits, self.noise_params.gate_time_g["t_1Q_gm"])

        # ideal CZ gate
        self.append("CZ", targets)

        # 両原子に跨るDEP2^dualエラー
        if self.noise_enabled:
            dual_depolarizing_rate = self.noise_params.p_2_dual
            if dual_depolarizing_rate > 0:
                self.append("DEPOLARIZE2", targets, dual_depolarizing_rate)

            # 各原子に個別のノイズを適用
            self._apply_individual_rydberg_noise(targets)

            # Clock excitation noise
            if "171g" in isotopes:
                self._apply_clock_excitation_noise(qubits_171g, "171", "g")
                if idling_qubits:
                    with self._suspend_tracking():
                        self.idling(
                            idling_qubits,
                            self.noise_params.gate_time_g["t_1Q_gm"],
                        )

    def _apply_individual_rydberg_noise(self, targets: int | list[int]) -> None:
        """個別qubitのノイズ適用"""
        if not self.noise_enabled:
            return
        isotopes = [self.qubit_manager.get_qubit_type(target)[0] for target in targets]  # type: ignore[union-attr]
        # isotopes is built element-wise from targets, so strict=True holds.
        qubits174 = [
            target
            for target, isotope in zip(targets, isotopes, strict=True)  # type: ignore[arg-type]
            if isotope == "174"
        ]
        qubits171 = [
            target
            for target, isotope in zip(targets, isotopes, strict=True)  # type: ignore[arg-type]
            if isotope == "171"
        ]
        grouped_qubits = {
            "174": qubits174,
            "171": qubits171,
        }
        for isotope, grouped_qubit in sorted(
            grouped_qubits.items(), key=lambda x: x[0]
        ):
            if isotope == "174":
                self._apply_174_two_qubit_noise(grouped_qubit)

            elif isotope == "171":
                self._apply_171m_two_qubit_noise(grouped_qubit)

    def _apply_clock_excitation_noise(
        self, qubit: int | list[int], isotope: str, qubit_type: str
    ) -> None:
        """Clock excitationノイズの適用"""
        if isinstance(qubit, int):
            qubit = [qubit]

        if not (isotope == "171" and qubit_type == "g"):
            raise ValueError(
                "Clock excitation noise can only be applied to 171Yb g-qubits"
            )

        # g-dual gateの場合、g↔m変換が必要。DEP1^{gm} (strict GPTA twirl) 部分は
        # _clock_excitation_noise() と共通の実装に委譲する。qubit は全て
        # ("171", "g") なので、そちら側の171m専用ブロックは素通りする。
        self._clock_excitation_noise(qubit)

        # CZ gate中のclock pulse時間 (gate_time_g["t_1Q_gm"]) 分の DECAY_mg
        t_clock = self.noise_params.gate_time_g["t_1Q_gm"]
        p_DECAY_mg_clock = self.noise_params.get_time_dependent_rate(
            t_clock,
            self.noise_params.gamma_mg_g,
        )
        if p_DECAY_mg_clock > 0:
            self.append("X_ERROR", qubit, p_DECAY_mg_clock)
            self.append("HERALDED_ERASE", qubit, p_DECAY_mg_clock)

        # Trap loss over the clock-pulse window.
        self._apply_trap_loss(qubit, t_clock)

    def without_noise(self, removed_MPADs: bool = False) -> "YbCircuit":
        """Remove noise from the circuit"""
        without_noise_circuit = self.copy()
        without_noise_circuit = without_noise_circuit.without_noise()
        if removed_MPADs:
            return self._remove_MPADs(without_noise_circuit)
        return without_noise_circuit

    def _remove_MPADs(self, circuit: "YbCircuit") -> "YbCircuit":
        """Remove MPADs from the circuit"""
        circuit_description = str(circuit)
        # Remove lines that start with "MPAD"
        lines = circuit_description.split("\n")
        filtered_lines = [line for line in lines if not line.strip().startswith("MPAD")]
        circuit_description = "\n".join(filtered_lines)
        circuit = YbCircuit(self.qubit_manager, self.noise_params)
        circuit += stim.Circuit(circuit_description)
        return circuit
