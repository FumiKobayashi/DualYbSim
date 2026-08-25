"""Qubit Manager for Dual Yb Quantum Devices

Manages qubit properties including atomic species (171Yb, 174Yb),
qubit types (m, g, gm), and roles (data, ancilla) in hybrid qubit arrays.
"""

from typing import overload


class QubitManager:
    """Manager for atom isotope and qubit array
    Attributes:
        qubit_registry: Dictionary mapping qubit ID to (isotope, encoding_type)
        data_qubits: List of data qubit IDs
        ancilla_qubits: List of ancilla qubit IDs
        qubit_roles: Dictionary mapping qubit ID to role (data or ancilla)
    """

    def __init__(self):
        """Initialize QubitManager"""
        self.qubit_registry: dict[
            int, tuple[str, str]
        ] = {}  # qubit_id -> (isotope, qubit_type)
        self.data_qubits: list[int] = []  # data qubit IDs
        self.ancilla_qubits: list[int] = []  # ancilla qubit IDs
        self.qubit_roles: dict[int, str] = {}  # qubit_id -> role

    def add_qubit(
        self, qubit_id: int, isotope: str, qubit_type: str, role: str = "data"
    ) -> None:
        """Register qubit

        Args:
            qubit_id: Qubit ID
            isotope: 原子種 ('171' or '174')
            qubit_type: Qubitタイプ ('m', 'g', 'gm')
            role: 役割 ('data' or 'ancilla')
        """
        # 入力値の検証
        if isotope not in ["171", "174"]:
            raise ValueError(f"Invalid isotope '{isotope}'. Must be '171' or '174'.")

        if qubit_type not in ["m", "g", "gm"]:
            raise ValueError(
                f"Invalid qubit_type '{qubit_type}'. Must be 'm', 'g', or 'gm'."
            )

        if role not in ["data", "ancilla"]:
            raise ValueError(f"Invalid role '{role}'. Must be 'data' or 'ancilla'.")

        # 174Ybの場合はgmタイプのみ許可
        if isotope == "174" and qubit_type != "gm":
            raise ValueError(f"174Yb qubits must have type 'gm', got '{qubit_type}'.")

        # 171Ybの場合はmまたはgタイプのみ許可
        if isotope == "171" and qubit_type not in ["m", "g"]:
            raise ValueError(
                f"171Yb qubits must have type 'm' or 'g', got '{qubit_type}'."
            )

        # qubitの登録
        self.qubit_registry[qubit_id] = (isotope, qubit_type)
        self.qubit_roles[qubit_id] = role

        if role == "data":
            if qubit_id not in self.data_qubits:
                self.data_qubits.append(qubit_id)
        else:  # role == 'ancilla'
            if qubit_id not in self.ancilla_qubits:
                self.ancilla_qubits.append(qubit_id)

    @overload
    def get_qubit_type(self, qubit_id: int) -> tuple[str, str]: ...

    @overload
    def get_qubit_type(self, qubit_id: list[int]) -> dict[int, tuple[str, str]]: ...

    def get_qubit_type(
        self, qubit_id: int | list[int]
    ) -> tuple[str, str] | dict[int, tuple[str, str]]:
        """Get atom isotope and encoding_type of qubit

        Args:
            qubit_id: Qubit ID or list of qubit IDs

        Returns:
            Tuple[str, str] or Dict[int, Tuple[str, str]]: (isotope, qubit_type) or dict mapping qubit_id to (isotope, qubit_type)
        """
        if isinstance(qubit_id, int):
            if qubit_id not in self.qubit_registry:
                raise ValueError(f"Qubit {qubit_id} is not registered.")
            return self.qubit_registry[qubit_id]
        else:  # List[int]
            result = {}
            for qid in qubit_id:
                if qid not in self.qubit_registry:
                    raise ValueError(f"Qubit {qid} is not registered.")
                result[qid] = self.qubit_registry[qid]
            return result

    @overload
    def get_qubit_role(self, qubit_id: int) -> str: ...

    @overload
    def get_qubit_role(self, qubit_id: list[int]) -> dict[int, str]: ...

    def get_qubit_role(self, qubit_id: int | list[int]) -> str | dict[int, str]:
        """qubitの役割を取得

        Args:
            qubit_id: Qubit ID or list of qubit IDs

        Returns:
            str or Dict[int, str]: role ('data' or 'ancilla') or dict mapping qubit_id to role
        """
        if isinstance(qubit_id, int):
            if qubit_id not in self.qubit_roles:
                raise ValueError(f"Qubit {qubit_id} is not registered.")
            return self.qubit_roles[qubit_id]
        else:  # List[int]
            result = {}
            for qid in qubit_id:
                if qid not in self.qubit_roles:
                    raise ValueError(f"Qubit {qid} is not registered.")
                result[qid] = self.qubit_roles[qid]
            return result

    def is_dual_gate(self, qubit1: int, qubit2: int) -> bool:
        """異なる原子種間のゲートかどうか判定

        Args:
            qubit1: 第1のqubit ID
            qubit2: 第2のqubit ID

        Returns:
            bool: 異なる原子種間のゲートの場合True
        """
        isotope1, _ = self.get_qubit_type(qubit1)
        isotope2, _ = self.get_qubit_type(qubit2)
        return isotope1 != isotope2

    def get_data_qubits(
        self, isotope: str | None = None, qubit_type: str | None = None
    ) -> list[int]:
        """データqubitのリストを取得

        Args:
            isotope: 指定した原子種のみ取得 (Noneの場合は全て)
            qubit_type: 指定したencoding typeのみ取得 (Noneの場合は全て)

        Returns:
            List[int]: データqubit IDのリスト
        """
        if isotope is None and qubit_type is None:
            return self.data_qubits.copy()

        if isotope is None:
            return [
                qid
                for qid in self.data_qubits
                if self.get_qubit_type(qid)[1] == qubit_type
            ]
        if qubit_type is None:
            return [
                qid
                for qid in self.data_qubits
                if self.get_qubit_type(qid)[0] == isotope
            ]

        return [
            qid
            for qid in self.data_qubits
            if self.get_qubit_type(qid)[0] == isotope
            and self.get_qubit_type(qid)[1] == qubit_type
        ]

    def get_ancilla_qubits(self, isotope: str | None = None) -> list[int]:
        """補助qubitのリストを取得

        Args:
            isotope: 指定した原子種のみ取得 (Noneの場合は全て)

        Returns:
            List[int]: 補助qubit IDのリスト
        """
        if isotope is None:
            return self.ancilla_qubits.copy()

        return [
            qid for qid in self.ancilla_qubits if self.get_qubit_type(qid)[0] == isotope
        ]

    def get_all_qubits(self) -> list[int]:
        """全てのqubitのリストを取得

        Returns:
            List[int]: 全qubit IDのリスト
        """
        return list(self.qubit_registry.keys())

    def get_qubit_count(self) -> dict[str, int]:
        """原子種別のqubit数を取得

        Returns:
            Dict[str, int]: 原子種別のqubit数
        """
        count = {"171": 0, "174": 0}
        for isotope, _ in self.qubit_registry.values():
            count[isotope] += 1
        return count

    def print_qubit_summary(self) -> None:
        """qubit配列の概要を出力"""
        print("Qubit Array Summary:")
        print(f"  Total qubits: {len(self.qubit_registry)}")

        count = self.get_qubit_count()
        print(f"  171Yb qubits: {count['171']}")
        print(f"  174Yb qubits: {count['174']}")

        print(f"  Data qubits: {len(self.data_qubits)}")
        print(f"  Ancilla qubits: {len(self.ancilla_qubits)}")

        print("\nDetailed qubit information:")
        for qid in sorted(self.qubit_registry.keys()):
            isotope, qtype = self.get_qubit_type(qid)
            role = self.get_qubit_role(qid)
            print(f"  Qubit {qid}: {isotope}Yb-{qtype} ({role})")

    @property
    def num_qubits(self) -> int:
        """登録されているqubitの総数"""
        return len(self.qubit_registry)

    def group_qubits_by_type(
        self, qubits: list[int]
    ) -> dict[tuple[str, str], list[int]]:
        """qubitを原子種とタイプでグループ化

        Args:
            qubits: 対象qubit IDのリスト

        Returns:
            Dict[Tuple[str, str], List[int]]: (isotope, qubit_type) -> qubit_list のマッピング
        """
        groups: dict[tuple[str, str], list[int]] = {}
        for qubit in qubits:
            isotope, qubit_type = self.get_qubit_type(qubit)
            key = (isotope, qubit_type)
            if key not in groups:
                groups[key] = []
            groups[key].append(qubit)
        return groups

    def clear(self) -> None:
        """全てのqubit登録をクリア"""
        self.qubit_registry.clear()
        self.data_qubits.clear()
        self.ancilla_qubits.clear()
        self.qubit_roles.clear()
