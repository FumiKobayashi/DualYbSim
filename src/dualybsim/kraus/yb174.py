# author: Toshi Kusano
# date: 2025-07-14
from numpy import array, complexfloating, conj, exp, eye, kron, ndarray, sqrt, zeros
from numpy.typing import NDArray


class Kraus1Q_174:
    """Kraus operators for 1-qubit gate of 174Yb.

    This class provides methods to generate Kraus operators for various error channels during single-qubit operations.

    The basis of density matrix are |0> = |1S0>, |1> = |3P0>, |r>, and |L>.
    The density matrix is a 4x4 matrix, which basis is following:
        |0> = [1,0,0,0]^T
        |1> = [0,1,0,0]^T
        |r> = [0,0,1,0]^T
        |L> = [0,0,0,1]^T.
    """

    def __init__(
        self,
        p_dep1: float,
        T2: float,
        gate_time: float,
        lifetime_gs: float,
        lifetime_es: float,
        leaktime_eg: float,
        lifetime_ryd: float,
        leaktime_ryd_gs: float,
        leaktime_ryd_es: float,
        idling_flag: bool,
    ):
        """Initialize Kraus1Q_174 with noise parameters.

        Parameters
        ----------
        p_dep1 : float
            Depolarization probability for 1-qubit gate.
        T2 : float
            Coherence time (seconds).
        gate_time : float
            Gate time (seconds).
        lifetime_gs : float
            Trap lifetime of the ground state (seconds).
        lifetime_es : float
            Trap lifetime of the excited state (seconds).
        leaktime_eg : float
            Leakage time from the 3P0 state to 1S0 (seconds).
        lifetime_ryd : float
            Radiative lifetime of the rydberg state to leakage (seconds).
        leaktime_ryd_gs : float
            Leakage time from the rydberg state to 1S0 (seconds).
        leaktime_ryd_es : float
            Leakage time from the rydberg state to 3P0 (seconds).
        idling_flag : bool
            Whether to use DEP1_c or ZERR_c for depolarization channel. If True, use ZERR_c, else use DEP1_c.
        """
        # Initialize parameters
        self.p_dep1 = p_dep1
        self.T2 = T2
        self.gate_time = gate_time
        self.lifetime_gs = lifetime_gs
        self.lifetime_es = lifetime_es
        self.leaktime_eg = leaktime_eg
        self.lifetime_ryd = lifetime_ryd
        self.leaktime_ryd_gs = leaktime_ryd_gs
        self.leaktime_ryd_es = leaktime_ryd_es
        self.idling_flag = idling_flag

        # Initialize the noise channels (collection of Kraus operators) for the 1-qubit gate
        self.noise_channels = {
            "DEP1_c": self.DEP1_c(p_dep1, idling_flag),
            "ZERR_c": self.ZERR_c(T2, gate_time, idling_flag),
            "LOSS_g": self.LOSS_g(lifetime_gs, gate_time),
            "LOSS_m": self.LOSS_m(lifetime_es, gate_time),
            "DECAY_mg": self.DECAY_mg(leaktime_eg, gate_time),
            "LOSS_r": self.LOSS_r(lifetime_ryd, gate_time),
            "DECAY_rg": self.DECAY_rg(leaktime_ryd_gs, gate_time),
            "DECAY_rm": self.DECAY_rm(leaktime_ryd_es, gate_time),
        }

        # Note: Each individual channel already satisfies CPTP condition
        # When applying all channels, they are applied sequentially
        # No need for normalized_kraus_operators as each channel is independent

    def CPTP(
        self,
        density_matrix: ndarray,
        channel: str | None = None,
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channels as Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied. Size should be (4, 4) for a 1-qubit system.
        channel : str, optional
            The name of the channel to apply. If None, applies all channels. Valid options are 'DEP1_c', 'ZERR_c', 'LOSS_g', 'LOSS_m', 'DECAY_mg', 'LOSS_r', 'DECAY_rg', 'DECAY_rm'.

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators. Size will be (4, 4) for a 1-qubit system.
        """
        if channel is None:
            # If no channel is specified, apply all channels sequentially
            # Each channel is applied independently and satisfies CPTP condition
            result = density_matrix.copy()
            for kraus_ops in self.noise_channels.values():
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    temp_result = zeros(result.shape, dtype=complex)
                    for kraus_op in kraus_ops:
                        temp_result += kraus_op @ result @ conj(kraus_op.T)
                    result = temp_result
                else:
                    # Single Kraus operator
                    result = kraus_ops @ result @ conj(kraus_ops.T)
            return result

        else:
            # Apply the specified channel
            result = zeros(density_matrix.shape, dtype=density_matrix.dtype)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    for kraus_op in kraus_ops:
                        result += kraus_op @ density_matrix @ conj(kraus_op.T)
                else:
                    # Single Kraus operator
                    result += kraus_ops @ density_matrix @ conj(kraus_ops.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def DEP1_c(self, p: float, idling_flag: bool) -> NDArray[complexfloating]:
        """Depolarizing channel for 1-qubit gate of 174Yb.

        Assuming equal depolarization in all directions, i.e., px = py = pz = p / 3.

        Parameters
        ----------
        p : float
            Depolarization probability in the range [0, 1].
        idling_flag : bool
            If True, use ZERR_c; else use DEP1_c.

        Returns:
        -------
        NDArray[complexfloating]
            4 Kraus operators for the depolarizing channel. Each operator is a 4x4 matrix representing the channel.
        """
        if not (0 <= p <= 1):
            raise ValueError(
                "Depolarization probability p must be in the range [0, 1]."
            )

        p = 0 if idling_flag else p

        px = py = pz = p / 3
        identity = sqrt(1 - p) * eye(4)
        X_err = sqrt(px) * array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        Y_err = sqrt(py) * array(
            [[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        Z_err = sqrt(pz) * array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        return array([identity, X_err, Y_err, Z_err])

    def ZERR_c(
        self,
        T2: float,
        gate_time: float,
        idling_flag: bool,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Z error channel for 1-qubit gate of 174Yb.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2). This assumes the decoherence is Markovian and follows an exponential decay model.

        Parameters
        ----------
        T2 : float
            Coherence time of the qubit in seconds.
        gate_time : float
            Gate time for the operation in seconds.
        idling_flag : bool
            If True, use ZERR_c; else use DEP1_c.
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the Z error channel. Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )
        p = p if idling_flag else 0

        identity = sqrt(1 - p) * eye(4)
        Z_err = sqrt(p) * array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        return array([identity, Z_err])

    def LOSS_g(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Trap loss error channel for 1-qubit gate of 174Yb with ground state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the ground state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides lifetime and idling_time. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the ground-state loss error channel. Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[sqrt(1 - p), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [sqrt(p), 0, 0, 0]])

        return array([identity, leak_err])

    def LOSS_m(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Trap loss error channel for 1-qubit gate of 174Yb with excited state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the excited state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the 3P0 loss error channel. Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, sqrt(1 - p), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, sqrt(p), 0, 0]])

        return array([identity, leak_err])

    def DECAY_mg(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Leakage error channel from 3P0 to 1S0 for 1-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Photon scattering time of the excited state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the leakage error from 3P0 to 1S0. Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, sqrt(1 - p), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, sqrt(p), 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return array([identity, leak_err])

    def LOSS_r(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Radiative loss error channel from rydberg for 1-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the trap loss of rydberg states. Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sqrt(1 - p), 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, sqrt(p), 0]])

        return array([identity, leak_err])

    def DECAY_rg(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to ground state for 1-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Leakage lifetime from the rydberg state to 1S0 in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]  # size = 2
            2 Kraus operators for the leakage error from rydberg to ground state.
            Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sqrt(1 - p), 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, sqrt(p), 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return array([identity, leak_err])

    def DECAY_rm(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to 3P0 for 1-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Leakage lifetime from the rydberg state to 3P0 in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the leakage error from rydberg to 3P0. Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sqrt(1 - p), 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, sqrt(p), 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return array([identity, leak_err])


class Kraus2Q_174174:
    """Kraus operators for 2-qubit gate between 174Yb atoms.

    This class provides methods to generate Kraus operators for various error channels.
    """

    def __init__(
        self,
        p_dep2: float,  # depolarization probability for 2-qubit gate
        T2: float,  # coherence time (seconds)
        gate_time: float,  # gate time (seconds)
        lifetime_gs: float,  # trap lifetime of the ground state (seconds)
        lifetime_es: float,  # trap lifetime of the excited state (seconds)
        leaktime_eg: float,  # leakage time from the 3P0 state to 1S0 (seconds)
        lifetime_ryd: float,  # radiative lifetime of the rydberg state to leakage (seconds)
        leaktime_ryd_gs: float,  # leakage time from the rydberg state to 1S0 (seconds)
        leaktime_ryd_es: float,  # leakage time from the rydberg state to 3P0 (seconds)
        idling_flag: bool,  # whether to use DEP2_c or ZERR_c for depolarization channel, if True, use ZERR_c, else use DEP2_c
    ):
        """Initialize Kraus2Q_174174 with noise parameters.

        Parameters
        ----------
        p_dep2 : float
            Depolarization probability for 2-qubit gate.
        T2 : float
            Coherence time of the clock transition (seconds).
        gate_time : float
            Gate time for the operation (seconds).
        lifetime_gs : float
            Trap lifetime of the ground state (seconds).
        lifetime_es : float
            Trap lifetime of the excited state (seconds).
        leaktime_eg : float
            Leakage time from the 3P0 state to 1S0 (seconds).
        lifetime_ryd : float
            Radiative lifetime of the rydberg state to leakage (seconds).
        leaktime_ryd_gs : float
            Leakage time from the rydberg state to 1S0 (seconds).
        leaktime_ryd_es : float
            Leakage time from the rydberg state to 3P0 (seconds).
        idling_flag : bool
            Whether to use DEP2_c or ZERR_c for depolarization channel. If True, use ZERR_c, else use DEP2_c.
        """
        # Initialize parameters
        self.p_dep2 = p_dep2
        self.T2 = T2
        self.gate_time = gate_time
        self.lifetime_gs = lifetime_gs
        self.lifetime_es = lifetime_es
        self.leaktime_eg = leaktime_eg
        self.lifetime_ryd = lifetime_ryd
        self.leaktime_ryd_gs = leaktime_ryd_gs
        self.leaktime_ryd_es = leaktime_ryd_es
        self.idling_flag = idling_flag

        # Initialize the Kraus operators for the 2-qubit gate
        self.noise_channels = {
            "DEP2_c": self.DEP2_c(p_dep2, idling_flag),
            "ZERR_c": self.ZERR_c(T2, gate_time, idling_flag),
            "LOSS_g": self.LOSS_g(lifetime_gs, gate_time),
            "LOSS_m": self.LOSS_m(lifetime_es, gate_time),
            "DECAY_mg": self.DECAY_mg(leaktime_eg, gate_time),
            "LOSS_r": self.LOSS_r(lifetime_ryd, gate_time),
            "DECAY_rg": self.DECAY_rg(leaktime_ryd_gs, gate_time),
            "DECAY_rm": self.DECAY_rm(leaktime_ryd_es, gate_time),
        }

        # normalize the Kraus operators
        TotalNumOfKraus = sum(1 for ops in self.noise_channels.values())
        self.normalized_kraus_operators = {
            "DEP2_c": self.DEP2_c(p_dep2, idling_flag) / sqrt(TotalNumOfKraus),
            "ZERR_c": self.ZERR_c(T2, gate_time, idling_flag) / sqrt(TotalNumOfKraus),
            "LOSS_g": self.LOSS_g(lifetime_gs, gate_time) / sqrt(TotalNumOfKraus),
            "LOSS_m": self.LOSS_m(lifetime_es, gate_time) / sqrt(TotalNumOfKraus),
            "DECAY_mg": self.DECAY_mg(leaktime_eg, gate_time) / sqrt(TotalNumOfKraus),
            "LOSS_r": self.LOSS_r(lifetime_ryd, gate_time) / sqrt(TotalNumOfKraus),
            "DECAY_rg": self.DECAY_rg(leaktime_ryd_gs, gate_time)
            / sqrt(TotalNumOfKraus),
            "DECAY_rm": self.DECAY_rm(leaktime_ryd_es, gate_time)
            / sqrt(TotalNumOfKraus),
        }

    def CPTP(
        self,
        density_matrix: ndarray,
        channel: str | None = None,
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied. Size should be (16, 16) for a 2-qubit system.
        channel : str, optional
            The name of the channel to apply. If None, applies all channels. Valid options are 'DEP2_c', 'ZERR_c', 'LOSS_g', 'LOSS_m', 'DECAY_mg', 'LOSS_r', 'DECAY_rg', 'DECAY_rm'.

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators. Size will be (16, 16) for a 2-qubit system.
        """
        if channel is None:
            # If no channel is specified, apply all channels
            result = zeros(density_matrix.shape, dtype=complex)
            for kraus_ops in self.normalized_kraus_operators.values():
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    for kraus_op in kraus_ops:
                        result += kraus_op @ density_matrix @ conj(kraus_op.T)
                else:
                    # Single Kraus operator
                    result += kraus_ops @ density_matrix @ conj(kraus_ops.T)
            return result

        else:
            # Apply the specified channel
            result = zeros(density_matrix.shape, dtype=complex)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    for kraus_op in kraus_ops:
                        result += kraus_op @ density_matrix @ conj(kraus_op.T)
                else:
                    # Single Kraus operator
                    result += kraus_ops @ density_matrix @ conj(kraus_ops.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def DEP2_c(
        self,
        p: float,
        idling_flag: bool,
    ) -> NDArray[complexfloating]:
        """Depolarizing channel for 2-qubit gate of 174Yb.

        Assuming equal depolarization in all directions.
        This error channel applied if idling_flag is True.

        Parameters
        ----------
        p : float
            Depolarization probability in the range [0, 1].
        idling_flag : bool
            If True, use ZERR_c; else use DEP2_c.

        Returns:
        -------
        NDArray[complexfloating]
            16 Kraus operators for the depolarizing channel. Each operator is a 16x16 matrix representing the channel.
        """
        if not (0 <= p <= 1):
            raise ValueError(
                "Depolarization probability p must be in the range [0, 1]."
            )

        p = 0 if idling_flag else p

        p_each = p / 15
        identity = eye(4)

        X = array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        Y = array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        Z = array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        out = []
        paulis = [identity, X, Y, Z]
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0:
                    out.append(sqrt(1 - p) * kron(identity, identity))
                else:
                    out.append(sqrt(p_each) * kron(paulis[i], paulis[j]))

        return array(out)

    def ZERR_c(
        self,
        T2: float,
        gate_time: float,
        idling_flag: bool,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Z error channel for 2-qubit gate of 174Yb.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2).
        This assumes the decoherence is Markovian and follows an exponential decay model.
        This error channel applied if idling_flag is True.

        Parameters
        ----------
        T2 : float
            Coherence time of the qubit in seconds.
        gate_time : float
            Gate time for the operation in seconds.
        idling_flag : bool
            If True, use ZERR_c; else use DEP2_c.
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            4 Kraus operators for the Z error channel. Each operator is a 16x16 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )
        p = p if idling_flag else 0

        identity = sqrt(1 - p) * eye(4)
        Z_err = sqrt(p) * array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, Z_err),
                kron(Z_err, identity),
                kron(Z_err, Z_err),
            ]
        )

    def LOSS_g(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Trap loss error channel for 2-qubit gate of 174Yb with ground state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the ground state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides lifetime and idling_time. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            4 Kraus operators for the ground-state loss error channel. Each operator is a 16x16 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[sqrt(1 - p), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [sqrt(p), 0, 0, 0]])

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err),
                kron(leak_err, identity),
                kron(leak_err, leak_err),
            ]
        )

    def LOSS_m(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Trap loss error channel for 2-qubit gate of 174Yb with excited state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the excited state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            4 Kraus operators for the 3P0 loss error channel. Each operator is a 16x16 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, sqrt(1 - p), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, sqrt(p), 0, 0]])

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err),
                kron(leak_err, identity),
                kron(leak_err, leak_err),
            ]
        )

    def DECAY_mg(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Leakage error channel from 3P0 to 1S0 for 2-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Photon scattering time of the excited state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            4 Kraus operators for the leakage error from 3P0 to 1S0. Each operator is a 16x16 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, sqrt(1 - p), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, sqrt(p), 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err),
                kron(leak_err, identity),
                kron(leak_err, leak_err),
            ]
        )

    def LOSS_r(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Radiative loss error channel from rydberg for 2-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            4 Kraus operators for the trap loss of rydberg states. Each operator is a 16x16 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sqrt(1 - p), 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, sqrt(p), 0]])

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err),
                kron(leak_err, identity),
                kron(leak_err, leak_err),
            ]
        )

    def DECAY_rg(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to ground state for 2-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to ground state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            4 Kraus operators for the leakage error from rydberg to ground state. Each operator is a 16x16 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sqrt(1 - p), 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, sqrt(p), 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err),
                kron(leak_err, identity),
                kron(leak_err, leak_err),
            ]
        )

    def DECAY_rm(
        self,
        lifetime: float,
        gate_time: float,
        p: float | None = None,
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to 3P0 for 2-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to metastable state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly. Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            4 Kraus operators for the leakage error from rydberg to 3P0. Each operator is a 16x16 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sqrt(1 - p), 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, sqrt(p), 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err),
                kron(leak_err, identity),
                kron(leak_err, leak_err),
            ]
        )


class KrausRESET_174:
    """Kraus operators for |0> state preparation of 174Yb.

    This class provides methods to generate Kraus operators for various error channels during the reset operation.
    """

    def __init__(self, p_loss: float):
        """Initialize KrausRESET_174 with atomic loss probability.

        Parameters
        ----------
        p_loss : float
            Atomic loss probability after reset.
        """
        # Initialize parameters
        self.p_loss = p_loss

        # Initialize the Kraus operators for the reset operation to |0> state of 174Yb
        self.noise_channels = {"LOSS_g_reset": self.LOSS_g_reset(p_loss)}

        # normalize the Kraus operators
        TotalNumOfKraus = sum(
            1 for ops in self.noise_channels.values()
        )  # total number of sub-channels is 1
        self.normalized_kraus_operators = {
            "LOSS_g_reset": self.LOSS_g_reset(p_loss) / sqrt(TotalNumOfKraus)
        }

    def CPTP(
        self,
        density_matrix: ndarray,
        channel: str | None = None,
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied. Size should be (4, 4) for a 1-qubit system.
        channel : str, optional
            The name of the channel to apply. If None, applies all channels. Valid option is 'LOSS_g_reset'. (Note: This is a single channel for reset operation.)

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators. Size will be (4, 4) for a 1-qubit system.
        """
        if channel is None:
            # If no channel is specified, apply all channels
            result = zeros(density_matrix.shape, dtype=complex)
            for kraus_ops in self.normalized_kraus_operators.values():
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    for kraus_op in kraus_ops:
                        result += kraus_op @ density_matrix @ conj(kraus_op.T)
                else:
                    # Single Kraus operator
                    result += kraus_ops @ density_matrix @ conj(kraus_ops.T)
            return result

        else:
            # Apply the specified channel
            result = zeros(density_matrix.shape, dtype=complex)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    for kraus_op in kraus_ops:
                        result += kraus_op @ density_matrix @ conj(kraus_op.T)
                else:
                    # Single Kraus operator
                    result += kraus_ops @ density_matrix @ conj(kraus_ops.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def LOSS_g_reset(self, p_loss: float) -> NDArray[complexfloating]:
        """Ground state loss error channel for |0> state preparation of 174Yb.

        The probability of loss is given by p_loss.

        Parameters
        ----------
        p_loss : float
            Atomic loss probability after reset in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the ground state loss error channel. Each operator is a 4x4 matrix representing the channel.
        """
        if not (0 <= p_loss <= 1):
            raise ValueError(
                "Ground state loss probability p_loss must be in the range [0, 1]."
            )
        identity = array(
            [[sqrt(1 - p_loss), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        leak_err = array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [sqrt(p_loss), 0, 0, 0]]
        )
        return array([identity, leak_err])


class KrausMEASURE_DISC_174:
    """Kraus operators for measurement of 174Yb.

    This class provides the discrimination error of measurement operation.
    The measurement error is modeled as a Kraus channel with a single Kraus operator.
    The measurement error probability is given by p_meas.
    """

    def __init__(self, p_meas: float, q: float = 1.0):
        """Initialize KrausMEASURE_DISC_174 with measurement error probability.

        Parameters
        ----------
        p_meas : float
            Measurement error probability.
        q : float
            Probability of assigning the ambiguous BD outcome to g instead of m.
        """
        # Initialize parameters
        self.p_meas = p_meas
        self.q = q

        # Initialize the Kraus operators for the measurement operation of 174Yb
        self.noise_channels = {"MERR": self.MERR(p_meas, q)}

        # normalize the Kraus operators
        TotalNumOfKraus = sum(
            1 for ops in self.noise_channels.values()
        )  # total number of sub-channels is 1
        self.normalized_kraus_operators = {
            "MERR": self.MERR(p_meas, q) / sqrt(TotalNumOfKraus)
        }

    def CPTP(
        self,
        density_matrix: ndarray,
        channel: str | None = None,
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied. Size should be (4, 4) for a 1-qubit system.
        channel : str, optional
            The name of the channel to apply. If None, applies all channels. Valid option is 'MERR'. (Note: This is a single channel for measurement operation.)

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators. Size will be (4, 4) for a 1-qubit system.
        """
        if channel is None:
            # If no channel is specified, apply all channels
            result = zeros(density_matrix.shape, dtype=complex)
            for kraus_ops in self.normalized_kraus_operators.values():
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    for kraus_op in kraus_ops:
                        result += kraus_op @ density_matrix @ conj(kraus_op.T)
                else:
                    # Single Kraus operator
                    result += kraus_ops @ density_matrix @ conj(kraus_ops.T)
            return result

        else:
            # Apply the specified channel
            result = zeros(density_matrix.shape, dtype=complex)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    for kraus_op in kraus_ops:
                        result += kraus_op @ density_matrix @ conj(kraus_op.T)
                else:
                    # Single Kraus operator
                    result += kraus_ops @ density_matrix @ conj(kraus_ops.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def MERR(self, p: float, q: float | None = None) -> NDArray[complexfloating]:
        """Measurement error channel for 174Yb.

        The probability of measurement error is given by p.

        Parameters
        ----------
        p : float
            Measurement error probability in the range [0, 1].
        q : float, optional
            Probability of assigning the ambiguous BD outcome to g.

        Returns:
        -------
        NDArray[complexfloating]
            9 Kraus operators for the measurement error channel. Each operator is a 4x4 matrix representing the channel.
        """
        if not (0 <= p <= 1):
            raise ValueError(
                "Measurement error probability p_meas must be in the range [0, 1]."
            )
        if q is None:
            q = self.q
        if not (0 <= q <= 1):
            raise ValueError("BD assignment probability q must be in the range [0, 1].")

        p_g_as_g = (1 - p) ** 2 + q * (1 - p) * p
        p_g_as_m = p**2 + (1 - q) * (1 - p) * p
        p_g_as_l = p * (1 - p)

        p_m_as_g = p**2 + q * (1 - p) * p
        p_m_as_m = (1 - p) ** 2 + (1 - q) * (1 - p) * p
        p_m_as_l = (1 - p) * p

        p_l_as_g = p * (1 - p) + q * p**2
        p_l_as_m = (1 - p) * p + (1 - q) * p**2
        p_l_as_l = (1 - p) ** 2

        g_as_m = array(
            [[0, 0, 0, 0], [sqrt(p_g_as_m), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        g_as_l = array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [sqrt(p_g_as_l), 0, 0, 0]]
        )

        m_as_g = array(
            [[0, sqrt(p_m_as_g), 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        m_as_l = array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, sqrt(p_m_as_l), 0, 0]]
        )

        l_as_g = array(
            [[0, 0, 0, sqrt(p_l_as_g)], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        l_as_m = array(
            [[0, 0, 0, 0], [0, 0, 0, sqrt(p_l_as_m)], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        # we treat |r> as |L>.
        r_as_g = array(
            [[0, 0, sqrt(p_l_as_g), 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        r_as_m = array(
            [[0, 0, 0, 0], [0, 0, sqrt(p_l_as_m), 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        identity = array(
            [
                [sqrt(p_g_as_g), 0, 0, 0],
                [0, sqrt(p_m_as_m), 0, 0],
                [0, 0, sqrt(p_l_as_l), 0],
                [0, 0, 0, sqrt(p_l_as_l)],
            ]
        )

        return array(
            [identity, g_as_m, g_as_l, m_as_g, m_as_l, l_as_g, l_as_m, r_as_g, r_as_m]
        )


class KrausMEASURE_174:
    """Kraus operators for measurement of 174Yb.

    This class provides the measurement operation without discrimination error.
    The measurement operation is modeled as a Kraus channel with a single Kraus operator.
    """

    def __init__(
        self,
        p_loss: float,  # atomic loss probability during the measurement operation
        gate_time: float,  # gate time for the measurement operation (seconds)
        lifetime_es: float,  # Trap lifetime of the excited state (3P0) in seconds
        leaktime_eg: float,  # Leakage lifetime from 3P0 to 1S0 in seconds
        lifetime_ryd: float,  # Radiative lifetime of the rydberg state to leakage in seconds
        leaktime_ryd_gs: float,  # Leakage lifetime from rydberg to ground state in seconds
        leaktime_ryd_es: float,  # Leakage lifetime from rydberg to excited state in seconds
    ):
        """Initialize the Kraus operators for the measurement operation of 174Yb.

        Parameters
        ----------
        p_loss : float
            Atomic loss probability during the measurement operation.
        gate_time : float
            Gate time for the measurement operation in seconds.
        lifetime_es : float
            Trap lifetime of the excited state (3P0) in seconds.
        leaktime_eg : float
            Leakage lifetime from 3P0 to 1S0 in seconds.
        lifetime_ryd : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        leaktime_ryd_gs : float
            Leakage lifetime from rydberg to ground state in seconds.
        leaktime_ryd_es : float
            Leakage lifetime from rydberg to excited state in seconds.
        """
        # Initialize the Kraus operators for the measurement operation of 174Yb
        self.noise_channels = {
            "LOSS_g_meas": self.LOSS_g_meas(p_loss),
            "LOSS_m": self.LOSS_m(lifetime_es, gate_time),
            "DECAY_mg": self.DECAY_mg(leaktime_eg, gate_time),
            "LOSS_r": self.LOSS_r(lifetime_ryd, gate_time),
            "DECAY_rg": self.DECAY_rg(leaktime_ryd_gs, gate_time),
            "DECAY_rm": self.DECAY_rm(leaktime_ryd_es, gate_time),
        }

        # normalize the Kraus operators
        TotalNumOfKraus = sum(
            1 for ops in self.noise_channels.values()
        )  # total number of sub-channels is 6
        self.normalized_kraus_operators = {
            "LOSS_g_meas": self.LOSS_g_meas(p_loss) / sqrt(TotalNumOfKraus),
            "LOSS_m": self.LOSS_m(lifetime_es, gate_time) / sqrt(TotalNumOfKraus),
            "DECAY_mg": self.DECAY_mg(leaktime_eg, gate_time) / sqrt(TotalNumOfKraus),
            "LOSS_r": self.LOSS_r(lifetime_ryd, gate_time) / sqrt(TotalNumOfKraus),
            "DECAY_rg": self.DECAY_rg(leaktime_ryd_gs, gate_time)
            / sqrt(TotalNumOfKraus),
            "DECAY_rm": self.DECAY_rm(leaktime_ryd_es, gate_time)
            / sqrt(TotalNumOfKraus),
        }

    def CPTP(
        self,
        density_matrix: ndarray,  # input density matrix
        channel: str
        | None = None,  # channel name, e.g., 'LOSS_g_meas', 'LOSS_m', 'DECAY_mg', 'LOSS_r', 'DECAY_rg', 'DECAY_rm'
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied.
            Size should be (4, 4) for a 1-qubit system.
        channel : Optional[str], optional
            The name of the channel to apply. If None, applies all channels.
            Valid options are 'LOSS_g_meas', 'LOSS_m', 'DECAY_mg', 'LOSS_r', 'DECAY_rg', 'DECAY_rm'.

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators.
            Size will be (4, 4) for a 1-qubit system.
        """
        if channel is None:
            # If no channel is specified, apply all channels
            result = zeros(density_matrix.shape, dtype=complex)
            for kraus_ops in self.normalized_kraus_operators.values():
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    for kraus_op in kraus_ops:
                        result += kraus_op @ density_matrix @ conj(kraus_op.T)
                else:
                    # Single Kraus operator
                    result += kraus_ops @ density_matrix @ conj(kraus_ops.T)
            return result

        else:
            # Apply the specified channel
            result = zeros(density_matrix.shape, dtype=complex)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                if isinstance(kraus_ops, ndarray) and kraus_ops.ndim == 3:
                    # Array of Kraus operators
                    for kraus_op in kraus_ops:
                        result += kraus_op @ density_matrix @ conj(kraus_op.T)
                else:
                    # Single Kraus operator
                    result += kraus_ops @ density_matrix @ conj(kraus_ops.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def LOSS_g_meas(
        self, p_loss: float
    ) -> NDArray[
        complexfloating
    ]:  # atomic loss probability during the measurement operation
        """Ground state loss error channel for measurement operation of 174Yb.

        The probability of loss is given by p_loss.

        Parameters
        ----------
        p_loss : float
            Atomic loss probability during the measurement operation in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the ground state loss error channel.
            Each operator is a 4x4 matrix representing the channel.
        """
        if not (0 <= p_loss <= 1):
            raise ValueError(
                "Ground state loss probability p_loss must be in the range [0, 1]."
            )

        identity = array(
            [[sqrt(1 - p_loss), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        leak_err = array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [sqrt(p_loss), 0, 0, 0]]
        )

        return array([identity, leak_err])

    def LOSS_m(
        self,
        lifetime: float,  # trap lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Trap loss error channel for 1-qubit gate of 174Yb with excited state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the excited state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : Optional[float], optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the 3P0 loss error channel.
            Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, sqrt(1 - p), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, sqrt(p), 0, 0]])

        return array([identity, leak_err])

    def DECAY_mg(
        self,
        lifetime: float,  # 3P0 lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from 3P0 to 1S0 for 1-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Photon scattering time of the excited state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : Optional[float], optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the leakage error from 3P0 to 1S0.
            Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, sqrt(1 - p), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, sqrt(p), 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return array([identity, leak_err])

    def LOSS_r(
        self,
        lifetime: float,  # radiative lifetime of the rydberg state to leakage in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Radiative loss error channel from rydberg for 1-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : Optional[float], optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the trap loss of rydberg states.
            Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sqrt(1 - p), 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, sqrt(p), 0]])

        return array([identity, leak_err])

    def DECAY_rg(
        self,
        lifetime: float,  # leakage lifetime from the rydberg state to 1S0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to ground state for 1-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to ground state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : Optional[float], optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the leakage error from rydberg to ground state.
            Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sqrt(1 - p), 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, sqrt(p), 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return array([identity, leak_err])

    def DECAY_rm(
        self,
        lifetime: float,  # leakage lifetime from the rydberg state to 3P0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to 3P0 for 1-qubit gate of 174Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to metastable state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : Optional[float], optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the leakage error from rydberg to 3P0.
            Each operator is a 4x4 matrix representing the channel.
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sqrt(1 - p), 0], [0, 0, 0, 1]]
        )
        leak_err = array([[0, 0, 0, 0], [0, 0, sqrt(p), 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        return array([identity, leak_err])
