# author: Toshi Kusano
# date: 2025-07-17
from numpy import array, complexfloating, conj, exp, eye, kron, ndarray, sqrt, zeros
from numpy.typing import NDArray


class Kraus1QClock_171m:
    """Kraus operators for clock excitation for metastable qubit of 171Yb.

    This class provides methods to generate Kraus operators for various error channels.
    The basis of the density matrix are defined as follows:
    |0g> = [1,0,0,0,0,0]^T,
    |1g> = [0,1,0,0,0,0]^T,
    |0m> = [0,0,1,0,0,0]^T,
    |1m> = [0,0,0,1,0,0]^T,
    |r> = [0,0,0,0,1,0]^T,
    |L> = [0,0,0,0,0,1]^T
    """

    def __init__(
        self,
        p_dep1: float,  # depolarization probability for 1-qubit gate
        gate_time: float,  # gate time (seconds)
        lifetime_gs: float,  # trap lifetime of the ground state (seconds)
        lifetime_es: float,  # trap lifetime of the excited state (seconds)
        leaktime_eg: float,  # leakage time from the 3P0 state to 1S0 (seconds)
        lifetime_ryd: float,  # radiative lifetime of the rydberg state to leakage (seconds)
        leaktime_ryd_gs: float,  # leakage time from the rydberg state to 1S0 (seconds)
        leaktime_ryd_es: float,  # leakage time from the rydberg state to 3P0 (seconds)
    ):
        """Initialize Kraus1QClock_171m with noise parameters.

        Parameters
        ----------
        p_dep1 : float
            Depolarization probability for 1-qubit gate.
        gate_time : float
            Gate time in seconds.
        lifetime_gs : float
            Trap lifetime of the ground state in seconds.
        lifetime_es : float
            Trap lifetime of the excited state in seconds.
        leaktime_eg : float
            Leakage time from the 3P0 state to 1S0 in seconds.
        lifetime_ryd : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        leaktime_ryd_gs : float
            Leakage time from the rydberg state to 1S0 in seconds.
        leaktime_ryd_es : float
            Leakage time from the rydberg state to 3P0 in seconds.
        """
        # Initialize parameters
        self.p_dep1 = p_dep1
        self.gate_time = gate_time
        self.lifetime_gs = lifetime_gs
        self.lifetime_es = lifetime_es
        self.leaktime_eg = leaktime_eg
        self.lifetime_ryd = lifetime_ryd
        self.leaktime_ryd_gs = leaktime_ryd_gs
        self.leaktime_ryd_es = leaktime_ryd_es

        # Initialize the noise channels (collection of Kraus operators) for the 1-qubit gate
        self.noise_channels = {
            "DEP1_gm": self.DEP1_gm(p_dep1),
            "LOSS_g": self.LOSS_g(lifetime_gs, gate_time),
            "LOSS_m": self.LOSS_m(lifetime_es, gate_time),
            "DECAY_mg": self.DECAY_mg(leaktime_eg, gate_time),
            "LOSS_r": self.LOSS_r(lifetime_ryd, gate_time),
            "DECAY_rg": self.DECAY_rg(leaktime_ryd_gs, gate_time),
            "DECAY_rm": self.DECAY_rm(leaktime_ryd_es, gate_time),
        }

    def CPTP(
        self,
        density_matrix: ndarray,  # input density matrix
        channel: str
        | None = None,  # channel name, e.g., 'DEP1_gm', 'ZERR', 'LOSS_g', etc.
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied.
            Size should be (6, 6) for a 1-qubit system. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        channel : Optional[str], optional
            The name of the channel to apply. If None, applies all channels.
            Valid options are 'DEP1_gm', 'LOSS_g', 'LOSS_m', 'DECAY_mg', 'LOSS_r', 'DECAY_rg', 'DECAY_rm'.

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators.
            Size will be (6, 6) for a 1-qubit system.
        """
        if channel is None:
            """If no channel is specified, apply all channels"""
            # apply all channels in the order they were added (the order of CPTP maps did not matter)
            result = zeros(density_matrix.shape, dtype=complex)
            tmp_result = zeros(
                density_matrix.shape, dtype=complex
            )  # Initialize tmp_result
            for idx, kraus_ops in enumerate(self.noise_channels.values()):
                tmp_rho = (
                    density_matrix if idx == 0 else tmp_result
                )  # use the result of the previous kraus as input for the next kraus
                tmp_result = zeros(
                    tmp_rho.shape, dtype=tmp_rho.dtype
                )  # temporary result for the current channel
                # apply the Kraus operator for each channel
                for kraus_op in kraus_ops:
                    tmp_result += kraus_op @ tmp_rho @ conj(kraus_op.T)

            result = tmp_result
            return result

        else:
            """Apply the specified channel"""
            result = zeros(density_matrix.shape, dtype=density_matrix.dtype)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                for kraus_op in kraus_ops:
                    result += kraus_op @ density_matrix @ conj(kraus_op.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def DEP1_gm(
        self,
        p: float,  # depolarization probability for the clock excitation
    ) -> NDArray[complexfloating]:
        """Depolarizing channel during the clock excitation for m-qubits of 171Yb.

        Assuming equal depolarization in all directions with probability p.

        Parameters
        ----------
        p : float
            Depolarization probability in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            13 Kraus operators for the depolarizing channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        if not (0 <= p <= 1):
            raise ValueError(
                "Depolarization probability p must be in the range [0, 1]."
            )

        px = py = pz = p / 3
        """ qubit spanned by |0g>,|1g>,|0m>,|1m> + rest unchanged """
        X_err00 = sqrt(px / 2) * array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Y_err00 = sqrt(py / 2) * array(
            [
                [0, 0, -1j, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1j, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Z_err00 = sqrt(pz / 2) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        """ qubit spanned by |0g> and |1m>"""
        X_err01 = sqrt(px / 2) * array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Y_err01 = sqrt(py / 2) * array(
            [
                [0, 0, 0, -1j, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1j, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Z_err01 = sqrt(pz / 2) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        """ qubit spanned by |1g> and |0m>"""
        X_err10 = sqrt(px / 2) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Y_err10 = sqrt(py / 2) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, -1j, 0, 0, 0],
                [0, 1j, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Z_err10 = sqrt(pz / 2) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        """ qubit spanned by |1g> and |1m>"""
        X_err11 = sqrt(px / 2) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Y_err11 = sqrt(py / 2) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 1j, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Z_err11 = sqrt(pz / 2) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        out = array(
            [
                X_err00,
                Y_err00,
                Z_err00,
                X_err01,
                Y_err01,
                Z_err01,
                X_err10,
                Y_err10,
                Z_err10,
                X_err11,
                Y_err11,
                Z_err11,
            ]
        )
        identity = sqrt(eye(6) - sum(conj(op).T @ op for op in out))
        out = array([identity, *out])

        return out

    def LOSS_g(
        self,
        lifetime: float,  # trap lifetime of the ground state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Trap loss error channel during clock excitation for m-qubit of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the ground state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : Optional[float], optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            3 Kraus operators for the ground-state loss error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [sqrt(1 - p), 0, 0, 0, 0, 0],
                [0, sqrt(1 - p), 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [sqrt(p), 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, sqrt(p), 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def LOSS_m(
        self,
        lifetime: float,  # trap lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Trap loss error channel during clock excitation for m-qubit of 171Yb.

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
            3 Kraus operators for the 3P0 loss error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p), 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p), 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def DECAY_mg(
        self,
        lifetime: float,  # 3P0 lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from 3P0 to 1S0 during clock excitation.

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
            5 Kraus operators for the leakage error from 3P0 to 1S0.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err00 = array(
            [
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err01 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err10 = array(
            [
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err11 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err00, leak_err01, leak_err10, leak_err11])

    def LOSS_r(
        self,
        lifetime: float,  # radiative lifetime of the rydberg state to leakage in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Radiative loss error channel from rydberg during clock excitation.

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
            2 Kraus operators for the radiative loss of rydberg states.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p), 0],
            ]
        )

        return array([identity, leak_err])

    def DECAY_rg(
        self,
        lifetime: float,  # leakage lifetime from the rydberg state to 1S0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to ground state during clock excitation.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            radiative lifetime of the rydberg state to 1S0 in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : Optional[float], optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            3 Kraus operators for the leakage error from rydberg to ground state.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def DECAY_rm(
        self,
        lifetime: float,  # leakage lifetime from the rydberg state to 3P0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to 3P0 during clock excitation.

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
            3 Kraus operators for the leakage error from rydberg to 3P0.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])


class Kraus1Q_171m:
    """Kraus operators for 1Q gate of the metastable qubit of 171Yb.

    This class provides methods to generate Kraus operators for various error channels.
    """

    def __init__(
        self,
        p_dep1: float,  # depolarization probability for 1-qubit gate
        p_leak: float,  # leakage probability from 3P0 to 1S0 by 1-qubit gate
        T2_g: float,  # coherence time of g-qubit (seconds)
        T1_g: float,  # T1 time of g-qubit (seconds)
        T2_m: float,  # coherence time of m-qubit (seconds)
        T1_m: float,  # T1 time of m-qubit (seconds)
        T2_c: float,  # coherence time of optical qubit (seconds)
        gate_time: float,  # gate time (seconds)
        lifetime_gs: float,  # trap lifetime of the ground state (seconds)
        lifetime_es: float,  # trap lifetime of the excited state (seconds)
        leaktime_eg: float,  # leakage time from the 3P0 state to 1S0 (seconds)
        lifetime_ryd: float,  # radiative lifetime of the rydberg state to L (seconds)
        leaktime_ryd_gs: float,  # leakage time from the rydberg state to 1S0 (seconds)
        leaktime_ryd_es: float,  # leakage time from the rydberg state to 3P0 (seconds)
        idling_flag: bool,  # whether to use DEP1_m or (ZERR_m & XERR_m) for error channel, if True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate)
    ):
        """Initialize Kraus1Q_171m with noise parameters.

        Parameters
        ----------
        p_dep1 : float
            Depolarization probability for 1-qubit gate.
        p_leak : float
            Leakage probability from 3P0 to 1S0 by 1-qubit gate.
        T2_g : float
            Coherence time of g-qubit in seconds.
        T1_g : float
            T1 time of g-qubit in seconds.
        T2_m : float
            Coherence time of m-qubit in seconds.
        T1_m : float
            T1 time of m-qubit in seconds.
        T2_c : float
            Coherence time of optical qubit in seconds.
        gate_time : float
            Gate time in seconds.
        lifetime_gs : float
            Trap lifetime of the ground state in seconds.
        lifetime_es : float
            Trap lifetime of the excited state in seconds.
        leaktime_eg : float
            Leakage time from the 3P0 state to 1S0 in seconds.
        lifetime_ryd : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        leaktime_ryd_gs : float
            Leakage time from the rydberg state to 1S0 in seconds.
        leaktime_ryd_es : float
            Leakage time from the rydberg state to 3P0 in seconds.
        idling_flag : bool
            Whether to use DEP1_m or (ZERR_m & XERR_m) for error channel.
            If True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate).
        """
        # Initialize parameters
        self.p_dep1 = p_dep1
        self.p_leak = p_leak
        self.T2_g = T2_g
        self.T1_g = T1_g
        self.T2_m = T2_m
        self.T1_m = T1_m
        self.T2_c = T2_c
        self.gate_time = gate_time
        self.lifetime_gs = lifetime_gs
        self.lifetime_es = lifetime_es
        self.leaktime_eg = leaktime_eg
        self.lifetime_ryd = lifetime_ryd
        self.leaktime_ryd_gs = leaktime_ryd_gs
        self.leaktime_ryd_es = leaktime_ryd_es
        self.idling_flag = idling_flag

        # Initialize the Kraus operators for the 1-qubit gate
        self.noise_channels = {
            "DEP1_m": self.DEP1_m(p_dep1, idling_flag),
            "DECAY_mg_gate": self.DECAY_mg_gate(p_leak, idling_flag),
            "ZERR_g": self.ZERR_g(T2_g, gate_time),
            "XERR_g": self.XERR_g(T1_g, gate_time),
            "ZERR_m": self.ZERR_m(T2_m, gate_time, idling_flag),
            "XERR_m": self.XERR_m(T1_m, gate_time, idling_flag),
            "ZERR_gm": self.ZERR_gm(T2_c, gate_time),
            "LOSS_g": self.LOSS_g(lifetime_gs, gate_time),
            "LOSS_m": self.LOSS_m(lifetime_es, gate_time),
            "DECAY_mg": self.DECAY_mg(leaktime_eg, gate_time),
            "LOSS_r": self.LOSS_r(lifetime_ryd, gate_time),
            "DECAY_rg": self.DECAY_rg(leaktime_ryd_gs, gate_time),
            "DECAY_rm": self.DECAY_rm(leaktime_ryd_es, gate_time),
        }

    def CPTP(
        self,
        density_matrix: ndarray,  # input density matrix
        channel: str
        | None = None,  # channel name, e.g., 'DEP1_m', 'ZERR', 'LOSS_g', etc.
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied.
            Size should be (6, 6) for a 1-qubit system. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        channel : Optional[str], optional
            The name of the channel to apply. If None, applies all channels.
            Valid options are 'DEP1_m', 'DECAY_mg_gate', 'ZERR_g', 'XERR_g', 'ZERR_m', 'XERR_m', 'ZERR_gm', 'LOSS_g', 'LOSS_m', 'DECAY_mg', 'LOSS_r', 'DECAY_rg', 'DECAY_rm'.

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators.
            Size will be (6, 6) for a 1-qubit system.
        """
        if channel is None:
            """If no channel is specified, apply all channels"""
            # apply all channels in the order they were added (the order of CPTP maps did not matter)
            result = zeros(density_matrix.shape, dtype=complex)
            tmp_result = zeros(
                density_matrix.shape, dtype=complex
            )  # Initialize tmp_result
            for idx, kraus_ops in enumerate(self.noise_channels.values()):
                tmp_rho = (
                    density_matrix if idx == 0 else tmp_result
                )  # use the result of the previous kraus as input for the next kraus
                tmp_result = zeros(
                    tmp_rho.shape, dtype=complex
                )  # temporary result for the current channel
                # apply the Kraus operator for each channel
                for kraus_op in kraus_ops:
                    tmp_result += kraus_op @ tmp_rho @ conj(kraus_op.T)

            result = tmp_result
            return result

        else:
            """Apply the specified channel"""
            result = zeros(density_matrix.shape, dtype=complex)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                for kraus_op in kraus_ops:
                    result += kraus_op @ density_matrix @ conj(kraus_op.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def DEP1_m(
        self,
        p: float,  # depolarization probability for 1Q gate
        idling_flag: bool,  # if True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate)
    ) -> NDArray[complexfloating]:
        """Depolarizing channel during 1Q gate for m-qubits of 171Yb.

        Assuming equal depolarization in all directions with probability p.

        Parameters
        ----------
        p : float
            Depolarization probability in the range [0, 1].
        idling_flag : bool
            Whether to use DEP1_m or (ZERR_m & XERR_m) for error channel.
            If True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate).

        Returns:
        -------
        NDArray[complexfloating]
            4 Kraus operators for the depolarizing channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        if not (0 <= p <= 1):
            raise ValueError(
                "Depolarization probability p must be in the range [0, 1]."
            )

        p = 0 if idling_flag else p

        px = py = pz = p / 3
        """ qubit spanned by |0m> and |1m> + rest unchanged """
        X_err = sqrt(px) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Y_err = sqrt(py) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0],
                [0, 0, 1j, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Z_err = sqrt(pz) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        out = array(
            [
                X_err,
                Y_err,
                Z_err,
            ]
        )
        identity = sqrt(eye(6) - sum(conj(op).T @ op for op in out))
        out = array([identity, *out])

        return out

    def DECAY_mg_gate(
        self,
        p: float,  # depolarization probability for 1Q gate
        idling_flag: bool,  # if True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate)
    ) -> NDArray[complexfloating]:
        """Leakage error channel from 3P0 to 1S0 induced by 1Q gate.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        p : float
            Leakage probability in the range [0, 1].
        idling_flag : bool
            Whether to use DEP1_m or (ZERR_m & XERR_m) for error channel.
            If True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate).

        Returns:
        -------
        NDArray[complexfloating]
            5 Kraus operators for the leakage error from 3P0 to 1S0.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")
        p = 0 if idling_flag else p

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err00 = array(
            [
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err01 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err10 = array(
            [
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err11 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err00, leak_err01, leak_err10, leak_err11])

    def ZERR_g(
        self,
        T2: float,  # coherence time of g-qubit (seconds)
        gate_time: float,  # gate time (seconds)
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """Z error channel for g-qubit during 1Q gate for m-qubit with T2 coherence time and gate time.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2).
        This assumes the decoherence is Marcovian and follows an exponential decay model.

        Parameters
        ----------
        T2 : float
            Coherence time of the g-qubit in seconds.
        gate_time : float
            Gate time for the 1Q gate operation for m-qubit in seconds.
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the Z error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)
        Z_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, Z_err])

    def XERR_g(
        self,
        T1: float,  # T1 time of g-qubit (seconds)
        gate_time: float,  # gate time (seconds)
        p: float | None = None,  # optional, if provided, overrides T1 and gate_time
    ) -> NDArray[complexfloating]:
        """X error channel for g-qubit during 1Q gate for m-qubit with T1 time and gate time.

        The probability of X error is calculated as p = 1 - exp(-gate_time / T1).

        Parameters
        ----------
        T1 : float
            T1 time of the g-qubit in seconds.
        gate_time : float
            Gate time for the 1Q gate operation for m-qubit in seconds.
        p : float, optional
            If provided, overrides the T1 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the X error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / T1) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated X error probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)

        X_err = sqrt(p) * array(
            [
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, X_err])

    def ZERR_m(
        self,
        T2: float,  # coherence time of m-qubit (seconds)
        gate_time: float,  # gate time (seconds)
        idling_flag: bool,  # if True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate)
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """Z error channel for m-qubit during 1Q gate for m-qubit with T2 coherence time and gate time.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2).
        This assumes the decoherence is Marcovian and follows an exponential decay model.

        Parameters
        ----------
        T2 : float
            Coherence time of the m-qubit in seconds.
        gate_time : float
            Gate time for the 1Q gate operation for m-qubit in seconds.
        idling_flag : bool
            Whether to use (DEP1_m & DECAY_mg_gate) or (ZERR_m & XERR_m) for error channel.
            If True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate).
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the Z error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )
        p = p if idling_flag else 0

        identity = sqrt(1 - p) * eye(6)
        Z_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, Z_err])

    def XERR_m(
        self,
        T1: float,  # T1 time of m-qubit (seconds)
        gate_time: float,  # gate time (seconds)
        idling_flag: bool,  # if True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate)
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """X error channel for m-qubit during 1Q gate for m-qubit with T1 time and gate time.

        The probability of X error is calculated as p = 1 - exp(-gate_time / T1).

        Parameters
        ----------
        T1 : float
            T1 time of the m-qubit in seconds.
        gate_time : float
            Gate time for the 1Q gate operation for m-qubit in seconds.
        idling_flag : bool
            Whether to use (DEP1_m & DECAY_mg_gate) or (ZERR_m & XERR_m) for error channel.
            If True, use (ZERR_m & XERR_m), else use (DEP1_m & DECAY_mg_gate).
        p : float, optional
            If provided, overrides the T1 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the X error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / T1) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated X error probability p must be in the range [0, 1]."
            )
        p = p if idling_flag else 0

        identity = sqrt(1 - p) * eye(6)
        X_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, X_err])

    def ZERR_gm(
        self,
        T2: float,  # coherence time of optical qubit (seconds)
        gate_time: float,  # gate time (seconds)
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """Z error channel for optical qubit during 1Q gate for m-qubit with T2 coherence time and gate time.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2).
        This assumes the decoherence is Marcovian and follows an exponential decay model.

        Parameters
        ----------
        T2 : float
            Coherence time of the optical qubit in seconds.
        gate_time : float
            Gate time for the 1Q gate operation for m-qubit in seconds.
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            2 Kraus operators for the Z error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)
        Z_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, Z_err])

    def LOSS_g(
        self,
        lifetime: float,  # trap lifetime of the ground state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Trap loss error channel during 1Q gate for m-qubit of 171Yb with ground state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the ground state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            3 Kraus operators for the ground-state loss error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [sqrt(1 - p), 0, 0, 0, 0, 0],
                [0, sqrt(1 - p), 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [sqrt(p), 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, sqrt(p), 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def LOSS_m(
        self,
        lifetime: float,  # trap lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Trap loss error channel during 1Q gate for m-qubit of 171Yb with excited state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the excited state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            3 Kraus operators for the 3P0 loss error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p), 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p), 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def DECAY_mg(
        self,
        lifetime: float,  # 3P0 lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from 3P0 to 1S0 during 1Q gate for m-qubit of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Photon scattering time of the excited state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            5 Kraus operators for the leakage error from 3P0 to 1S0.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err00 = array(
            [
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err01 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err10 = array(
            [
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err11 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err00, leak_err01, leak_err10, leak_err11])

    def LOSS_r(
        self,
        lifetime: float,  # radiative lifetime of the rydberg state to leakage in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Radiative loss error channel from rydberg during 1Q gate for m-qubit of 171Yb.

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
            2 Kraus operators for the radiative loss of rydberg states.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p), 0],
            ]
        )

        return array([identity, leak_err])

    def DECAY_rg(
        self,
        lifetime: float,  # leakage lifetime from the rydberg state to 1S0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to ground state during 1Q gate for m-qubit of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to 1S0 in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            3 Kraus operators for the leakage error from rydberg to ground state.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def DECAY_rm(
        self,
        lifetime: float,  # leakage lifetime from the rydberg state to 3P0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to 3P0 during 1Q gate for m-qubit of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to metastable state in seconds.
        gate_time : float
            Idling time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and idling_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        NDArray[complexfloating]
            3 Kraus operators for the leakage error from rydberg to 3P0.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])


class Kraus2Q_171m171m:
    """Kraus operators for 2-qubit gate between 171Yb m-qubits.

    This class provides methods to generate Kraus operators for various error channels.
    """

    def __init__(
        self,
        p_dep2: float,  # depolarization probability for 2-qubit gate
        T2_g: float,  # coherence time of g-qubit (seconds)
        T1_g: float,  # T1 time of g-qubit (seconds)
        T2_m: float,  # coherence time of m-qubit (seconds)
        T1_m: float,  # T1 time of m-qubit (seconds)
        T2_c: float,  # coherence time of optical qubit (seconds)
        gate_time: float,  # gate time (seconds)
        lifetime_gs: float,  # trap lifetime of the ground state (seconds)
        lifetime_es: float,  # trap lifetime of the excited state (seconds)
        leaktime_eg: float,  # leakage time from the 3P0 state to 1S0 (seconds)
        lifetime_ryd: float,  # radiative lifetime of the rydberg state to leakage (seconds)
        leaktime_ryd_gs: float,  # leakage time from the rydberg state to 1S0 (seconds)
        leaktime_ryd_es: float,  # leakage time from the rydberg state to 3P0 (seconds)
        idling_flag: bool,  # whether to use (DEP2_m&XERR) or (ZERR_m&XERR_m) for error channel, if True, use (ZERR_m&XERR_m), else use (DEP2_m&XERR)
    ):
        """Initialize Kraus2Q_171m171m with noise parameters.

        Parameters
        ----------
        p_dep2 : float
            Depolarization probability for 2-qubit gate.
        T2_g : float
            Coherence time of g-qubit in seconds.
        T1_g : float
            T1 time of g-qubit in seconds.
        T2_m : float
            Coherence time of m-qubit in seconds.
        T1_m : float
            T1 time of m-qubit in seconds.
        T2_c : float
            Coherence time of optical qubit in seconds.
        gate_time : float
            Gate time in seconds.
        lifetime_gs : float
            Trap lifetime of the ground state in seconds.
        lifetime_es : float
            Trap lifetime of the excited state in seconds.
        leaktime_eg : float
            Leakage time from the 3P0 state to 1S0 in seconds.
        lifetime_ryd : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        leaktime_ryd_gs : float
            Leakage time from the rydberg state to 1S0 in seconds.
        leaktime_ryd_es : float
            Leakage time from the rydberg state to 3P0 in seconds.
        idling_flag : bool
            Whether to use (DEP2_m&XERR) or (ZERR_m&XERR_m) for error channel. If True, use (ZERR_m&XERR_m), else use (DEP2_m&XERR).
        """
        # Initialize parameters
        self.p_dep2 = p_dep2
        self.T2_g = T2_g
        self.T1_g = T1_g
        self.T2_m = T2_m
        self.T1_m = T1_m
        self.T2_c = T2_c
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
            "DEP2_m": self.DEP2_m(p_dep2, idling_flag),
            "ZERR_g": self.ZERR_g(T2_g, gate_time),
            "XERR_g": self.XERR_g(T1_g, gate_time),
            "ZERR_m": self.ZERR_m(T2_m, gate_time, idling_flag),
            "XERR_m": self.XERR_m(T1_m, gate_time, idling_flag),
            "ZERR_gm": self.ZERR_gm(T2_c, gate_time),
            "LOSS_g": self.LOSS_g(lifetime_gs, gate_time),
            "LOSS_m": self.LOSS_m(lifetime_es, gate_time),
            "DECAY_mg": self.DECAY_mg(leaktime_eg, gate_time),
            "LOSS_r": self.LOSS_r(lifetime_ryd, gate_time),
            "DECAY_rg": self.DECAY_rg(leaktime_ryd_gs, gate_time),
            "DECAY_rm": self.DECAY_rm(leaktime_ryd_es, gate_time),
        }

    def CPTP(
        self,
        density_matrix: ndarray,  # input density matrix
        channel: str | None = None,  # channel name, e.g., 'DEP2_m', etc.
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied.
            Size should be (36, 36) for a 2-qubit system.
        channel : str, optional
            The name of the channel to apply. If None, applies all channels.
            Valid options are 'DEP2_m', 'XERR', 'ZERR_g', 'XERR_g', 'ZERR_m', 'XERR_m',
            'ZERR_gm', 'LOSS_g', 'LOSS_m', 'DECAY_mg', 'LOSS_r', 'DECAY_rg', 'DECAY_rm'.
            If the channel is not found, a ValueError will be raised.

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators.
            Size will be (36, 36) for a 2-qubit system.
        """
        if channel is None:
            """If no channel is specified, apply all channels"""
            # apply all channels in the order they were added (the order of CPTP maps did not matter)
            result = zeros(density_matrix.shape, dtype=complex)
            tmp_result = zeros(
                density_matrix.shape, dtype=complex
            )  # Initialize tmp_result
            for idx, kraus_ops in enumerate(self.noise_channels.values()):
                tmp_rho = (
                    density_matrix if idx == 0 else tmp_result
                )  # use the result of the previous kraus as input for the next kraus
                tmp_result = zeros(
                    tmp_rho.shape, dtype=tmp_rho.dtype
                )  # temporary result for the current channel
                # apply the Kraus operator for each channel
                for kraus_op in kraus_ops:
                    tmp_result += kraus_op @ tmp_rho @ conj(kraus_op.T)

            result = tmp_result
            return result

        else:
            """Apply the specified channel"""
            result = zeros(density_matrix.shape, dtype=density_matrix.dtype)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                for kraus_op in kraus_ops:
                    result += kraus_op @ density_matrix @ conj(kraus_op.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def DEP2_m(
        self,
        p: float,
        idling_flag: bool,  # if True, use (ZERR_m&XERR_m), else use (DEP2_m&XERR)
    ) -> NDArray[complexfloating]:
        """Depolarizing channel for 2-qubit gate of 171Yb m-qubit with depolarization probability p.

        Assuming equal depolarization in all directions.
        This error channel applied if idling_flag is False.

        Parameters
        ----------
        p : float
            Depolarization probability in the range [0, 1].
        idling_flag : bool
            If True, use (ZERR_m&XERR_m), else use (DEP2_m&XERR).

        Returns:
        -------
        ndarray[ndarray], size=16
            16 Kraus operators for the depolarizing channel.
            Each operator is a 36x36 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        if not (0 <= p <= 1):
            raise ValueError(
                "Depolarization probability p must be in the range [0, 1]."
            )

        p = 0 if idling_flag else p

        p_each = p / 15
        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        X = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Y = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0],
                [0, 0, 1j, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        Z = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        out = []
        paulis = [identity, X, Y, Z]
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0:
                    pass
                else:
                    out.append(sqrt(p_each) * kron(paulis[i], paulis[j]))

        kraus0 = sqrt(eye(36) - sum([conj(op).T @ op for op in out]))
        out.insert(0, kraus0)
        return array(out)

    def ZERR_g(
        self,
        T2: float,  # coherence time (seconds)
        gate_time: float,  # gate time (seconds)
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """Z error channel for g-qubit during 2-qubit gate of 171Yb m-qubit with T2 coherence time and gate time.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2).
        This assumes the decoherence is Markovian and follows an exponential decay model.

        Parameters
        ----------
        T2 : float
            Coherence time of the g-qubit in seconds.
        gate_time : float
            Gate time for 2Q gate of m-qubit in seconds.
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=4
            4 Kraus operators for the Z error channel.
            Each operator is a 36x36 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)
        Z_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, Z_err),
                kron(Z_err, identity),
                kron(Z_err, Z_err),
            ]
        )

    def XERR_g(
        self,
        T1: float,  # T1 time of g-qubit (seconds)
        gate_time: float,  # gate time (seconds)
        p: float | None = None,  # optional, if provided, overrides T1 and gate_time
    ) -> NDArray[complexfloating]:
        """X error channel for g-qubit during 2Q gate for m-qubit with T1 time and gate time.

        The probability of X error is calculated as p = 1 - exp(-gate_time / T1).

        Parameters
        ----------
        T1 : float
            T1 time of the g-qubit in seconds.
        gate_time : float
            Gate time for the 2Q gate operation for m-qubit in seconds.
        p : float, optional
            If provided, overrides the T1 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=4
            4 Kraus operators for the X error channel.
            Each operator is a 36x36 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / T1) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated X error probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)
        X_err = sqrt(p) * array(
            [
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, X_err),
                kron(X_err, identity),
                kron(X_err, X_err),
            ]
        )

    def ZERR_m(
        self,
        T2: float,  # coherence time of m-qubit (seconds)
        gate_time: float,  # gate time (seconds)
        idling_flag: bool,  # if True, use (ZERR_m & XERR_m), else use DEP2_m
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """Z error channel for m-qubit during 2Q gate for m-qubit with T2 coherence time and gate time.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2).
        This assumes the decoherence is Markovian and follows an exponential decay model.

        Parameters
        ----------
        T2 : float
            Coherence time of the m-qubit in seconds.
        gate_time : float
            Gate time for the 2Q gate operation for m-qubit in seconds.
        idling_flag : bool
            Whether to use DEP2_m or (ZERR_m & XERR_m) for error channel. If True, use (ZERR_m & XERR_m), else use DEP2_m.
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=4
            4 Kraus operators for the Z error channel.
                Each operator is a 36x36 matrix representing the channel.
                (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )
        p = p if idling_flag else 0

        identity = sqrt(1 - p) * eye(6)
        Z_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, Z_err),
                kron(Z_err, identity),
                kron(Z_err, Z_err),
            ]
        )

    def XERR_m(
        self,
        T1: float,  # T1 time of m-qubit (seconds)
        gate_time: float,  # gate time (seconds)
        idling_flag: bool,  # if True, use (ZERR_m & XERR_m), else use DEP2_m
        p: float | None = None,  # optional, if provided, overrides T1 and gate_time
    ) -> NDArray[complexfloating]:
        """X error channel for m-qubit during 2Q gate for m-qubit with T1 time and gate time.

        The probability of X error is calculated as p = 1 - exp(-gate_time / T1).

        Parameters
        ----------
        T1 : float
            T1 time of the m-qubit in seconds.
        gate_time : float
            Gate time for the 2Q gate operation for m-qubit in seconds.
        idling_flag : bool
            Whether to use DEP2_m or (ZERR_m & XERR_m) for error channel. If True, use (ZERR_m & XERR_m), else use DEP2_m.
        p : float, optional
            If provided, overrides the T1 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=4
            4 Kraus operators for the X error channel.
                Each operator is a 36x36 matrix representing the channel.
                (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / T1) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated X error probability p must be in the range [0, 1]."
            )
        p = p if idling_flag else 0

        identity = sqrt(1 - p) * eye(6)
        X_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, X_err),
                kron(X_err, identity),
                kron(X_err, X_err),
            ]
        )

    def ZERR_gm(
        self,
        T2: float,  # coherence time of optical qubit (seconds)
        gate_time: float,  # gate time (seconds)
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """Z error channel for optical qubit during 2Q gate for m-qubit with T2 coherence time and gate time.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2).
        This assumes the decoherence is Markovian and follows an exponential decay model.

        Parameters
        ----------
        T2 : float
            Coherence time of the optical qubit in seconds.
        gate_time : float
            Gate time for the 2Q gate operation for m-qubit in seconds.
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=4
            4 Kraus operators for the Z error channel.
                Each operator is a 36x36 matrix representing the channel.
                (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)
        Z_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
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
        lifetime: float,  # trap lifetime of the ground state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Trap loss error channel during 2Q gate for m-qubit of 171Yb with ground state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the ground state in seconds.
        gate_time : float
            Gate time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=9
            9 Kraus operators for the ground-state loss error channel.
            Each operator is a 36x36 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [sqrt(1 - p), 0, 0, 0, 0, 0],
                [0, sqrt(1 - p), 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [sqrt(p), 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, sqrt(p), 0, 0, 0, 0],
            ]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err0),
                kron(identity, leak_err1),
                kron(leak_err0, identity),
                kron(leak_err0, leak_err0),
                kron(leak_err0, leak_err1),
                kron(leak_err1, identity),
                kron(leak_err1, leak_err0),
                kron(leak_err1, leak_err1),
            ]
        )

    def LOSS_m(
        self,
        lifetime: float,  # trap lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Trap loss error channel during 2Q gate for m-qubit of 171Yb with excited state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the excited state in seconds.
        gate_time : float
            Gate time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=9
            9 Kraus operators for the excited-state loss error channel.
            Each operator is a 36x36 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p), 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p), 0, 0],
            ]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err0),
                kron(identity, leak_err1),
                kron(leak_err0, identity),
                kron(leak_err0, leak_err0),
                kron(leak_err0, leak_err1),
                kron(leak_err1, identity),
                kron(leak_err1, leak_err0),
                kron(leak_err1, leak_err1),
            ]
        )

    def DECAY_mg(
        self,
        lifetime: float,  # 3P0 lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from 3P0 to 1S0 during 2Q gate for m-qubit of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Photon scattering time of the excited state in seconds.
        gate_time : float
            Gate time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 25
            25 Kraus operators for the leakage error from 3P0 to 1S0.
            Each operator is a 36x36 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err00 = array(
            [
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err01 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err10 = array(
            [
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err11 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err00),
                kron(identity, leak_err01),
                kron(identity, leak_err10),
                kron(identity, leak_err11),
                kron(leak_err00, identity),
                kron(leak_err00, leak_err00),
                kron(leak_err00, leak_err01),
                kron(leak_err00, leak_err10),
                kron(leak_err00, leak_err11),
                kron(leak_err01, identity),
                kron(leak_err01, leak_err00),
                kron(leak_err01, leak_err01),
                kron(leak_err01, leak_err10),
                kron(leak_err01, leak_err11),
                kron(leak_err10, identity),
                kron(leak_err10, leak_err00),
                kron(leak_err10, leak_err01),
                kron(leak_err10, leak_err10),
                kron(leak_err10, leak_err11),
                kron(leak_err11, identity),
                kron(leak_err11, leak_err00),
                kron(leak_err11, leak_err01),
                kron(leak_err11, leak_err10),
                kron(leak_err11, leak_err11),
            ]
        )

    def LOSS_r(
        self,
        lifetime: float,  # radiative lifetime of the rydberg state to leakage in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Radiative loss error channel from rydberg during 2Q gate for m-qubit of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        gate_time : float
            Gate time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 4
            4 Kraus operators for the radiative loss of rydberg states.
            Each operator is a 36x36 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p), 0],
            ]
        )

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
        lifetime: float,  # leakage lifetime from the rydberg state to 1S0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to ground state during 2Q gate for m-qubit of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to 1S0 in seconds.
        gate_time : float
            Gate time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 9
            9 Kraus operators for the leakage error from rydberg to 1S0.
            Each operator is a 36x36 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err0),
                kron(identity, leak_err1),
                kron(leak_err0, identity),
                kron(leak_err0, leak_err0),
                kron(leak_err0, leak_err1),
                kron(leak_err1, identity),
                kron(leak_err1, leak_err0),
                kron(leak_err1, leak_err1),
            ]
        )

    def DECAY_rm(
        self,
        lifetime: float,  # leakage lifetime from the rydberg state to 3P0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to 3P0 during 2Q gate for m-qubit of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to metastable state in seconds.
        gate_time : float
            Gate time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 9
            9 Kraus operators for the leakage error from rydberg to 3P0.
            Each operator is a 36x36 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array(
            [
                kron(identity, identity),
                kron(identity, leak_err0),
                kron(identity, leak_err1),
                kron(leak_err0, identity),
                kron(leak_err0, leak_err0),
                kron(leak_err0, leak_err1),
                kron(leak_err1, identity),
                kron(leak_err1, leak_err0),
                kron(leak_err1, leak_err1),
            ]
        )


class KrausRESET_171m:
    """Kraus operators for |0m> state preparation of 171Yb.

    This class provides methods to generate Kraus operators for various reset operations.
    """

    def __init__(
        self,
        p_mloss: float,  # probability of m-qubit loss during reset,
        p_mflip: float,  # probability of m-qubit flip during reset
    ):
        """Initialize KrausRESET_171m with reset error parameters.

        Parameters
        ----------
        p_mloss : float
            Probability of m-qubit loss during reset.
        p_mflip : float
            Probability of m-qubit flip during reset.
        """
        # Initialize parameters
        self.p_loss = p_mloss
        self.p_flip = p_mflip

        # Initialize the Kraus operators for the reset operations to |0m> state of 171Yb
        self.noise_channels = {
            "LOSS_m_reset": self.LOSS_m_reset(p_mloss),
            "FLIP_m": self.FLIP_m(p_mflip),
        }

    def CPTP(
        self,
        density_matrix: ndarray,  # input density matrix
        channel: str
        | None = None,  # channel name, e.g., 'DEP2_m', 'ZERR_m', 'LOSS_g', etc.
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied.
            Size should be (6, 6) for a 1-qubit system. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        channel : str, optional
            The name of the channel to apply. If None, applies all channels.
            Valid options are 'LOSS_m_reset' and 'FLIP_m'.

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators.
            Size will be (6, 6) for a 1-qubit system.
        """
        if channel is None:
            """If no channel is specified, apply all channels"""
            # apply all channels in the order they were added (the order of CPTP maps did not matter)
            result = zeros(density_matrix.shape, dtype=complex)
            tmp_result = zeros(
                density_matrix.shape, dtype=complex
            )  # Initialize tmp_result
            for idx, kraus_ops in enumerate(self.noise_channels.values()):
                tmp_rho = (
                    density_matrix if idx == 0 else tmp_result
                )  # use the result of the previous kraus as input for the next kraus
                tmp_result = zeros(
                    tmp_rho.shape, dtype=tmp_rho.dtype
                )  # temporary result for the current channel
                # apply the Kraus operator for each channel
                for kraus_op in kraus_ops:
                    tmp_result += kraus_op @ tmp_rho @ conj(kraus_op.T)

            result = tmp_result
            return result

        else:
            """Apply the specified channel"""
            result = zeros(density_matrix.shape, dtype=density_matrix.dtype)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                for kraus_op in kraus_ops:
                    result += kraus_op @ density_matrix @ conj(kraus_op.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def LOSS_m_reset(self, p: float) -> NDArray[complexfloating]:
        """Kraus operators for m-qubit loss during reset to |0m> state.

        The probability of m-qubit loss is given by p.

        Parameters
        ----------
        p : float
            Probability of m-qubit loss during reset, should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 3
            3 Kraus operators for m-qubit loss during reset.
            Each operator is a 6x6 matrix representing the channel.
            (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        if not (0 <= p <= 1):
            raise ValueError("Loss error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p), 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p), 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def FLIP_m(self, p: float) -> NDArray[complexfloating]:
        """Kraus operators for m-qubit flip during reset to |0m> state.

        The probability of m-qubit flip is given by p.

        Parameters
        ----------
        p : float
            Probability of m-qubit flip during reset, should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 2
            2 Kraus operators for m-qubit flip during reset.
            Each operator is a 6x6 matrix representing the channel.
                (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        identity = sqrt(1 - p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        X_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, X_err])


class KrausMEASURE_DISC_171m:
    """Kraus operators for measurement of 171Yb m-qubit.

    This class provides methods to generate Kraus operators for various measurement operations.
    """

    def __init__(
        self,
        p_meas: float,  # probability of measurement error per fluorescence pulse
        q: float = 1.0,  # probability of assigning the ambiguous BD outcome to "0" instead of "1"
    ):
        """Initialize KrausMEASURE_DISC_171m with measurement error parameters.

        Parameters
        ----------
        p_meas : float
            Probability of bright/dark misdiscrimination per fluorescence pulse.
        q : float
            Probability of assigning the ambiguous BD outcome (both pulses bright)
            to "0" instead of "1". Defaults to 1.0.
        """
        # Initialize parameters
        self.p_meas = p_meas
        self.q = q

        # Initialize the Kraus operators for the measurement operations to |0m> state of 171Yb
        self.noise_channels = {"MERR": self.MERR(p_meas, q)}

    def CPTP(
        self,
        density_matrix: ndarray,  # input density matrix
        channel: str | None = None,  # channel name, e.g., 'MERR'
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied.
            Size should be (6, 6) for a 1-qubit system. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        channel : str, optional
            The name of the channel to apply. If None, applies all channels.
            Valid options are 'MERR'.

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators.
            Size will be (6, 6) for a 1-qubit system.
        """
        if channel is None:
            """If no channel is specified, apply all channels"""
            # apply all channels in the order they were added (the order of CPTP maps did not matter)
            result = zeros(density_matrix.shape, dtype=complex)
            tmp_result = zeros(
                density_matrix.shape, dtype=complex
            )  # Initialize tmp_result
            for idx, kraus_ops in enumerate(self.noise_channels.values()):
                tmp_rho = (
                    density_matrix if idx == 0 else tmp_result
                )  # use the result of the previous kraus as input for the next kraus
                tmp_result = zeros(
                    tmp_rho.shape, dtype=tmp_rho.dtype
                )  # temporary result for the current channel
                # apply the Kraus operator for each channel
                for kraus_op in kraus_ops:
                    tmp_result += kraus_op @ tmp_rho @ conj(kraus_op.T)

            result = tmp_result
            return result

        else:
            """Apply the specified channel"""
            result = zeros(density_matrix.shape, dtype=density_matrix.dtype)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                for kraus_op in kraus_ops:
                    result += kraus_op @ density_matrix @ conj(kraus_op.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def MERR(
        self,
        p: float,
        q: float | None = None,
    ) -> NDArray[complexfloating]:
        """Measurement error channel for m-qubit of 171Yb with the BD readout model.

        The 171m readout protocol first transfers |0m>, |1m> to |0g>, |1g> and then
        applies two state-selective fluorescence pulses (one bright for "0", one
        bright for "1"). The bright/dark misdiscrimination probability per pulse is
        ``p``. ``q`` controls how the ambiguous "both bright" event is assigned.

        The same discrimination error therefore acts on the m-subspace as on the
        g-subspace; this implementation keeps the post-measurement state in its
        original subspace (m -> m, g -> g).

        Parameters
        ----------
        p : float
            Bright/dark misdiscrimination probability per pulse, in [0, 1].
        q : float, optional
            Probability of assigning the ambiguous BD outcome to "0".
            Defaults to ``self.q``.

        Returns:
        -------
        ndarray[ndarray], size = 13
            13 Kraus operators for measurement error.
            Each operator is a 6x6 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        if not (0 <= p <= 1):
            raise ValueError(
                "Measurement error probability p must be in the range [0, 1]."
            )
        if q is None:
            q = self.q
        if not (0 <= q <= 1):
            raise ValueError("BD assignment probability q must be in the range [0, 1].")

        # Per-outcome probabilities (true state -> readout label).
        #
        # Two pulses: pulse 1 is bright for "0", pulse 2 is bright for "1", each
        # misdiscriminating with probability p. (bright, bright) is the ambiguous
        # branch split by q; (dark, dark) means no fluorescence at all and is
        # therefore indistinguishable from atom loss, i.e. read as "L". For a
        # true |0>: (bright, dark) = (1-p)^2 -> "0", (bright, bright) = (1-p)p
        # -> q-split, (dark, bright) = p^2 -> "1", (dark, dark) = p(1-p) -> "L".
        # Mirrors KrausMEASURE_DISC_174.MERR with 0 <-> g, 1 <-> m.
        p_0_as_0 = (1 - p) ** 2 + q * (1 - p) * p
        p_0_as_1 = p**2 + (1 - q) * (1 - p) * p
        p_0_as_l = p * (1 - p)

        p_1_as_0 = p**2 + q * (1 - p) * p
        p_1_as_1 = (1 - p) ** 2 + (1 - q) * (1 - p) * p
        p_1_as_l = (1 - p) * p

        p_l_as_0 = p * (1 - p) + q * p**2
        p_l_as_1 = (1 - p) * p + (1 - q) * p**2
        p_l_as_l = (1 - p) ** 2

        # Identity (correct readout, state stays in its original subspace)
        identity = array(
            [
                [sqrt(p_0_as_0), 0, 0, 0, 0, 0],
                [0, sqrt(p_1_as_1), 0, 0, 0, 0],
                [0, 0, sqrt(p_0_as_0), 0, 0, 0],
                [0, 0, 0, sqrt(p_1_as_1), 0, 0],
                [0, 0, 0, 0, sqrt(p_l_as_l), 0],
                [0, 0, 0, 0, 0, sqrt(p_l_as_l)],
            ]
        )

        # g-subspace: |0g> misread as "1" or "L"
        g0_as_g1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [sqrt(p_0_as_1), 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        g0_as_l = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [sqrt(p_0_as_l), 0, 0, 0, 0, 0],
            ]
        )

        # g-subspace: |1g> misread as "0" or "L"
        g1_as_g0 = array(
            [
                [0, sqrt(p_1_as_0), 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        g1_as_l = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, sqrt(p_1_as_l), 0, 0, 0, 0],
            ]
        )

        # m-subspace: |0m> misread as "1" or "L" (analogous to g-subspace after m->g transfer)
        m0_as_m1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p_0_as_1), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        m0_as_l = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p_0_as_l), 0, 0, 0],
            ]
        )

        # m-subspace: |1m> misread as "0" or "L"
        m1_as_m0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p_1_as_0), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        m1_as_l = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p_1_as_l), 0, 0],
            ]
        )

        # Loss state misread as "0" or "1" (treated as ending in the g-subspace)
        l_as_g0 = array(
            [
                [0, 0, 0, 0, 0, sqrt(p_l_as_0)],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        l_as_g1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, sqrt(p_l_as_1)],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        # |r> is treated like |L>.
        r_as_g0 = array(
            [
                [0, 0, 0, 0, sqrt(p_l_as_0), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        r_as_g1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p_l_as_1), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array(
            [
                identity,
                g0_as_g1,
                g0_as_l,
                g1_as_g0,
                g1_as_l,
                m0_as_m1,
                m0_as_l,
                m1_as_m0,
                m1_as_l,
                l_as_g0,
                l_as_g1,
                r_as_g0,
                r_as_g1,
            ]
        )


class KrausMEASURE_171m:
    """Kraus operators for measurement of 171Yb.

    This class provides methods to generate Kraus operators for various measurement operations.
    """

    def __init__(
        self,
        p_gflip: float,  # probability of ground state flip during measurement
        p_gloss: float,  # probability of ground state loss during measurement
        T2_m: float,  # T2 time of the m-qubit in seconds
        T1_m: float,  # T1 time of the m-qubit in seconds
        T2_c: float,  # T2 time of the optical qubit in seconds
        readout_time: float,  # readout time in seconds
        lifetime_es: float,  # trap lifetime of the excited state (seconds)
        leaktime_eg: float,  # leakage time from the 3P0 state to 1S0 (seconds)
        lifetime_ryd: float,  # radiative lifetime of the rydberg state to leakage (seconds)
        leaktime_ryd_gs: float,  # leakage time from the rydberg state to 1S0 (seconds)
        leaktime_ryd_es: float,  # leakage time from the rydberg state to 3P0 (seconds)
    ):
        """Initialize KrausMEASURE_171m with measurement parameters.

        Parameters
        ----------
        p_gflip : float
            Probability of ground state flip during measurement.
        p_gloss : float
            Probability of ground state loss during measurement.
        T2_m : float
            T2 time of the m-qubit in seconds.
        T1_m : float
            T1 time of the m-qubit in seconds.
        T2_c : float
            T2 time of the optical qubit in seconds.
        readout_time : float
            Readout time in seconds.
        lifetime_es : float
            Trap lifetime of the excited state in seconds.
        leaktime_eg : float
            Leakage time from the 3P0 state to 1S0 in seconds.
        lifetime_ryd : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        leaktime_ryd_gs : float
            Leakage time from the rydberg state to 1S0 in seconds.
        leaktime_ryd_es : float
            Leakage time from the rydberg state to 3P0 in seconds.
        """
        # Initialize parameters
        self.p_gflip = p_gflip
        self.p_gloss = p_gloss
        self.T2_m = T2_m
        self.T1_m = T1_m
        self.T2_c = T2_c
        self.readout_time = readout_time
        self.lifetime_es = lifetime_es
        self.leaktime_eg = leaktime_eg
        self.lifetime_ryd = lifetime_ryd
        self.leaktime_ryd_gs = leaktime_ryd_gs
        self.leaktime_ryd_es = leaktime_ryd_es

        # Initialize the Kraus operators for the measurement operations to |0m> state of 171Yb
        self.noise_channels = {
            # readout |0g> or |1g> state
            "FLIP_g": self.FLIP_g(p_gflip),
            "LOSS_g_meas": self.LOSS_g_meas(p_gloss),
            "ZERR_m": self.ZERR_m(T2_m, readout_time),
            "XERR_m": self.XERR_m(T1_m, readout_time),
            "ZERR_gm": self.ZERR_gm(T2_c, readout_time),
            "LOSS_m": self.LOSS_m(lifetime_es, readout_time),
            "DECAY_mg": self.DECAY_mg(leaktime_eg, readout_time),
            "LOSS_r": self.LOSS_r(lifetime_ryd, readout_time),
            "DECAY_rg": self.DECAY_rg(leaktime_ryd_gs, readout_time),
            "DECAY_rm": self.DECAY_rm(leaktime_ryd_es, readout_time),
        }

    def CPTP(
        self,
        density_matrix: ndarray,  # input density matrix
        channel: str | None = None,  # channel name, e.g., 'FLIP_g', 'LOSS_g_meas', etc.
    ) -> ndarray:
        """Apply the Kraus operators to a given density matrix.

        This method applies the specified channel's Kraus operators to the input density matrix.
        If no channel is specified, it applies all channels.

        Parameters
        ----------
        density_matrix : ndarray
            Input density matrix to which the Kraus operators will be applied.
            Size should be (6, 6) for a 1-qubit system. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        channel : str, optional
            The name of the channel to apply. If None, applies all channels.
            Valid options are 'FLIP_g', 'LOSS_g_meas', 'ZERR_m', 'XERR_m', 'ZERR_gm', 'LOSS_m', 'DECAY_mg', 'LOSS_r', 'DECAY_rg', 'DECAY_rm'.

        Returns:
        -------
        ndarray
            The resulting density matrix after applying the Kraus operators.
            Size will be (6, 6) for a 1-qubit system.
        """
        if channel is None:
            """If no channel is specified, apply all channels"""
            # apply all channels in the order they were added (the order of CPTP maps did not matter)
            result = zeros(density_matrix.shape, dtype=complex)
            tmp_result = zeros(
                density_matrix.shape, dtype=complex
            )  # Initialize tmp_result
            for idx, kraus_ops in enumerate(self.noise_channels.values()):
                tmp_rho = (
                    density_matrix if idx == 0 else tmp_result
                )  # use the result of the previous kraus as input for the next kraus
                tmp_result = zeros(
                    tmp_rho.shape, dtype=tmp_rho.dtype
                )  # temporary result for the current channel
                # apply the Kraus operator for each channel
                for kraus_op in kraus_ops:
                    tmp_result += kraus_op @ tmp_rho @ conj(kraus_op.T)

            result = tmp_result
            return result

        else:
            """Apply the specified channel"""
            result = zeros(density_matrix.shape, dtype=density_matrix.dtype)
            if channel in self.noise_channels:
                kraus_ops = self.noise_channels[channel]
                for kraus_op in kraus_ops:
                    result += kraus_op @ density_matrix @ conj(kraus_op.T)
            else:
                raise ValueError(f"Channel '{channel}' not found in Kraus operators.")
            return result

    def FLIP_g(
        self, p: float
    ) -> NDArray[
        complexfloating
    ]:  # probability of ground state flip during measurement
        """Kraus operators for ground state flip during readout of 171Yb.

        The probability of ground state flip is given by p.

        Parameters
        ----------
        p : float
            Probability of ground state flip during readout, should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 2
            2 Kraus operators for ground state flip during readout.
            Each operator is a 6x6 matrix representing the channel.
            (spanned by {|0g>, |1g>, |0m>, |1m>, |r>, |L>} tensor product {|0g>, |1g>, |0m>, |1m>, |r>, |L>})
        """
        if not (0 <= p <= 1):
            raise ValueError(
                "Ground state flip probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)
        X_err = sqrt(p) * array(
            [
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, X_err])

    def LOSS_g_meas(
        self, p: float
    ) -> NDArray[
        complexfloating
    ]:  # probability of ground state loss during measurement
        """Kraus operators for ground state loss during readout of 171Yb.

        The probability of ground state loss is given by p.

        Parameters
        ----------
        p : float
            Probability of ground state loss during readout, should be in the range [0, 1].

        Returns:
        -------
        list[ndarray], size = 3
            3 Kraus operators for ground state loss during readout.
            Each operator is a 6x6 matrix representing the channel.
            (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        if not (0 <= p <= 1):
            raise ValueError(
                "Ground state loss probability p must be in the range [0, 1]."
            )

        identity = array(
            [
                [sqrt(1 - p), 0, 0, 0, 0, 0],
                [0, sqrt(1 - p), 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [sqrt(p), 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, sqrt(p), 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def ZERR_m(
        self,
        T2: float,  # coherence time of m-qubit (seconds)
        gate_time: float,  # readout time (seconds)
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """Z error channel for m-qubit during g-qubit readout with T2 coherence time and gate time.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2).
        This assumes the decoherence is Markovian and follows an exponential decay model.

        Parameters
        ----------
        T2 : float
            Coherence time of the m-qubit in seconds.
        gate_time : float
            Readout time for g-qubit in seconds.
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=2
            2 Kraus operators for the Z error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)
        Z_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, Z_err])

    def XERR_m(
        self,
        T1: float,  # T1 time of m-qubit (seconds)
        gate_time: float,  # readout time (seconds)
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """X error channel for m-qubit during g-qubit readout with T1 time and gate time.

        The probability of X error is calculated as p = 1 - exp(-gate_time / T1).

        Parameters
        ----------
        T1 : float
            T1 time of the m-qubit in seconds.
        gate_time : float
            Readout time for g-qubit in seconds.
        p : float, optional
            If provided, overrides the T1 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=2
            2 Kraus operators for the X error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / T1) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated X error probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)
        X_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, X_err])

    def ZERR_gm(
        self,
        T2: float,  # coherence time of optical qubit (seconds)
        gate_time: float,  # readout time (seconds)
        p: float | None = None,  # optional, if provided, overrides T2 and gate_time
    ) -> NDArray[complexfloating]:
        """Z error channel for optical qubit during g-qubit readout with T2 coherence time and gate time.

        The probability of Z error is calculated as p = 1 - exp(-gate_time / T2).
        This assumes the decoherence is Markovian and follows an exponential decay model.

        Parameters
        ----------
        T2 : float
            Coherence time of the optical qubit in seconds.
        gate_time : float
            Readout time for g-qubit in seconds.
        p : float, optional
            If provided, overrides the T2 and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=2
            2 Kraus operators for the Z error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / T2) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError(
                "Calculated Z error probability p must be in the range [0, 1]."
            )

        identity = sqrt(1 - p) * eye(6)
        Z_err = sqrt(p) * array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return array([identity, Z_err])

    def LOSS_m(
        self,
        lifetime: float,  # trap lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Trap loss error channel during g-qubit readout of 171Yb with excited state trap lifetime.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Trap lifetime of the excited state in seconds.
        gate_time : float
            Readout time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and gate_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size=3
            3 Kraus operators for the 3P0 loss error channel.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p), 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p), 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def DECAY_mg(
        self,
        lifetime: float,  # 3P0 lifetime of the excited state in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from 3P0 to 1S0 during g-qubit readout of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Photon scattering time of the excited state in seconds.
        gate_time : float
            Readout time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and readout_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 5
            5 Kraus operators for the leakage error from 3P0 to 1S0.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, sqrt(1 - p), 0, 0, 0],
                [0, 0, 0, sqrt(1 - p), 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        leak_err00 = array(
            [
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err01 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, sqrt(p / 2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err10 = array(
            [
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err11 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, sqrt(p / 2), 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err00, leak_err01, leak_err10, leak_err11])

    def LOSS_r(
        self,
        lifetime: float,  # radiative lifetime of the rydberg state to leakage in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Radiative loss error channel from rydberg during g-qubit readout of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to leakage in seconds.
        gate_time : float
            Readout time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and readout_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 2
            2 Kraus operators for the radiative loss of rydberg states.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p), 0],
            ]
        )

        return array([identity, leak_err])

    def DECAY_rg(
        self,
        lifetime: float,  # leakage lifetime from the rydberg state to 1S0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to ground state during g-qubit readout of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to 1S0 in seconds.
        gate_time : float
            Readout time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and readout_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 3
            3 Kraus operators for the leakage error from rydberg to ground state.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])

    def DECAY_rm(
        self,
        lifetime: float,  # leakage lifetime from the rydberg state to 3P0 in seconds
        gate_time: float,  # idling time in seconds
        p: float
        | None = None,  # optional, if provided, overrides lifetime and idling_time
    ) -> NDArray[complexfloating]:
        """Leakage error channel from rydberg to 3P0 during g-qubit readout of 171Yb.

        The probability of leakage is calculated as p = 1 - exp(-gate_time / lifetime).

        Parameters
        ----------
        lifetime : float
            Radiative lifetime of the rydberg state to metastable state in seconds.
        gate_time : float
            Readout time for the operation in seconds.
        p : float, optional
            If provided, overrides the lifetime and readout_time to use this probability directly.
            Should be in the range [0, 1].

        Returns:
        -------
        ndarray[ndarray], size = 3
            3 Kraus operators for the leakage error from rydberg to 3P0.
            Each operator is a 6x6 matrix representing the channel. (spanned by |0g>, |1g>, |0m>, |1m>, |r>, |L>)
        """
        p = 1 - exp(-gate_time / lifetime) if p is None else float(p)
        if not (0 <= p <= 1):
            raise ValueError("Leakage error probability p must be in the range [0, 1].")

        identity = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, sqrt(1 - p), 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        leak_err0 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        leak_err1 = array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, sqrt(p / 2), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        return array([identity, leak_err0, leak_err1])
