"""Ties the closed forms back to the Kraus operators they claim to summarise.

``NoiseModelParameters`` carries closed-form expressions for the twirled
measurement channels. Those are only trustworthy if they agree with running the
generalised Pauli twirling approximation directly on the Kraus operators, which
is what these tests check. Marked slow because the twirl builds and
eigendecomposes operators on the full 4- and 6-dimensional Hilbert spaces.
"""

import numpy as np
import pytest

from dualybsim import NoiseModelParameters
from dualybsim.kraus import YbNoiseChannelFactory
from dualybsim.twirling import GeneralizedTwirlingApproximation

pytestmark = pytest.mark.slow


def gpta_rates(channel, subchannel: str) -> dict[str, float]:
    """Run GPTA and split the result into loss and Pauli probabilities."""
    gta = GeneralizedTwirlingApproximation(channel, subchannel=subchannel)
    probs = gta.derive_conditional_probabilities()
    loss = sum(v for k, v in probs.items() if "->L:" in k or "->r:" in k)
    pauli = {k[-1]: v for k, v in probs.items() if k.startswith("c->c:")}
    return {
        "p_loss": float(loss),
        "p_X": float(pauli.get("X", 0.0)),
        "p_Y": float(pauli.get("Y", 0.0)),
        "p_Z": float(pauli.get("Z", 0.0)),
    }


@pytest.mark.parametrize("p_meas", [1e-4, 1e-3, 1e-2])
@pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
class TestMerrClosedFormMatchesGpta:
    """The MERR closed forms must reproduce the twirl of the Kraus channel.

    The closed forms truncate at second order in ``p_meas``, so the residual is
    expected to scale as ``p_meas**3``.
    """

    def test_174(self, p_meas: float, q: float):
        channel = YbNoiseChannelFactory.create_174Yb_MEASURE_DISC_channel(
            p_meas=p_meas, q=q
        )
        got = gpta_rates(channel, "MERR")
        want = NoiseModelParameters().get_twirled_174_measurement_merr_rates(
            p_meas=p_meas, q=q
        )
        for key in ("p_loss", "p_X", "p_Y", "p_Z"):
            assert got[key] == pytest.approx(want[key], abs=5 * p_meas**3), key

    def test_171m(self, p_meas: float, q: float):
        channel = YbNoiseChannelFactory.create_171Yb_MEASURE_DISC_channel(
            p_meas=p_meas, q=q
        )
        got = gpta_rates(channel, "MERR")
        want = NoiseModelParameters().get_twirled_171m_measurement_merr_rates(
            p_meas=p_meas, q=q
        )
        for key in ("p_loss", "p_X", "p_Y", "p_Z"):
            assert got[key] == pytest.approx(want[key], abs=5 * p_meas**3), key


class TestMerrIsExactWhereClaimed:
    """``p_loss = p(1-p)`` is exact; ``p_Z`` carries an ``O(p^3)`` residue."""

    @pytest.mark.parametrize("p_meas", [1e-4, 1e-3, 1e-2])
    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
    def test_loss_is_exact(self, p_meas: float, q: float):
        channel = YbNoiseChannelFactory.create_174Yb_MEASURE_DISC_channel(
            p_meas=p_meas, q=q
        )
        got = gpta_rates(channel, "MERR")
        assert got["p_loss"] == pytest.approx(p_meas * (1 - p_meas), rel=1e-12)

    @pytest.mark.parametrize("p_meas", [1e-4, 1e-3, 1e-2])
    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
    def test_pz_matches_to_third_order(self, p_meas: float, q: float):
        channel = YbNoiseChannelFactory.create_174Yb_MEASURE_DISC_channel(
            p_meas=p_meas, q=q
        )
        got = gpta_rates(channel, "MERR")
        q_bb = 2 * q - 1
        assert got["p_Z"] == pytest.approx((q_bb**2 / 16) * p_meas**2, abs=p_meas**3)


class TestCptp:
    """Every factory-built channel must be a valid quantum channel."""

    FACTORIES = [
        ("174_1Q", YbNoiseChannelFactory.create_174Yb_1Q_channel),
        ("174_2Q", YbNoiseChannelFactory.create_174Yb_2Q_channel),
        ("174_RESET", YbNoiseChannelFactory.create_174Yb_RESET_channel),
        ("174_MEAS_DISC", YbNoiseChannelFactory.create_174Yb_MEASURE_DISC_channel),
        ("174_MEAS_READ", YbNoiseChannelFactory.create_174Yb_MEASURE_READ_channel),
        ("171_1Q", YbNoiseChannelFactory.create_171Yb_1Q_channel),
        ("171_1Q_CLOCK", YbNoiseChannelFactory.create_171Yb_1Q_clock_channel),
        ("171_2Q", YbNoiseChannelFactory.create_171Yb_2Q_channel),
        ("171_RESET", YbNoiseChannelFactory.create_171Yb_RESET_channel),
        ("171_MEAS_DISC", YbNoiseChannelFactory.create_171Yb_MEASURE_DISC_channel),
        ("171_MEAS_READ", YbNoiseChannelFactory.create_171Yb_MEASURE_READ_channel),
    ]

    @pytest.mark.parametrize(("label", "factory"), FACTORIES)
    def test_each_subchannel_is_trace_preserving(self, label: str, factory):
        channel = factory()
        per_channel = channel.get_kraus_operators()
        assert isinstance(per_channel, dict)
        for name, ops in per_channel.items():
            dim = ops[0].shape[0]
            completeness = sum(K.conj().T @ K for K in ops)
            np.testing.assert_allclose(
                completeness,
                np.eye(dim),
                atol=1e-10,
                err_msg=f"{label}/{name} is not trace preserving",
            )


def absolute_rates(channel, subchannel: str) -> dict[str, float]:
    """As :func:`gpta_rates`, but with the Pauli terms de-conditioned.

    ``derive_conditional_probabilities`` returns the Pauli weights conditioned on
    staying inside the computational subspace. Multiplying by ``P(c -> c)`` gives
    the absolute per-operation probability, which is what a Stim instruction
    takes.
    """
    rates = gpta_rates(channel, subchannel)
    p_cc = 1.0 - rates["p_loss"]
    return {
        "p_loss": rates["p_loss"],
        "p_X": p_cc * rates["p_X"],
        "p_Y": p_cc * rates["p_Y"],
        "p_Z": p_cc * rates["p_Z"],
    }


class TestDep1GmStrictTwirl:
    """The twirl of DEP1_gm depends on which manifold holds the qubit.

    Both encodings lose ``2p/3`` and keep no X or Y, but the surviving phase flip
    differs: the pair-resolved ``Z_{g_j,m_k}`` operators act as the identity on
    the ground manifold and only their complement parts contribute, whereas on
    the metastable manifold they act as a genuine ``Z``.
    """

    @pytest.mark.parametrize("p_dep", [1e-4, 1e-3, 1e-2])
    @pytest.mark.parametrize(
        ("qubit_type", "pz_coefficient"), [("ground", 1 / 3), ("metastable", 1.0)]
    )
    def test_twirl(self, p_dep: float, qubit_type: str, pz_coefficient: float):
        channel = YbNoiseChannelFactory.create_171Yb_1Q_clock_channel(
            p_dep1=p_dep, qubit_type=qubit_type
        )
        got = absolute_rates(channel, "DEP1_gm")
        assert got["p_X"] == pytest.approx(0.0, abs=1e-14)
        assert got["p_Y"] == pytest.approx(0.0, abs=1e-14)
        assert got["p_Z"] == pytest.approx(pz_coefficient * p_dep, rel=1e-12)
        assert got["p_loss"] == pytest.approx(2 * p_dep / 3, rel=1e-12)

    def test_ground_manifold_conserves_the_depolarising_weight(self):
        """For a g qubit the surviving Z plus the loss add back up to p."""
        p_dep = 1e-3
        channel = YbNoiseChannelFactory.create_171Yb_1Q_clock_channel(
            p_dep1=p_dep, qubit_type="ground"
        )
        got = absolute_rates(channel, "DEP1_gm")
        assert got["p_Z"] + got["p_loss"] == pytest.approx(p_dep, rel=1e-12)

    def test_clock_pulse_uses_the_metastable_twirl_for_both_encodings(self):
        """The g qubit's clock pulse takes the m-qubit rates, deliberately.

        ``DEP1_gm`` only reaches a g qubit while the clock pulse is driving it
        into the metastable manifold, so the population it acts on is the
        metastable one. Twirling onto the ground manifold instead would describe
        an atom that stayed put, which is not what the pulse does. Both encodings
        therefore use the metastable rates: ``p_Z = p_1_gm``, loss ``2/3 p_1_gm``.
        """
        from dualybsim import NoiseModelParameters, QubitManager, YbCircuit

        expected = absolute_rates(
            YbNoiseChannelFactory.create_171Yb_1Q_clock_channel(
                p_dep1=NoiseModelParameters().p_1_gm, qubit_type="metastable"
            ),
            "DEP1_gm",
        )

        qm = QubitManager()
        for q in (0, 1):
            qm.add_qubit(q, isotope="171", qubit_type="g", role="data")
        p = NoiseModelParameters()
        circuit = YbCircuit(qm, p)
        circuit.two_qubit_gate("CZ", [0, 1])

        z_rates = [
            inst.gate_args_copy()[0]
            for inst in circuit.flattened()
            if inst.name == "Z_ERROR"
        ]
        assert z_rates, "the clock pulse must emit a phase flip"
        assert z_rates[0] == pytest.approx(expected["p_Z"], rel=1e-12)
        assert expected["p_Z"] == pytest.approx(p.p_1_gm, rel=1e-12)
