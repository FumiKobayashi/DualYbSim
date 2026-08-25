"""Pins the library to the paper it implements.

Every value here is transcribed from the paper's noise-model appendix rather than
read back from the code, so a drift in either direction shows up as a failure.
The final class asserts the handful of places where the library departs from the
paper on purpose, so that changing one of those is a visible change rather than a
silent one.
"""

import numpy as np
import pytest

from dualybsim import NoiseModelParameters
from dualybsim.kraus import YbNoiseChannelFactory
from dualybsim.kraus.yb171 import KrausMEASURE_DISC_171m
from dualybsim.kraus.yb174 import KrausMEASURE_DISC_174

PCT = 1e-2  # the paper quotes several probabilities as percentages


# ---------------------------------------------------------------------------
# Table of noise channels: default parameter values
# ---------------------------------------------------------------------------


class TestPaperTableDefaults:
    """``NoiseModelParameters()`` must reproduce the paper's tabulated values."""

    @pytest.fixture
    def p(self) -> NoiseModelParameters:
        return NoiseModelParameters()

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            # Coherent-control: p_1^(c/gm/g/m) = 0.01 %, p_2^(c/m/dual) = 0.1 %
            ("p_1_c", 0.01 * PCT),
            ("p_1_gm", 0.01 * PCT),
            ("p_1_g", 0.01 * PCT),
            ("p_1_m", 0.01 * PCT),
            ("p_2_c", 0.1 * PCT),
            ("p_2_m", 0.1 * PCT),
            ("p_2_dual", 0.1 * PCT),
            # DECAY_mg^(gate) = 0.1 %
            ("p_m_g_gate", 0.1 * PCT),
            # Measurement: p_(g->L)^(meas) = 0.1 %, p_flip^(g) = 0.1 %,
            # p_meas = 0.01 %
            ("p_g_L_meas_c", 0.1 * PCT),
            ("p_g_L_meas_g", 0.1 * PCT),
            ("p_g_L_meas_m", 0.1 * PCT),
            ("p_flip_g_g", 0.1 * PCT),
            ("p_flip_g_m", 0.1 * PCT),
            ("p_meas_c", 0.01 * PCT),
            ("p_meas_g", 0.01 * PCT),
            ("p_meas_m", 0.01 * PCT),
            # Reset: p_(g->L)^(reset) = 0.1 %, p_(m->L)^(reset) = 0.6 %,
            # p_flip^(m) = 0.1 %
            ("p_g_L_reset_c", 0.1 * PCT),
            ("p_g_L_reset_g", 0.1 * PCT),
            ("p_m_L_reset_m", 0.6 * PCT),
            ("p_flip_m_m", 0.1 * PCT),
            # Transportation: p_(g/m->L)^(hand) = 0.1 %
            ("p_hand_c", 0.1 * PCT),
            ("p_hand_g", 0.1 * PCT),
            ("p_hand_m", 0.1 * PCT),
        ],
    )
    def test_probability(self, p: NoiseModelParameters, name: str, expected: float):
        assert getattr(p, name) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        ("name", "timescale_s"),
        [
            # Idling: T_2^(c) = T_2^(gm) = 5 s, T_2^(g) = T_2^(m) = 10 s,
            # T_1^(g) = T_1^(m) = 200 s
            ("gamma_Z_c", 5.0),
            ("gamma_Z_gm", 5.0),
            ("gamma_Z_g", 10.0),
            ("gamma_Z_m", 10.0),
            ("gamma_X_g", 200.0),
            ("gamma_X_m", 200.0),
            # Decay: Gamma_gL = Gamma_mL = (30 s)^-1, Gamma_Ryd = (50 us)^-1
            ("gamma_gL", 30.0),
            ("gamma_mL", 30.0),
            ("gamma_Ryd", 50e-6),
        ],
    )
    def test_rate_is_inverse_timescale(
        self, p: NoiseModelParameters, name: str, timescale_s: float
    ):
        assert getattr(p, name) == pytest.approx(1.0 / timescale_s, rel=1e-12)

    def test_rydberg_branching(self, p: NoiseModelParameters):
        # 0.42 to the ground manifold, 0.07 to the metastable manifold, 0.51 lost.
        assert p.ryd_branching == {
            "LOSS_r": 0.51,
            "DECAY_rg": 0.42,
            "DECAY_rm": 0.07,
        }
        assert sum(p.ryd_branching.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("tag", "key", "expected_s"),
        [
            # t_1Q^(c) = 100 us, t_1Q^(gm) = 10 us, t_1Q^(g) = t_1Q^(m) = 1 us
            ("c", "t_1Q", 100e-6),
            ("g", "t_1Q_gm", 10e-6),
            ("g", "t_1Q", 1e-6),
            ("m", "t_1Q", 1e-6),
            # t_2Q = 0.3 us for every pattern
            ("c", "t_2Q", 0.3e-6),
            ("g", "t_2Q", 0.3e-6),
            ("m", "t_2Q", 0.3e-6),
            # t_read = 1 ms, t_reset = 2 ms
            ("c", "t_read", 1e-3),
            ("g", "t_read", 1e-3),
            ("m", "t_read", 1e-3),
            ("c", "t_reset", 2e-3),
            ("g", "t_reset", 2e-3),
            ("m", "t_reset", 2e-3),
        ],
    )
    def test_operation_time(
        self, p: NoiseModelParameters, tag: str, key: str, expected_s: float
    ):
        assert getattr(p, f"gate_time_{tag}")[key] == pytest.approx(expected_s)

    def test_transport_geometry(self, p: NoiseModelParameters):
        assert p.l_site == pytest.approx(3e-6)  # site spacing 3 um
        assert p.l_zone == pytest.approx(100e-6)  # zoned separation 100 um
        assert p.t_hand == pytest.approx(200e-6)  # handover 200 us
        assert p.a == pytest.approx(5500)  # acceleration 5500 m/s^2

    def test_bb_assignment_is_even(self, p: NoiseModelParameters):
        # q_BB = 1/2 in the paper, which makes the MERR p_Z vanish.
        assert p.q_BB_c == 0.5
        assert p.q_BB_g == 0.5
        assert p.q_BB_m == 0.5


# ---------------------------------------------------------------------------
# Transport timing
# ---------------------------------------------------------------------------


class TestTransportTiming:
    def test_move_time_has_the_factor_of_two(self):
        # t_move(l) = 2 sqrt(l / a)
        p = NoiseModelParameters()
        for length in (1e-6, 22e-6, 1e-3):
            assert p.transportation_time(length) == pytest.approx(
                2 * np.sqrt(length / p.a)
            )

    def test_readout_distance(self):
        # l = d * l_site + l_zone
        p = NoiseModelParameters()
        for d in (3, 5, 7, 11):
            assert p.readout_transport_one_way_distance(d) == pytest.approx(
                d * p.l_site + p.l_zone
            )


# ---------------------------------------------------------------------------
# Readout detection table and its generalised Pauli twirl
# ---------------------------------------------------------------------------


def _detection_table(kraus, labels: dict[str, int]) -> dict[str, float]:
    """p_{a,b} = sum_i |<b|K_i|a>|^2 read straight off the Kraus operators."""
    ops = [kraus[i] for i in range(kraus.shape[0])]
    return {
        f"{an}->{bn}": float(sum(abs(K[bi, ai]) ** 2 for K in ops))
        for an, ai in labels.items()
        for bn, bi in labels.items()
    }


def _expected_table(p: float, q: float) -> dict[str, float]:
    """The paper's readout detection table, for general BD assignment ratio q.

    Enumerating the two imaging steps: a true |0> is ideally (bright, dark), so
    the record DB needs both steps wrong and carries p^2, while DD needs only the
    first step wrong and carries p(1-p). BB is unphysical and is split by q.
    """
    return {
        "0->0": (1 - p) ** 2 + q * p * (1 - p),
        "0->1": p**2 + (1 - q) * p * (1 - p),
        "0->L": p * (1 - p),
        "1->0": p**2 + q * p * (1 - p),
        "1->1": (1 - p) ** 2 + (1 - q) * p * (1 - p),
        "1->L": p * (1 - p),
        "L->0": p * (1 - p) + q * p**2,
        "L->1": p * (1 - p) + (1 - q) * p**2,
        "L->L": (1 - p) ** 2,
    }


@pytest.mark.parametrize("p_meas", [1e-4, 1e-3, 1e-2])
@pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
class TestReadoutDetectionTable:
    """The Kraus operators must match the paper's detection table."""

    def test_174(self, p_meas: float, q: float):
        kraus = KrausMEASURE_DISC_174(p_meas, q).MERR(p_meas, q)
        got = _detection_table(kraus, {"0": 0, "1": 1, "L": 3})
        want = _expected_table(p_meas, q)
        for key in want:
            assert got[key] == pytest.approx(want[key], abs=1e-14), key

    def test_171m(self, p_meas: float, q: float):
        kraus = KrausMEASURE_DISC_171m(p_meas, q).MERR(p_meas, q)
        got = _detection_table(kraus, {"0": 0, "1": 1, "L": 5})
        want = _expected_table(p_meas, q)
        for key in want:
            assert got[key] == pytest.approx(want[key], abs=1e-14), key

    def test_rows_sum_to_one(self, p_meas: float, q: float):
        want = _expected_table(p_meas, q)
        for a in ("0", "1", "L"):
            total = sum(want[f"{a}->{b}"] for b in ("0", "1", "L"))
            assert total == pytest.approx(1.0)


class TestTwirledMerr:
    """The twirled MERR rates must match the first-principles derivation.

    ``p_loss = p(1-p)`` and ``p_Z = ((2q-1)^2/16) p^2`` are exact; ``p_X = p_Y``
    is exact to second order.

    ``p_Z`` is written in terms of the asymmetry ``2q - 1`` between the two ways
    of resolving an ambiguous bright-bright readout record, which is the quantity
    that survives the twirl. It therefore vanishes at an even split rather than
    at ``q = 0``.
    """

    @pytest.mark.parametrize("p_meas", [1e-4, 1e-3, 1e-2])
    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
    def test_174_closed_form(self, p_meas: float, q: float):
        rates = NoiseModelParameters().get_twirled_174_measurement_merr_rates(
            p_meas=p_meas, q=q
        )
        q_bb = 2 * q - 1
        assert rates["p_loss"] == pytest.approx(p_meas * (1 - p_meas))
        assert rates["p_X"] == pytest.approx(p_meas / 4 + p_meas**2 / 2)
        assert rates["p_Y"] == rates["p_X"]
        assert rates["p_Z"] == pytest.approx((q_bb**2 / 16) * p_meas**2)

    def test_pz_vanishes_at_even_split(self):
        rates = NoiseModelParameters().get_twirled_174_measurement_merr_rates(
            p_meas=1e-3, q=0.5
        )
        assert rates["p_Z"] == 0.0

    def test_171m_composite_first_order(self):
        """The 6-input closed form's linear response, coefficient by coefficient.

        p_loss = p_meas + (2/3) p_1^(gm) + p_(g->L)^(meas) + p_(m->g)(t_read)
        p_X    = (1/4) p_meas + p_flip^(g)
        p_Y    = (1/4) p_meas
        p_Z    = p_1^(gm) + p_Z^(m)(t_read)
        """
        p = NoiseModelParameters()
        t_read = p.gate_time_m["t_read"]
        # One input at a time, small enough that the quadratic terms are
        # negligible against the linear ones.
        eps = 1e-6
        zero = {
            "p_meas": 0.0,
            "p_dep_gm": 0.0,
            "p_X_g": 0.0,
            "p_loss_RO": 0.0,
            "T2_m": 1e12,
            "leaktime_eg": 1e12,
            "gate_time_measure": t_read,
            "q": 0.5,
        }

        def rates(**over):
            return p.get_twirled_171m_measurement_error_rates(**{**zero, **over})

        base = rates()
        for key in ("p_loss", "p_X", "p_Y", "p_Z"):
            assert base[key] == pytest.approx(0.0, abs=1e-15), key

        r = rates(p_meas=eps)
        assert r["p_loss"] == pytest.approx(eps, rel=1e-3)
        assert r["p_X"] == pytest.approx(eps / 4, rel=1e-3)
        assert r["p_Y"] == pytest.approx(eps / 4, rel=1e-3)

        r = rates(p_dep_gm=eps)
        assert r["p_loss"] == pytest.approx(2 * eps / 3, rel=1e-3)
        assert r["p_Z"] == pytest.approx(eps, rel=1e-3)

        r = rates(p_X_g=eps)
        assert r["p_X"] == pytest.approx(eps, rel=1e-3)

        r = rates(p_loss_RO=eps)
        assert r["p_loss"] == pytest.approx(eps, rel=1e-3)

        # T2_m and leaktime_eg enter through 1 - exp(-t_read / T).
        r = rates(T2_m=t_read / eps)
        assert r["p_Z"] == pytest.approx(eps, rel=1e-2)
        r = rates(leaktime_eg=t_read / eps)
        assert r["p_loss"] == pytest.approx(eps, rel=1e-2)


# ---------------------------------------------------------------------------
# Channel names
# ---------------------------------------------------------------------------


class TestChannelNames:
    """Every Kraus channel key must be a channel name used by the paper."""

    PAPER_CHANNELS = {
        # Coherent-control
        "DEP1_c",
        "DEP1_g",
        "DEP1_m",
        "DEP1_gm",
        "DECAY_mg_gate",
        "DEP2_c",
        "DEP2_m",
        "DEP2_dual",
        # Measurement
        "LOSS_g_meas",
        "FLIP_g",
        "MERR",
        # Reset
        "LOSS_g_reset",
        "LOSS_m_reset",
        "FLIP_m",
        # Idling
        "ZERR_c",
        "ZERR_g",
        "ZERR_m",
        "ZERR_gm",
        "XERR_g",
        "XERR_m",
        # Decay
        "LOSS_r",
        "DECAY_rg",
        "DECAY_rm",
        "DECAY_mg",
        "LOSS_g",
        "LOSS_m",
        # Transportation
        "LOSS_g_hand",
        "LOSS_m_hand",
    }

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
    def test_every_channel_is_a_paper_channel(self, label: str, factory):
        channels = set(factory().get_available_channels())
        unknown = channels - self.PAPER_CHANNELS
        assert not unknown, f"{label} exposes non-paper channel names: {unknown}"

    def test_no_leak_naming_survives(self):
        """The paper's conventions reject *leakage* as a way to classify errors."""
        for label, factory in self.FACTORIES:
            for name in factory().get_available_channels():
                assert "LEAK" not in name.upper(), f"{label}: {name}"


# ---------------------------------------------------------------------------
# Twirled amplitude damping
# ---------------------------------------------------------------------------


class TestTwirledAmplitudeDamping:
    """p_X = p_Y = p_1/4, p_Z = p_2/2 - p_1/4, reproducing exp(-t/T_2)."""

    def test_reproduces_the_transverse_decay(self):
        p = NoiseModelParameters()
        T_1, T_2 = 1 / p.gamma_mg_c, 1 / p.gamma_Z_c
        for t in (1e-4, 1e-3, 2e-3, 3.2e-3):
            p_X, p_Y, p_Z = p.twirled_amplitude_damping(
                t, T_1_inv=1 / T_1, T_2_inv=1 / T_2
            )
            # A Pauli channel maps the transverse Bloch component by
            # 1 - 2 p_Y - 2 p_Z; that must equal exp(-t / T_2).
            assert 1 - 2 * p_Y - 2 * p_Z == pytest.approx(np.exp(-t / T_2), rel=1e-12)
            assert p_X == p_Y

    def test_damping_limit_has_no_extra_dephasing(self):
        p = NoiseModelParameters()
        assert p.twirled_amplitude_damping(1e-3, T_1_inv=p.gamma_mg_c)[2] == 0.0

    def test_rejects_t2_above_twice_t1(self):
        p = NoiseModelParameters()
        with pytest.raises(ValueError, match="T_2 <= 2 T_1"):
            p.twirled_amplitude_damping(1e-3, T_1_inv=1.0, T_2_inv=1 / 5.0)

    def test_clock_qubit_parameters_are_realisable(self):
        """T_2^(c) <= 2 T_1: the reason gamma_mg_c is the natural 3P0 lifetime."""
        p = NoiseModelParameters()
        assert 1 / p.gamma_Z_c <= 2 / p.gamma_mg_c


# ---------------------------------------------------------------------------
# Rydberg branch renormalisation
# ---------------------------------------------------------------------------


class TestRydbergBranches:
    """Each branch is renormalised by the population the earlier ones removed."""

    def test_matches_the_paper_expression(self):
        p = NoiseModelParameters()
        t = p.gate_time_c["t_2Q"]
        total = 1 - np.exp(-t * p.gamma_Ryd)
        hat_L = p.ryd_branching["LOSS_r"] * total
        hat_g = p.ryd_branching["DECAY_rg"] * total
        hat_m = p.ryd_branching["DECAY_rm"] * total

        got = p.rydberg_branch_rates(t)
        assert got["LOSS_r"] == pytest.approx(hat_L)
        assert got["DECAY_rg"] == pytest.approx(hat_g / (1 - hat_L))
        assert got["DECAY_rm"] == pytest.approx(hat_m / ((1 - hat_L) * (1 - hat_g)))

    def test_zero_duration(self):
        got = NoiseModelParameters().rydberg_branch_rates(0.0)
        assert got == {"LOSS_r": 0.0, "DECAY_rg": 0.0, "DECAY_rm": 0.0}

    def test_sequential_application_hits_the_aggregate_branch_fractions(self):
        """Applied in order, the absolute fraction reaching each destination.

        The renormalisation is exact for the first two branches. The third is
        exact only to leading order, because the paper divides by
        ``(1 - p_hat[r->L])(1 - p_hat[r->g])`` where the population actually
        surviving the first two channels is
        ``(1 - p_hat[r->L]) - p_hat[r->g]``. The two differ at
        ``O(p_hat^2)``, which at the two-qubit gate time is a relative 8e-6 on a
        branch of 4e-4, i.e. 3e-9 absolute.
        """
        p = NoiseModelParameters()
        t = p.gate_time_c["t_2Q"]
        total = 1 - np.exp(-t * p.gamma_Ryd)
        hat = {k: v * total for k, v in p.ryd_branching.items()}
        r = p.rydberg_branch_rates(t)

        reaching_L = r["LOSS_r"]
        reaching_g = (1 - r["LOSS_r"]) * r["DECAY_rg"]
        reaching_m = (1 - r["LOSS_r"]) * (1 - r["DECAY_rg"]) * r["DECAY_rm"]

        assert reaching_L == pytest.approx(hat["LOSS_r"], rel=1e-15)
        assert reaching_g == pytest.approx(hat["DECAY_rg"], rel=1e-15)
        assert reaching_m == pytest.approx(hat["DECAY_rm"], rel=1e-4)


# ---------------------------------------------------------------------------
# Documented deviations
# ---------------------------------------------------------------------------


class TestDeliberateDepartures:
    """Where the library departs from the paper on purpose.

    Pinned so that changing any of these is a deliberate edit to this file rather
    than an unnoticed drift.
    """

    def test_gamma_mg_is_per_encoding(self):
        p = NoiseModelParameters()
        # The clock qubit idles in a shallow trap, so it uses the natural 3P0
        # lifetime rather than the imaging-time rate the paper tabulates.
        assert p.gamma_mg_c == pytest.approx(1 / 20.0)
        assert p.gamma_mg_g == pytest.approx(1.0)
        assert p.gamma_mg_m == pytest.approx(1.0)

    def test_171_zerr_follows_the_papers_literal_definition(self):
        """171Yb ZERR uses p_Z = 1 - exp(-t/T_2), which dephases at 2/T_2.

        A Pauli-Z channel of probability p maps the transverse Bloch component by
        1 - 2p, so the paper's definition decays coherence at twice the rate its
        T_2 implies; reproducing exp(-t/T_2) would need p_Z / 2. The library
        follows the definition as written, so the effective coherence time of the
        171Yb encodings is about T_2 / 2. Pinned here so that changing the
        convention is deliberate.
        """
        p = NoiseModelParameters()
        t, T_2 = 1e-3, 1 / p.gamma_Z_m
        p_Z = p.get_time_dependent_rate(t, p.gamma_Z_m)
        assert p_Z == pytest.approx(1 - np.exp(-t / T_2))
        # The transverse decay this produces is twice as fast as exp(-t/T_2).
        transverse = 1 - 2 * p_Z
        assert transverse == pytest.approx(np.exp(-2 * t / T_2), rel=1e-5)
