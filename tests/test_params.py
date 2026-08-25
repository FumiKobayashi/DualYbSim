"""Tests for NoiseModelParameters: grouped access, presets and serialisation."""

import numpy as np
import pytest

from dualybsim import ENCODINGS, NoiseModelParameters


@pytest.fixture
def p() -> NoiseModelParameters:
    return NoiseModelParameters()


class TestForQubit:
    @pytest.mark.parametrize(("isotope", "qubit_type"), sorted(ENCODINGS))
    def test_view_drops_the_encoding_tag(
        self, p: NoiseModelParameters, isotope: str, qubit_type: str
    ):
        tag = ENCODINGS[(isotope, qubit_type)]
        view = p.for_qubit(isotope, qubit_type)
        assert view.tag == tag
        assert view.isotope == isotope
        assert view.qubit_type == qubit_type
        for field in ("p_1", "p_2", "p_meas", "q_BB", "gamma_Z", "gate_time"):
            assert getattr(view, field) == getattr(p, f"{field}_{tag}")

    def test_absent_channels_read_zero(self, p: NoiseModelParameters):
        clock = p.for_qubit("174", "gm")
        # The clock qubit has no nuclear spin, so no in-manifold flips, and no
        # T_1 bit flip channel.
        assert clock.p_flip_g == 0.0
        assert clock.p_flip_m == 0.0
        assert clock.gamma_X == 0.0
        # DEP1_gm / DECAY_mg^(gate) are 171Yb clock-transition channels.
        assert clock.p_1_gm == 0.0
        assert clock.p_m_g_gate == 0.0

    def test_clock_transition_channels_shared_across_171_encodings(
        self, p: NoiseModelParameters
    ):
        g = p.for_qubit("171", "g")
        m = p.for_qubit("171", "m")
        assert g.p_1_gm == m.p_1_gm == p.p_1_gm
        assert g.p_m_g_gate == m.p_m_g_gate == p.p_m_g_gate

    def test_view_is_live(self, p: NoiseModelParameters):
        p.p_1_m = 0.0123
        assert p.for_qubit("171", "m").p_1 == 0.0123

    @pytest.mark.parametrize(
        ("isotope", "qubit_type"),
        [("171", "gm"), ("174", "g"), ("174", "m"), ("173", "m"), ("171", "x")],
    )
    def test_rejects_unknown_encodings(
        self, p: NoiseModelParameters, isotope: str, qubit_type: str
    ):
        with pytest.raises(ValueError, match="Unknown qubit encoding"):
            p.for_qubit(isotope, qubit_type)


class TestGetGateTime:
    def test_returns_the_encoding_specific_value(self, p: NoiseModelParameters):
        assert p.get_gate_time("t_1Q", "174", "gm") == p.gate_time_c["t_1Q"]
        assert p.get_gate_time("t_1Q", "171", "m") == p.gate_time_m["t_1Q"]

    def test_g_only_key_resolves_for_g(self, p: NoiseModelParameters):
        """t_1Q_gm exists only for the g encoding and must not be rejected."""
        assert p.get_gate_time("t_1Q_gm", "171", "g") == p.gate_time_g["t_1Q_gm"]

    def test_missing_key_raises_value_error(self, p: NoiseModelParameters):
        with pytest.raises(ValueError, match="Unsupported gate type"):
            p.get_gate_time("t_1Q_gm", "174", "gm")

    def test_mismatched_encoding_raises_value_error(self, p: NoiseModelParameters):
        with pytest.raises(ValueError, match="Unknown qubit encoding"):
            p.get_gate_time("t_1Q", "174", "m")


class TestPresets:
    def test_paper_defaults_is_the_constructor(self):
        assert (
            NoiseModelParameters.paper_defaults().get_parameters_dict()
            == NoiseModelParameters().get_parameters_dict()
        )

    def test_legacy_differs_only_where_documented(self):
        paper = NoiseModelParameters().get_parameters_dict()
        legacy = NoiseModelParameters.legacy_defaults().get_parameters_dict()
        differing = {
            k for k in paper if not isinstance(paper[k], dict) and paper[k] != legacy[k]
        }
        assert differing == {
            "p_1_c",
            "p_1_g",
            "p_1_m",
            "p_1_gm",
            "p_2_c",
            "p_2_g",
            "p_2_m",
            "p_2_dual",
            "p_meas_c",
            "p_meas_g",
            "p_meas_m",
            "p_g_L_reset_g",
            "p_m_L_reset_m",
            "gamma_gL",
            "gamma_mL",
            "gamma_Z_gm",
            "gamma_X_g",
            "gamma_X_m",
        }

    def test_legacy_switches_off_the_channels_it_never_injected(self):
        legacy = NoiseModelParameters.legacy_defaults()
        assert legacy.gamma_gL == 0.0
        assert legacy.gamma_mL == 0.0
        assert legacy.gamma_Z_gm == 0.0
        assert legacy.gamma_X_g == 0.0
        assert legacy.gamma_X_m == 0.0

    def test_legacy_does_not_restore_the_unphysical_gamma_mg_c(self):
        """The legacy T_2^(c) = 5 s with Gamma_mg = 1 s^-1 is not realisable."""
        legacy = NoiseModelParameters.legacy_defaults()
        assert legacy.gamma_mg_c == NoiseModelParameters().gamma_mg_c
        assert 1 / legacy.gamma_Z_c <= 2 / legacy.gamma_mg_c

    def test_legacy_uses_one_value_for_every_cz_pattern(self):
        legacy = NoiseModelParameters.legacy_defaults()
        assert legacy.p_2_c == legacy.p_2_g == legacy.p_2_m == legacy.p_2_dual


class TestSerialisation:
    def test_round_trip(self, p: NoiseModelParameters, tmp_path):
        p.set_parameters(p_1_c=3e-4, gamma_mg_m=1.5)
        path = tmp_path / "params.txt"
        p.save_noise_params_to_file(str(path))

        loaded = NoiseModelParameters()
        loaded.load_noise_params_from_file(str(path))
        assert loaded.get_parameters_dict() == p.get_parameters_dict()

    def test_every_parameter_is_written(self, p: NoiseModelParameters, tmp_path):
        path = tmp_path / "params.txt"
        p.save_noise_params_to_file(str(path))
        written = {
            line.split(":", 1)[0].split("/", 1)[0]
            for line in path.read_text().splitlines()
            if ":" in line and not line.startswith("#")
        }
        assert written == set(p.get_parameter_names())

    def test_slash_notation(self, p: NoiseModelParameters):
        p.set_parameters(**{"gate_time_c/t_1Q": 200e-6})
        assert p.gate_time_c["t_1Q"] == 200e-6

    def test_unknown_parameter_is_rejected(self, p: NoiseModelParameters):
        with pytest.raises(ValueError, match="Unknown parameter"):
            p.set_parameters(p_gm_174=1.0)

    def test_slash_notation_on_a_scalar_is_rejected(self, p: NoiseModelParameters):
        with pytest.raises(ValueError, match="not a dictionary"):
            p.set_parameters(**{"p_1_c/oops": 1.0})


class TestRescale:
    def test_scales_probabilities_and_rates(self, p: NoiseModelParameters):
        scaled = p.rescale_error_params(0.5)
        assert scaled.p_1_c == pytest.approx(p.p_1_c / 2)
        assert scaled.gamma_gL == pytest.approx(p.gamma_gL / 2)

    def test_leaves_ratios_geometry_and_times_alone(self, p: NoiseModelParameters):
        scaled = p.rescale_error_params(0.5)
        assert scaled.q_BB_c == p.q_BB_c
        assert scaled.l_site == p.l_site
        assert scaled.l_zone == p.l_zone
        assert scaled.a == p.a
        assert scaled.t_hand == p.t_hand
        assert scaled.gate_time_c == p.gate_time_c

    def test_not_inplace_by_default(self, p: NoiseModelParameters):
        before = p.p_1_c
        p.rescale_error_params(0.5)
        assert p.p_1_c == before

    def test_negative_scale_is_rejected(self, p: NoiseModelParameters):
        with pytest.raises(ValueError, match="non-negative"):
            p.rescale_error_params(-1.0)


class TestTimeDependentRate:
    def test_exponential(self, p: NoiseModelParameters):
        assert p.get_time_dependent_rate(1e-3, 1 / 30) == pytest.approx(
            1 - np.exp(-1e-3 / 30)
        )

    def test_branching_ratio_scales_the_rate(self, p: NoiseModelParameters):
        assert p.get_time_dependent_rate(1e-3, 2.0, branching_ratio=0.5) == (
            pytest.approx(p.get_time_dependent_rate(1e-3, 1.0))
        )


class TestReadoutTransport:
    def test_round_trip_includes_handovers_and_measurement(
        self, p: NoiseModelParameters
    ):
        d = 5
        move = p.readout_transport_one_way_time(d)
        total = p.readout_transport_round_trip_time(d)
        expected = 2 * move + 4 * p.t_hand + p.gate_time_c["t_read"]
        assert total == pytest.approx(expected)

    def test_optional_components(self, p: NoiseModelParameters):
        d = 5
        move_only = p.readout_transport_round_trip_time(
            d, include_handover=False, include_measurement=False
        )
        assert move_only == pytest.approx(2 * p.readout_transport_one_way_time(d))
