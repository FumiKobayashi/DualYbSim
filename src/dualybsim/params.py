"""Noise parameters for dual-isotope Yb quantum devices.

Holds :class:`NoiseModelParameters`, the single source of truth for every rate,
probability and operation time used by the noise model, together with the
closed-form twirled measurement channels derived from the Kraus operators in
:mod:`dualybsim.kraus`.

:class:`~dualybsim.circuit.YbCircuit` reads these values to decide which Stim
noise instructions to emit for each operation.
"""

import ast
import copy
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# 6-input twirled 171Yb m-qubit measurement closed-form coefficients.
# Inputs: (p_meas, p_dep_gm, p_X_g, p_loss_RO, p_MZ_idl, p_MGL_idl)
# Outputs: (p_loss, p_X, p_Y, p_Z)
# Validated against GTA to <= 1.0e-6 over [0, 1e-2]^6 at q in {0, 0.5, 1.0}.
_TWIRLED_171M_6INPUT_INPUTS = (
    "p_meas",
    "p_dep_gm",
    "p_X_g",
    "p_loss_RO",
    "p_MZ_idl",
    "p_MGL_idl",
)
_TWIRLED_171M_6INPUT_OUTPUTS = ("p_loss", "p_X", "p_Y", "p_Z")
# Documents the closed form evaluated by the twirled-measurement helpers below,
# together with the M_POLY / Q_POLY tables that implement it. Kept as a comment
# block rather than a string literal: a bare string after a module-level assignment
# is an unreachable PEP 258 attribute docstring, not documentation anyone can read.
# The closed form is expressed in matrix / tensor form so that physical
# correspondence is direct:
#
#     y = M(q) p  +  [p^T Q_i(q) p]_i
#
# where p is the (6,) input vector ordered as INPUTS, y is the (4,) output
# vector ordered as OUTPUTS, M(q) is a (4, 6) linear response matrix, and
# each Q_i(q) is a (6, 6) symmetric coupling matrix for output i. Each of
# M and Q_i is itself a quadratic polynomial in q:
#
#     M(q)   = M_POLY[0]   + q M_POLY[1]   + q^2 M_POLY[2]
#     Q_i(q) = Q_POLY[0,i] + q Q_POLY[1,i] + q^2 Q_POLY[2,i]
#
# The quadratic tensor is pruned at |c| < 1e-2 (34 of 252 entries survive);
# the dominants are
# p_loss: |p_meas*p_loss_RO * a|     = 2.96
# p_X   : |p_dep_gm*p_X_g * a|       = 1.00
# p_Y   : |p_X_g*p_MZ_idl * a|       = 0.99
# p_Z   : |p_dep_gm*p_MZ_idl * a|    = 2.00 ).
# Packed and re-validated offline:
# worst-case residual of the pruned+snapped table versus fresh GTA samples is
# 3.5e-6 over the operating envelope p_i in [0, 1e-2] and q in [0, 1]. All six
# input channels retain at least one surviving entry, so the closed form
# remains structurally responsive if p_X_g or p_loss_RO are later calibrated
# to non-zero values.
# Linear response polynomial in q: M(q) = M_POLY[0] + q*M_POLY[1].
# Shape (2, 4, 6); leading axis = [a, b] (constant / linear in q).
# Middle axis = output (p_loss, p_X, p_Y, p_Z); last axis = input
# (p_meas, p_dep_gm, p_X_g, p_loss_RO, p_MZ_idl, p_MGL_idl). Each row M[oi, :]
# is the linear slope vector for output `oi`.
#
# After snapping to clean physical values: every nonzero entry is one of
# {1, 2/3, 1/4}. Both the b (q^1) and c (q^2) slices came out identically
# zero, i.e. the linear response is now completely independent of q -- the
# corrected BD MERR channel contributes p_loss = p(1-p) and
# p_X = p_Y = p/4 + p^2/2 with no q dependence at all, and the only surviving
# q dependence in the whole closed form is the MERR p_Z = ((2q-1)^2/16) p_meas^2
# term, which lives in the quadratic table. The all-zero b slice is retained
# so the M(q) = M_POLY[0] + q*M_POLY[1] evaluation stays general. Reading guide:
# p_loss = p_meas + (2/3) p_dep_gm + p_loss_RO + p_MGL_idl + (Q quadratic)
#                 <- BD MERR p(1-p); the -p^2 sits in Q as p_meas^2 ~ -1
# p_X    = (1/4) p_meas + p_X_g                            + (Q quadratic)
#                 <- BD MERR p/4;   the +p^2/2 sits in Q as p_meas^2 ~ +1/2
# p_Y    = (1/4) p_meas                                    + (Q quadratic)
# p_Z    = p_dep_gm + p_MZ_idl                             + (Q quadratic)
#                 <- BD MERR p_Z is purely quadratic: p_meas^2 entries
#                    (a, b, c) ~ (1/16, -1/4, +1/4) = ((2q-1)^2)/16
# fmt: off
_TWIRLED_171M_6INPUT_M_POLY = np.array([
    # ---- a (constant) ----
    [
        [ 1.000000000000e+00,  6.666666666667e-01,  0.000000000000e+00,  1.000000000000e+00,  0.000000000000e+00,  1.000000000000e+00],  # p_loss
        [ 2.500000000000e-01,  0.000000000000e+00,  1.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00],  # p_X
        [ 2.500000000000e-01,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00],  # p_Y
        [ 0.000000000000e+00,  1.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  1.000000000000e+00,  0.000000000000e+00],  # p_Z
    ],
    # ---- b (linear in q): identically zero after the MERR transposition fix ----
    [
        [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00],  # p_loss
        [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00],  # p_X
        [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00],  # p_Y
        [ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00],  # p_Z
    ]
])
# fmt: on

# Index list for the 21 unique upper-triangular pairs (i, j) with i <= j,
# indexing into INPUTS. Used to extract pair products p[i]*p[j] for the
# packed quadratic form below.
_TWIRLED_171M_6INPUT_Q_PAIRS = (
    (0, 0),  # p_meas^2
    (0, 1),  # p_meas*p_dep_gm
    (0, 2),  # p_meas*p_X_g
    (0, 3),  # p_meas*p_loss_RO
    (0, 4),  # p_meas*p_MZ_idl
    (0, 5),  # p_meas*p_MGL_idl
    (1, 1),  # p_dep_gm^2
    (1, 2),  # p_dep_gm*p_X_g
    (1, 3),  # p_dep_gm*p_loss_RO
    (1, 4),  # p_dep_gm*p_MZ_idl
    (1, 5),  # p_dep_gm*p_MGL_idl
    (2, 2),  # p_X_g^2
    (2, 3),  # p_X_g*p_loss_RO
    (2, 4),  # p_X_g*p_MZ_idl
    (2, 5),  # p_X_g*p_MGL_idl
    (3, 3),  # p_loss_RO^2
    (3, 4),  # p_loss_RO*p_MZ_idl
    (3, 5),  # p_loss_RO*p_MGL_idl
    (4, 4),  # p_MZ_idl^2
    (4, 5),  # p_MZ_idl*p_MGL_idl
    (5, 5),  # p_MGL_idl^2
)
# Precomputed numpy arrays of pair indices for fast extraction of
# p[i] * p[j] across the 21 pairs.
_TWIRLED_171M_6INPUT_Q_PAIR_I = np.array([p[0] for p in _TWIRLED_171M_6INPUT_Q_PAIRS])
_TWIRLED_171M_6INPUT_Q_PAIR_J = np.array([p[1] for p in _TWIRLED_171M_6INPUT_Q_PAIRS])

# Sparse quadratic coupling: list of (slice, output, pair, coef) tuples.
# slice in {0, 1, 2} = q^0, q^1, q^2; output in {0..3} = (p_loss, p_X, p_Y,
# p_Z); pair in {0..20} indexes _TWIRLED_171M_6INPUT_Q_PAIRS; coef is the
# cross-term coefficient c_ij directly (not c_ij / 2). The quadratic
# contribution to output oi is the sum over (s, oi, k, v) of
#   v * q^s * p[i_k] * p[j_k],   (i_k, j_k) = _TWIRLED_171M_6INPUT_Q_PAIRS[k].
# Originating dense (3, 4, 21) tensor was pruned at |c| < 1e-2; 34 of 252
# entries survive and the remainder are treated as exact zeros. The pruned +
# snapped table was re-validated against fresh GTA samples at 3.5e-6 worst-case
# absolute error over p_i in [0, 1e-2] and q in [0, 1].
# The (2, 3, 0) / (1, 3, 0) / (0, 3, 0) entries below are the BD MERR
# p_Z = ((2q-1)^2/16) p_meas^2 term, i.e. (a, b, c) ~ (1/16, -1/4, +1/4); the
# few remaining b / c entries near the 1e-2 threshold are fit noise.
# fmt: off
_TWIRLED_171M_6INPUT_Q_SPARSE = (
    # ---- a (constant in q) ----
    (0, 0,  0, -9.834325257597e-01),  # p_loss <- p_meas*p_meas
    (0, 0,  1, -6.510470023137e-01),  # p_loss <- p_meas*p_dep_gm
    (0, 0,  3, -2.958771305920e+00),  # p_loss <- p_meas*p_loss_RO
    (0, 0,  5, -9.771072362682e-01),  # p_loss <- p_meas*p_MGL_idl
    (0, 0,  8, -6.587514252074e-01),  # p_loss <- p_dep_gm*p_loss_RO
    (0, 0, 10, -6.630297354974e-01),  # p_loss <- p_dep_gm*p_MGL_idl
    (0, 0, 17, -9.891082883817e-01),  # p_loss <- p_loss_RO*p_MGL_idl
    (0, 1,  0, +4.951253527773e-01),  # p_X    <- p_meas*p_meas
    (0, 1,  2, -7.704009340716e-01),  # p_X    <- p_meas*p_X_g
    (0, 1,  3, +4.918839416805e-01),  # p_X    <- p_meas*p_loss_RO
    (0, 1,  7, -9.954334202225e-01),  # p_X    <- p_dep_gm*p_X_g
    (0, 1, 13, -9.882748116114e-01),  # p_X    <- p_X_g*p_MZ_idl
    (0, 2,  0, +4.992821686714e-01),  # p_Y    <- p_meas*p_meas
    (0, 2,  2, -2.570614405420e-01),  # p_Y    <- p_meas*p_X_g
    (0, 2,  3, +5.026225651289e-01),  # p_Y    <- p_meas*p_loss_RO
    (0, 2,  7, +9.964189386582e-01),  # p_Y    <- p_dep_gm*p_X_g
    (0, 2, 13, +9.899892974436e-01),  # p_Y    <- p_X_g*p_MZ_idl
    (0, 3,  0, +5.596003827313e-02),  # p_Z    <- p_meas*p_meas
    (0, 3,  1, -5.147350141864e-01),  # p_Z    <- p_meas*p_dep_gm
    (0, 3,  2, +2.582738714525e-01),  # p_Z    <- p_meas*p_X_g
    (0, 3,  3, +4.890608579237e-01),  # p_Z    <- p_meas*p_loss_RO
    (0, 3,  4, -5.108653753049e-01),  # p_Z    <- p_meas*p_MZ_idl
    (0, 3,  6, +6.603950860850e-01),  # p_Z    <- p_dep_gm*p_dep_gm
    (0, 3,  7, -9.968731125794e-01),  # p_Z    <- p_dep_gm*p_X_g
    (0, 3,  9, -1.999018923412e+00),  # p_Z    <- p_dep_gm*p_MZ_idl
    (0, 3, 13, -9.903997737090e-01),  # p_Z    <- p_X_g*p_MZ_idl
    # ---- b (q^1) ----
    (1, 0,  9, -1.236816421427e-02),  # p_loss <- p_dep_gm*p_MZ_idl
    (1, 0, 17, +1.648322620699e-02),  # p_loss <- p_loss_RO*p_MGL_idl
    (1, 1, 19, -1.112930768832e-02),  # p_X    <- p_MZ_idl*p_MGL_idl
    (1, 3,  0, -2.508415761357e-01),  # p_Z    <- p_meas*p_meas
    # ---- c (q^2) ----
    (2, 0, 17, -1.386262221677e-02),  # p_loss <- p_loss_RO*p_MGL_idl
    (2, 1,  1, +1.065914212128e-02),  # p_X    <- p_meas*p_dep_gm
    (2, 1, 19, +1.077219239662e-02),  # p_X    <- p_MZ_idl*p_MGL_idl
    (2, 3,  0, +2.516132869218e-01),  # p_Z    <- p_meas*p_meas
)
# fmt: on

# Precomputed flat arrays for vectorized sparse evaluation. The pair index
# is expanded into (i, j) here so the eval path needs no further indirection.
_TWIRLED_171M_6INPUT_Q_SPARSE_SLICE = np.array(
    [t[0] for t in _TWIRLED_171M_6INPUT_Q_SPARSE], dtype=np.int8
)
_TWIRLED_171M_6INPUT_Q_SPARSE_OUTPUT = np.array(
    [t[1] for t in _TWIRLED_171M_6INPUT_Q_SPARSE], dtype=np.int8
)
_TWIRLED_171M_6INPUT_Q_SPARSE_PAIR_I = _TWIRLED_171M_6INPUT_Q_PAIR_I[
    np.array([t[2] for t in _TWIRLED_171M_6INPUT_Q_SPARSE])
]
_TWIRLED_171M_6INPUT_Q_SPARSE_PAIR_J = _TWIRLED_171M_6INPUT_Q_PAIR_J[
    np.array([t[2] for t in _TWIRLED_171M_6INPUT_Q_SPARSE])
]
_TWIRLED_171M_6INPUT_Q_SPARSE_COEF = np.array(
    [t[3] for t in _TWIRLED_171M_6INPUT_Q_SPARSE], dtype=np.float64
)


def _eval_171m_6input_outputs(p_in: np.ndarray, q: float) -> np.ndarray:
    """Evaluate the 4 twirled 171m measurement outputs from the 6-input
    closed form at probabilities ``p_in`` and BD assignment ratio ``q``.

    Returns:
    -------
    np.ndarray, shape (4,)
        ``[p_loss, p_X, p_Y, p_Z] = M(q) @ p_in + p_in^T Q_oi(q) p_in``.

    The linear part uses the full M(q) (4, 6) matrix. The quadratic part
    is summed sparsely over the 43 surviving (slice, output, pair, coef)
    entries of ``_TWIRLED_171M_6INPUT_Q_SPARSE`` rather than via a dense
    (4, 21) @ (21,) gemv: ~83%% of the underlying tensor was pruned at
    |c| < 1e-2, so a bincount over the surviving entries is both more
    compact and slightly faster than the dense path.

    The caller is responsible for rejecting ``q`` outside ``[0, 1]``.
    """
    M = _TWIRLED_171M_6INPUT_M_POLY[0] + q * _TWIRLED_171M_6INPUT_M_POLY[1]
    q_pow = np.array([1.0, q, q * q])
    pp = (
        p_in[_TWIRLED_171M_6INPUT_Q_SPARSE_PAIR_I]
        * p_in[_TWIRLED_171M_6INPUT_Q_SPARSE_PAIR_J]
    )
    contribs = (
        _TWIRLED_171M_6INPUT_Q_SPARSE_COEF
        * q_pow[_TWIRLED_171M_6INPUT_Q_SPARSE_SLICE]
        * pp
    )
    quad = np.bincount(_TWIRLED_171M_6INPUT_Q_SPARSE_OUTPUT, contribs, minlength=4)
    return M @ p_in + quad


#: The three qubit encodings the noise model distinguishes, as
#: ``(isotope, qubit_type)`` and the suffix used on the flat parameter names.
#: ``c`` is the 174Yb optical clock qubit, ``g`` and ``m`` are the 171Yb
#: ground- and metastable-manifold nuclear-spin qubits.
ENCODINGS: dict[tuple[str, str], str] = {
    ("174", "gm"): "c",
    ("171", "g"): "g",
    ("171", "m"): "m",
}


@dataclass(frozen=True)
class QubitNoiseView:
    """Parameters of a single qubit encoding, with the encoding suffix dropped.

    Returned by :meth:`NoiseModelParameters.for_qubit`. Values are read out of
    the owning :class:`NoiseModelParameters` at call time, so the view always
    reflects the current parameter values.

    A value of ``0.0`` means the channel does not fire for this encoding, either
    because it is physically absent (``p_flip_m`` on the 174Yb clock qubit) or
    because it has been calibrated away. Callers can therefore keep using the
    ``if p > 0`` guard style throughout.
    """

    isotope: str
    qubit_type: str
    #: Suffix this view was built from: ``"c"``, ``"g"`` or ``"m"``.
    tag: str

    # --- Coherent-control ---
    p_1: float
    """DEP1_c / DEP1_g / DEP1_m depolarising probability."""
    p_2: float
    """DEP2_c / DEP2_m depolarising probability for same-encoding CZ."""
    p_1_gm: float
    """DEP1_gm probability per clock pi-pulse. Zero for 174Yb."""
    p_m_g_gate: float
    """DECAY_mg^(gate) probability, from the nuclear-spin control laser."""

    # --- Measurement ---
    p_meas: float
    """MERR discrimination probability per fluorescence imaging step."""
    q_BB: float
    """Probability of assigning the ambiguous bright-bright record to |0>."""
    p_g_L_meas: float
    """LOSS_g^(meas) atom loss during fluorescence imaging."""
    p_flip_g: float
    """FLIP_g in-manifold bit flip induced by the imaging lasers."""
    p_depol_meas_idling: float
    """Depolarisation while another isotope is measured in place."""

    # --- Reset ---
    p_g_L_reset: float
    """LOSS_g^(reset) loss during preparation or motional reset."""
    p_m_L_reset: float
    """LOSS_m^(reset) loss during metastable spin reset."""
    p_flip_m: float
    """FLIP_m nuclear-spin flip in the metastable manifold during reset."""

    # --- Idling ---
    gamma_Z: float
    """1/T_2 of this encoding, driving ZERR_c / ZERR_g / ZERR_m."""
    gamma_X: float
    """1/T_1 of this encoding, driving XERR_g / XERR_m."""
    gamma_mg: float
    """Gamma_mg, the metastable -> ground decay rate seen by this encoding."""

    # --- Transportation ---
    p_hand: float
    """LOSS^(hand) loss per trap handover."""

    # --- Operation times (seconds) ---
    gate_time: dict[str, float]
    """Operation times keyed by ``t_1Q``, ``t_1Q_gm``, ``t_2Q``, ``t_reset``, ``t_read``."""


class NoiseModelParameters:
    """Noise parameters for a dual-isotope Yb device.

    Parameter names follow the noise-model appendix of the dual-Yb surface-code
    paper; see ``docs/channel_reference.md`` for the notation-to-identifier
    mapping and ``docs/parameter_reference.md`` for the full list.

    Per-encoding parameters are stored flat with a trailing encoding tag:
    ``_c`` for the 174Yb clock qubit, ``_g`` and ``_m`` for the 171Yb ground and
    metastable qubits. Use :meth:`for_qubit` to get them grouped:

        >>> params = NoiseModelParameters()
        >>> params.p_1_m == params.for_qubit("171", "m").p_1
        True

    Rates that are properties of the atom rather than of the encoding
    (``gamma_Ryd`` and its branching ratios, ``gamma_gL``, ``gamma_mL``) are
    stored once as shared physical constants. ``gamma_mg`` is per encoding,
    because the effective metastable lifetime depends on the trap depth each
    encoding operates in. Whether they
    apply to a given operation is decided by the circuit builder, not by
    carrying a separate copy per encoding.
    """

    def __init__(self):
        """Populate every parameter with the value tabulated in the paper."""
        # ------------------------------------------------------------------
        # Shared physical constants (properties of the atom, not the encoding)
        # ------------------------------------------------------------------
        #: Total Rydberg decay rate Gamma_Ryd [1/s].
        self.gamma_Ryd = 1 / (50e-6)
        #: Branching of Gamma_Ryd into the three Rydberg decay channels.
        self.ryd_branching = {
            "LOSS_r": 0.51,
            "DECAY_rg": 0.42,
            "DECAY_rm": 0.07,
        }
        #: Gamma_gL / Gamma_mL, loss to |L> from the finite trap lifetime [1/s].
        self.gamma_gL = 1 / 30.0
        self.gamma_mL = 1 / 30.0
        #: 1/T_2^(gm), clock-transition coherence of 171Yb [1/s].
        self.gamma_Z_gm = 1 / 5.0

        # ------------------------------------------------------------------
        # Transport geometry and schedule
        # ------------------------------------------------------------------
        self.l_site = 3e-6  # site spacing [m]
        self.l_zone = 100e-6  # computation-to-readout zone separation [m]
        self.t_hand = 200e-6  # handover time [s]
        self.a = 5500  # transport acceleration [m/s^2]

        # ------------------------------------------------------------------
        # 174Yb clock qubit (tag: c)
        # ------------------------------------------------------------------
        self.p_1_c = 1e-4  # DEP1_c
        self.p_2_c = 1e-3  # DEP2_c
        self.p_meas_c = 1e-4  # MERR
        self.q_BB_c = 0.5  # BD assignment ratio
        self.p_g_L_meas_c = 0.001  # LOSS_g^(meas)
        self.p_g_L_reset_c = 0.001  # LOSS_g^(reset)
        self.p_hand_c = 0.001  # LOSS^(hand)
        self.gamma_Z_c = 1 / 5.0  # 1/T_2^(c), ZERR_c
        self.gamma_mg_c = 1 / 20.0
        self.gate_time_c = {
            "t_1Q": 100e-6,
            "t_2Q": 300e-9,
            "t_reset": 2e-3,
            "t_read": 1e-3,
        }

        # ------------------------------------------------------------------
        # 171Yb ground-manifold qubit (tag: g)
        # ------------------------------------------------------------------
        self.p_1_g = 1e-4  # DEP1_g
        self.p_2_g = 1e-3  # DEP2 for g-g CZ (via the metastable manifold)
        self.p_meas_g = 1e-4  # MERR
        self.q_BB_g = 0.5  # BD assignment ratio
        self.p_g_L_meas_g = 0.001  # LOSS_g^(meas)
        self.p_flip_g_g = 0.001  # FLIP_g, reset and measurement
        self.p_g_L_reset_g = 1e-3  # LOSS_g^(reset)
        self.p_hand_g = 0.001  # LOSS^(hand)
        self.gamma_Z_g = 0.1  # 1/T_2^(g), ZERR_g
        self.gamma_X_g = 1 / 200.0  # 1/T_1^(g), XERR_g
        #: Gamma_mg seen by the g encoding [1/s]. Only reached through the clock
        #: pulse and the readout shelving step, both of which run in the deep
        #: imaging trap, so this keeps the scattering-dominated rate.
        self.gamma_mg_g = 1.0
        #: Depolarisation on an idle g qubit while 174Yb is measured in place.
        #: Not part of the paper's channel set; held at 0.
        self.p_depol_meas_idling_g = 0.0
        self.gate_time_g = {
            "t_1Q": 1e-6,
            "t_1Q_gm": 10e-6,
            "t_2Q": 300e-9,
            "t_reset": 2e-3,
            "t_read": 1e-3,
        }

        # ------------------------------------------------------------------
        # 171Yb metastable-manifold qubit (tag: m)
        # ------------------------------------------------------------------
        self.p_1_m = 1e-4  # DEP1_m
        self.p_2_m = 1e-3  # DEP2_m
        self.p_1_gm = 1e-4  # DEP1_gm, clock-transition excitation
        self.p_m_g_gate = 0.001  # DECAY_mg^(gate)
        self.p_meas_m = 1e-4  # MERR
        self.q_BB_m = 0.5  # BD assignment ratio
        self.p_g_L_meas_m = 0.001  # LOSS_g^(meas) during the m readout
        #: FLIP_g during the imaging step of the m readout. Provisional default
        #: pending experimental calibration.
        self.p_flip_g_m = 0.001
        self.p_m_L_reset_m = 6e-3  # LOSS_m^(reset)
        self.p_flip_m_m = 0.001  # FLIP_m during motional reset
        self.p_hand_m = 0.001  # LOSS^(hand)
        self.gamma_Z_m = 0.1  # 1/T_2^(m), ZERR_m
        self.gamma_X_m = 1 / 200.0  # 1/T_1^(m), XERR_m
        #: Gamma_mg for the m encoding [1/s]. Here m -> g leaves the
        #: computational subspace, so it becomes a loss channel rather than
        #: amplitude damping and the T_2 <= 2 T_1 bound does not apply.
        self.gamma_mg_m = 1.0
        #: Depolarisation on an idle m qubit while 174Yb is measured in place.
        self.p_depol_meas_idling_m = 0.0
        self.gate_time_m = {
            "t_1Q": 1e-6,
            "t_2Q": 300e-9,
            "t_reset": 2e-3,
            "t_read": 1e-3,
        }

        # ------------------------------------------------------------------
        # Cross-encoding
        # ------------------------------------------------------------------
        self.p_2_dual = 1e-3  # DEP2_dual, 171Yb-m with 174Yb-clock

    # ----------------------------------------------------------------------
    # Presets
    # ----------------------------------------------------------------------

    @classmethod
    def paper_defaults(cls) -> "NoiseModelParameters":
        """The values tabulated in the paper. Same as ``NoiseModelParameters()``."""
        return cls()

    @classmethod
    def legacy_defaults(cls) -> "NoiseModelParameters":
        """The legacy parameter set that predates this library.

        Provided so that numbers produced before this library existed can be
        reproduced. These are *not* the paper's values: the single-qubit and
        measurement error rates are one to two orders of magnitude larger, and
        the channels the paper defines but the legacy code never injected
        (``LOSS_g`` / ``LOSS_m`` trap loss, ``XERR_g`` / ``XERR_m``,
        ``ZERR_gm``) are switched off by setting their rates to zero.

        Note that this restores *parameters* only. Formula-level corrections are
        not undone -- in particular the transport time keeps the factor of two
        in ``t_move = 2 sqrt(l / a)``.
        """
        p = cls()

        # Coherent-control: the legacy code ran 10-20x the paper's gate error rates.
        p.p_1_c = 0.002
        p.p_1_g = 0.001
        p.p_1_m = 0.001
        p.p_1_gm = 0.002
        # A single p_2_dep drove every CZ pattern.
        p.p_2_c = p.p_2_g = p.p_2_m = p.p_2_dual = 0.002
        p.p_m_g_gate = 0.001

        # Measurement.
        p.p_meas_c = 0.001
        p.p_meas_g = 0.001
        # Calibrated so that the MERR-only total p_loss + 2 p_X + p_Z came to
        # 0.005; that total is exactly (3/2) p, hence 0.005 / 1.5.
        p.p_meas_m = 1.0 / 300.0

        # Reset: the legacy code held the g-qubit reset loss at zero and used a
        # metastable reset loss six times smaller than the paper's.
        p.p_g_L_reset_g = 0.0
        p.p_m_L_reset_m = 0.001

        # Channels the legacy code never injected into the Stim circuit.
        p.gamma_gL = 0.0
        p.gamma_mL = 0.0
        p.gamma_Z_gm = 0.0
        p.gamma_X_g = 0.0
        p.gamma_X_m = 0.0

        # Operation times: the legacy code used 100 us for every single-qubit gate,
        # including the 171Yb in-manifold gates and the clock pulse.
        p.gate_time_g["t_1Q"] = 100e-6
        p.gate_time_g["t_1Q_gm"] = 100e-6
        p.gate_time_m["t_1Q"] = 100e-6

        return p

    # ----------------------------------------------------------------------
    # Grouped access
    # ----------------------------------------------------------------------

    def for_qubit(self, isotope: str, qubit_type: str = "gm") -> QubitNoiseView:
        """Return the parameters of one qubit encoding, grouped.

        Args:
            isotope: ``"171"`` or ``"174"``.
            qubit_type: ``"gm"`` for 174Yb, ``"g"`` or ``"m"`` for 171Yb.

        Returns:
            A :class:`QubitNoiseView` whose attribute names drop the encoding
            tag, so the same code can serve every encoding.

        Raises:
            ValueError: if the ``(isotope, qubit_type)`` pair is not one of the
                three modelled encodings.
        """
        key = (isotope, qubit_type)
        if key not in ENCODINGS:
            raise ValueError(
                f"Unknown qubit encoding {key!r}. Expected one of {sorted(ENCODINGS)}."
            )
        tag = ENCODINGS[key]

        def get(name: str, default: float = 0.0) -> float:
            return getattr(self, f"{name}_{tag}", default)

        return QubitNoiseView(
            isotope=isotope,
            qubit_type=qubit_type,
            tag=tag,
            p_1=get("p_1"),
            p_2=get("p_2"),
            # DEP1_gm and DECAY_mg^(gate) are 171Yb clock-transition channels;
            # they are stored on the m encoding and shared with g, which reaches
            # the Rydberg state through the same clock pulse.
            p_1_gm=self.p_1_gm if isotope == "171" else 0.0,
            p_m_g_gate=self.p_m_g_gate if isotope == "171" else 0.0,
            p_meas=get("p_meas"),
            q_BB=get("q_BB"),
            p_g_L_meas=get("p_g_L_meas"),
            p_flip_g=get("p_flip_g"),
            p_depol_meas_idling=get("p_depol_meas_idling"),
            p_g_L_reset=get("p_g_L_reset"),
            p_m_L_reset=get("p_m_L_reset"),
            p_flip_m=get("p_flip_m"),
            gamma_Z=get("gamma_Z"),
            gamma_X=get("gamma_X"),
            gamma_mg=get("gamma_mg"),
            p_hand=get("p_hand"),
            gate_time=get("gate_time", {}),  # type: ignore[arg-type]
        )

    def get_twirled_174_measurement_merr_rates(
        self, p_meas: float | None = None, q: float | None = None
    ) -> dict[str, float]:
        """Return twirled 174Yb MERR rates for Stim-based simulation.

        The fitted channel is parameterized by the underlying discrimination
        probability ``p_meas`` and BD assignment probability ``q``. Closed
        form derived from the corrected ``KrausMEASURE_DISC_174.MERR``
        channel:
        ``p_loss`` is exact; ``p_X = p_Y`` is exactly independent of ``q``
        to this order; ``p_Z`` scales as ``q_BB^2`` with ``q_BB = 2q - 1``.
        """
        p = self.p_meas_c if p_meas is None else p_meas
        q_val = self.q_BB_c if q is None else q

        if not (0 <= p <= 1):
            raise ValueError(
                "Measurement discrimination probability must be in [0, 1]."
            )
        if not (0 <= q_val <= 1):
            raise ValueError("BD assignment probability q must be in [0, 1].")

        q_bb = 2 * q_val - 1
        p_loss = p * (1 - p)
        p_x = p / 4 + p**2 / 2
        p_y = p_x
        p_z = (q_bb**2 / 16) * p**2

        return {
            "p_loss": max(0.0, p_loss),
            "p_X": max(0.0, p_x),
            "p_Y": max(0.0, p_y),
            "p_Z": max(0.0, p_z),
        }

    def get_twirled_171m_measurement_merr_rates(
        self, p_meas: float | None = None, q: float | None = None
    ) -> dict[str, float]:
        """Return twirled 171Yb m-qubit MERR rates for Stim-based simulation.

        The 171Yb m-qubit readout transfers the population to the g-qubit and
        applies state-selective fluorescence on `|0>` and `|1>` separately, so the
        BD discrimination model used for 174Yb applies symmetrically here. The
        analytical Pauli/leakage breakdown is therefore identical in form to
        :meth:`get_twirled_174_measurement_merr_rates`: ``p_loss`` is exact,
        ``p_X = p_Y`` is exactly independent of ``q`` to this order, and ``p_Z``
        scales as ``q_BB^2`` with ``q_BB = 2q - 1``.
        """
        p = self.p_meas_m if p_meas is None else p_meas
        q_val = self.q_BB_m if q is None else q

        if not (0 <= p <= 1):
            raise ValueError(
                "Measurement discrimination probability must be in [0, 1]."
            )
        if not (0 <= q_val <= 1):
            raise ValueError("BD assignment probability q must be in [0, 1].")

        q_bb = 2 * q_val - 1
        p_loss = p * (1 - p)
        p_x = p / 4 + p**2 / 2
        p_y = p_x
        p_z = (q_bb**2 / 16) * p**2

        return {
            "p_loss": max(0.0, p_loss),
            "p_X": max(0.0, p_x),
            "p_Y": max(0.0, p_y),
            "p_Z": max(0.0, p_z),
        }

    def get_twirled_171m_measurement_error_rates(
        self,
        p_meas: float | None = None,
        p_dep_gm: float | None = None,
        p_X_g: float | None = None,
        p_loss_RO: float | None = None,
        T2_m: float | None = None,
        leaktime_eg: float | None = None,
        gate_time_measure: float | None = None,
        q: float | None = None,
    ) -> dict[str, float]:
        """Return twirled 171Yb m-qubit measurement noise rates from the
        empirical 6-input closed form.

        The closed form is a (linear + 21 symmetric quadratic) fit to GTA
        applied on the 8-step single-application protocol
            DEP1_gm -> ZERR_m -> DECAY_mg -> U_{m<->g}
                    -> FLIP_g -> LOSS_g^(meas) -> MERR -> U_{m<->g}.
        It reproduces the GTA output to within 1.0e-6 over [0, 1e-2]^6 at
        q in {0.0, 0.5, 1.0}; intermediate q is linearly interpolated.

        Inputs (all optional; falls back to ``self`` attributes when None):

        - ``p_meas`` : MERR per-fluorescence-pulse discrimination probability
          (default ``self.p_meas_m``).
        - ``p_dep_gm`` : DEP1^{gm} clock-pulse depolarization probability
          (default ``self.p_1_gm``).
        - ``p_X_g`` : FLIP_g imaging-driven g-state bit-flip probability
          (default ``self.p_flip_g_m``).
        - ``p_loss_RO`` : LOSS_g^(meas) g-state atomic loss during imaging
          (default ``self.p_g_L_meas_m``).
        - ``T2_m`` : m-qubit Z-dephasing time T2 in seconds, used to derive
          ``p_MZ_idl = 1 - exp(-gate_time_measure / T2_m)``
          (default ``1 / self.gamma_Z_m``).
        - ``leaktime_eg`` : m -> g radiative lifetime in seconds, used to
          derive ``p_MGL_idl = 1 - exp(-gate_time_measure / leaktime_eg)``
          (default ``1 / self.gamma_mg_m``).
        - ``gate_time_measure`` : measurement / readout duration in seconds
          (default ``self.gate_time_m["t_read"]``).
        - ``q`` : BD assignment probability (default ``self.q_BB_m``).

        Returns:
        -------
        dict[str, float]
            ``{"p_loss", "p_X", "p_Y", "p_Z"}`` Pauli + leakage rates,
            clipped to non-negative values.
        """
        p_meas = self.p_meas_m if p_meas is None else p_meas
        p_dep_gm = self.p_1_gm if p_dep_gm is None else p_dep_gm
        p_X_g = self.p_flip_g_m if p_X_g is None else p_X_g
        p_loss_RO = self.p_g_L_meas_m if p_loss_RO is None else p_loss_RO
        if T2_m is None:
            T2_m = 1.0 / self.gamma_Z_m
        if leaktime_eg is None:
            leaktime_eg = 1.0 / self.gamma_mg_m
        if gate_time_measure is None:
            gate_time_measure = self.gate_time_m["t_read"]
        q_val = self.q_BB_m if q is None else q

        for name, val in (
            ("p_meas", p_meas),
            ("p_dep_gm", p_dep_gm),
            ("p_X_g", p_X_g),
            ("p_loss_RO", p_loss_RO),
        ):
            if not (0 <= val <= 1):
                raise ValueError(f"{name} must be in [0, 1].")
        if not (0 <= q_val <= 1):
            raise ValueError("BD assignment probability q must be in [0, 1].")
        if T2_m <= 0:
            raise ValueError("T2_m must be positive.")
        if leaktime_eg <= 0:
            raise ValueError("leaktime_eg must be positive.")
        if gate_time_measure < 0:
            raise ValueError("gate_time_measure must be non-negative.")

        p_MZ_idl = 1.0 - np.exp(-gate_time_measure / T2_m)
        p_MGL_idl = 1.0 - np.exp(-gate_time_measure / leaktime_eg)

        # Closed form: y = M(q) p + [p^T Q_i(q) p]_i evaluated via the
        # sparse helper (linear M @ p plus 43-entry bincount over Q).
        p_in = np.array([p_meas, p_dep_gm, p_X_g, p_loss_RO, p_MZ_idl, p_MGL_idl])
        out = _eval_171m_6input_outputs(p_in, q_val)

        return {
            "p_loss": max(0.0, float(out[0])),
            "p_X": max(0.0, float(out[1])),
            "p_Y": max(0.0, float(out[2])),
            "p_Z": max(0.0, float(out[3])),
        }

    def __str__(self):
        return "\n".join(
            f"{key}: {value}"
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        )

    def twirled_amplitude_damping(
        self,
        duration: float,
        T_1_inv: float,
        T_2_inv: float | None = None,
        branching_ratio: float | None = 1.0,
    ) -> list[float]:
        r"""Pauli approximation of amplitude damping, after Tomita and Svore.

        With :math:`p_1(t)=1-e^{-t/T_1}` and :math:`p_2(t)=1-e^{-t/T_2}`,

        .. math::
            p_X = p_Y = \frac{p_1(t)}{4}, \qquad
            p_Z = \frac{p_2(t)}{2} - \frac{p_1(t)}{4}.

        The factor of two on the :math:`T_2` term is what makes the channel
        reproduce the transverse Bloch decay :math:`e^{-t/T_2}`: a Pauli channel
        maps that component by :math:`1-2p_Y-2p_Z`, so
        :math:`1-p_1/2-2p_Z = 1-p_2` gives exactly the expression above.

        ``T_2_inv = None`` means the damping limit, where the only source of
        coherence loss is the damping itself and there is no extra dephasing to
        add. That is the case for every coherent-control operation and for reset,
        because the paper disables the idling ``ZERR`` channels there -- their
        contribution is already inside the operation's own error channel -- so
        ``p_Z`` is reported as exactly zero rather than as the :math:`O(t^2)`
        residue that substituting :math:`T_2 = 2T_1` would leave.

        Args:
            duration: Exposure time in seconds.
            T_1_inv: Amplitude damping rate, ``1 / T_1``.
            T_2_inv: Dephasing rate ``1 / T_2``. ``None`` selects the damping
                limit described above.
            branching_ratio: Optional scale on both rates.

        Returns:
            ``[p_X, p_Y, p_Z]``.

        Raises:
            ValueError: if ``T_2 > 2 T_1``, which no single qubit can satisfy --
                damping at ``T_1`` alone already decoheres at ``2 T_1`` -- and
                which would make ``p_Z`` negative.
        """
        p_1 = 1 - np.exp(-duration * T_1_inv * branching_ratio)  # type: ignore[operator]
        p_X = p_Y = p_1 / 4

        if T_2_inv is None:
            return [p_X, p_Y, 0.0]

        p_2 = 1 - np.exp(-duration * T_2_inv * branching_ratio)  # type: ignore[operator]
        p_Z = p_2 / 2 - p_1 / 4
        if p_Z < 0:
            raise ValueError(
                "Twirled amplitude damping got a negative p_Z "
                f"({p_Z:.3e}) at duration={duration:.3e} s, "
                f"T_1={1 / T_1_inv:.3e} s, T_2={1 / T_2_inv:.3e} s. "
                "The Pauli approximation needs T_2 <= 2 T_1; amplitude damping "
                "at T_1 alone already decoheres at 2 T_1, so a longer T_2 is "
                "not physically realisable."
            )
        return [p_X, p_Y, p_Z]

    def set_parameters(self, **kwargs):
        """Set free parameters with validation.

        Supports both direct parameter setting and nested dictionary updates.
        For nested dictionary parameters (e.g., gate_time_c), use slash notation:
        gate_time_c/t_1Q = 100e-6

        Note: When using slash notation in Python code, you need to pass it as a dictionary
        since '/' cannot be used in keyword argument names:
        params.set_parameters(**{"gate_time_c/t_1Q": 200e-6})

        Args:
            **kwargs: Parameter name-value pairs

        Examples:
            >>> params = NoiseModelParameters()
            >>> params.set_parameters(p_1_c=0.003, gamma_mg_c=1.5)
            >>> params.set_parameters(**{"gate_time_c/t_1Q": 200e-6})
        """
        # Valid parameter names are the public attributes set in __init__,
        # plus any class-level property that exposes a parameter.
        valid_params = {k for k in self.__dict__ if not k.startswith("_")}
        for attr_name in dir(type(self)):
            if isinstance(getattr(type(self), attr_name, None), property):
                valid_params.add(attr_name)

        for key, value in kwargs.items():
            # Check for nested dictionary update (e.g., gate_time_c/t_1Q)
            if "/" in key:
                parts = key.split("/", 1)
                param_name = parts[0]
                dict_key = parts[1]

                if param_name not in valid_params:
                    raise ValueError(
                        f"Unknown parameter: {param_name}. "
                        f"Available parameters: {sorted(valid_params)}"
                    )

                param_value = getattr(self, param_name)
                if not isinstance(param_value, dict):
                    raise ValueError(
                        f"Parameter '{param_name}' is not a dictionary, "
                        f"cannot use nested update syntax"
                    )

                # Update nested dictionary
                param_value[dict_key] = value
            else:
                # Direct parameter setting
                if key not in valid_params:
                    raise ValueError(
                        f"Unknown parameter: {key}. "
                        f"Available parameters: {sorted(valid_params)}"
                    )
                setattr(self, key, value)

    def show_parameters(self):
        """Show all parameters"""
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            print(f"{key}: {value}")

    def rescale_error_params(
        self,
        scale: float,
        inplace: bool = False,
        include_prefixes: tuple[str, ...] = ("p_", "gamma_"),
        exclude: Iterable[str] | None = None,
    ) -> "NoiseModelParameters":
        """Rescale scalar error parameters such as probabilities and decay rates.

        By default, parameters whose names start with ``p_`` or ``gamma_`` are
        multiplied by ``scale``. Timing and transport settings are left unchanged.

        Args:
            scale: Multiplicative scale factor applied to matching parameters.
            inplace: If True, update this instance directly.
            include_prefixes: Parameter name prefixes to rescale.
            exclude: Parameter names to skip even if their prefix matches.

        Returns:
            NoiseModelParameters: The updated instance.
        """
        if scale < 0:
            raise ValueError(f"scale must be non-negative, got {scale}")

        target = self if inplace else copy.deepcopy(self)
        exclude_set = set(exclude or [])

        for name, value in target.__dict__.items():
            if name in exclude_set:
                continue
            if isinstance(value, (int, float)) and name.startswith(include_prefixes):
                setattr(target, name, value * scale)

        return target

    def to_string(self):
        """Convert parameters to string"""
        return "\n".join(
            f"{key}: {value}"
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        )

    def get_time_dependent_rate(
        self, duration: float, gamma: float, branching_ratio: float | None = 1.0
    ) -> float:
        r"""Compute time-dependent error rate using the exponential decay,
        $$
            p = 1 - \exp(-t * \gamma * branching_ratio),
        $$
        where $t$ is the duration like gate time and $\gamma$ is the decay rate.
        $branching_ratio$ is set to 1.0 by default.

        Args:
            duration: Duration
            gamma: decay rate or inverse of T_1
            branching_ratio: Branching ratio (optional)

        Returns:
            float: Time-dependent error rate
        """
        return 1 - np.exp(-duration * gamma * branching_ratio)  # type: ignore[operator]

    def rydberg_branch_rates(self, duration: float) -> dict[str, float]:
        """Return the three Rydberg decay probabilities over *duration* seconds.

        The aggregate Rydberg channel is ``p_hat[r->i] = b_i (1 - exp(-Gamma_Ryd
        t))``. Factorising it into three sequential channels needs each branch
        renormalised by the population the earlier branches already removed,
        otherwise the later branches act on a depleted population and come out
        too small. Applied in the order ``LOSS_r``, ``DECAY_rg``, ``DECAY_rm``:

            p[r->L] = p_hat[r->L]
            p[r->g] = p_hat[r->g] / (1 - p_hat[r->L])
            p[r->m] = p_hat[r->m] / ((1 - p_hat[r->L]) (1 - p_hat[r->g]))

        Args:
            duration: Exposure time in seconds, normally the two-qubit gate time.

        Returns:
            ``{"LOSS_r": ..., "DECAY_rg": ..., "DECAY_rm": ...}``, clipped to
            ``[0, 1]``.
        """
        if duration <= 0:
            return {"LOSS_r": 0.0, "DECAY_rg": 0.0, "DECAY_rm": 0.0}

        total = 1.0 - np.exp(-duration * self.gamma_Ryd)
        hat_L = self.ryd_branching["LOSS_r"] * total
        hat_g = self.ryd_branching["DECAY_rg"] * total
        hat_m = self.ryd_branching["DECAY_rm"] * total

        p_L = hat_L
        p_g = hat_g / (1.0 - hat_L) if hat_L < 1.0 else 1.0
        denom = (1.0 - hat_L) * (1.0 - hat_g)
        p_m = hat_m / denom if denom > 0 else 1.0

        return {
            "LOSS_r": float(min(max(p_L, 0.0), 1.0)),
            "DECAY_rg": float(min(max(p_g, 0.0), 1.0)),
            "DECAY_rm": float(min(max(p_m, 0.0), 1.0)),
        }

    def get_gate_time(self, gate_type: str, isotope: str, qubit_type: str) -> float:
        """Return the operation time of one gate type for one qubit encoding.

        Args:
            gate_type: One of the keys of the encoding's ``gate_time`` dict
                (``t_1Q``, ``t_1Q_gm``, ``t_2Q``, ``t_reset``, ``t_read``).
                Which keys exist depends on the encoding: only the 171Yb-g
                qubit needs a separate clock-pulse time ``t_1Q_gm``.
            isotope: ``"171"`` or ``"174"``.
            qubit_type: ``"gm"`` for 174Yb, ``"g"`` or ``"m"`` for 171Yb.

        Returns:
            The operation time in seconds.

        Raises:
            ValueError: for an unknown encoding, or a gate type this encoding
                does not define.
        """
        gate_time = self.for_qubit(isotope, qubit_type).gate_time
        if gate_type not in gate_time:
            raise ValueError(
                f"Unsupported gate type {gate_type!r} for ({isotope}, {qubit_type}). "
                f"Available gate types are {sorted(gate_time)}."
            )
        return gate_time[gate_type]

    def transportation_time(self, distance: float) -> float:
        """Calculate transportation time excluding handover time using the constant jerk acceleration model.
        The unit of distance is meter.
        """
        return 2 * np.sqrt(distance / self.a)

    def readout_transport_one_way_distance(self, code_distance: int) -> float:
        """One-way distance from computation zone to readout zone: l*d + A."""
        return self.l_site * code_distance + self.l_zone

    def readout_transport_one_way_time(self, code_distance: int) -> float:
        """Pure motion time for one-way readout transport (excludes handover)."""
        return self.transportation_time(
            self.readout_transport_one_way_distance(code_distance)
        )

    def readout_transport_round_trip_time(
        self,
        code_distance: int,
        include_handover: bool = True,
        include_measurement: bool = True,
    ) -> float:
        r"""Total latency of the readout transport round trip.

        Protocol sequence:
            handover -> move_out -> handover -> measurement
            -> handover -> move_back -> handover

        Returns:
            Total time in seconds.
        """
        move_time = self.readout_transport_one_way_time(code_distance)
        total = 2 * move_time
        if include_handover:
            total += 4 * self.t_hand
        if include_measurement:
            total += max(
                self.gate_time_c.get("t_read", 0),
                self.gate_time_m.get("t_read", 0),
                self.gate_time_g.get("t_read", 0),
            )
        return total

    def load_noise_params_from_file(self, filepath: str):
        """Load noise parameters from a text file and set the parameters to the NoiseModelParameters instance.

        File format:
            # Comments start with #
            parameter_name: value
            nested_parameter/key: value  # For dictionary parameters

        Examples:
            p_1_c: 0.003
            gamma_mg_c: 1.5
            gate_time_c/t_1Q: 200e-6
            ryd_branching/LOSS_r: 0.52

        Args:
            filepath: Path to the parameter file
        """
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse key: value format
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # valueをstrから対応する型に変換
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(
                        f"Failed to parse value for '{key}': {value}. Error: {e}"
                    ) from e

                self.set_parameters(**{key: value})

    def save_noise_params_to_file(self, filepath: str):
        """Save current noise parameters to a text file.

        Args:
            filepath: Path to save the parameter file
        """
        with open(filepath, "w") as f:
            f.write("# Noise Model Parameters\n")
            f.write("# Format: parameter_name: value\n")
            f.write("# For nested dictionary parameters, use: param/key: value\n\n")

            # Group parameters by category
            categories = {
                "Shared physical constants": [
                    "gamma_Ryd",
                    "ryd_branching",
                    "gamma_gL",
                    "gamma_mL",
                    "gamma_Z_gm",
                ],
                "Transport geometry and schedule": [
                    "l_site",
                    "l_zone",
                    "t_hand",
                    "a",
                ],
                "174Yb clock qubit": [
                    "p_1_c",
                    "p_2_c",
                    "p_meas_c",
                    "q_BB_c",
                    "p_g_L_meas_c",
                    "p_g_L_reset_c",
                    "p_hand_c",
                    "gamma_Z_c",
                    "gamma_mg_c",
                    "gate_time_c",
                ],
                "171Yb ground-manifold qubit": [
                    "p_1_g",
                    "p_2_g",
                    "p_meas_g",
                    "q_BB_g",
                    "p_g_L_meas_g",
                    "p_flip_g_g",
                    "p_g_L_reset_g",
                    "p_hand_g",
                    "gamma_Z_g",
                    "gamma_X_g",
                    "gamma_mg_g",
                    "p_depol_meas_idling_g",
                    "gate_time_g",
                ],
                "171Yb metastable-manifold qubit": [
                    "p_1_m",
                    "p_2_m",
                    "p_1_gm",
                    "p_m_g_gate",
                    "p_meas_m",
                    "q_BB_m",
                    "p_g_L_meas_m",
                    "p_flip_g_m",
                    "p_m_L_reset_m",
                    "p_flip_m_m",
                    "p_hand_m",
                    "gamma_Z_m",
                    "gamma_X_m",
                    "gamma_mg_m",
                    "p_depol_meas_idling_m",
                    "gate_time_m",
                ],
                "Cross-encoding": [
                    "p_2_dual",
                ],
            }

            for category, param_names in categories.items():
                f.write(f"\n# {category}\n")
                for param_name in param_names:
                    if hasattr(self, param_name):
                        value = getattr(self, param_name)
                        if isinstance(value, dict):
                            # Write nested dictionary parameters
                            for key, val in value.items():
                                f.write(f"{param_name}/{key}: {repr(val)}\n")
                        else:
                            f.write(f"{param_name}: {repr(value)}\n")

    def get_parameter_names(self) -> list[str]:
        """Get list of all parameter names.

        Returns:
            List of parameter names
        """
        return sorted(k for k in self.__dict__.keys() if not k.startswith("_"))

    def get_parameters_dict(self) -> dict:
        """Get all parameters as a dictionary.

        Returns:
            Dictionary of all parameters
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
