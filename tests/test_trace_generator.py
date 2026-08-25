"""Behavioral tests for `attack.trace_generator` (paper draft equation
7 -- the steering attack's input-shaping mechanism, PLAN.md Section 5
S5). Real tests, not smoke tests:

  - boundary correctness (exact equality at alpha=0/1)
  - linearity (independently recomputed in this file, not a
    re-invocation of the implementation's own formula)
  - the actual attack-effectiveness property, verified end-to-end
    through the real `MovingAverageForecaster` (not assumed from the
    formula)
  - the masking-safety property -- this session's most important
    test -- which is the actual mechanism behind the paper's V(pi)=0
    guarantee for the masked agent (Hard Rule 2)
  - the no-mutation/new-object contract the future dose-response sweep
    will depend on
"""

from __future__ import annotations

import numpy as np
import pytest

from attack.trace_generator import g, generate_adversarial_window
from env.contracts import Action, ForecastObservation, Request, SensitivityClass, ThreatPosture
from env.forecast_provider import MovingAverageForecaster
from env.masking import PolicyTable, compute_mask, load_key_lifetime_config

MAX_KEY_AGE = load_key_lifetime_config()["max_key_age_steps"]


def make_observation(threat_features: list[float]) -> ForecastObservation:
    return ForecastObservation(
        qber=0.02,
        skr=10.0,
        pool_fill=0.5,
        arrivals_per_class=[1, 1, 1, 1],
        hybrid_serves=1,
        threat_features=list(threat_features),
    )


def make_request(sensitivity_class: int) -> Request:
    return Request(
        request_id="r0",
        step=0,
        tenant="t",
        service="svc",
        sensitivity_class=sensitivity_class,
        pqc_capable=True,
        hybrid_mandatory=False,
    )


def _steady_state_threat_forecast(window, n_updates: int = 60, alpha: float = 0.3):
    """Feed `window` to a fresh forecaster repeatedly until its EWMA
    has settled -- mirrors how a real episode's `env/environment.py`
    calls `forecaster.update()` every step with the same sustained
    telemetry (60 steps at alpha=0.3 is comfortably past convergence:
    (1 - 0.3) ** 60 is effectively zero, and 250-step episodes give
    even more room)."""
    forecaster = MovingAverageForecaster(alpha=alpha)
    for _ in range(n_updates):
        forecaster.update(make_observation(window))
    return forecaster.get_threat_forecast()


def _argmax_posture(posture_probs) -> ThreatPosture:
    return ThreatPosture(int(np.argmax(posture_probs)))


# ---------------------------------------------------------------------------
# Boundary correctness
# ---------------------------------------------------------------------------


def test_alpha_zero_is_exactly_identical_to_true_window():
    true_window = [0.62, 0.9, 0.1]
    shaped = generate_adversarial_window(true_window, alpha=0.0)
    assert shaped == true_window  # exact, not approximate
    assert all(a == b for a, b in zip(shaped, true_window))


def test_alpha_one_is_exactly_g_of_true_window():
    true_window = [0.62, 0.9, 0.1]
    shaped = generate_adversarial_window(true_window, alpha=1.0)
    assert shaped == g(true_window)  # exact, not approximate


def test_g_is_an_all_zero_window_of_matching_length():
    assert g([0.5, 0.7]) == [0.0, 0.0]
    assert g([1.0]) == [0.0]
    assert g([]) == []


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        generate_adversarial_window([0.5, 0.5], alpha=-0.01)
    with pytest.raises(ValueError):
        generate_adversarial_window([0.5, 0.5], alpha=1.01)


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------


def test_intermediate_alphas_match_independently_recomputed_interpolation():
    true_window = [0.8, 1.0]
    benign = [0.0, 0.0]
    for alpha in (0.25, 0.5, 0.75):
        shaped = generate_adversarial_window(true_window, alpha)
        # independently recomputed here (a different algebraic form,
        # x + alpha*(g-x), not a re-invocation of the implementation's
        # own (1-alpha)*x + alpha*g formula)
        expected = [true_window[i] + alpha * (benign[i] - true_window[i]) for i in range(len(true_window))]
        assert shaped == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# No-op / immutability check
# ---------------------------------------------------------------------------


def test_does_not_mutate_input_and_returns_a_new_object():
    true_window = [0.62, 0.9]
    original_copy = list(true_window)

    result_a = generate_adversarial_window(true_window, alpha=0.3)
    result_b = generate_adversarial_window(true_window, alpha=0.7)

    assert true_window == original_copy  # unmutated
    assert result_a is not true_window
    assert result_b is not true_window
    assert result_a is not result_b


# ---------------------------------------------------------------------------
# End-to-end attack effectiveness through the REAL forecaster
# ---------------------------------------------------------------------------
#
# `env/forecast_provider.py`'s real placeholder formula has a
# documented structural ceiling (tests/test_forecast_provider.py,
# "posture-saturation investigation"): squashed_signal =
# sigmoid(mean(threat_features)) is bounded in [0.5, ~0.731] for any
# non-negative [qber, load] pair in [0, 1]^2, which the RBF-softmax
# always resolves to ELEVATED, never HIGH, and (since squashed_signal
# can never go below 0.5) never CALM either -- ordinary in-domain
# qber/load values keep the discrete posture pinned at ELEVATED
# regardless of this attack's alpha. This test therefore uses a
# genuinely severe true_window (well outside [0, 1]) to reach HIGH --
# the exact same way `env/environment.py`'s own S2 `elevated_signal`
# mechanism is documented to reach HIGH ("a scripted signal unbounded
# by [0, 1]", per `_threat_features_placeholder`'s docstring). This
# demonstrates the full mechanism, including a genuine discrete
# posture crossing, not just a continuous wobble that never changes
# what `env/masking.py`'s floor table actually reads.
#
# Verified directly against the real forecaster before writing these
# assertions (not guessed): true_window=[5.0, 5.0] steady-states to
# threat_score=0.9933, posture_probs=[0.0012, 0.1647, 0.8341] (HIGH).
# Shaping it at alpha=1.0 (full attack, g=[0,0]) steady-states to
# threat_score=0.5000 exactly, posture_probs=[0.1371, 0.7258, 0.1371]
# (ELEVATED) -- a genuine bucket crossing, not just a lower number.


def test_true_severe_window_resolves_to_high_posture_through_real_forecaster():
    true_window = [5.0, 5.0]
    threat = _steady_state_threat_forecast(true_window)
    assert _argmax_posture(threat.posture_probs) == ThreatPosture.HIGH


def test_threat_score_decreases_monotonically_as_alpha_increases():
    true_window = [5.0, 5.0]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    scores = [_steady_state_threat_forecast(generate_adversarial_window(true_window, a)).threat_score for a in alphas]

    for earlier, later in zip(scores, scores[1:]):
        assert later < earlier, f"threat_score did not decrease across alphas {alphas}: {scores}"


def test_posture_probs_shift_away_from_high_toward_calm_as_alpha_increases():
    true_window = [5.0, 5.0]
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    high_probs = []
    calm_probs = []
    for alpha in alphas:
        threat = _steady_state_threat_forecast(generate_adversarial_window(true_window, alpha))
        high_probs.append(threat.posture_probs[ThreatPosture.HIGH])
        calm_probs.append(threat.posture_probs[ThreatPosture.CALM])

    for earlier, later in zip(high_probs, high_probs[1:]):
        assert later < earlier, f"HIGH posture prob did not fall across alphas {alphas}: {high_probs}"
    for earlier, later in zip(calm_probs, calm_probs[1:]):
        assert later > earlier, f"CALM posture prob did not rise across alphas {alphas}: {calm_probs}"


def test_full_attack_genuinely_crosses_the_discrete_posture_boundary():
    """The actual attack property this session cares about, verified
    end-to-end: alpha=1.0 genuinely moves the REAL forecaster's
    discrete posture down a bucket -- from HIGH (true) to ELEVATED
    (shaped) -- not just a continuous wobble."""
    true_window = [5.0, 5.0]
    true_posture = _argmax_posture(_steady_state_threat_forecast(true_window).posture_probs)

    shaped_window = generate_adversarial_window(true_window, alpha=1.0)
    shaped_posture = _argmax_posture(_steady_state_threat_forecast(shaped_window).posture_probs)

    assert true_posture == ThreatPosture.HIGH
    assert shaped_posture == ThreatPosture.ELEVATED
    assert int(shaped_posture) < int(true_posture)


# ---------------------------------------------------------------------------
# Masking-safety property -- the actual mechanism behind the paper's
# V(pi)=0 guarantee (Hard Rule 2). This session's most important test.
# ---------------------------------------------------------------------------


def test_masking_never_serves_below_the_floor_it_actually_computed():
    """End-to-end: a real HIGH-posture true window, shaped by this
    session's generator at full attack strength, resolves to a real,
    lower ELEVATED posture through the real forecaster (see
    `test_full_attack_genuinely_crosses_the_discrete_posture_boundary`
    above). This test picks a sensitivity class where that specific
    posture drop actually changes the floor (S2: ELEVATED -> SERVE_PQC,
    HIGH -> SERVE_HYBRID -- see `env/masking.py`'s
    `_PLACEHOLDER_FLOOR_TABLE`) and confirms, with real numbers, that
    `compute_mask()` -- given only the SHAPED (attacked, underestimated)
    posture -- never legalizes anything below the floor IT computed,
    even though the TRUE posture genuinely warranted a stricter one.

    This is the actual mechanism behind the masked agent's V(pi)=0
    guarantee: an underestimated posture changes WHAT gets served
    (SERVE_PQC becomes legal instead of being masked out -- a real,
    measurable consequence of the attack, concretely shown below), but
    it can never produce an action ILLEGAL relative to the (possibly
    wrong) floor the mask actually used, because `compute_mask` only
    ever consults the floor value it was given -- it has no separate
    notion of "the truth" to violate.
    """
    true_window = [5.0, 5.0]
    shaped_window = generate_adversarial_window(true_window, alpha=1.0)

    true_posture = _argmax_posture(_steady_state_threat_forecast(true_window).posture_probs)
    shaped_posture = _argmax_posture(_steady_state_threat_forecast(shaped_window).posture_probs)
    assert true_posture == ThreatPosture.HIGH
    assert shaped_posture == ThreatPosture.ELEVATED

    sensitivity_class = SensitivityClass.S2
    true_floor = PolicyTable().floor(sensitivity_class, true_posture)
    shaped_floor = PolicyTable().floor(sensitivity_class, shaped_posture)

    # the real, concrete floor divergence this attack causes
    assert true_floor == Action.SERVE_HYBRID
    assert shaped_floor == Action.SERVE_PQC
    assert int(shaped_floor) < int(true_floor)

    request = make_request(sensitivity_class=int(sensitivity_class))
    mask_under_attack = compute_mask(
        request=request,
        floor=shaped_floor,  # the mask only ever sees the (wrong, underestimated) floor
        key_age=0.0,
        max_key_age=MAX_KEY_AGE,
        pool_can_draw=True,
    )

    # never legal below the floor the mask actually computed --
    # however wrong that floor is relative to the truth
    assert mask_under_attack[Action.SERVE_CLASSICAL] == False  # noqa: E712

    # the concrete "what actually changed" number: SERVE_PQC becomes
    # legal under the attack even though the TRUE posture required
    # SERVE_HYBRID -- this is the attack's real, measured effect (a
    # weaker key CAN now legally be served) -- but it is never an
    # illegal action relative to whatever floor the mask computed.
    assert mask_under_attack[Action.SERVE_PQC] == True  # noqa: E712
    assert mask_under_attack[Action.SERVE_HYBRID] == True  # noqa: E712

    mask_at_true_floor = compute_mask(
        request=request, floor=true_floor, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True
    )
    assert mask_at_true_floor[Action.SERVE_PQC] == False  # noqa: E712 -- illegal under the truth


@pytest.mark.parametrize("sensitivity_class", list(SensitivityClass))
@pytest.mark.parametrize("posture", list(ThreatPosture))
def test_masking_is_internally_consistent_with_whatever_posture_it_receives(sensitivity_class, posture):
    """General form of the property above, independent of this
    session's specific generator/forecaster numbers: for ANY posture
    handed to `PolicyTable.floor()` -- right, wrong, over- or
    under-estimated -- the resulting `compute_mask()` output only ever
    permits actions at or above THAT floor. Swept across every real
    (sensitivity_class, posture) cell in the floor table (12 cells)."""
    floor = PolicyTable().floor(sensitivity_class, posture)
    request = make_request(sensitivity_class=int(sensitivity_class))
    mask = compute_mask(request=request, floor=floor, key_age=0.0, max_key_age=MAX_KEY_AGE, pool_can_draw=True)

    for action in Action:
        if int(action) < int(floor):
            assert mask[action] == False  # noqa: E712
