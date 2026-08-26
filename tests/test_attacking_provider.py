"""
tests/test_attacking_provider.py

Tests for `attack/attacking_provider.py::AttackingForecastProvider`
(PLAN.md §5 S5 dose-response sweep session, Part 1). Per that
session's instruction, these are real tests proven BEFORE any real
sweep is run.
"""

from __future__ import annotations

import math

import pytest

from attack.attacking_provider import AttackingForecastProvider
from attack.trace_generator import g
from env.contracts import ForecastObservation
from env.forecast_provider import MovingAverageForecaster


def _obs(threat_features: list[float]) -> ForecastObservation:
    return ForecastObservation(
        qber=0.1,
        skr=0.2,
        pool_fill=0.5,
        arrivals_per_class=[1, 0, 0, 0],
        hybrid_serves=0,
        threat_features=threat_features,
    )


class TestAlphaZeroExactEquality:
    """alpha=0.0 must reproduce the unwrapped base provider's output
    exactly (bit-for-bit), on both the threat and pool heads, across a
    real multi-step sequence -- not just a single call."""

    def test_threat_and_pool_forecasts_match_exactly(self):
        plain = MovingAverageForecaster()
        wrapped = AttackingForecastProvider(base_provider=MovingAverageForecaster(), alpha=0.0)

        # A real, varying sequence -- not a single repeated observation --
        # so this genuinely exercises the EWMA's running state, not just
        # its fresh-instance defaults.
        windows = [[0.1, 0.2], [0.4, 0.1], [0.9, 0.9], [0.0, 0.05], [2.0, 3.0]]
        for w in windows:
            obs = _obs(w)
            plain.update(obs)
            wrapped.update(obs)

            plain_threat = plain.get_threat_forecast()
            wrapped_threat = wrapped.get_threat_forecast()
            assert wrapped_threat.threat_score == plain_threat.threat_score
            assert list(wrapped_threat.posture_probs) == list(plain_threat.posture_probs)
            assert list(wrapped_threat.horizon_scores) == list(plain_threat.horizon_scores)

            plain_pool = plain.get_pool_forecast()
            wrapped_pool = wrapped.get_pool_forecast()
            assert list(wrapped_pool.pool_level_hat) == list(plain_pool.pool_level_hat)
            assert list(wrapped_pool.skr_mean_hat) == list(plain_pool.skr_mean_hat)
            assert list(wrapped_pool.hybrid_demand_hat) == list(plain_pool.hybrid_demand_hat)

    def test_fresh_instance_matches_before_any_update(self):
        plain = MovingAverageForecaster()
        wrapped = AttackingForecastProvider(base_provider=MovingAverageForecaster(), alpha=0.0)
        assert wrapped.get_threat_forecast() == plain.get_threat_forecast()
        assert wrapped.get_pool_forecast() == plain.get_pool_forecast()


class TestAlphaOneFullReplacement:
    """alpha=1.0 must reproduce feeding `g(window)` directly into the
    base provider, exactly -- the other boundary already proven by
    `attack/trace_generator.py`'s own tests, re-verified here through
    the wrapper."""

    def test_matches_feeding_g_directly(self):
        plain = MovingAverageForecaster()
        wrapped = AttackingForecastProvider(base_provider=MovingAverageForecaster(), alpha=1.0)

        windows = [[5.0, 5.0], [1.0, 9.0], [0.3, 0.3]]
        for w in windows:
            plain.update(_obs(g(w)))
            wrapped.update(_obs(w))

            assert wrapped.get_threat_forecast() == plain.get_threat_forecast()
            assert wrapped.get_pool_forecast() == plain.get_pool_forecast()


class TestConstructorValidation:
    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError):
            AttackingForecastProvider(base_provider=MovingAverageForecaster(), alpha=1.5)
        with pytest.raises(ValueError):
            AttackingForecastProvider(base_provider=MovingAverageForecaster(), alpha=-0.1)


class TestHandVerifiedSyntheticCase:
    """The single most important test this session produces (per
    instruction): a constructed synthetic case where the expected TRUE
    posture and ATTACKED posture are computed independently (via a
    separate, direct `MovingAverageForecaster` instance never touched
    by `AttackingForecastProvider`), then confirmed to be exactly what
    the dual-tracking mechanism (`get_true_threat_forecast()` vs.
    `get_threat_forecast()`) reports.

    Reuses the exact same severe true_window (`[5.0, 5.0]`) and
    alpha=1.0 full attack that `attack/trace_generator.py`'s own
    2026-08-25 end-to-end test already verified against the real
    forecaster in isolation (true steady-state threat_score=0.9933,
    posture argmax HIGH; shaped steady-state threat_score=0.5000
    exactly, posture argmax ELEVATED -- a genuine discrete crossing) --
    this test re-derives those same two numbers completely independently
    (fresh MovingAverageForecaster instances, fed directly, no import of
    or dependency on the prior session's recorded numbers) and confirms
    AttackingForecastProvider's dual-tracking reports exactly them.
    """

    N_STEPS_TO_STEADY_STATE = 60  # matches trace_generator's own test convention

    def test_true_and_attacked_posture_diverge_across_a_discrete_boundary(self):
        true_window = [5.0, 5.0]
        alpha = 1.0

        # --- independently computed expected values (no AttackingForecastProvider involved) ---
        expected_true = MovingAverageForecaster()
        expected_attacked = MovingAverageForecaster()
        for _ in range(self.N_STEPS_TO_STEADY_STATE):
            expected_true.update(_obs(true_window))
            expected_attacked.update(_obs(g(true_window)))  # g([5,5]) == [0.0, 0.0]

        expected_true_forecast = expected_true.get_threat_forecast()
        expected_attacked_forecast = expected_attacked.get_threat_forecast()

        # sanity: these are the exact figures already independently verified
        # in attack/trace_generator.py's own 2026-08-25 test session
        assert expected_true_forecast.threat_score == pytest.approx(0.9933071490757153)
        assert expected_attacked_forecast.threat_score == pytest.approx(0.5, abs=1e-9)

        import numpy as np
        from env.contracts import ThreatPosture

        expected_true_posture = ThreatPosture(int(np.argmax(expected_true_forecast.posture_probs)))
        expected_attacked_posture = ThreatPosture(int(np.argmax(expected_attacked_forecast.posture_probs)))
        assert expected_true_posture == ThreatPosture.HIGH
        assert expected_attacked_posture == ThreatPosture.ELEVATED
        assert expected_true_posture != expected_attacked_posture  # the discrete crossing itself

        # --- now drive the actual mechanism under test ---
        shadow = MovingAverageForecaster()
        attacking = AttackingForecastProvider(
            base_provider=MovingAverageForecaster(), alpha=alpha, shadow_provider=shadow
        )
        for _ in range(self.N_STEPS_TO_STEADY_STATE):
            attacking.update(_obs(true_window))

        actual_attacked_forecast = attacking.get_threat_forecast()
        actual_true_forecast = attacking.get_true_threat_forecast()

        # exact equality against the independently-computed expected values --
        # not approximate, not re-derived from the same code path a second time
        assert actual_attacked_forecast == expected_attacked_forecast
        assert actual_true_forecast == expected_true_forecast

        actual_attacked_posture = ThreatPosture(int(np.argmax(actual_attacked_forecast.posture_probs)))
        actual_true_posture = ThreatPosture(int(np.argmax(actual_true_forecast.posture_probs)))
        assert actual_attacked_posture == ThreatPosture.ELEVATED
        assert actual_true_posture == ThreatPosture.HIGH

    def test_get_true_threat_forecast_raises_without_shadow_provider(self):
        attacking = AttackingForecastProvider(base_provider=MovingAverageForecaster(), alpha=0.5)
        attacking.update(_obs([1.0, 1.0]))
        with pytest.raises(ValueError):
            attacking.get_true_threat_forecast()
