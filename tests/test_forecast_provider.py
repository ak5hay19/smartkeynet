"""Behavioral tests for `env.forecast_provider` (PLAN.md Addition A:
`MovingAverageForecaster` is the EWMA fallback that lets the env run
before the real LSTM forecaster exists).
"""

from __future__ import annotations

import pytest

from env.contracts import ForecastObservation
from env.forecast_provider import MovingAverageForecaster


def make_observation(
    threat_features: list[float],
    pool_fill: float = 0.5,
    skr: float = 10.0,
    hybrid_serves: int = 1,
    qber: float = 0.02,
    arrivals_per_class: tuple[int, ...] = (1, 1, 1, 1),
) -> ForecastObservation:
    return ForecastObservation(
        qber=qber,
        skr=skr,
        pool_fill=pool_fill,
        arrivals_per_class=list(arrivals_per_class),
        hybrid_serves=hybrid_serves,
        threat_features=list(threat_features),
    )


# ---------------------------------------------------------------------------
# Fresh instance
# ---------------------------------------------------------------------------


def test_fresh_instance_does_not_crash_and_is_sensible():
    forecaster = MovingAverageForecaster(alpha=0.3)

    threat = forecaster.get_threat_forecast()
    assert threat.threat_score == pytest.approx(0.0)
    assert sum(threat.posture_probs) == pytest.approx(1.0)
    assert len(threat.posture_probs) == 3
    assert len(threat.horizon_scores) == 5

    pool = forecaster.get_pool_forecast()
    assert len(pool.pool_level_hat) == 3
    assert len(pool.skr_mean_hat) == 3
    assert len(pool.hybrid_demand_hat) == 3
    assert pool.pool_level_hat == [0.0, 0.0, 0.0]
    assert pool.skr_mean_hat == [0.0, 0.0, 0.0]
    assert pool.hybrid_demand_hat == [0.0, 0.0, 0.0]


def test_alpha_must_be_in_valid_range():
    with pytest.raises(ValueError):
        MovingAverageForecaster(alpha=0.0)
    with pytest.raises(ValueError):
        MovingAverageForecaster(alpha=1.5)
    with pytest.raises(ValueError):
        MovingAverageForecaster(alpha=-0.1)


# ---------------------------------------------------------------------------
# EWMA smoothing behavior
# ---------------------------------------------------------------------------


def test_update_smooths_rather_than_snapping_to_newest_value():
    forecaster = MovingAverageForecaster(alpha=0.3)
    for _ in range(3):
        forecaster.update(make_observation(threat_features=[0.0]))

    # a single big jump shouldn't fully snap the smoothed score to
    # sigmoid(10) (~0.99995) -- EWMA should still be dragging it down
    forecaster.update(make_observation(threat_features=[10.0]))
    threat_score = forecaster.get_threat_forecast().threat_score

    assert 0.0 < threat_score < 0.9


def test_higher_alpha_reacts_faster_than_lower_alpha():
    obs_sequence = [make_observation(threat_features=[0.0])] * 3 + [make_observation(threat_features=[10.0])]

    slow = MovingAverageForecaster(alpha=0.1)
    fast = MovingAverageForecaster(alpha=0.9)
    for obs in obs_sequence:
        slow.update(obs)
        fast.update(obs)

    assert fast.get_threat_forecast().threat_score > slow.get_threat_forecast().threat_score


def test_alpha_one_snaps_directly_to_latest_observation():
    forecaster = MovingAverageForecaster(alpha=1.0)
    forecaster.update(make_observation(threat_features=[0.0], pool_fill=500.0, skr=20.0, hybrid_serves=4))

    pool = forecaster.get_pool_forecast()
    assert pool.pool_level_hat[0] == pytest.approx(500.0)
    assert pool.skr_mean_hat[0] == pytest.approx(20.0)
    assert pool.hybrid_demand_hat[0] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Threat forecast invariants
# ---------------------------------------------------------------------------


def test_posture_probs_always_sum_to_one():
    forecaster = MovingAverageForecaster(alpha=0.5)
    for features in ([0.0], [5.0], [-5.0], [100.0], [-100.0]):
        forecaster.update(make_observation(threat_features=features))
        probs = forecaster.get_threat_forecast().posture_probs
        assert sum(probs) == pytest.approx(1.0)
        assert all(p >= 0.0 for p in probs)


def test_posture_probs_shift_toward_high_as_threat_score_rises():
    forecaster = MovingAverageForecaster(alpha=1.0)  # snap directly for a clean comparison
    forecaster.update(make_observation(threat_features=[-100.0]))
    calm_probs = forecaster.get_threat_forecast().posture_probs

    forecaster.update(make_observation(threat_features=[100.0]))
    high_probs = forecaster.get_threat_forecast().posture_probs

    # index 0 = CALM anchor, index 2 = HIGH anchor (PLAN.md ThreatPosture order)
    assert calm_probs[0] > high_probs[0]
    assert high_probs[2] > calm_probs[2]


# ---------------------------------------------------------------------------
# Pool forecast flat-hold design
# ---------------------------------------------------------------------------


def test_pool_forecast_horizons_are_flat_held():
    forecaster = MovingAverageForecaster(alpha=0.4)
    forecaster.update(make_observation(threat_features=[0.0], pool_fill=123.0, skr=45.0, hybrid_serves=2))
    forecaster.update(make_observation(threat_features=[0.0], pool_fill=200.0, skr=50.0, hybrid_serves=3))

    pool = forecaster.get_pool_forecast()
    assert pool.pool_level_hat[0] == pool.pool_level_hat[1] == pool.pool_level_hat[2]
    assert pool.skr_mean_hat[0] == pool.skr_mean_hat[1] == pool.skr_mean_hat[2]
    assert pool.hybrid_demand_hat[0] == pool.hybrid_demand_hat[1] == pool.hybrid_demand_hat[2]


def test_threat_horizon_scores_are_also_flat_held():
    forecaster = MovingAverageForecaster(alpha=0.6)
    forecaster.update(make_observation(threat_features=[3.0]))

    horizon_scores = forecaster.get_threat_forecast().horizon_scores
    assert len(horizon_scores) == 5
    assert all(s == horizon_scores[0] for s in horizon_scores)
