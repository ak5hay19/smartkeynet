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


# ---------------------------------------------------------------------------
# Posture-saturation investigation (2026-08-24, S3 recalibration session).
#
# PROGRESS.md/SESSION_LOG.md 2026-08-19 found that under real S1/S3
# config, floor *and* posture sequences were byte-identical -- read at
# the time as "posture is already saturated at ELEVATED, load
# dominates QBER." This session investigated the precise mechanism,
# directly against this real class (env/environment.py's
# `_threat_features_placeholder` returns `[qber, load]`, both in
# [0, 1]; `MovingAverageForecaster.update` averages them, squashes via
# sigmoid, then argmax's the RBF-softmax posture_probs -- see
# env/environment.py line ~753). Finding, precise and two-sided (not
# fixed here -- read/investigate only, per instruction; flagged as a
# separate, standing item, distinct from S3's pool/scarcity fix above):
#   (a) QBER genuinely, measurably moves threat_score and posture_probs
#       -- it is NOT negligible, contrary to a "load totally dominates"
#       reading.
#   (b) But the *discrete* posture classification (argmax(posture_probs),
#       the only thing env/masking.py's floor table actually reads)
#       can mathematically never reach HIGH for ANY (qber, load) in
#       [0, 1]^2 -- both features are bounded in [0, 1], so
#       raw_signal = mean(qber, load) is bounded in [0, 1], so
#       squashed_signal = sigmoid(raw_signal) is bounded in
#       [0.5, ~0.731] (sigmoid(0)=0.5, sigmoid(1)=0.731) -- a range
#       that, under the RBF-softmax's anchors {0.0, 0.5, 1.0} and
#       temperature 0.15, always sits closer to the ELEVATED anchor
#       (0.5) than the HIGH anchor (1.0), even at the theoretical
#       maximum of both features simultaneously. This is a real,
#       structural property of the current placeholder formula, not a
#       load-vs-QBER competition -- HIGH is architecturally
#       unreachable via this path regardless of which feature
#       dominates. (S2's threat_schedule sidesteps this entirely by
#       injecting a scripted signal unbounded by [0, 1] -- see
#       `_threat_features_placeholder`'s S2 branch -- which is exactly
#       why S2 can and does reach HIGH while S3, driven by real
#       qber/load, cannot.)
# Not fixed this session: env/forecast_provider.py is investigate-only
# per instruction; S3's own purpose is pool/scarcity budgeting (this
# module's row above), not posture/floor behavior (S2's territory).
# Flagged in PROGRESS.md as a new, separate open item.
# ---------------------------------------------------------------------------


def test_qber_alone_measurably_moves_threat_score_and_posture_probs():
    """(a) above: QBER is not negligible when load is held fixed --
    contrary to a naive "load totally dominates, QBER doesn't matter"
    reading of the 2026-08-19 byte-identical-posture finding."""
    for load in (0.0, 0.3, 0.5, 1.0):
        low_qber_forecaster = MovingAverageForecaster(alpha=0.3)
        high_qber_forecaster = MovingAverageForecaster(alpha=0.3)
        for _ in range(80):  # let the EWMA settle to steady state
            low_qber_forecaster.update(make_observation(threat_features=[0.02, load], qber=0.02))
            high_qber_forecaster.update(make_observation(threat_features=[0.62, load], qber=0.62))

        low = low_qber_forecaster.get_threat_forecast()
        high = high_qber_forecaster.get_threat_forecast()

        # threat_score moves by a real, non-trivial margin purely from qber
        assert high.threat_score - low.threat_score > 0.03
        # the HIGH-anchor posture probability (index 2) rises measurably too
        assert high.posture_probs[2] > low.posture_probs[2] * 1.3


def test_posture_argmax_can_never_reach_high_via_qber_load_placeholder():
    """(b) above: the *discrete* posture (what env/masking.py's floor
    table actually consumes) is structurally capped at ELEVATED for
    every possible (qber, load) combination in [0, 1]^2 -- swept
    exhaustively across the corners and center of the space, including
    the theoretical worst case (both features maxed simultaneously)."""
    HIGH_INDEX = 2
    ELEVATED_INDEX = 1
    for qber in (0.0, 0.02, 0.5, 0.62, 0.9, 0.999):
        for load in (0.0, 0.3, 0.5, 0.7, 1.0):
            forecaster = MovingAverageForecaster(alpha=0.3)
            for _ in range(80):
                forecaster.update(make_observation(threat_features=[qber, load], qber=qber))
            probs = forecaster.get_threat_forecast().posture_probs
            argmax = max(range(len(probs)), key=lambda i: probs[i])
            assert argmax != HIGH_INDEX, (
                f"qber={qber} load={load} reached HIGH posture ({probs}) -- "
                "this would contradict the confirmed structural ceiling"
            )
            assert argmax == ELEVATED_INDEX or argmax == 0
