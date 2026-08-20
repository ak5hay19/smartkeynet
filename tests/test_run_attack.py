"""Behavioral tests for `attack.run_attack` -- the S5 dose-response
experiment (PLAN2 §7.5, the headline contribution).

The claim under test is Hard Rule 2's: a threat signal may only ever
raise a floor, never lower one. These tests pin the *measurement*, so
that the headline number cannot quietly stop meaning what it says.
"""

from __future__ import annotations

import pytest

from agents.baselines import AlwaysHybridPolicy, AlwaysPQCPolicy
from attack.run_attack import (
    class_floor,
    escalated_floor,
    run_dose_response,
)
from attack.steering_trace import default_honest_trace
from env.contracts import Action, SensitivityClass, ThreatPosture
from experiments.train import load_full_config


def test_class_floor_is_the_calm_row_and_is_signal_independent():
    """This is the level Hard Rule 2 guarantees -- the ratchet starts
    here and only ever moves up, so no threat signal can put a request
    below it."""
    assert class_floor(SensitivityClass.S3) is Action.SERVE_HYBRID
    assert class_floor(SensitivityClass.S2) is Action.SERVE_PQC
    assert class_floor(SensitivityClass.S0) is Action.SERVE_CLASSICAL


def test_escalated_floor_is_never_below_the_class_floor():
    for sensitivity in SensitivityClass:
        for posture in ThreatPosture:
            assert int(escalated_floor(sensitivity, posture)) >= int(class_floor(sensitivity))


@pytest.mark.slow
def test_dose_response_masked_arm_is_flat_at_zero_and_soft_reward_arm_is_not():
    """The headline result, as an executable assertion.

    Deliberately asserts the *structural* claim (masked arm identically
    zero at every dose) tightly, and the empirical one (soft-reward arm
    is non-zero and non-decreasing) loosely -- the second is a measured
    outcome that may move with the environment, the first is a guarantee
    that must not.
    """
    config = load_full_config()
    response = run_dose_response(config, doses=[0.0, 0.5, 1.0], max_steps=800, seed=0)

    masked = response.results["masked architecture (security is a constraint)"]
    soft = response.results["soft-reward (security IS the reward, no mask)"]

    assert [r.below_class_floor_share for r in masked] == [0.0, 0.0, 0.0]

    shares = [r.below_class_floor_share for r in soft]
    assert shares == sorted(shares), "soft-reward arm should not improve under a stronger attack"
    assert max(shares) > 0.0, "soft-reward arm never served below the class floor at all"
    assert max(shares) > min(shares), "the attack had no effect on the soft-reward arm"


@pytest.mark.slow
def test_every_arm_is_scored_on_the_same_number_of_establishments():
    """Both arms share one trajectory precisely so the comparison cannot
    be confounded by the trajectory each would have induced."""
    config = load_full_config()
    response = run_dose_response(config, doses=[0.0, 1.0], max_steps=600, seed=0)
    for dose_index in range(2):
        counts = {name: runs[dose_index].decisions for name, runs in response.results.items()}
        assert len(set(counts.values())) == 1, counts


@pytest.mark.slow
def test_dose_response_serializes_the_numbers_the_dashboard_plots():
    config = load_full_config()
    response = run_dose_response(config, doses=[0.0, 1.0], max_steps=400, seed=0)
    payload = response.to_json()
    assert '"below_class_floor_share"' in payload
    assert '"below_escalated_floor_share"' in payload
    assert '"tier_counts"' in payload
