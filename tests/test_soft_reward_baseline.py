"""Behavioral tests for `agents.soft_reward_baseline` -- the reproduced
soft-reward design that the steering attack targets (PLAN.md §2, §5 S5).

The property under test is not that this agent is *good*. It is that it
is a faithful reproduction of the design being critiqued: security is a
preference priced against cost, so the protection it selects is a
function of the threat score it is shown -- which is exactly what makes
it steerable.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.soft_reward_baseline import (
    SoftRewardAgent,
    SoftRewardConfig,
    soft_reward,
)
from env.contracts import N_ACTIONS, Action


def _state(sensitivity_class: int = 3, pool_fill: float = 0.8, threat_score: float = 0.5) -> dict:
    return {
        "sensitivity_class": sensitivity_class,
        "pool_fill": pool_fill,
        "threat_score": threat_score,
    }


def _mask(*legal: Action) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for action in legal:
        mask[int(action)] = True
    return mask


_ALL_TIERS = _mask(Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)


# ---------------------------------------------------------------------------
# The reward itself -- this is where the steerability lives
# ---------------------------------------------------------------------------


def test_stronger_tiers_are_preferred_when_the_threat_score_is_high():
    """The design works as intended on an honest signal. That is what
    makes the steering result interesting rather than a strawman."""
    rewards = {
        action: soft_reward(_state(), action, threat_score=1.0)
        for action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)
    }
    assert rewards[Action.SERVE_HYBRID] > rewards[Action.SERVE_PQC]
    assert rewards[Action.SERVE_PQC] > rewards[Action.SERVE_CLASSICAL]


def test_the_preference_inverts_when_the_threat_score_is_suppressed():
    """The whole attack, visible in one assertion: drive the score to
    calm and the cheapest -- weakest -- tier becomes the best choice."""
    rewards = {
        action: soft_reward(_state(), action, threat_score=0.02)
        for action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)
    }
    assert rewards[Action.SERVE_CLASSICAL] > rewards[Action.SERVE_PQC]
    assert rewards[Action.SERVE_PQC] > rewards[Action.SERVE_HYBRID]


def test_the_security_term_vanishes_entirely_at_zero_threat_score():
    """`w_sec * threat_score * tier_strength` -> 0 leaves only costs, so
    protection stops being represented in the objective at all."""
    for action in Action:
        assert soft_reward(_state(), action, threat_score=0.0) == pytest.approx(
            soft_reward(_state(), action, threat_score=0.0, config=SoftRewardConfig(w_sec=0.0))
        )


def test_the_reward_does_not_depend_on_sensitivity_class():
    """The design has no per-class floor -- protection is priced purely
    off the observed threat. This is precisely the property the policy
    table replaces."""
    for action in Action:
        assert soft_reward(_state(sensitivity_class=0), action, 0.5) == pytest.approx(
            soft_reward(_state(sensitivity_class=3), action, 0.5)
        )


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


def test_q_rows_initialize_to_the_myopic_value_not_to_zeros():
    """Zeros are actively wrong here: every real soft-reward value is
    negative, so a zero-initialized row makes every *unvisited* action
    strictly preferred to every learned one and a greedy policy selects
    whatever it has never tried. That produced a dose-response curve
    flat at 44% across all doses -- the agent reporting its
    initialization rather than its learning."""
    agent = SoftRewardAgent(seed=0)
    state = _state(threat_score=0.9)
    row = agent._row(agent._key(state, 0.9), state, 0.9)
    expected = [soft_reward(state, Action(i), 0.9) for i in range(N_ACTIONS)]
    assert row == pytest.approx(expected)
    assert all(value < 0.5 for value in row)


def test_unmasked_agent_ignores_the_mask_by_default():
    """`respect_mask=False` is what makes this the unmasked comparison
    point -- the design under critique has no policy table."""
    agent = SoftRewardAgent(seed=0, respect_mask=False)
    only_hybrid = _mask(Action.SERVE_HYBRID)
    chosen = agent.act(_state(threat_score=0.02), only_hybrid)
    assert not only_hybrid[int(chosen)]  # it picked something the mask forbade


def test_masked_agent_stays_inside_the_mask_when_asked_to():
    agent = SoftRewardAgent(seed=0, respect_mask=True)
    for _ in range(20):
        mask = _mask(Action.SERVE_PQC, Action.REUSE)
        assert mask[int(agent.act(_state(), mask))]


def test_greedy_tier_choice_tracks_the_threat_score():
    """The agent, not just the reward function, is steerable: same state,
    same mask, different observed threat -> different protection."""
    agent = SoftRewardAgent(seed=0, respect_mask=True)
    high = agent.act(_state(threat_score=0.95), _ALL_TIERS)
    low = agent.act(_state(threat_score=0.02), _ALL_TIERS)
    assert int(high) > int(low)


def test_learning_moves_q_values_and_reports_the_td_error():
    agent = SoftRewardAgent(seed=0)
    state, next_state = _state(threat_score=0.9), _state(threat_score=0.9)
    before = agent._row(agent._key(state, 0.9), state, 0.9).copy()
    metrics = agent.learn(state, Action.SERVE_HYBRID, 0.9, next_state, 0.9)
    after = agent._row(agent._key(state, 0.9), state, 0.9)

    assert "reward" in metrics and "td_error" in metrics
    assert not np.allclose(before, after)


def test_same_seed_reproduces_the_same_exploration_sequence():
    a, b = SoftRewardAgent(seed=5), SoftRewardAgent(seed=5)
    sequence_a = [a.act_exploring(_state(), _ALL_TIERS) for _ in range(50)]
    sequence_b = [b.act_exploring(_state(), _ALL_TIERS) for _ in range(50)]
    assert sequence_a == sequence_b


def test_agent_satisfies_the_shared_policy_interface():
    """So `experiments/harness.py` and `attack/run_attack.py` can run it
    like any other policy."""
    from agents.baselines import Policy

    agent: Policy = SoftRewardAgent(seed=0, respect_mask=True)
    assert isinstance(agent.act(_state(), _ALL_TIERS), Action)
