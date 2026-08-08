"""Behavioral tests for `agents.baselines` -- the four tuned non-RL
baselines (PLAN.md Hard Rule 7; §10 kickoff step 6). Must be built and
comparable *before* the DQN is tuned.

The one property that actually matters, tested first and adversarially:
none of these policies can ever return an action outside the mask they
were given, no matter how contrived the mask.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from agents.baselines import (
    AlwaysHybridPolicy,
    AlwaysPQCPolicy,
    RandomPolicy,
    StaticThresholdPolicy,
)
from env.contracts import N_ACTIONS, Action


def _mask(*legal_actions: Action) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for action in legal_actions:
        mask[int(action)] = True
    return mask


def _all_nonempty_masks() -> list[np.ndarray]:
    """Every one of the 2**5 - 1 possible non-empty legal-action sets --
    including contrived ones no real `compute_mask` output would ever
    produce (e.g. only REUSE legal) -- because the adversarial-mask
    guarantee must hold regardless of where the mask came from."""
    masks = []
    for r in range(1, N_ACTIONS + 1):
        for combo in itertools.combinations(list(Action), r):
            masks.append(_mask(*combo))
    return masks


ALL_MASKS = _all_nonempty_masks()


def _dummy_state(pool_fill: float = 0.5) -> dict:
    """Minimal `StateDict` stand-in -- only `pool_fill` is ever read by
    any policy in this module (`StaticThresholdPolicy`); the others
    ignore `state` entirely."""
    return {"pool_fill": pool_fill}


# ---------------------------------------------------------------------------
# Adversarial: never return an action outside the mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask", ALL_MASKS)
def test_always_pqc_never_illegal(mask):
    action = AlwaysPQCPolicy().act(_dummy_state(), mask)
    assert mask[int(action)]


@pytest.mark.parametrize("mask", ALL_MASKS)
def test_always_hybrid_never_illegal(mask):
    action = AlwaysHybridPolicy().act(_dummy_state(), mask)
    assert mask[int(action)]


@pytest.mark.parametrize("mask", ALL_MASKS)
@pytest.mark.parametrize("pool_fill", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_static_threshold_never_illegal(mask, pool_fill):
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)
    action = policy.act(_dummy_state(pool_fill), mask)
    assert mask[int(action)]


@pytest.mark.parametrize("mask", ALL_MASKS)
def test_random_policy_never_illegal(mask):
    policy = RandomPolicy(seed=0)
    for _ in range(20):
        action = policy.act(_dummy_state(), mask)
        assert mask[int(action)]


# ---------------------------------------------------------------------------
# AlwaysPQCPolicy
# ---------------------------------------------------------------------------


def test_always_pqc_prefers_pqc_when_legal():
    mask = _mask(*list(Action))
    assert AlwaysPQCPolicy().act(_dummy_state(), mask) == Action.SERVE_PQC


def test_always_pqc_never_voluntarily_draws_hybrid_when_pqc_legal():
    """Never draws from the pool unless the floor actually forces it --
    i.e. whenever SERVE_PQC is legal at all, it's chosen over
    SERVE_HYBRID, even if HYBRID is also legal."""
    policy = AlwaysPQCPolicy()
    for mask in ALL_MASKS:
        if mask[int(Action.SERVE_PQC)]:
            assert policy.act(_dummy_state(), mask) == Action.SERVE_PQC


def test_always_pqc_falls_back_to_lowest_legal_tier_when_floor_forces_hybrid():
    # floor forces >= SERVE_HYBRID: CLASSICAL/PQC illegal
    mask = _mask(Action.SERVE_HYBRID, Action.REKEY_NOW)
    assert AlwaysPQCPolicy().act(_dummy_state(), mask) == Action.SERVE_HYBRID


# ---------------------------------------------------------------------------
# AlwaysHybridPolicy
# ---------------------------------------------------------------------------


def test_always_hybrid_draws_whenever_legal():
    policy = AlwaysHybridPolicy()
    for mask in ALL_MASKS:
        if mask[int(Action.SERVE_HYBRID)]:
            assert policy.act(_dummy_state(), mask) == Action.SERVE_HYBRID


def test_always_hybrid_falls_back_to_lowest_legal_tier_when_hybrid_masked():
    mask = _mask(Action.SERVE_CLASSICAL, Action.SERVE_PQC)
    assert AlwaysHybridPolicy().act(_dummy_state(), mask) == Action.SERVE_CLASSICAL


# ---------------------------------------------------------------------------
# StaticThresholdPolicy
# ---------------------------------------------------------------------------


def test_static_threshold_flips_at_boundary():
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)
    mask = _mask(*list(Action))  # nothing else constrains the choice

    assert policy.act(_dummy_state(0.5 + 1e-9), mask) == Action.SERVE_HYBRID
    assert policy.act(_dummy_state(0.5), mask) == Action.SERVE_PQC  # exactly at threshold: not "exceeds"
    assert policy.act(_dummy_state(0.5 - 1e-9), mask) == Action.SERVE_PQC


def test_static_threshold_falls_back_when_pqc_also_masked():
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)
    mask = _mask(Action.SERVE_CLASSICAL)  # floor forces classical; nothing else legal
    assert policy.act(_dummy_state(0.9), mask) == Action.SERVE_CLASSICAL


def test_grid_search_picks_genuinely_best_scoring_candidate():
    candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
    true_best = 0.7

    def eval_fn(policy: StaticThresholdPolicy) -> float:
        return -abs(policy.pool_fill_threshold - true_best)

    result = StaticThresholdPolicy.grid_search(candidates, eval_fn)
    assert result.pool_fill_threshold == pytest.approx(true_best)


def test_grid_search_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        StaticThresholdPolicy.grid_search([], lambda policy: 0.0)


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------


def test_random_policy_reproducible_with_same_seed():
    mask = _mask(*list(Action))
    policy_a = RandomPolicy(seed=42)
    policy_b = RandomPolicy(seed=42)
    seq_a = [policy_a.act(_dummy_state(), mask) for _ in range(50)]
    seq_b = [policy_b.act(_dummy_state(), mask) for _ in range(50)]
    assert seq_a == seq_b


def test_random_policy_different_seeds_diverge():
    mask = _mask(*list(Action))
    policy_a = RandomPolicy(seed=1)
    policy_b = RandomPolicy(seed=2)
    seq_a = [policy_a.act(_dummy_state(), mask) for _ in range(50)]
    seq_b = [policy_b.act(_dummy_state(), mask) for _ in range(50)]
    assert seq_a != seq_b


def test_random_policy_roughly_uniform_over_legal_actions():
    mask = _mask(*list(Action))  # all 5 legal
    policy = RandomPolicy(seed=7)
    n = 10_000
    counts = {action: 0 for action in Action}
    for _ in range(n):
        counts[policy.act(_dummy_state(), mask)] += 1

    expected = n / len(Action)
    for action, count in counts.items():
        assert abs(count - expected) < expected * 0.2, f"{action} drawn {count}/{n} times, expected ~{expected}"


def test_random_policy_single_legal_action_always_returned():
    mask = _mask(Action.REKEY_NOW)  # a contrived, single-option mask
    policy = RandomPolicy(seed=3)
    for _ in range(20):
        assert policy.act(_dummy_state(), mask) == Action.REKEY_NOW
