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
    GreedyRecommenderPolicy,
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


MAX_KEY_AGE = 500.0
"""Key-lifetime cap `L` for the dummy states below. Fixed here rather
than read from config so these unit tests stay independent of config
edits; policies that read it are constructed with a matching explicit
`max_key_age`."""


def _dummy_state(
    pool_fill: float = 0.5,
    key_age: float = MAX_KEY_AGE,
    sensitivity_class: int = 3,
) -> dict:
    """Minimal `StateDict` stand-in covering the fields the policies read.

    `key_age` defaults to the cap -- i.e. "the existing key is stale, so
    `StaticThresholdPolicy`'s key-lifetime rule does not fire" -- which
    keeps these tests aimed at the *tier* half of that policy, exactly
    as they were written before the `rho`/REUSE rule was added on
    2026-08-15. `sensitivity_class` defaults to the highest class for
    the same reason: it clears any `min_hybrid_class` gate so the tier
    rule is what is under test.
    """
    return {
        "pool_fill": pool_fill,
        "key_age": key_age,
        "sensitivity_class": sensitivity_class,
    }


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
    # Inclusive at the threshold: SMARTKEYNET_BUILD_SPEC.md §S7 defines
    # the rule as "serve hybrid iff pool_fill >= tau". This was a strict
    # ">" until 2026-08-15; the boundary itself is not load-bearing (the
    # threshold is grid-searched over a coarse grid either way), but
    # matching the spec exactly keeps the baseline defensible.
    assert policy.act(_dummy_state(0.5), mask) == Action.SERVE_HYBRID
    assert policy.act(_dummy_state(0.5 - 1e-9), mask) == Action.SERVE_PQC
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


# ---------------------------------------------------------------------------
# StaticThresholdPolicy's key-lifetime rule (added 2026-08-15)
#
# Regression coverage for the strawman-baseline bug: without the
# `rho`/REUSE half of the spec's rule, this policy re-established key
# material on every single decision and the DQN beat it by an order of
# magnitude purely by discovering REUSE.
# ---------------------------------------------------------------------------


def test_reuses_a_fresh_key_instead_of_rekeying():
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5, rekey_age_frac=0.9, max_key_age=MAX_KEY_AGE)
    mask = _mask(*list(Action))
    state = _dummy_state(pool_fill=1.0, key_age=0.1 * MAX_KEY_AGE)
    assert policy.act(state, mask) == Action.REUSE


def test_rekeys_once_the_key_passes_the_age_fraction():
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5, rekey_age_frac=0.9, max_key_age=MAX_KEY_AGE)
    mask = _mask(*list(Action))

    just_under = _dummy_state(pool_fill=1.0, key_age=0.9 * MAX_KEY_AGE - 1)
    just_over = _dummy_state(pool_fill=1.0, key_age=0.9 * MAX_KEY_AGE)

    assert policy.act(just_under, mask) == Action.REUSE
    assert policy.act(just_over, mask) != Action.REUSE


def test_never_reuses_when_reuse_is_masked_however_fresh_the_key():
    """The mask always wins over the policy's own preference."""
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5, rekey_age_frac=0.9, max_key_age=MAX_KEY_AGE)
    mask = _mask(Action.SERVE_PQC, Action.SERVE_HYBRID, Action.REKEY_NOW)
    state = _dummy_state(pool_fill=1.0, key_age=0.0)
    assert policy.act(state, mask) != Action.REUSE


def test_min_hybrid_class_gates_pool_spending():
    """`c_min`: a class below the gate must not spend the pool even
    with a full pool and hybrid legal."""
    policy = StaticThresholdPolicy(
        pool_fill_threshold=0.0, min_hybrid_class=2, rekey_age_frac=0.0, max_key_age=MAX_KEY_AGE
    )
    mask = _mask(*list(Action))

    below_gate = _dummy_state(pool_fill=1.0, key_age=MAX_KEY_AGE, sensitivity_class=1)
    at_gate = _dummy_state(pool_fill=1.0, key_age=MAX_KEY_AGE, sensitivity_class=2)

    assert policy.act(below_gate, mask) == Action.SERVE_PQC
    assert policy.act(at_gate, mask) == Action.SERVE_HYBRID


@pytest.mark.parametrize("mask", ALL_MASKS)
def test_static_threshold_with_all_three_params_never_illegal(mask):
    policy = StaticThresholdPolicy(
        pool_fill_threshold=0.5, min_hybrid_class=2, rekey_age_frac=0.5, max_key_age=MAX_KEY_AGE
    )
    for key_age in (0.0, 0.4 * MAX_KEY_AGE, MAX_KEY_AGE):
        for sensitivity_class in range(4):
            action = policy.act(_dummy_state(0.7, key_age, sensitivity_class), mask)
            assert mask[int(action)]


# ---------------------------------------------------------------------------
# GreedyRecommenderPolicy (spec §S7 diagnostic 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask", ALL_MASKS)
def test_greedy_recommender_never_illegal(mask):
    action = GreedyRecommenderPolicy().act(_dummy_state(), mask)
    assert mask[int(action)]


def test_greedy_recommender_prefers_reuse_then_cheapest_tier():
    policy = GreedyRecommenderPolicy()
    state = _dummy_state()

    assert policy.act(state, _mask(*list(Action))) == Action.REUSE
    assert policy.act(state, _mask(Action.SERVE_CLASSICAL, Action.SERVE_HYBRID)) == Action.SERVE_CLASSICAL
    assert policy.act(state, _mask(Action.SERVE_PQC, Action.SERVE_HYBRID)) == Action.SERVE_PQC
    # hybrid is the last resort: it is the only action that pays w_qkd
    assert policy.act(state, _mask(Action.SERVE_HYBRID)) == Action.SERVE_HYBRID
