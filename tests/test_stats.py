"""Tests for `metrics.aggregate` -- the §9 statistical protocol."""

from __future__ import annotations

import numpy as np
import pytest

from metrics.aggregate import (
    Estimate,
    bootstrap_ci,
    holm_bonferroni,
    interquartile_mean,
    paired_difference,
)


def test_iqm_matches_a_hand_computed_value():
    # sorted: 1..8 -> middle 50% is indices 2..5 -> {3,4,5,6} -> 4.5
    assert interquartile_mean([5, 1, 8, 3, 6, 2, 7, 4]) == pytest.approx(4.5)


def test_iqm_discards_a_diverged_seed_that_would_wreck_the_mean():
    """The reason IQM is the point estimate: this project measured per-seed
    results spanning -1,326 to -3,015,813 on one configuration."""
    seeds = [-1300, -1400, -1350, -1450, -1500, -1250, -1380, -3_000_000]
    assert interquartile_mean(seeds) > -1600
    assert float(np.mean(seeds)) < -300_000


def test_bootstrap_ci_brackets_the_point_estimate():
    values = [10.0, 12.0, 11.0, 13.0, 9.0, 11.5, 10.5, 12.5]
    estimate = bootstrap_ci(values, seed=0)
    assert estimate.low <= estimate.point <= estimate.high
    assert estimate.n_seeds == len(values)


def test_bootstrap_ci_is_deterministic_under_seed():
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_bootstrap_ci_narrows_with_more_seeds():
    rng = np.random.default_rng(0)
    narrow = bootstrap_ci(rng.normal(0, 1, 200), seed=0)
    wide = bootstrap_ci(rng.normal(0, 1, 8), seed=0)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_paired_difference_detects_a_real_effect():
    """A consistent per-seed gap must produce an interval excluding zero,
    even when each policy's own spread is large."""
    rng = np.random.default_rng(0)
    shared_seed_noise = rng.normal(0, 50, 10)
    policy_a = shared_seed_noise + 100.0
    policy_b = shared_seed_noise + 90.0  # always exactly 10 better
    difference = paired_difference(policy_a, policy_b, seed=0)
    assert difference.excludes(0.0)
    assert difference.point == pytest.approx(10.0, abs=1.0)


def test_paired_difference_reports_no_effect_when_there_is_none():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 10)
    b = rng.normal(0, 1, 10)
    assert not paired_difference(a, b, seed=0).excludes(0.0)


def test_pairing_actually_uses_the_pairing():
    """Permuting one policy's seed order must change the result -- otherwise
    the comparison is not paired at all (spec §S12's own check)."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 50, 10)
    a, b = noise + 100.0, noise + 90.0
    paired = paired_difference(a, b, seed=0)
    shuffled = paired_difference(a, rng.permutation(b), seed=0)
    assert (paired.high - paired.low) < (shuffled.high - shuffled.low)


def test_paired_difference_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_difference([1.0, 2.0], [1.0])


def test_holm_bonferroni_is_more_conservative_than_uncorrected():
    p_values = [0.001, 0.02, 0.03, 0.04, 0.049]
    rejected = holm_bonferroni(p_values, alpha=0.05)
    assert rejected[0] is True  # smallest survives
    assert sum(rejected) < sum(p < 0.05 for p in p_values)


def test_holm_bonferroni_preserves_input_order():
    rejected = holm_bonferroni([0.6, 0.0001, 0.7], alpha=0.05)
    assert rejected == [False, True, False]


def test_estimate_excludes_is_the_claim_test():
    estimate = Estimate(point=5.0, low=2.0, high=8.0, n_seeds=10)
    assert not estimate.excludes(0.0) or True  # 0 is outside [2,8]
    assert estimate.excludes(0.0)
    assert not estimate.excludes(5.0)
