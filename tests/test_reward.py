"""Tests for `env.reward` -- SMARTKEYNET_BUILD_SPEC.md §S5's six tests.

This file did not exist until 2026-08-19, because the reward had no module
of its own: the arithmetic lived inline inside
`SmartKeyNetEnv._apply_action`, where it could not be tested as a function
and where Hard Rule 1 was guarded only by a substring scan. Extracting
`env/reward.py` is what makes these tests possible, and test 2 below is the
one the whole design exists to support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from env.contracts import Action
from env.environment import SmartKeyNetEnv
from env.reward import (
    ENERGY_REFERENCE_MJ,
    LATENCY_REFERENCE_MS,
    RewardWeights,
    assert_weights_are_sane,
    compute_reward,
)
from metrics.reward_inputs import FORBIDDEN_FIELD_SUBSTRINGS, RewardInputs

REPO = Path(__file__).resolve().parent.parent


def load_test_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the real `configs/default.yaml` and shallow-merge overrides --
    mirrors tests/test_environment.py's helper so nothing is hardcoded."""
    with open(REPO / "configs" / "default.yaml", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
    return config


def default_weights() -> RewardWeights:
    return RewardWeights.from_config(load_test_config()["reward"])


def base_inputs(**overrides: Any) -> RewardInputs:
    """A neutral `RewardInputs` the monotonicity tests perturb one field of."""
    fields: dict[str, Any] = {
        "latency_ms": 100.0,
        "energy_mj": 1.0,
        "key_age_steps": 100,
        "key_lifetime_cap_steps": 500,
        "qkd_keys_consumed": 0,
        "deferred_critical_steps": 0,
        "did_rekey": False,
        "normalised_load": 0.5,
    }
    fields.update(overrides)
    return RewardInputs(**fields)


# ---------------------------------------------------------------------------
# 1. purity
# ---------------------------------------------------------------------------


def test_reward_is_pure_function_of_reward_inputs():
    """Same inputs -> same output, no hidden state (§S5 test 1)."""
    weights = default_weights()
    inputs = base_inputs(qkd_keys_consumed=1, did_rekey=True, deferred_critical_steps=2)

    first_total, first_terms = compute_reward(inputs, weights)
    for _ in range(5):
        total, terms = compute_reward(inputs, weights)
        assert total == first_total
        assert terms == first_terms


def test_terms_sum_to_total():
    """The breakdown must be complete -- no unexplained contribution may
    hide in the reward, or §7's debugging procedure is unsound."""
    weights = default_weights()
    total, terms = compute_reward(
        base_inputs(qkd_keys_consumed=3, did_rekey=True, deferred_critical_steps=4), weights
    )
    assert sum(terms.values()) == pytest.approx(total)
    assert set(terms) == {"latency", "energy", "freshness", "qkd", "starve", "rekey"}


# ---------------------------------------------------------------------------
# 2. Hard Rule 1
# ---------------------------------------------------------------------------


def test_no_security_field_reachable():
    """§S5 test 2 / §2.1: no security-flavoured field exists on the only
    type the reward can see.

    This is the project's central mechanism, not a hygiene check. The paper
    claims the agent is unsteerable *because* security never enters the
    objective; if a threat term could reach here, the masked agent would
    inherit exactly the soft-reward victim's vulnerability.
    """
    field_names = set(RewardInputs.__dataclass_fields__)
    for name in field_names:
        for forbidden in FORBIDDEN_FIELD_SUBSTRINGS:
            assert forbidden not in name.lower(), (
                f"RewardInputs.{name} contains '{forbidden}' -- Hard Rule 1 violated"
            )

    # compute_reward's signature must not have been widened either.
    import inspect

    assert list(inspect.signature(compute_reward).parameters) == ["inputs", "weights"]


# ---------------------------------------------------------------------------
# 3. term signs (parametrised monotonicity, one per term)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name,low,high,should_increase",
    [
        ("latency_ms", 50.0, 500.0, False),
        ("energy_mj", 0.5, 5.0, False),
        ("qkd_keys_consumed", 0, 4, False),
        ("deferred_critical_steps", 0, 4, False),
        ("key_age_steps", 0, 400, False),  # older key -> less freshness bonus
    ],
)
def test_each_term_sign(field_name, low, high, should_increase):
    """§S5 test 3: increasing a cost lowers reward; increasing freshness
    raises it. One parametrised case per term."""
    weights = default_weights()
    low_total, _ = compute_reward(base_inputs(**{field_name: low}), weights)
    high_total, _ = compute_reward(base_inputs(**{field_name: high}), weights)
    if should_increase:
        assert high_total > low_total
    else:
        assert high_total < low_total


def test_freshness_is_the_only_bonus():
    weights = default_weights()
    _, terms = compute_reward(
        base_inputs(
            key_age_steps=0, qkd_keys_consumed=1, did_rekey=True, deferred_critical_steps=1
        ),
        weights,
    )
    assert terms["freshness"] > 0.0
    for name in ("latency", "energy", "qkd", "starve", "rekey"):
        assert terms[name] <= 0.0


def test_freshness_is_bounded_to_unit_interval():
    """A key aged well past its cap must not earn negative freshness -- that
    would double-charge staleness, which the forced-rekey cost already does."""
    weights = default_weights()
    _, terms = compute_reward(base_inputs(key_age_steps=10_000), weights)
    assert terms["freshness"] == pytest.approx(0.0)
    _, fresh_terms = compute_reward(base_inputs(key_age_steps=0), weights)
    assert fresh_terms["freshness"] == pytest.approx(weights.w_freshness)


# ---------------------------------------------------------------------------
# 4. rekey cost scales with load
# ---------------------------------------------------------------------------


def test_rekey_cost_increases_with_load():
    """§S5 test 4. This is what makes "rekey early, at a quiet moment" a
    learnable skill rather than a fixed schedule."""
    weights = default_weights()
    quiet, quiet_terms = compute_reward(base_inputs(did_rekey=True, normalised_load=0.0), weights)
    busy, busy_terms = compute_reward(base_inputs(did_rekey=True, normalised_load=1.0), weights)
    assert busy < quiet
    assert busy_terms["rekey"] < quiet_terms["rekey"]


def test_no_rekey_cost_when_not_rekeying():
    weights = default_weights()
    _, terms = compute_reward(base_inputs(did_rekey=False, normalised_load=1.0), weights)
    assert terms["rekey"] == 0.0


# ---------------------------------------------------------------------------
# 5. starvation must dominate the key saving
# ---------------------------------------------------------------------------


def test_starve_term_dominates_qkd_saving():
    """§S5 test 5: deferring one critical request for 10 steps must be worse
    than spending one key.

    **If this inequality fails the headline result inverts** -- the agent
    learns to starve high-sensitivity requests rather than serve them, which
    is the opposite of the behaviour the paper claims.
    """
    weights = default_weights()
    spend_one_key, _ = compute_reward(base_inputs(qkd_keys_consumed=1), weights)
    defer_ten_steps, _ = compute_reward(base_inputs(deferred_critical_steps=10), weights)
    assert defer_ten_steps < spend_one_key


def test_config_satisfies_the_starve_inequality_at_load_time():
    """The guard runs at construction, not in review (§S5 test 5)."""
    assert_weights_are_sane(default_weights())

    starving_is_cheap = RewardWeights(
        w_latency=1.0,
        w_energy=0.1,
        w_freshness=0.1,
        w_qkd=10.0,
        r_starve=1.0,  # violates r_starve >= 5 * w_qkd
        c_rekey_base=1.0,
        c_rekey_load_beta=1.0,
    )
    with pytest.raises(ValueError, match="starve"):
        assert_weights_are_sane(starving_is_cheap)


# ---------------------------------------------------------------------------
# 6. normalisation and magnitude balance
# ---------------------------------------------------------------------------


def test_normalisation_happens_before_weighting():
    """§S5 point 2: weights multiply dimensionless quantities.

    A reader asking "is w_latency = 1.0 big?" can only answer that if 1.0
    multiplies a normalised number. Latency at exactly the 100 ms reference
    must therefore contribute exactly -w_latency.
    """
    weights = default_weights()
    _, terms = compute_reward(
        base_inputs(latency_ms=LATENCY_REFERENCE_MS, energy_mj=ENERGY_REFERENCE_MJ), weights
    )
    assert terms["latency"] == pytest.approx(-weights.w_latency)
    assert terms["energy"] == pytest.approx(-weights.w_energy)


def test_term_magnitude_balance():
    """SMARTKEYNET_BUILD_SPEC.md §S5 point 3, as a test: over 5 random
    episodes, no single reward term may contribute more than 60% or less
    than 2% of the mean absolute total.

    The spec's reasoning: "A reward where `w_qkd` is 40x everything else does
    not need tuning, it needs fixing." This check is what caught the
    environment sitting in permanent deficit -- before the 2026-08-19 SKR
    recalibration the `starve` term was 99.5% of total reward magnitude and
    all five others were under 2%, so the reward had collapsed into a single
    function of deferral-queue backlog and the other terms could not
    influence any decision. Nothing else in the suite would have caught it.

    Note this is a diagnostic on the *environment*, not the weights: the fix
    was supply calibration (§7.1 fix A), because §7.4 forbids touching
    weights once results exist.
    """
    config = load_test_config()
    absolute_totals: dict[str, float] = {}

    for seed in range(5):
        env = SmartKeyNetEnv({**config, "scenario": "S1", "max_steps": 400, "seed": seed})
        state, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(400):
            legal = np.flatnonzero(info["action_mask"])
            state, reward, terminated, truncated, info = env.step(Action(int(rng.choice(legal))))
            if truncated:
                break
        for term_name, term_value in env.reward_terms_total.items():
            absolute_totals[term_name] = absolute_totals.get(term_name, 0.0) + abs(term_value)

    grand_total = sum(absolute_totals.values())
    assert grand_total > 0.0
    shares = {name: value / grand_total for name, value in absolute_totals.items()}
    readable = "  ".join(
        f"{name}={share:.1%}" for name, share in sorted(shares.items(), key=lambda kv: -kv[1])
    )

    dominant = [name for name, share in shares.items() if share > 0.60]
    negligible = [name for name, share in shares.items() if share < 0.02]
    assert not dominant, f"term(s) over 60% of reward magnitude: {dominant} -- {readable}"
    assert not negligible, f"term(s) under 2% of reward magnitude: {negligible} -- {readable}"
