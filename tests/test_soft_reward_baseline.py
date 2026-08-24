"""Unit tests for `agents.soft_reward_baseline` -- the Noetzold-style
soft-reward baseline agent's reward function (PLAN.md §5 S5's future
steering-attack target; PLAN2.md §7.5). See that module's docstring for
the Hard Rule 1 tension/resolution this file's boundary-check test
verifies at the code level, not just documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.soft_reward_baseline import (
    SoftRewardConfig,
    compute_soft_reward,
    delivered_tier,
    load_soft_reward_config,
    resolved_cost_action,
    security_score,
)
from env.contracts import Action


def _make_state(*, key_type_onehot: list[float], policy_floor: int) -> dict[str, Any]:
    """Minimal `StateDict` stand-in -- mirrors tests/test_train.py's own
    `_make_state` helper, trimmed to only what `compute_soft_reward`
    reads (`key_type_onehot`, `policy_floor`)."""
    return {
        "threat_score": 0.0,
        "threat_forecast": [0.0] * 5,
        "qber": 0.01,
        "skr": 500.0,
        "pool_fill": 0.5,
        "arrival_rate": 1.0,
        "load": 0.2,
        "avg_latency": 1.0,
        "key_age": 10.0,
        "key_type_onehot": key_type_onehot,
        "sensitivity_class": 1,
        "policy_floor": policy_floor,
        "pool_level_hat": [0.0] * 3,
        "skr_mean_hat": [0.0] * 3,
        "hybrid_demand_hat": [0.0] * 3,
        "regret_event_recent": False,
    }


# ---------------------------------------------------------------------------
# security_score mapping (Hard Rule 4 provenance: HANDOFF_C's own spec,
# not invented fresh for this project -- see module docstring)
# ---------------------------------------------------------------------------


def test_security_score_matches_the_reproduced_specs_own_values():
    assert security_score(Action.SERVE_CLASSICAL) == 0.2
    assert security_score(Action.SERVE_PQC) == 0.6
    assert security_score(Action.SERVE_HYBRID) == 1.0


# ---------------------------------------------------------------------------
# tier resolution -- REUSE/REKEY_NOW get security_score via whatever tier
# they actually deliver, never an invented 4th/5th constant
# ---------------------------------------------------------------------------


def test_delivered_tier_for_tier_actions_is_the_action_itself():
    floor = Action.SERVE_PQC
    for tier_action in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID):
        assert delivered_tier(tier_action, [0.0, 0.0, 0.0], floor) is tier_action


def test_delivered_tier_for_reuse_is_the_existing_session_tier_even_below_floor():
    """The property that distinguishes this agent from the masked one:
    env/masking.py's compute_mask forbids REUSE-ing a stale below-floor
    key for the masked agent (the 2026-08-19 fix). This agent's own
    delivered_tier has no such restriction -- it just reports whatever
    tier is actually still active, honestly, even below the floor passed
    in (which is exactly what makes the resulting security_score
    genuinely lower -- the property this agent exists to demonstrate)."""
    classical_onehot = [1.0, 0.0, 0.0]  # KeyType.CLASSICAL
    high_floor = Action.SERVE_HYBRID
    assert delivered_tier(Action.REUSE, classical_onehot, high_floor) is Action.SERVE_CLASSICAL


def test_delivered_tier_for_rekey_now_is_max_of_existing_and_floor():
    pqc_onehot = [0.0, 1.0, 0.0]  # KeyType.PQC
    assert delivered_tier(Action.REKEY_NOW, pqc_onehot, Action.SERVE_CLASSICAL) is Action.SERVE_PQC
    assert delivered_tier(Action.REKEY_NOW, pqc_onehot, Action.SERVE_HYBRID) is Action.SERVE_HYBRID


def test_delivered_tier_for_rekey_now_cold_start_adopts_floor():
    no_session = [0.0, 0.0, 0.0]
    assert delivered_tier(Action.REKEY_NOW, no_session, Action.SERVE_PQC) is Action.SERVE_PQC


def test_resolved_cost_action_reuse_costs_against_itself_not_a_tier():
    """Distinguishes cost (resolved_cost_action) from security_score
    lookup (delivered_tier): REUSE is cheap (a cache hit) regardless of
    which tier it happens to deliver."""
    classical_onehot = [1.0, 0.0, 0.0]
    assert resolved_cost_action(Action.REUSE, classical_onehot, Action.SERVE_HYBRID) is Action.REUSE


def test_resolved_cost_action_rekey_now_costs_against_its_resolved_tier():
    no_session = [0.0, 0.0, 0.0]
    assert resolved_cost_action(Action.REKEY_NOW, no_session, Action.SERVE_HYBRID) is Action.SERVE_HYBRID


# ---------------------------------------------------------------------------
# compute_soft_reward -- r = -w_lat*latency - w_en*energy + w_sec*security_score(tier)
# ---------------------------------------------------------------------------


def test_compute_soft_reward_matches_the_formula_directly():
    cfg = SoftRewardConfig(w_lat=1.0, w_en=0.1, w_sec=1.0)
    state = _make_state(key_type_onehot=[0.0, 0.0, 0.0], policy_floor=int(Action.SERVE_HYBRID))

    reward = compute_soft_reward(state, Action.SERVE_CLASSICAL, cfg)

    # SERVE_CLASSICAL: latency=1.0, energy=1.0 (env/environment.py's real
    # _LATENCY_UNITS/_ENERGY_UNITS), security_score=0.2 -- served BELOW
    # the state's own policy_floor (SERVE_HYBRID), which this agent's
    # reward doesn't forbid or additionally penalize beyond the security
    # term itself (no masking, no floor-violation cost -- Hard Rule 2 is
    # deliberately absent here, see agents/soft_reward_baseline.py's
    # module docstring).
    expected = -1.0 * 1.0 - 0.1 * 1.0 + 1.0 * 0.2
    assert reward == expected


def test_compute_soft_reward_security_term_genuinely_rewards_higher_tiers():
    """Not a claim that higher tiers always win overall (latency/energy
    cost more too, by design) -- isolates the security term's own
    sign/direction: at zero latency/energy weight, a higher-security
    tier strictly increases reward via w_sec alone."""
    cfg = SoftRewardConfig(w_lat=0.0, w_en=0.0, w_sec=1.0)
    state = _make_state(key_type_onehot=[0.0, 0.0, 0.0], policy_floor=int(Action.SERVE_CLASSICAL))

    classical_reward = compute_soft_reward(state, Action.SERVE_CLASSICAL, cfg)
    pqc_reward = compute_soft_reward(state, Action.SERVE_PQC, cfg)
    hybrid_reward = compute_soft_reward(state, Action.SERVE_HYBRID, cfg)

    assert classical_reward < pqc_reward < hybrid_reward
    assert hybrid_reward - classical_reward == 1.0 - 0.2


def test_compute_soft_reward_is_independent_of_the_real_floor_value():
    """The defining property under test: this agent's reward is the SAME
    regardless of what the real (bypassed) floor happens to be for a
    given action/key-state -- unlike the masked agent, nothing here reads
    the floor to decide legality, only REKEY_NOW/REUSE's tier resolution
    reads it (see delivered_tier), and neither of those actions is
    involved in this particular check."""
    cfg = SoftRewardConfig(w_lat=1.0, w_en=0.1, w_sec=1.0)
    low_floor_state = _make_state(key_type_onehot=[0.0, 0.0, 0.0], policy_floor=int(Action.SERVE_CLASSICAL))
    high_floor_state = _make_state(key_type_onehot=[0.0, 0.0, 0.0], policy_floor=int(Action.SERVE_HYBRID))

    assert compute_soft_reward(low_floor_state, Action.SERVE_CLASSICAL, cfg) == compute_soft_reward(
        high_floor_state, Action.SERVE_CLASSICAL, cfg
    )


def test_load_soft_reward_config_reads_the_real_committed_file():
    cfg = load_soft_reward_config()
    assert isinstance(cfg, SoftRewardConfig)
    assert cfg.w_lat > 0
    assert cfg.w_sec > 0


# ---------------------------------------------------------------------------
# Hard Rule 1 boundary check (code-level, not assumed) -- this agent's
# security term must be fully contained to this module (and its own
# config/tests), and nowhere in the masked agent's own code paths.
# ---------------------------------------------------------------------------


def test_hard_rule_1_dqn_agent_contains_no_security_term():
    dqn_source = Path("agents/dqn.py").read_text(encoding="utf-8")
    for forbidden in ("security_score", "w_sec", "TIER_SECURITY_SCORE", "soft_reward"):
        assert forbidden not in dqn_source, f"{forbidden!r} leaked into agents/dqn.py"


def test_hard_rule_1_environment_apply_action_contains_no_security_term():
    """Scoped to `_apply_action`'s body specifically (the actual reward
    computation), not the whole file -- env/environment.py's own module
    docstring legitimately *discusses* security_masking/the security
    floor in prose (design decision 16); the reward computation itself
    must still be completely clean. Mirrors this repo's existing
    Hard-Rule-3-boundary-check convention (see
    tests/test_environment.py::test_hard_rule_3_no_scenario_aware_branching_downstream_of_request_generation)."""
    env_source = Path("env/environment.py").read_text(encoding="utf-8")
    apply_action_start = env_source.index("def _apply_action")
    apply_action_body = env_source[apply_action_start : apply_action_start + 3000]

    for forbidden in ("security_score", "w_sec", "TIER_SECURITY_SCORE"):
        assert forbidden not in apply_action_body, f"{forbidden!r} leaked into _apply_action"
