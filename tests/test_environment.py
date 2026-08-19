"""Behavioral tests for `env.environment` -- the spine (PLAN.md §10
kickoff step 5; split.md Gate W2: "env step() runs end-to-end with
random agent across a full S1 episode; regret events logged").

Covers the unit-level guarantees this session's wiring is responsible
for (reset validity, forced rekey, Hard Rule 9 pre-screening, illegal
action enforcement, the reward formula, foresight zeroing/population)
plus the W2 gate itself: a full S1 episode driven by a random *valid*
policy (sampled from the mask each step, never a real agent) with zero
floor violations, and -- under a small-pool config -- regret events
that do get logged and a deferred request that eventually gets served
without ever being downgraded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from env.reward import ENERGY_REFERENCE_MJ, LATENCY_REFERENCE_MS
from env.contracts import Action
from env.environment import (
    _ACTION_TO_KEY_TYPE,
    _ENERGY_MJ,
    _KEY_TYPE_TO_SERVE_ACTION,
    _LATENCY_MS,
    IllegalActionError,
    SmartKeyNetEnv,
)

_TIER_ACTIONS = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)

_SCARCE_OVERRIDES = {
    "pool": {
        "capacity_bits": 2_560.0,  # 10 ETSI keys
        "initial_fill_frac": 0.0,
        "bits_per_hybrid_draw": 256.0,  # the real ETSI draw, not an inflated one
    },
    "qkd": {"mean_skr_kbps": 0.02},  # ~0.078 keys/step: a link that cannot keep up
    # Synthetic threat signal: these tests are about POOL scarcity, and real
    # RT-IoT2022 traffic occasionally trips the posture ratchet, which raises
    # floors and changes how much hybrid demand exists -- a second moving part
    # this test is not trying to measure.
    "threat_source": "synthetic",
}
"""Config override that makes the pool genuinely bind, for the
deferral/regret tests below.

Two things changed here on 2026-08-15, for two different reasons.

**Why the draw size went back to 256 bits.** These tests previously
forced scarcity with `capacity_bits=500_000,
bits_per_hybrid_draw=300_000` -- a single draw consuming 60% of the
pool. That contrivance existed because the *real* physics had no
scarcity at all: refill ran at 781 keys/step against a ceiling of 1
key/step of demand, so no realistic draw could ever empty the pool
(see `configs/default.yaml`'s scarcity calibration block). It also
breaks outright against calibrated physics, for a mundane reason:
accumulating 300,000 bits at 220 bits/step takes ~1,364 steps, so
nothing drains inside a 300-step test.

**Why refill is throttled rather than the draw inflated.** The tests
below drive the env with a *random valid* policy, which serves hybrid
on roughly one decision in five -- about 0.2 keys/step against the
calibrated 0.859 keys/step of refill. That is a scarcity ratio near
0.23, and at that ratio a random policy correctly *never* exhausts the
pool: under the calibration, only always-hybrid (rho 1.14) runs a
deficit. So provoking a deferral needs a link that cannot keep up with
even modest demand, which is what `mean_skr_kbps: 0.02` expresses.
That is the same knob scenario S3 turns, just held constant instead of
ramped.
"""


def load_test_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the real `configs/default.yaml` (nothing hardcoded here)
    and shallow-merge any per-section overrides a test needs (e.g. a
    small pool to force scarcity)."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value

    return config


def take_random_valid_step(env: SmartKeyNetEnv, rng: np.random.Generator, info: dict[str, Any]):
    mask = info["action_mask"]
    legal = [a for a in Action if mask[int(a)]]
    action = rng.choice(legal)
    return action, env.step(action)


# ---------------------------------------------------------------------------
# reset() validity
# ---------------------------------------------------------------------------


def test_reset_produces_valid_initial_state_and_mask():
    env = SmartKeyNetEnv(load_test_config())
    state, info = env.reset(seed=0)
    mask = info["action_mask"]

    assert mask.shape == (5,)
    assert mask.dtype == bool
    assert mask.any()  # never a deadlock

    assert len(state["threat_forecast"]) == 5
    assert len(state["pool_level_hat"]) == 3
    assert len(state["skr_mean_hat"]) == 3
    assert len(state["hybrid_demand_hat"]) == 3
    # 4-wide since 2026-08-18 (spec §4.2): {none, classical, pqc, hybrid}.
    # A cold-start session now sets the explicit "none" slot rather than
    # flattening to all-zeros, which was indistinguishable from holding a
    # classical key.
    assert len(state["key_type_onehot"]) == 4
    assert sum(state["key_type_onehot"]) == 1.0
    assert state["key_type_onehot"][0] == 1.0  # cold start
    assert len(state["posture_probs"]) == 4
    assert len(state["request_class_onehot"]) == 4
    assert len(state["floor_onehot"]) == 4
    assert state["pqc_capable"] in (0.0, 1.0)
    assert 0.0 <= state["queue_len_norm"] <= 1.0
    assert 0.0 <= state["queue_head_wait_norm"] <= 1.0
    assert 0.0 <= state["steps_since_rekey_norm"] <= 1.0
    assert 0.0 <= state["pool_fill"] <= 1.0
    assert state["policy_floor"] in (int(Action.SERVE_CLASSICAL), int(Action.SERVE_PQC), int(Action.SERVE_HYBRID))
    assert isinstance(state["regret_event_recent"], bool)


def test_reset_is_reproducible_with_same_seed():
    env_a = SmartKeyNetEnv(load_test_config())
    env_b = SmartKeyNetEnv(load_test_config())
    state_a, info_a = env_a.reset(seed=99)
    state_b, info_b = env_b.reset(seed=99)

    assert state_a == state_b
    assert (info_a["action_mask"] == info_b["action_mask"]).all()


# ---------------------------------------------------------------------------
# Forced rekey (Addition C: "Staleness is NOT a reward term")
# ---------------------------------------------------------------------------


def test_forced_rekey_triggers_and_logs_when_key_age_hits_cap():
    config = load_test_config(overrides={"key_lifetime": {"max_key_age_steps": 3}})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    mask = info["action_mask"]

    # cold-start session -> key_age initialized at the cap -> REUSE illegal
    assert not mask[Action.REUSE]
    legal_non_reuse = [a for a in Action if mask[int(a)] and a is not Action.REUSE]
    action = legal_non_reuse[0]

    state, reward, terminated, truncated, info = env.step(action)

    assert "forced_rekey" in info
    event = info["forced_rekey"]
    assert event["key_age_at_rekey"] == pytest.approx(3.0)
    assert isinstance(event["request_id"], str) and event["request_id"]
    assert event["cost"] >= 0.0


def test_no_forced_rekey_when_action_is_discretionary():
    """A rekey taken while REUSE was still legal (age under the cap)
    must not be logged as forced."""
    config = load_test_config(overrides={"key_lifetime": {"max_key_age_steps": 1000}})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    mask = info["action_mask"]

    # first decision is still cold-start (REUSE illegal by design) --
    # take whatever non-REUSE action is legal to establish a session,
    # then drive it forward while it's still fresh.
    action = next(a for a in Action if mask[int(a)] and a is not Action.REUSE)
    state, reward, terminated, truncated, info = env.step(action)
    assert "forced_rekey" in info  # that first one *was* the cold-start forced case

    # now the session has a fresh key (age reset to 0 last step); if we
    # get offered the same session again soon and choose to rekey
    # early, it must not be "forced".
    for _ in range(20):
        mask = info["action_mask"]
        if mask[Action.REUSE]:
            action = next(a for a in Action if mask[int(a)] and a is not Action.REUSE)
            state, reward, terminated, truncated, info = env.step(action)
            assert "forced_rekey" not in info
            return
        action = next(a for a in Action if mask[int(a)])
        state, reward, terminated, truncated, info = env.step(action)


# ---------------------------------------------------------------------------
# Hard Rule 9 pre-screening
# ---------------------------------------------------------------------------


def test_hybrid_mandatory_request_with_uncoverable_pool_never_reaches_agent():
    """Structural invariant, not a single-shot check: over a whole run
    of a scarce-pool config, whatever request is currently offered to
    the agent (`env._current_request`) must never be hybrid-mandatory
    while the pool can't cover its draw -- if it were, it should have
    been diverted to the deferral queue instead (Hard Rule 9)."""
    config = load_test_config(overrides=_SCARCE_OVERRIDES)
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    rng = np.random.default_rng(0)

    saw_hybrid_mandatory_request = False
    saw_regret_event = False

    for _ in range(300):
        current = env._current_request
        if current["hybrid_mandatory"]:
            saw_hybrid_mandatory_request = True
            # if it's being shown to the agent right now, the pool
            # must actually be able to cover it right now
            assert env._pool_sim.can_draw(env._bits_per_hybrid_draw)

        _action, (state, reward, terminated, truncated, info) = take_random_valid_step(env, rng, info)
        if info["regret_events"]:
            saw_regret_event = True

    assert saw_hybrid_mandatory_request  # the scenario actually exercised this path
    assert saw_regret_event  # and scarcity actually forced at least one deferral


# ---------------------------------------------------------------------------
# Illegal actions
# ---------------------------------------------------------------------------


def test_illegal_action_raises():
    env = SmartKeyNetEnv(load_test_config())
    state, info = env.reset(seed=0)
    assert not info["action_mask"][Action.REUSE]  # cold start guarantees this

    with pytest.raises(IllegalActionError):
        env.step(Action.REUSE)


def test_step_before_reset_raises():
    env = SmartKeyNetEnv(load_test_config())
    with pytest.raises(RuntimeError):
        env.step(Action.REKEY_NOW)


# ---------------------------------------------------------------------------
# Reward formula
# ---------------------------------------------------------------------------


def test_reward_matches_documented_formula_for_known_action():
    config = load_test_config()  # ample default pool -> deferral queue stays empty
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    mask = info["action_mask"]
    assert mask[Action.REKEY_NOW]  # always legal at cold start with an ample pool
    assert len(env._deferral_queue) == 0

    floor = env._current_floor
    reward_cfg = config["reward"]

    resulting_key_type = _ACTION_TO_KEY_TYPE[floor]
    keys_consumed = 1.0 if resulting_key_type.name == "HYBRID" else 0.0
    cost_action = _KEY_TYPE_TO_SERVE_ACTION[resulting_key_type]
    latency = _LATENCY_MS[cost_action]
    energy = _ENERGY_MJ[cost_action]
    freshness = 1.0  # age resets to 0 this step
    load_before = env._current_load()
    rekey_cost = reward_cfg["c_rekey_base"] * (1.0 + reward_cfg["c_rekey_load_beta"] * load_before)

    # The reward now lives in env/reward.py behind `RewardInputs` (Hard Rule
    # 1), and normalises before weighting: latency per 100 ms, energy per
    # reference mJ. Reproduce that here rather than the pre-2026-08-19 inline
    # formula, which weighted raw quantities.
    expected_reward = (
        -reward_cfg["w_lat"] * (latency / LATENCY_REFERENCE_MS)
        - reward_cfg["w_en"] * (energy / ENERGY_REFERENCE_MJ)
        + reward_cfg["w_fr"] * freshness
        - reward_cfg["w_qkd"] * keys_consumed
        - rekey_cost
    )

    state, reward, terminated, truncated, info = env.step(Action.REKEY_NOW)

    assert len(info["deferred_critical_steps"]) == 0  # nothing was queued to age
    assert reward == pytest.approx(expected_reward)


def test_qkd_scarcity_price_is_charged_per_key_not_per_bit():
    """Regression test for the 2026-08-15 units bug (design decision
    11). `w_qkd` is documented as a price per 256-bit ETSI key, but was
    multiplied by the raw bit count -- making the term 256x its
    intended size.

    The bug survived undetected because
    `test_reward_matches_documented_formula_for_known_action` happens
    to land on a non-hybrid floor at seed 0, so `bits_consumed` was
    zero there and the term was never actually exercised. This test
    forces a hybrid serve and pins the magnitude.
    """
    config = load_test_config()
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)

    # walk to a decision where SERVE_HYBRID is legal
    for _ in range(200):
        if info["action_mask"][Action.SERVE_HYBRID]:
            break
        action = next(a for a in Action if info["action_mask"][int(a)])
        state, reward, terminated, truncated, info = env.step(action)
    else:
        pytest.fail("no SERVE_HYBRID-legal decision appeared within the search window")

    fill_before = env._pool_sim.fill
    state, reward, terminated, truncated, info = env.step(Action.SERVE_HYBRID)

    # exactly one key's worth of bits left the pool...
    drawn_bits = fill_before - env._pool_sim.fill
    assert drawn_bits > 0

    # ...and the scarcity price charged is w_qkd * 1 key, not w_qkd * 256 bits.
    # Bound the whole reward below by the per-key charge plus the other
    # (small, bounded) terms; a per-bit charge would be 256x larger and
    # blow straight through this.
    reward_cfg = config["reward"]
    worst_case_other_terms = (
        reward_cfg["w_lat"] * (max(_LATENCY_MS.values()) / LATENCY_REFERENCE_MS)
        + reward_cfg["w_en"] * (max(_ENERGY_MJ.values()) / ENERGY_REFERENCE_MJ)
        + reward_cfg["c_rekey_base"] * (1.0 + reward_cfg["c_rekey_load_beta"])
        + reward_cfg["r_starve"] * len(info["deferred_critical_steps"])
    )
    assert reward >= -(reward_cfg["w_qkd"] * 1.0 + worst_case_other_terms)


def test_starve_term_dominates_qkd_saving():
    """SMARTKEYNET_BUILD_SPEC.md §S5 test 5: deferring one critical
    request for ten steps must cost more than spending one key.
    Otherwise the agent learns to starve rather than spend, and the
    headline result inverts."""
    reward_cfg = load_test_config()["reward"]
    cost_of_deferring_ten_steps = reward_cfg["r_starve"] * 10
    cost_of_spending_one_key = reward_cfg["w_qkd"] * 1.0
    assert cost_of_deferring_ten_steps > cost_of_spending_one_key
    assert reward_cfg["r_starve"] >= 5.0 * reward_cfg["w_qkd"]


def test_env_rejects_a_config_where_starving_is_cheaper_than_spending():
    """The same inequality, enforced at construction so it cannot
    silently regress via a config edit."""
    config = load_test_config(overrides={"reward": {"w_qkd": 100.0, "r_starve": 10.0}})
    with pytest.raises(ValueError, match="r_starve"):
        SmartKeyNetEnv(config)


def test_reuse_reward_has_no_rekey_cost_or_pool_draw():
    config = load_test_config(overrides={"key_lifetime": {"max_key_age_steps": 1000}})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)

    # get past the forced cold-start rekey first
    action = next(a for a in Action if info["action_mask"][int(a)] and a is not Action.REUSE)
    state, reward, terminated, truncated, info = env.step(action)

    # find a moment where REUSE is legal (same or another session, still fresh)
    for _ in range(30):
        if info["action_mask"][Action.REUSE]:
            break
        action = next(a for a in Action if info["action_mask"][int(a)])
        state, reward, terminated, truncated, info = env.step(action)
    else:
        pytest.skip("no REUSE-eligible session appeared within the search window")

    fill_before = env._pool_sim.fill
    state, reward, terminated, truncated, info = env.step(Action.REUSE)

    # REUSE never draws. Asserting `fill == fill_before` would be wrong:
    # the pool also *refills* during the step, so the level legitimately
    # rises. (That assertion passed before the 2026-08-15 recalibration
    # only because the pool was permanently pinned at capacity, where
    # refill is clipped to zero -- it was testing pool saturation, not
    # REUSE.) A draw costs 256 bits against ~220 bits/step of refill, so
    # any draw would show up as a net *decrease*: a non-decreasing level
    # is a genuine discriminator here.
    assert env._pool_sim.fill >= fill_before
    assert "forced_rekey" not in info


# ---------------------------------------------------------------------------
# Foresight zeroing / population (use_foresight: off vs ewma)
# ---------------------------------------------------------------------------


def test_foresight_fields_zeroed_under_off():
    config = load_test_config(overrides={"use_foresight": "off"})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)

    assert state["threat_score"] == 0.0
    assert list(state["threat_forecast"]) == [0.0] * 5
    assert list(state["pool_level_hat"]) == [0.0] * 3
    assert list(state["skr_mean_hat"]) == [0.0] * 3
    assert list(state["hybrid_demand_hat"]) == [0.0] * 3


def test_foresight_fields_populated_under_ewma():
    config = load_test_config(overrides={"use_foresight": "ewma"})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)

    assert len(state["threat_forecast"]) == 5
    assert len(state["pool_level_hat"]) == 3
    # default pool starts at initial_fill_frac=0.5 and only grows -> EWMA must be nonzero
    assert state["pool_level_hat"][0] > 0.0
    assert state["skr_mean_hat"][0] > 0.0


def test_lstm_foresight_loads_a_trained_checkpoint():
    """`use_foresight: lstm` used to raise `NotImplementedError`; as of
    2026-08-15 it loads `forecaster.model.LSTMForecastProvider` from a
    checkpoint. Skipped rather than failed when no checkpoint has been
    trained yet, so a fresh clone still gets a green suite -- training
    one takes minutes and is not a unit-test concern."""
    from pathlib import Path

    checkpoint = Path("checkpoints/forecaster.pt")
    if not checkpoint.exists():
        pytest.skip("no trained forecaster checkpoint -- run `python -m forecaster.train`")

    config = load_test_config(overrides={"use_foresight": "lstm"})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)

    assert len(state["pool_level_hat"]) == 3
    assert len(state["threat_forecast"]) == 5
    assert info["action_mask"].any()


def test_lstm_foresight_without_a_checkpoint_fails_loudly():
    """A missing checkpoint must not silently fall back to EWMA -- that
    would make an E-A ablation row secretly a duplicate of another."""
    config = load_test_config(
        overrides={"use_foresight": "lstm", "forecaster_checkpoint": "checkpoints/_absent.pt"}
    )
    env = SmartKeyNetEnv(config)
    with pytest.raises(FileNotFoundError, match="forecaster.train"):
        env.reset(seed=0)


# ---------------------------------------------------------------------------
# Gate test (split.md Gate W2)
# ---------------------------------------------------------------------------


def test_gate_full_s1_episode_random_valid_policy_zero_floor_violations():
    config = load_test_config()
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=123)
    rng = np.random.default_rng(123)

    floor_violations = 0
    for _ in range(250):
        floor = env._current_floor
        action, (state, reward, terminated, truncated, info) = take_random_valid_step(env, rng, info)

        if action in _TIER_ACTIONS and int(action) < int(floor):
            floor_violations += 1

        assert env._pool_sim.fill >= 0.0
        assert not terminated

    assert floor_violations == 0


def test_gate_regret_events_logged_and_deferred_request_eventually_served():
    config = load_test_config(overrides=_SCARCE_OVERRIDES)
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=7)
    rng = np.random.default_rng(7)

    total_regret_events = 0
    floor_violations = 0
    queue_len_history = [len(env._deferral_queue)]

    # 900 rather than 250 decisions: at the scarce override's refill rate
    # (~0.078 keys/step) draining a queue that has built to ~30 deep
    # takes on the order of 400 steps, so a 250-step run could observe
    # the queue filling but never emptying and would fail on a scenario
    # that is in fact behaving correctly.
    for _ in range(900):
        floor = env._current_floor
        action, (state, reward, terminated, truncated, info) = take_random_valid_step(env, rng, info)

        if action in _TIER_ACTIONS and int(action) < int(floor):
            floor_violations += 1

        total_regret_events += len(info["regret_events"])
        queue_len_history.append(len(env._deferral_queue))
        assert env._pool_sim.fill >= 0.0

    assert floor_violations == 0  # never downgraded, even under forced scarcity
    assert total_regret_events > 0  # exhaustion was actually forced at least once
    peak_queue_len = max(queue_len_history)
    assert peak_queue_len > 0  # something actually got queued
    assert queue_len_history[-1] < peak_queue_len  # and later drained -- i.e., served


# ---------------------------------------------------------------------------
# load_spike diagnostic wiring (2026-08-10 -- see request_generator.py's
# docstring: a diagnostic stand-in for temporary load surges, NOT real S4)
# ---------------------------------------------------------------------------

_LOAD_SPIKE_CFG = {
    "enabled": True,
    "period_steps": 500,
    "spike_duration_steps": 20,
    "spike_rate_multiplier": 3.0,
    "low_rate_multiplier": 0.3,
}


def test_load_spike_disabled_by_default_matches_flat_stream():
    """`configs/default.yaml`'s real default (`load_spike.enabled: false`)
    must reproduce the exact same request stream as before this
    feature existed -- backward compatible, opt-in only."""
    config = load_test_config()
    assert config["load_spike"]["enabled"] is False

    env = SmartKeyNetEnv(config)
    assert env._load_spike_cfg is None


def test_load_spike_enabled_raises_observed_load_during_the_window():
    """With the spike active, average `state["load"]` inside the
    spike window should be measurably higher than outside it -- this
    is the actual mechanism `experiments/train.py`'s diagnostic run
    depends on (env/environment.py's `load` feeds `c_rekey(load)`
    directly)."""
    config = load_test_config(overrides={"load_spike": _LOAD_SPIKE_CFG})
    env = SmartKeyNetEnv(config)
    assert env._load_spike_cfg == {
        "period_steps": 500,
        "spike_duration_steps": 20,
        "spike_rate_multiplier": 3.0,
        "low_rate_multiplier": 0.3,
    }

    state, info = env.reset(seed=5)
    rng = np.random.default_rng(5)

    # 3 full periods -- long enough for the low phase to actually drain
    # the backlog built up during the spike (see configs/default.yaml's
    # comment on why low_rate_multiplier must be < 1, empirically tuned
    # this session; a too-short sample window would catch the queue
    # mid-drain and understate the contrast).
    in_window_loads: list[float] = []
    out_window_loads: list[float] = []
    for _ in range(1500):
        step_now = env._step_count
        in_window = (step_now % _LOAD_SPIKE_CFG["period_steps"]) < _LOAD_SPIKE_CFG["spike_duration_steps"]
        (in_window_loads if in_window else out_window_loads).append(state["load"])

        _action, (state, reward, terminated, truncated, info) = take_random_valid_step(env, rng, info)

    assert in_window_loads and out_window_loads
    assert sum(in_window_loads) / len(in_window_loads) > sum(out_window_loads) / len(out_window_loads)
    # the mechanism this diagnostic exists to test: `load` should
    # genuinely recede during cooldown, not just be permanently pinned
    # at its cap once the first spike hits.
    assert min(out_window_loads) < 0.5


def test_load_spike_absent_key_behaves_same_as_disabled():
    config = load_test_config()
    del config["load_spike"]
    env = SmartKeyNetEnv(config)
    assert env._load_spike_cfg is None
