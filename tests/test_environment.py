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

from agents.baselines import AlwaysHybridPolicy, RandomPolicy, StaticThresholdPolicy
from env.contracts import Action, KeyType, SensitivityClass, ThreatPosture
from env.environment import (
    _ACTION_TO_KEY_TYPE,
    _ENERGY_UNITS,
    _KEY_TYPE_TO_SERVE_ACTION,
    _LATENCY_UNITS,
    IllegalActionError,
    SmartKeyNetEnv,
    _SessionKeyState,
)
from env.masking import PolicyTable
from env.request_generator import RequestGenerator, build_tenant_graph
from experiments.harness import run_scenario
from experiments.train import load_full_config

_TIER_ACTIONS = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)


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
    assert len(state["key_type_onehot"]) == 3
    assert sum(state["key_type_onehot"]) in (0.0, 1.0)  # cold start -> all zero
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
    config = load_test_config(
        overrides={"pool": {"capacity_bits": 500_000.0, "initial_fill_frac": 0.0, "bits_per_hybrid_draw": 300_000.0}}
    )
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
# REUSE/REKEY_NOW floor-enforcement gap (2026-08-19 Hard Rule 2 fix)
# ---------------------------------------------------------------------------


def test_rekey_now_escalates_a_stale_existing_tier_up_to_the_current_floor():
    """The crux of the fix: REKEY_NOW must never resolve below the
    current floor, even when the session's existing tier predates a
    since-ratcheted-up floor (`_resulting_key_type` unit-tested
    directly, bypassing the need to drive the env through a real S2
    episode to reach this exact state)."""
    env = SmartKeyNetEnv(load_test_config())
    session = _SessionKeyState(key_age=0.0, key_type=KeyType.PQC)  # stale: below a HYBRID floor

    resolved = env._resulting_key_type(Action.REKEY_NOW, session, floor=Action.SERVE_HYBRID)

    assert resolved == KeyType.HYBRID


def test_rekey_now_never_downgrades_an_existing_higher_tier():
    """Regression check: the fix must not trade Hard Rule 2 for design
    decision 4's "never downgrade" behavior -- a session already ABOVE
    the current floor keeps its existing (higher) tier unchanged."""
    env = SmartKeyNetEnv(load_test_config())
    session = _SessionKeyState(key_age=0.0, key_type=KeyType.HYBRID)  # already above floor

    resolved = env._resulting_key_type(Action.REKEY_NOW, session, floor=Action.SERVE_CLASSICAL)

    assert resolved == KeyType.HYBRID


def test_rekey_now_at_exactly_the_current_floor_is_unchanged():
    env = SmartKeyNetEnv(load_test_config())
    session = _SessionKeyState(key_age=0.0, key_type=KeyType.PQC)

    resolved = env._resulting_key_type(Action.REKEY_NOW, session, floor=Action.SERVE_PQC)

    assert resolved == KeyType.PQC


def test_rekey_now_cold_start_still_adopts_the_floor_tier():
    """Regression check: cold-start behavior (design decision 4) is
    unchanged by this fix."""
    env = SmartKeyNetEnv(load_test_config())
    session = _SessionKeyState(key_age=0.0, key_type=None)

    resolved = env._resulting_key_type(Action.REKEY_NOW, session, floor=Action.SERVE_HYBRID)

    assert resolved == KeyType.HYBRID


def test_s2_reuse_and_rekey_now_never_deliver_below_current_floor():
    """The empirical reproduction, kept as a permanent regression test
    (per instruction). Before this session's fix, a real S2 episode
    (posture/floor genuinely ratchets mid-episode -- see
    env/environment.py design decision 10) under `RandomPolicy` (which
    exercises REUSE/REKEY_NOW whenever legal, unlike the tier-favoring
    baselines) measured 64 of 279 REUSE/REKEY_NOW decisions (22.9%)
    delivering key material below the request's current floor -- 32
    via each action (see SESSION_LOG.md's 2026-08-19 entry for the
    full before/after comparison). This must now be exactly zero,
    verified by directly comparing the session's actual post-decision
    key tier against `env._current_floor` at the moment of that exact
    decision -- never by trusting the (also fixed, separately tested)
    floor_violations counter alone."""
    config = load_test_config(
        overrides={
            "scenario": "S2",
            "threat_schedule": {"elevate_at_step": 50, "elevated_signal": 6.0},
            "max_steps": 500,
        }
    )
    env = SmartKeyNetEnv(config)
    policy = RandomPolicy(seed=0)
    state, info = env.reset(seed=0)

    total_reuse_or_rekey = 0
    violations = 0
    truncated = False
    while not truncated:
        mask = info["action_mask"]
        action = policy.act(state, mask)
        floor = env._current_floor
        tenant_service = (env._current_request["tenant"], env._current_request["service"])
        session = env._sessions[tenant_service]

        state, reward, terminated, truncated, info = env.step(action)

        if action in (Action.REUSE, Action.REKEY_NOW):
            total_reuse_or_rekey += 1
            delivered = session.key_type  # post-step: what the session actually holds now
            if delivered is not None:
                delivered_action = _KEY_TYPE_TO_SERVE_ACTION[delivered]
                if int(delivered_action) < int(floor):
                    violations += 1

    assert total_reuse_or_rekey > 50  # the scenario actually exercised REUSE/REKEY_NOW meaningfully
    assert violations == 0


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
    bits_consumed = env._bits_per_hybrid_draw if resulting_key_type.name == "HYBRID" else 0.0
    cost_action = _KEY_TYPE_TO_SERVE_ACTION[resulting_key_type]
    latency = _LATENCY_UNITS[cost_action]
    energy = _ENERGY_UNITS[cost_action]
    freshness = 1.0  # age resets to 0 this step
    load_before = env._current_load()
    rekey_cost = reward_cfg["c_rekey_base"] * (1.0 + reward_cfg["c_rekey_load_beta"] * load_before)

    expected_reward = (
        -reward_cfg["w_lat"] * latency
        - reward_cfg["w_en"] * energy
        + reward_cfg["w_fr"] * freshness
        - reward_cfg["w_qkd"] * bits_consumed
        - rekey_cost
    )

    state, reward, terminated, truncated, info = env.step(Action.REKEY_NOW)

    assert len(info["deferred_critical_steps"]) == 0  # nothing was queued to age
    assert reward == pytest.approx(expected_reward)


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
    assert env._pool_sim.fill == pytest.approx(fill_before)  # REUSE never draws
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


def test_lstm_foresight_raises_not_implemented():
    config = load_test_config(overrides={"use_foresight": "lstm"})
    env = SmartKeyNetEnv(config)
    with pytest.raises(NotImplementedError):
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


# ---------------------------------------------------------------------------
# Hard Rule 3 swap test (2026-08-23) -- "deleting the NetworkX graph and
# replacing it with a plain arrival process must not change one line of
# agent code." This duplicates
# test_gate_full_s1_episode_random_valid_policy_zero_floor_violations
# verbatim except for the one line constructing SmartKeyNetEnv, which
# passes a request_stream_factory built from the real
# build_tenant_graph()/RequestGenerator instead of relying on the
# default (random_request_generator). Everything else -- the random
# *valid* policy, the floor-violation check, the assertions -- is
# identical, unmodified environment/agent-facing code.
# ---------------------------------------------------------------------------


def test_hard_rule_3_graph_driven_generator_is_a_drop_in_replacement():
    config = load_test_config()
    graph = build_tenant_graph(n_nodes=10, seed=123)
    env = SmartKeyNetEnv(
        config,
        request_stream_factory=lambda seed: iter(RequestGenerator(graph, seed=seed)),
    )
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
    config = load_test_config(
        overrides={"pool": {"capacity_bits": 500_000.0, "initial_fill_frac": 0.0, "bits_per_hybrid_draw": 300_000.0}}
    )
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=7)
    rng = np.random.default_rng(7)

    total_regret_events = 0
    floor_violations = 0
    queue_len_history = [len(env._deferral_queue)]

    for _ in range(250):
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


# ---------------------------------------------------------------------------
# S2/S3 real scenario dispatch (design decision 10, 2026-08-19)
# ---------------------------------------------------------------------------

_S2_THREAT_SCHEDULE = {"elevate_at_step": 50, "elevated_signal": 6.0}
_S3_QKD_DEGRADATION = {"spike_start": 50, "spike_duration": 150, "spike_magnitude": 0.6}
_SCARCITY_POOL = {"capacity_bits": 500_000.0, "initial_fill_frac": 0.0, "bits_per_hybrid_draw": 300_000.0}


def test_s1_scenario_dispatch_is_a_pure_no_op():
    """Regression check: S1 (the only scenario every other test in this
    file exercises) must be completely unaffected by S2/S3 dispatch --
    both new config attributes stay None, and a full random-valid-policy
    episode reproduces byte-for-byte identically across two fresh envs
    given the same seed, exactly as it did before this session."""
    config = load_test_config()
    env = SmartKeyNetEnv(config)
    assert env._scenario == "S1"
    assert env._threat_schedule_cfg is None
    assert env._qkd_degradation_cfg is None

    def run_fills(seed: int) -> list[float]:
        env = SmartKeyNetEnv(load_test_config())
        state, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        fills = []
        for _ in range(100):
            _action, (state, reward, terminated, truncated, info) = take_random_valid_step(env, rng, info)
            fills.append(env._pool_sim.fill)
        return fills

    assert run_fills(42) == run_fills(42)


def test_s2_requires_threat_schedule_config():
    config = load_test_config(overrides={"scenario": "S2"})
    assert "threat_schedule" not in config  # default.yaml never carries this key
    with pytest.raises(KeyError):
        SmartKeyNetEnv(config)


def test_s3_requires_qkd_degradation_config():
    config = load_test_config(overrides={"scenario": "S3"})
    assert "qkd_degradation" not in config
    with pytest.raises(KeyError):
        SmartKeyNetEnv(config)


def test_s2_posture_genuinely_elevates_not_flat_calm():
    config = load_test_config(overrides={"scenario": "S2", "threat_schedule": _S2_THREAT_SCHEDULE})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    mask = info["action_mask"]

    postures: list[ThreatPosture] = []
    truncated = False
    while not truncated and len(postures) < 200:
        action = next(a for a in Action if bool(mask[int(a)]))
        tf = env._forecaster.get_threat_forecast()
        postures.append(ThreatPosture(int(np.argmax(tf.posture_probs))))
        state, reward, terminated, truncated, info = env.step(action)
        mask = info["action_mask"]

    assert any(p != ThreatPosture.CALM for p in postures)  # not flat CALM throughout, unlike S1's early steps
    assert ThreatPosture.HIGH in postures  # the scripted elevation actually saturates to HIGH, not just ELEVATED


def test_s2_elevated_floor_matches_real_policy_table_across_representative_classes():
    """Cross-check step (not a hardcoded expected value): wherever S2's
    scripted elevation raises posture, the resulting floor for every
    sensitivity class actually seen in the request stream must equal
    env/masking.py's real, unmodified `PolicyTable.floor()` lookup for
    that exact (class, posture) pair."""
    config = load_test_config(
        overrides={"scenario": "S2", "threat_schedule": {"elevate_at_step": 10, "elevated_signal": 6.0}}
    )
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    mask = info["action_mask"]

    # run well past elevate_at_step so the ratchet has saturated to HIGH
    # for every class -- `observed[sens]` keeps overwriting with the
    # latest reading, so by the end each class's entry reflects a
    # post-saturation decision, not an early pre-elevation one.
    observed: dict[SensitivityClass, tuple[ThreatPosture, Action]] = {}
    truncated = False
    steps = 0
    while not truncated and steps < 150:
        posture = ThreatPosture(int(np.argmax(env._forecaster.get_threat_forecast().posture_probs)))
        sens = SensitivityClass(state["sensitivity_class"])
        observed[sens] = (posture, Action(state["policy_floor"]))

        action = next(a for a in Action if bool(mask[int(a)]))
        state, reward, terminated, truncated, info = env.step(action)
        mask = info["action_mask"]
        steps += 1

    assert len(observed) >= 3  # a real spread of sensitivity classes, not just one

    fresh_table = PolicyTable()  # unratcheted -- the scripted schedule only ever rises, so this equals the env's own ratcheted state
    for sens, (posture, floor) in observed.items():
        assert floor == fresh_table.floor(sens, posture), (
            f"S2 floor drifted from env/masking.py's real table at class={sens.name}, posture={posture.name}"
        )
    assert any(posture is ThreatPosture.HIGH for posture, _ in observed.values())


def test_s3_pool_trajectory_is_genuinely_worse_than_s1():
    """Same seed, same (scarcity-forcing) pool config -- only `scenario`
    differs. S3's degraded SKR/QBER trace must produce a genuinely
    worse trajectory than S1's undegraded one, read off pool_sim's own
    real state (regret-event count, minimum pool fill), never a
    fabricated expectation."""

    def run(scenario: str, extra: dict[str, Any] | None = None) -> tuple[int, float]:
        overrides: dict[str, Any] = {"scenario": scenario, "pool": _SCARCITY_POOL, "max_steps": 250}
        if extra:
            overrides.update(extra)
        env = SmartKeyNetEnv(load_test_config(overrides=overrides))
        state, info = env.reset(seed=0)
        policy = AlwaysHybridPolicy()
        regret_events = 0
        min_fill_frac = 1.0
        truncated = False
        while not truncated:
            mask = info["action_mask"]
            action = policy.act(state, mask)
            state, reward, terminated, truncated, info = env.step(action)
            regret_events += len(info["regret_events"])
            min_fill_frac = min(min_fill_frac, env._pool_sim.fill / env._pool_sim.capacity)
        return regret_events, min_fill_frac

    regret_s1, min_fill_s1 = run("S1")
    regret_s3, min_fill_s3 = run("S3", {"qkd_degradation": _S3_QKD_DEGRADATION})

    assert regret_s3 > regret_s1
    assert min_fill_s3 < min_fill_s1


def test_real_s3_config_file_genuinely_diverges_from_real_s1_config_no_test_override():
    """2026-08-24 Gate W3 recalibration: the *real*, committed
    configs/scenarios/s3_degradation.yaml (as `experiments/train.py`'s
    train() will actually load it, not a test-only scarcity override,
    and configs/default.yaml for S1, completely unmodified) must
    themselves produce a genuinely different pool trajectory under the
    same seed -- this is the property Gate W3's S3 training run
    actually depends on. Uses the same seeds this session's
    SESSION_LOG.md entry reports."""
    s1_config = load_full_config("configs/default.yaml")
    s3_config = load_full_config("configs/scenarios/s3_degradation.yaml")

    def run(config: dict[str, Any], seed: int) -> tuple[int, float, float]:
        cfg = {**config, "seed": seed, "max_steps": 250}
        env = SmartKeyNetEnv(cfg)
        state, info = env.reset(seed=seed)
        policy = AlwaysHybridPolicy()
        regret_events = len(info["regret_events"])
        min_fill = env._pool_sim.fill
        terminated = truncated = False
        while not (terminated or truncated):
            mask = info["action_mask"]
            action = policy.act(state, mask)
            state, reward, terminated, truncated, info = env.step(action)
            regret_events += len(info["regret_events"])
            min_fill = min(min_fill, env._pool_sim.fill)
        return regret_events, min_fill, env._pool_sim.capacity

    for seed in (0, 1, 4, 7):
        regret_s1, min_fill_s1, cap_s1 = run(s1_config, seed)
        regret_s3, min_fill_s3, cap_s3 = run(s3_config, seed)

        assert regret_s1 == 0  # S1's real pool (1,000,000 bits) never gets stressed -- pre-existing property
        assert regret_s3 > 0  # S3's recalibrated pool (20,000 bits) genuinely exhausts under sustained demand
        # S3's own capacity is smaller by design (point 3a) -- compare fill *fractions*, not raw bit counts
        assert (min_fill_s3 / cap_s3) < (min_fill_s1 / cap_s1)
        assert (min_fill_s3 / cap_s3) < 0.01  # near-total exhaustion, not merely "somewhat lower"


def test_real_s3_config_exhaustion_defers_never_downgrades_hard_rule_9():
    """Hard Rule 9, verified explicitly under the new, more severe
    real S3 config (not assumed to transfer from the pre-recalibration
    S3, and not just from the small-pool test override elsewhere in
    this file): a pool that genuinely exhausts under S3 must still
    produce zero floor violations -- every request is deferred, never
    served below its floor."""
    s3_config = load_full_config("configs/scenarios/s3_degradation.yaml")
    for seed in (0, 1, 4, 7):
        result = run_scenario(AlwaysHybridPolicy(), "S3", s3_config, seed=seed)
        assert result.pool_exhaustion_events > 0  # confirms this run actually exercised scarcity
        assert result.floor_violations == 0  # Hard Rule 9: deferred, never downgraded


# ---------------------------------------------------------------------------
# S4 (DDoS / noisy-neighbor) scenario dispatch (2026-08-24, design decision 13)
# ---------------------------------------------------------------------------

_DDOS_ON = {"graph_seed": 0, "tenant_index": 4, "extra_rate": 5.0}
_DDOS_OFF = {"graph_seed": 0, "tenant_index": 4, "extra_rate": 0.0}
_S4_NOISY_TENANT = "tenant_4"  # real S0 (low-sensitivity) tenant under graph_seed 0, n_nodes 10
_S4_CRITICAL_TENANT = "tenant_9"  # real S3 (highest-sensitivity) tenant under the same graph


def _run_s4_and_count_by_tenant(ddos_cfg: dict[str, Any], seed: int, n_steps: int, pool_override=None):
    """Drives a real S4 episode under AlwaysHybridPolicy (see rationale
    in the docstring below) and returns, per tenant: decision count,
    regret-event count, and summed reward -- plus the env's own final
    `_step_count` (real elapsed simulator time), for a fair flood-on vs
    flood-off comparison that isn't confounded by "less time elapsed."
    `AlwaysHybridPolicy` is used (not a grid-searched threshold or
    random policy) because it deterministically maximizes and
    stabilizes hybrid-draw demand -- the cleanest, least-confounded
    lens for measuring whether one tenant's flood degrades another
    tenant's access to a shared resource, isolating the flood's effect
    from any baseline-policy-choice variance."""
    overrides: dict[str, Any] = {"scenario": "S4", "ddos": ddos_cfg}
    if pool_override is not None:
        overrides["pool"] = pool_override
    env = SmartKeyNetEnv(load_test_config(overrides=overrides))
    state, info = env.reset(seed=seed)
    policy = AlwaysHybridPolicy()

    decisions: dict[str, int] = {}
    regret: dict[str, int] = {}
    reward_sum: dict[str, float] = {}
    for _ in range(n_steps):
        mask = info["action_mask"]
        tenant = env._current_request["tenant"]
        decisions[tenant] = decisions.get(tenant, 0) + 1
        action = policy.act(state, mask)
        state, reward, terminated, truncated, info = env.step(action)
        reward_sum[tenant] = reward_sum.get(tenant, 0.0) + reward
        for event in info["regret_events"]:
            regret[event["tenant"]] = regret.get(event["tenant"], 0) + 1

    return decisions, regret, reward_sum, env._step_count


def test_s4_requires_ddos_config():
    config = load_test_config(overrides={"scenario": "S4"})
    assert "ddos" not in config  # default.yaml never carries this key -- only s4_ddos.yaml does
    with pytest.raises(KeyError):
        SmartKeyNetEnv(config)


def test_s4_flooded_tenants_request_count_is_measurably_higher_than_unflooded():
    """The designated noisy tenant's own request count must be
    measurably higher with S4's flood active than with it inactive,
    same seed, same graph."""
    decisions_off, _, _, _ = _run_s4_and_count_by_tenant(_DDOS_OFF, seed=7, n_steps=500)
    decisions_on, _, _, _ = _run_s4_and_count_by_tenant(_DDOS_ON, seed=7, n_steps=500)

    assert decisions_on[_S4_NOISY_TENANT] > decisions_off[_S4_NOISY_TENANT] * 3


# NOTE on "other tenants isolated" -- the exact, byte-for-byte isolation
# guarantee (every OTHER tenant's own arrival stream is completely
# unaffected by flood activity) is a property of RequestGenerator's
# arrival process itself, and is tested directly against the generator
# in tests/test_request_generator.py (bypassing the environment).
# Tested here, THROUGH the environment, it does NOT hold at the
# decision-count level over a *fixed external step budget* -- not
# because the flood leaks into other tenants' own arrival draws (it
# provably doesn't, see the generator-level test), but because
# env/environment.py's `_advance_to_next_decision` renders exactly one
# decision per external env.step() call and drains the pending FIFO
# queue in arrival order: a much bigger backlog under flood means
# *every* tenant's realized share of a bounded external decision
# budget shifts, not just the deliberately-targeted critical tenant's.
# This was found empirically this session (see SESSION_LOG.md) and is
# an honest, additional confirmation that the flood's crowd-out effect
# generalizes across the whole non-flooded neighborhood, not narrowly
# to whichever one victim tenant a test happens to check.


def test_s4_critical_tenant_service_throughput_collapses_under_flood():
    """The scenario's actual point: does a HIGH-sensitivity tenant's
    service hold up under the flood? Under AlwaysHybridPolicy -- which
    has no notion of protecting any tenant -- it does not: within the
    same span of real elapsed simulator time (`env._step_count`, held
    equal between the two runs, not just the same external step
    budget), the critical tenant's own decision throughput collapses
    once the noisy neighbor floods. This is this environment's
    dominant, clearly measurable "noisy neighbor" effect (see
    configs/scenarios/s4_ddos.yaml's own comments for the full
    empirical numbers this session found)."""
    decisions_off, _, _, step_count_off = _run_s4_and_count_by_tenant(_DDOS_OFF, seed=7, n_steps=2000)
    decisions_on, _, _, step_count_on = _run_s4_and_count_by_tenant(_DDOS_ON, seed=7, n_steps=2000)

    # confirms the comparison isn't confounded by one run simply covering
    # less real simulator time than the other
    assert step_count_on == pytest.approx(step_count_off, rel=0.1)

    critical_off = decisions_off.get(_S4_CRITICAL_TENANT, 0)
    critical_on = decisions_on.get(_S4_CRITICAL_TENANT, 0)
    assert critical_off > 0  # sanity: the critical tenant must appear at all under no flood
    assert critical_on < critical_off * 0.5  # a real, large collapse, not a marginal dip


def test_s4_critical_tenant_regret_rate_is_visible_only_under_pool_scarcity():
    """Secondary check, reported honestly rather than oversold: under
    this file's default `pool:` scale, regret events stay at exactly 0
    regardless of the flood (same pre-existing calibration-headroom
    property S3's own regression test already found) -- so a
    per-decision regret-rate effect for the critical tenant is only
    even possible to observe under the same small-pool scarcity
    override this suite's other regret tests use."""
    decisions_default, regret_default, _, _ = _run_s4_and_count_by_tenant(_DDOS_ON, seed=7, n_steps=500)
    assert sum(regret_default.values()) == 0

    decisions_scarce, regret_scarce, _, _ = _run_s4_and_count_by_tenant(
        _DDOS_ON, seed=7, n_steps=500, pool_override=_SCARCITY_POOL
    )
    assert sum(regret_scarce.values()) > 0


def test_hard_rule_3_no_scenario_aware_branching_downstream_of_request_generation():
    """Code-level check, not just a behavioral assertion: no file the
    agent's decision actually depends on may branch on scenario/S4/ddos
    at all. `env/environment.py` itself legitimately branches on
    `self._scenario` (that's the one sanctioned dispatch point, same as
    S2/S3) -- masking.py, the reward calculation, and state
    construction must not."""
    masking_source = Path("env/masking.py").read_text(encoding="utf-8")
    assert "scenario" not in masking_source.lower()
    assert "ddos" not in masking_source.lower()
    assert "tenant" not in masking_source.lower()

    contracts_source = Path("env/contracts.py").read_text(encoding="utf-8")
    assert "ddos" not in contracts_source.lower()

    reward_calc_source = Path("env/environment.py").read_text(encoding="utf-8")
    apply_action_start = reward_calc_source.index("def _apply_action")
    apply_action_body = reward_calc_source[apply_action_start : apply_action_start + 3000]
    assert "scenario" not in apply_action_body.lower()
    assert "ddos" not in apply_action_body.lower()


def test_s5_scenario_is_not_yet_dispatched():
    """S5 (steering attack) still needs a mechanism this repo doesn't
    have yet (see PROGRESS.md/SESSION_LOG.md) -- deliberately deferred,
    not an oversight. Selecting it must currently be a pure no-op,
    identical to any other unrecognized scenario string, and must not
    require any new config block the way S2/S3/S4/S6 now do."""
    config = load_test_config(overrides={"scenario": "S5"})
    env = SmartKeyNetEnv(config)
    assert env._threat_schedule_cfg is None
    assert env._qkd_degradation_cfg is None
    assert env._ddos_cfg is None
    assert env._migration_graph_seed is None
    assert env._migration_schedule == []
    assert env._tenant_graph is None
    env.reset(seed=0)  # must not raise -- no scenario-specific config required


def test_scenario_config_files_load_and_construct_a_working_env():
    """The committed, standalone scenario files must actually be
    loadable and runnable, not just referenced."""
    for path, expected_scenario in (
        ("configs/scenarios/s2_hndl.yaml", "S2"),
        ("configs/scenarios/s3_degradation.yaml", "S3"),
        ("configs/scenarios/s4_ddos.yaml", "S4"),
        ("configs/scenarios/s6_migration.yaml", "S6"),
    ):
        config = load_full_config(path)
        config["max_steps"] = 20
        env = SmartKeyNetEnv(config)
        assert env._scenario == expected_scenario
        state, info = env.reset(seed=0)
        assert info["action_mask"].any()


# ---------------------------------------------------------------------------
# S6 (migration wave) scenario dispatch (2026-08-24, design decision 15)
#
# The mechanism tests below deliberately use a test-local
# migration_schedule (via load_test_config's shallow-merge), NOT the
# real committed configs/scenarios/s6_migration.yaml -- same convention
# S4's own mechanism tests already established (see
# _run_s4_and_count_by_tenant above): the committed file's real
# schedule is separately verified end-to-end by
# test_scenario_config_files_load_and_construct_a_working_env, but its
# tenants (tenant_0, weight ~1%; tenant_3, ~4%) are too low-traffic to
# reliably produce many real before/after decisions within a fast test
# budget -- tenant_5 (S0, real traffic_rate ~12.7% under graph_seed 0 --
# verified by build_tenant_graph(seed=0), not guessed) gives a robust,
# fast, deterministic sample on both sides of its scripted ratchet.
# ---------------------------------------------------------------------------

_S6_GRAPH_SEED = 0
_S6_RATCHET_STEP = 100
_S6_TEST_SCHEDULE = [{"step": _S6_RATCHET_STEP, "tenant_index": 5, "new_sensitivity_class": 3}]
_S6_RATCHET_TENANT = "tenant_5"  # real S0 tenant under graph_seed 0, n_nodes 10 (verified this session)


def _run_s6_and_collect_decisions(
    schedule: list[dict[str, Any]], seed: int, n_steps: int, use_foresight: str = "ewma"
) -> list[dict[str, Any]]:
    """Drives a real S6 episode under `RandomPolicy` (sampled from the
    real mask each step, same "valid random agent" convention the W2
    gate test and S4's own tests use) and returns, per decision, the
    real `Request` actually decided plus the `policy_floor` the env
    computed for it -- a direct, request-by-request record, not an
    aggregate statistic."""
    config = load_test_config(
        overrides={
            "scenario": "S6",
            "migration_graph_seed": _S6_GRAPH_SEED,
            "migration_schedule": schedule,
            "use_foresight": use_foresight,
        }
    )
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    decisions: list[dict[str, Any]] = []
    for _ in range(n_steps):
        request = env._current_request
        decisions.append(
            {
                "tenant": request["tenant"],
                "sensitivity_class": request["sensitivity_class"],
                "policy_floor": state["policy_floor"],
                "request_step": request["step"],
            }
        )
        action, (state, reward, terminated, truncated, info) = take_random_valid_step(env, rng, info)

    return decisions


def test_s6_requires_migration_graph_seed_config():
    config = load_test_config(overrides={"scenario": "S6"})
    assert "migration_graph_seed" not in config  # default.yaml never carries this key -- only s6_migration.yaml does
    with pytest.raises(KeyError):
        SmartKeyNetEnv(config)


def test_s6_tenant_sensitivity_class_changes_at_scripted_step_request_by_request():
    """Direct, request-by-request check (not an aggregate statistic):
    every real decision drawn from the ratcheted tenant carries the OLD
    sensitivity_class while its arrival step is before the scripted
    ratchet step, and the NEW one from the ratchet step onward."""
    decisions = _run_s6_and_collect_decisions(_S6_TEST_SCHEDULE, seed=7, n_steps=400)
    tenant_decisions = [d for d in decisions if d["tenant"] == _S6_RATCHET_TENANT]

    before = [d for d in tenant_decisions if d["request_step"] < _S6_RATCHET_STEP]
    after = [d for d in tenant_decisions if d["request_step"] >= _S6_RATCHET_STEP]
    assert before  # real evidence on both sides, not vacuously true
    assert after

    assert all(d["sensitivity_class"] == 0 for d in before)  # S0, the tenant's real pre-migration class
    assert all(d["sensitivity_class"] == 3 for d in after)  # S3, the scripted post-migration class


def test_s6_floor_changes_correspondingly_cross_checked_against_real_masking_table():
    """Confirm the resulting floor also changes, cross-checked against
    a real `PolicyTable().floor()` call (not a hardcoded expected
    value). `use_foresight: off` pins posture at CALM for the whole
    episode (env/environment.py's `_prepare_decision`: no forecaster ->
    `current_posture = ThreatPosture.CALM` always) so the floor is
    driven purely by sensitivity_class, isolating exactly the effect
    this scenario is supposed to produce."""
    decisions = _run_s6_and_collect_decisions(_S6_TEST_SCHEDULE, seed=7, n_steps=400, use_foresight="off")
    tenant_decisions = [d for d in decisions if d["tenant"] == _S6_RATCHET_TENANT]
    before = [d for d in tenant_decisions if d["request_step"] < _S6_RATCHET_STEP]
    after = [d for d in tenant_decisions if d["request_step"] >= _S6_RATCHET_STEP]
    assert before
    assert after

    fresh_table = PolicyTable()
    expected_floor_before = int(fresh_table.floor(SensitivityClass.S0, ThreatPosture.CALM))
    expected_floor_after = int(fresh_table.floor(SensitivityClass.S3, ThreatPosture.CALM))
    assert expected_floor_before != expected_floor_after  # the table must actually distinguish these

    assert all(d["policy_floor"] == expected_floor_before for d in before)
    assert all(d["policy_floor"] == expected_floor_after for d in after)


def test_s6_other_tenants_are_completely_unaffected_by_one_tenants_scheduled_ratchet():
    """Isolation check, same spirit as S4's own per-tenant isolation
    test: every OTHER tenant's sensitivity_class (and therefore floor)
    stays exactly at its real, graph-sampled value for the entire
    episode -- a scheduled change to one tenant must not leak into any
    other."""
    graph = build_tenant_graph(n_nodes=10, seed=_S6_GRAPH_SEED)
    real_classes = {n: attrs["sensitivity_class"] for n, attrs in graph.nodes(data=True) if attrs.get("kind") == "tenant"}

    decisions = _run_s6_and_collect_decisions(_S6_TEST_SCHEDULE, seed=7, n_steps=400)
    other_decisions = [d for d in decisions if d["tenant"] != _S6_RATCHET_TENANT]
    assert other_decisions  # real evidence some other tenant was actually decided during the run

    for d in other_decisions:
        assert d["sensitivity_class"] == real_classes[d["tenant"]]


def test_hard_rule_3_no_s6_scenario_aware_branching_downstream_of_request_generation():
    """Code-level check, same standard as S4's own equivalent test:
    masking.py, contracts.py, and the reward-calculation body must
    contain zero mentions of "migration"/"s6" -- env/environment.py's
    own sanctioned dispatch site (design decision 15) is the only place
    this scenario is allowed to be scenario-aware."""
    masking_source = Path("env/masking.py").read_text(encoding="utf-8")
    assert "migration" not in masking_source.lower()
    assert "s6" not in masking_source.lower()

    contracts_source = Path("env/contracts.py").read_text(encoding="utf-8")
    assert "migration" not in contracts_source.lower()

    reward_calc_source = Path("env/environment.py").read_text(encoding="utf-8")
    apply_action_start = reward_calc_source.index("def _apply_action")
    apply_action_body = reward_calc_source[apply_action_start : apply_action_start + 3000]
    assert "migration" not in apply_action_body.lower()
    assert "s6" not in apply_action_body.lower()


def test_s6_held_out_eval_sanity_run_via_harness():
    """The scenario's whole purpose: it must be genuinely usable for
    held-out evaluation via experiments/harness.py's existing
    run_scenario against a real baseline policy -- not a rigorous
    benchmark, just confirmation the episode runs to completion and
    Hard Rule 9's floor-violation guarantee still holds across the
    ratchet point, using the real, committed s6_migration.yaml (not a
    test-local override)."""
    s6_config = load_full_config("configs/scenarios/s6_migration.yaml")
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)

    result = run_scenario(policy, "S6", s6_config, seed=7)

    assert result.floor_violations == 0  # Hard Rule 2, holds across the ratchet too


# ---------------------------------------------------------------------------
# security_masking config flag (2026-08-25, design decision 16 --
# soft-reward baseline agent session)
# ---------------------------------------------------------------------------


class _AlwaysClassicalPolicy:
    """Test-local stub: always attempts the weakest tier regardless of
    state/mask. Would raise `IllegalActionError` under the default
    (`security_masking: true`) mask the instant the real floor rises
    above SERVE_CLASSICAL -- used specifically to prove `security_masking:
    false` genuinely lifts that restriction, not just documents it."""

    def act(self, state, mask):
        return Action.SERVE_CLASSICAL


def test_security_masking_defaults_to_true_and_is_unaffected_by_the_keys_absence():
    """`config.get("security_masking", True)` -- a config that doesn't
    set this key at all (e.g. any pre-existing caller written before
    this session) must behave byte-for-byte identically to one that sets
    it explicitly `True`. Same S2 elevated-posture config/seed/policy,
    every mask and reward across a full episode compared directly."""
    base_config = load_test_config(overrides={"scenario": "S2", "threat_schedule": _S2_THREAT_SCHEDULE})
    assert base_config["security_masking"] is True  # configs/default.yaml now sets this explicitly

    config_explicit_true = {**base_config, "security_masking": True}
    config_absent = {k: v for k, v in base_config.items() if k != "security_masking"}

    def run(config: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
        env = SmartKeyNetEnv(config)
        state, info = env.reset(seed=0)
        mask = info["action_mask"]
        masks = [mask.copy()]
        rewards: list[float] = []
        policy = RandomPolicy(seed=0)
        for _ in range(200):
            action = policy.act(state, mask)
            state, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            masks.append(mask.copy())
            rewards.append(reward)
        return masks, rewards

    masks_true, rewards_true = run(config_explicit_true)
    masks_absent, rewards_absent = run(config_absent)

    assert len(masks_true) == len(masks_absent)
    for m1, m2 in zip(masks_true, masks_absent):
        assert np.array_equal(m1, m2)
    assert rewards_true == rewards_absent


def test_security_masking_false_lets_a_policy_serve_below_the_real_floor():
    """The core proof design decision 16 exists for: with
    `security_masking: false`, a policy that always attempts
    SERVE_CLASSICAL genuinely serves below a real, ratcheted-up floor
    (Hard Rule 2's floor rule is not enforced for this config) -- while
    the SAME scenario/config, left at the default `security_masking:
    true`, still guarantees `floor_violations == 0` for a real masked
    policy (`RandomPolicy`, which only ever samples from the mask).
    Confirms `env/masking.py::compute_mask()` output itself is
    unaffected -- only which arguments `env/environment.py` passes to it
    differ (see that function's own unmodified test suite in
    tests/test_masking.py, which this session left completely untouched)."""
    base = {"scenario": "S2", "threat_schedule": _S2_THREAT_SCHEDULE, "max_steps": 200}

    masked_config = load_test_config(overrides=base)
    unmasked_config = load_test_config(overrides={**base, "security_masking": False})

    masked_result = run_scenario(RandomPolicy(seed=0), "S2", masked_config, seed=0)
    assert masked_result.floor_violations == 0

    unmasked_result = run_scenario(_AlwaysClassicalPolicy(), "S2", unmasked_config, seed=0)
    assert unmasked_result.floor_violations > 0


def test_security_masking_false_still_gates_serve_hybrid_on_pool_and_reuse_on_key_age():
    """Physical/protocol feasibility rules are NOT part of what
    `security_masking` lifts (design decision 16): a scarce pool still
    forces SERVE_HYBRID illegal, and a cold-start session (key_age ==
    max_key_age, see `_prepare_decision`'s session-creation default)
    still forces REUSE illegal, even with `security_masking: false`."""
    scarce_config = load_test_config(overrides={"security_masking": False, "pool": _SCARCITY_POOL})
    scarce_env = SmartKeyNetEnv(scarce_config)
    state, info = scarce_env.reset(seed=0)
    mask = info["action_mask"]

    # The actual lifted rule, sanity-checked: both non-hybrid tiers are
    # legal regardless of the real floor.
    assert bool(mask[Action.SERVE_CLASSICAL])
    assert bool(mask[Action.SERVE_PQC])

    # The NOT-lifted rules, cross-checked against real env state rather
    # than a hardcoded expectation:
    pool_can_draw = scarce_env._pool_sim.can_draw(scarce_env._bits_per_hybrid_draw)
    assert bool(mask[Action.SERVE_HYBRID]) == pool_can_draw
    assert not bool(mask[Action.REUSE])  # cold-start session: key_age == max_key_age


def test_security_masking_false_does_not_crash_a_full_random_valid_episode():
    """A full episode driven by a policy that samples uniformly from
    whatever the (floor-free) mask allows must run to completion without
    ever hitting `IllegalActionError` or a simulator-level crash (e.g.
    `PoolExhaustedError` from `_apply_action`/`pool_sim.draw()`) -- the
    concrete regression risk design decision 16's docstring flags."""
    config = load_test_config(overrides={"security_masking": False, "max_steps": 300})
    result = run_scenario(RandomPolicy(seed=0), "S1", config, seed=0)
    assert result.episode_metrics.regret_events >= 0  # ran to completion; no crash


# ---------------------------------------------------------------------------
# forecast_provider_factory (design decision 17, S5 dose-response sweep
# session) -- mirrors request_stream_factory's own swap-test precedent
# above (design decision 12).
# ---------------------------------------------------------------------------


def test_forecast_provider_factory_none_is_byte_for_byte_identical_to_default():
    """`forecast_provider_factory=None` (the default) must reproduce
    `reset()`'s prior forecaster-construction behavior exactly -- same
    full-episode trajectory as constructing `SmartKeyNetEnv` with no
    such argument at all."""
    from env.forecast_provider import MovingAverageForecaster

    config = load_test_config(overrides={"use_foresight": "ewma", "max_steps": 100})

    baseline_env = SmartKeyNetEnv(config)
    state_a, info_a = baseline_env.reset(seed=11)

    explicit_none_env = SmartKeyNetEnv(config, forecast_provider_factory=None)
    state_b, info_b = explicit_none_env.reset(seed=11)

    policy_a, policy_b = RandomPolicy(seed=5), RandomPolicy(seed=5)
    truncated_a = truncated_b = False
    while not truncated_a:
        action_a = policy_a.act(state_a, info_a["action_mask"])
        action_b = policy_b.act(state_b, info_b["action_mask"])
        assert action_a == action_b
        state_a, reward_a, _, truncated_a, info_a = baseline_env.step(action_a)
        state_b, reward_b, _, truncated_b, info_b = explicit_none_env.step(action_b)
        assert reward_a == reward_b
        assert state_a == state_b

    assert isinstance(baseline_env._forecaster, MovingAverageForecaster)
    assert isinstance(explicit_none_env._forecaster, MovingAverageForecaster)
    assert baseline_env._forecaster.get_threat_forecast() == explicit_none_env._forecaster.get_threat_forecast()


def test_forecast_provider_factory_swap_is_a_genuine_drop_in():
    """Hard Rule 3's swap test, for the forecaster this time: a caller
    supplying a `forecast_provider_factory` gets a genuinely different
    forecaster instance actually driving the episode's floor -- proof
    it's load-bearing, not silently ignored -- while no other line in
    this file needs to know or care which provider is in use."""

    class _ConstantHighForecastProvider:
        """A trivial stand-in ForecastProvider (not MovingAverageForecaster) --
        always reports HIGH posture, regardless of what it's updated with."""

        def update(self, observation):
            pass

        def get_threat_forecast(self):
            from env.contracts import ThreatForecast

            return ThreatForecast(threat_score=1.0, posture_probs=[0.0, 0.0, 1.0], horizon_scores=[1.0] * 5)

        def get_pool_forecast(self):
            from env.contracts import PoolForecast

            return PoolForecast(pool_level_hat=[0.0] * 3, skr_mean_hat=[0.0] * 3, hybrid_demand_hat=[0.0] * 3)

    config = load_test_config(overrides={"use_foresight": "ewma", "max_steps": 20})
    env = SmartKeyNetEnv(config, forecast_provider_factory=lambda seed: _ConstantHighForecastProvider())
    state, info = env.reset(seed=0)

    assert isinstance(env._forecaster, _ConstantHighForecastProvider)
    # HIGH posture, every sensitivity class -> floor is at least SERVE_PQC
    # (S0/S1) or SERVE_HYBRID (S2/S3) from the very first decision --
    # never CALM's SERVE_CLASSICAL floor, which a request_generator-driven
    # S0 request would otherwise get on a fresh episode.
    assert state["policy_floor"] >= int(Action.SERVE_PQC)


def test_forecast_provider_factory_receives_episode_seed():
    """The factory is called with the same `episode_seed` `reset()`
    itself resolves (`seed` argument if given, else `config["seed"]`) --
    mirroring `request_stream_factory`'s own contract exactly."""
    seen_seeds: list[int | None] = []

    def factory(episode_seed):
        seen_seeds.append(episode_seed)
        return None  # use_foresight handling: env tolerates a None forecaster (== "off" behavior)

    config = load_test_config(overrides={"use_foresight": "off", "max_steps": 10})
    env = SmartKeyNetEnv(config, forecast_provider_factory=factory)
    env.reset(seed=77)
    env.reset(seed=88)

    assert seen_seeds == [77, 88]
