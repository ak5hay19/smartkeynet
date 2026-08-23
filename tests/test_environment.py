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

from agents.baselines import AlwaysHybridPolicy, RandomPolicy
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


def test_s4_and_s6_scenarios_are_not_yet_dispatched():
    """S4 (DDoS/noisy-neighbor) and S6 (migration wave) both need a
    "which tenant is this" concept env/request_generator.py's current
    random stream doesn't have (see PROGRESS.md/SESSION_LOG.md
    2026-08-19) -- deliberately deferred, not an oversight. Selecting
    either must currently be a pure no-op, identical to any other
    unrecognized scenario string, and must not require any new config
    block the way S2/S3 now do."""
    for scenario in ("S4", "S5", "S6"):
        config = load_test_config(overrides={"scenario": scenario})
        env = SmartKeyNetEnv(config)
        assert env._threat_schedule_cfg is None
        assert env._qkd_degradation_cfg is None
        env.reset(seed=0)  # must not raise -- no scenario-specific config required


def test_scenario_config_files_load_and_construct_a_working_env():
    """The two committed, standalone scenario files must actually be
    loadable and runnable, not just referenced."""
    for path, expected_scenario in (
        ("configs/scenarios/s2_hndl.yaml", "S2"),
        ("configs/scenarios/s3_degradation.yaml", "S3"),
    ):
        config = load_full_config(path)
        config["max_steps"] = 20
        env = SmartKeyNetEnv(config)
        assert env._scenario == expected_scenario
        state, info = env.reset(seed=0)
        assert info["action_mask"].any()
