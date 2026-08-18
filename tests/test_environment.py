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

from agents.baselines import AlwaysHybridPolicy, AlwaysPQCPolicy
from env.contracts import Action, ThreatPosture
from env.environment import (
    _ACTION_TO_KEY_TYPE,
    _ENERGY_UNITS,
    _KEY_TYPE_TO_SERVE_ACTION,
    _LATENCY_UNITS,
    IllegalActionError,
    SmartKeyNetEnv,
    build_scenario_runtime,
)
from experiments.harness import run_scenario

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


_EXTREME_SCARCITY_POOL: dict[str, float] = {
    "capacity_bits": 500_000.0,
    "initial_fill_frac": 0.0,
    "bits_per_hybrid_draw": 300_000.0,
    # A hand-built draw size ~1,200x the production one needs a refill rate to
    # match, or the pool takes 1,500 steps to cover a single draw and nothing
    # these tests are actually about (deferral onset, then drain) is observable
    # inside a 250-300 step window. 200 kbps / 1 request-per-epoch == 200,000
    # bits/step, i.e. two steps of refill per draw. Before the 2026-08-19 pool
    # recalibration (see configs/default.yaml's `pool:` block) this was the
    # *production* rate, which is why these tests didn't state it themselves.
    "link_skr_kbps": 200.0,
    "kms_requests_per_decision_epoch": 1.0,
}
"""Deliberately extreme scarcity for the Hard Rule 9 structural tests below:
a pool that cannot cover a draw at reset and refills to cover one every two
steps. Not representative of the production `pool:` block -- these tests are
about the *mechanism* (never serve below floor; defer instead; drain later),
which has to hold at any pool sizing."""


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
        overrides={"pool": _EXTREME_SCARCITY_POOL}
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
    # REUSE never draws: the pool can only have gone *up* (trace refill during
    # the step's advance-to-next-decision phase). This was an equality check
    # until 2026-08-19, which passed only because the pre-recalibration pool sat
    # pinned at capacity every step -- refill was clamped away, so "no draw" and
    # "no change" were indistinguishable. With a pool that genuinely moves, the
    # real invariant is that nothing was *subtracted*.
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
        overrides={"pool": _EXTREME_SCARCITY_POOL}
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
# Scenario dispatch S1-S4 (2026-08-19 -- before this, config["scenario"] was
# read but never acted on, so every scenario silently ran S1)
# ---------------------------------------------------------------------------


def _run_scenario_probe(scenario: str, seed: int = 0, steps: int = 250):
    """Drive one scenario with a fixed, boring policy and collect the
    environment-side quantities each scenario is supposed to move.
    AlwaysPQC never draws from the pool, so pool/threat/load effects
    are attributable to the scenario rather than to policy behaviour."""
    config = load_test_config(overrides={"scenario": scenario, "max_steps": steps})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=seed)
    policy = AlwaysPQCPolicy()

    floors, postures, loads, skrs, tenants = [], [], [], [], []
    truncated = False
    while not truncated:
        floors.append(state["policy_floor"])
        loads.append(state["load"])
        skrs.append(state["skr"])
        tenants.append(env._current_request["tenant"])
        postures.append(
            ThreatPosture(int(np.argmax(env._forecaster.get_threat_forecast().posture_probs)))
        )
        action = policy.act(state, info["action_mask"])
        state, _reward, _terminated, truncated, info = env.step(action)

    return {
        "floors": floors,
        "postures": postures,
        "loads": loads,
        "skrs": skrs,
        "tenants": tenants,
        "ratcheted_posture": env._policy_table._ratcheted_posture,
    }


def test_unknown_scenario_is_an_error_not_a_silent_fallback_to_s1():
    """Silently running the wrong scenario is invisible in results --
    which is exactly the failure mode this repo had before scenario
    dispatch existed."""
    with pytest.raises(ValueError):
        SmartKeyNetEnv(load_test_config(overrides={"scenario": "S99"}))


def test_s1_stays_calm_for_a_whole_benign_episode():
    """S1 is the *benign* baseline, and every other scenario's claim is
    relative to it. Regression guard for the 2026-08-19 forecaster
    squash calibration: before it, S1 spent 249/250 decisions at
    ELEVATED and the CALM floor row was unreachable."""
    probe = _run_scenario_probe("S1")
    assert set(probe["postures"]) == {ThreatPosture.CALM}
    assert probe["ratcheted_posture"] is ThreatPosture.CALM


def test_s2_elevates_threat_and_ratchets_floors_above_s1():
    """S2's distinguishing behaviour (PLAN2 §9): threat elevates ->
    floors ratchet up."""
    s1 = _run_scenario_probe("S1")
    s2 = _run_scenario_probe("S2")

    assert s2["ratcheted_posture"] is ThreatPosture.HIGH
    assert ThreatPosture.HIGH in s2["postures"]
    assert np.mean(s2["floors"]) > np.mean(s1["floors"])

    # Hard Rule 2: the ratchet is one-way -- the resolved posture must
    # never step back down over the episode, even as the raw signal
    # ramps through intermediate values.
    ratchet_track = np.maximum.accumulate([int(p) for p in s2["postures"]])
    assert list(ratchet_track) == sorted(ratchet_track)


def test_s3_collapses_pool_refill_and_multiplies_exhaustion():
    """S3's distinguishing behaviour (PLAN2 §9): QBER up, SKR down,
    pool refill collapses."""
    s1 = _run_scenario_probe("S1")
    s3 = _run_scenario_probe("S3")

    assert np.mean(s3["skrs"]) < 0.75 * np.mean(s1["skrs"])
    assert min(s3["skrs"]) < min(s1["skrs"])

    config = load_test_config(overrides={"max_steps": 250})
    s1_hybrid = run_scenario(AlwaysHybridPolicy(), "S1", config, seed=0)
    s3_hybrid = run_scenario(AlwaysHybridPolicy(), "S3", config, seed=0)
    assert s3_hybrid.pool_exhaustion_events > s1_hybrid.pool_exhaustion_events
    assert (
        s3_hybrid.episode_metrics.deferred_critical_steps
        > s1_hybrid.episode_metrics.deferred_critical_steps
    )


def test_s4_floods_the_low_sensitivity_tenant_and_raises_load():
    """S4's distinguishing behaviour (PLAN2 §9): one low-sensitivity
    tenant floods the API. The flood must show up as a *share* shift
    toward that tenant and as higher aggregate load -- a noisy
    neighbour sends more traffic, it does not merely take a bigger
    slice of a fixed budget."""
    s1 = _run_scenario_probe("S1")
    s4 = _run_scenario_probe("S4")

    def share(probe, tenant):
        return sum(t == tenant for t in probe["tenants"]) / len(probe["tenants"])

    assert share(s4, "iot-telemetry") > 3 * share(s1, "iot-telemetry")
    assert np.mean(s4["loads"]) > np.mean(s1["loads"])
    # ...and it must not be a general load increase: the *other* tenants'
    # share has to fall, or "noisy neighbour" means nothing.
    assert share(s4, "hospital") < share(s1, "hospital")


@pytest.mark.parametrize("scenario", ["S1", "S2", "S3", "S4"])
def test_floor_violations_stay_structurally_impossible_in_every_scenario(scenario):
    """Hard Rule 2/9 hold under every perturbation, not just the benign
    baseline -- including S3, where the pool genuinely runs dry."""
    config = load_test_config(overrides={"max_steps": 200})
    for policy in (AlwaysHybridPolicy(), AlwaysPQCPolicy()):
        result = run_scenario(policy, scenario, config, seed=0)
        assert result.floor_violations == 0


@pytest.mark.parametrize("scenario", ["S1", "S2", "S3", "S4"])
def test_agent_visible_state_never_leaks_the_tenant_graph(scenario):
    """Hard Rule 3: the graph shapes which requests arrive and nothing
    else. A `StateDict` must carry no tenant identity, no node/edge
    reference, and no scenario name -- otherwise deleting the graph
    would change agent code."""
    config = load_test_config(overrides={"scenario": scenario, "max_steps": 50})
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    tenants = {data["tenant"] for _n, data in env._tenant_graph.nodes(data=True)}

    truncated = False
    while not truncated:
        for key, value in state.items():
            assert not isinstance(value, str), f"{key} leaked a string into agent-visible state"
            if isinstance(value, (list, tuple)):
                assert not any(isinstance(v, str) for v in value)
        assert tenants.isdisjoint(set(state))
        action = next(a for a in Action if info["action_mask"][int(a)])
        state, _r, _t, truncated, info = env.step(action)


def test_scenario_runtime_is_a_pure_function_of_step():
    """`_ScenarioRuntime` carries no internal state, so an episode is
    reproducible from its seed alone and the dashboard can re-derive a
    scenario's schedule without re-running the environment."""
    config = load_test_config()
    runtime = build_scenario_runtime("S2", config)
    first = [runtime.threat_level(s) for s in range(200)]
    second = [runtime.threat_level(s) for s in reversed(range(200))][::-1]
    assert first == second
    assert first == sorted(first)  # the HNDL ramp only ever rises


def test_external_threat_trace_overrides_the_scenario_signal():
    """S5's hook. Narrow by construction: it reaches the forecaster's
    threat features and nothing else."""
    config = load_test_config(overrides={"scenario": "S1", "max_steps": 30})
    env = SmartKeyNetEnv(config)
    env.set_external_threat_trace([4.0] * 500)
    state, info = env.reset(seed=0)

    for _ in range(20):
        action = next(a for a in Action if info["action_mask"][int(a)])
        state, _r, _t, _tr, info = env.step(action)

    # an adversarially high trace can only push the posture UP
    assert env._policy_table._ratcheted_posture is not ThreatPosture.CALM
    assert state["threat_score"] > 0.5
