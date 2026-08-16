"""Behavioral tests for `env.scenarios` -- the S1-S6 dispatch layer
(PLAN.md §5; SMARTKEYNET_BUILD_SPEC.md §S6's scenario table).

Covers the three exogenous perturbation channels (QBER drift, tenant
flood, threat windows), the Hard Rule 8 eval-only guard, and the Hard
Rule 2 property that a threat window can only ever *raise* a floor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from env.contracts import Action, ThreatPosture
from env.environment import SmartKeyNetEnv
from agents.baselines import StaticThresholdPolicy
from env.scenarios import (
    FloorChange,
    ScenarioError,
    ScenarioSpec,
    ThreatWindow,
    build_scenario,
    require_trainable,
)


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["S1", "S2", "S3", "S4", "S5", "S6"])
def test_every_scenario_builds(name):
    spec = build_scenario(name, load_config(), episode_steps=2500)
    assert isinstance(spec, ScenarioSpec)
    assert spec.name == name


@pytest.mark.parametrize("name", ["s3", " S3 ", "s3 "])
def test_scenario_names_are_case_and_whitespace_insensitive(name):
    assert build_scenario(name, load_config(), episode_steps=2500).name == "S3"


def test_unknown_scenario_raises_rather_than_falling_back():
    """A typo'd scenario name that silently ran S1 would invalidate a
    whole results table with no visible failure."""
    with pytest.raises(ScenarioError):
        build_scenario("S7", load_config(), episode_steps=2500)
    with pytest.raises(ScenarioError):
        build_scenario("baseline", load_config(), episode_steps=2500)


def test_only_the_expected_channel_is_populated_per_scenario():
    config = load_config()
    s1 = build_scenario("S1", config, episode_steps=2500)
    s2 = build_scenario("S2", config, episode_steps=2500)
    s3 = build_scenario("S3", config, episode_steps=2500)
    s4 = build_scenario("S4", config, episode_steps=2500)

    assert (s1.qber_drift, s1.tenant_flood, s1.threat_windows) == (None, None, ())
    assert s2.threat_windows and s2.qber_drift is None and s2.tenant_flood is None
    assert s3.qber_drift is not None and s3.tenant_flood is None and not s3.threat_windows
    assert s4.tenant_flood is not None and s4.qber_drift is None and not s4.threat_windows


# ---------------------------------------------------------------------------
# Hard Rule 8 -- S5/S6 are eval-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["S5", "S6"])
def test_eval_only_scenarios_are_marked_and_rejected_for_training(name):
    spec = build_scenario(name, load_config(), episode_steps=2500)
    assert spec.eval_only is True
    with pytest.raises(ScenarioError):
        require_trainable(spec)


@pytest.mark.parametrize("name", ["S1", "S2", "S3", "S4"])
def test_trainable_scenarios_pass_the_guard(name):
    spec = build_scenario(name, load_config(), episode_steps=2500)
    assert spec.eval_only is False
    assert require_trainable(spec) is spec


# ---------------------------------------------------------------------------
# Hard Rule 2 -- threat signals may only raise floors
# ---------------------------------------------------------------------------


def test_threat_window_rejects_negative_intensity():
    with pytest.raises(ValueError):
        ThreatWindow(start_step=0, end_step=10, intensity=-0.1)


def test_threat_window_rejects_empty_range():
    with pytest.raises(ValueError):
        ThreatWindow(start_step=10, end_step=10, intensity=1.0)


def test_threat_boost_is_never_negative_anywhere():
    """The machine-checked form of "threat signals may only raise
    floors": there is no step at which the scenario can subtract from
    the threat signal."""
    spec = build_scenario("S2", load_config(), episode_steps=2500)
    for step in range(0, 3000, 3):
        assert spec.threat_boost_at(step) >= 0.0


def test_threat_boost_is_zero_outside_windows_and_positive_inside():
    """`ramp_steps=0` gives the original rectangular behaviour, kept as
    a mode so the ramp can be ablated against it."""
    spec = ScenarioSpec(
        name="test",
        eval_only=False,
        threat_windows=(
            ThreatWindow(100, 200, 2.0, ramp_steps=0),
            ThreatWindow(150, 250, 3.0, ramp_steps=0),
        ),
    )
    assert spec.threat_boost_at(99) == pytest.approx(0.0)
    assert spec.threat_boost_at(100) == pytest.approx(2.0)
    assert spec.threat_boost_at(175) == pytest.approx(5.0)  # overlapping windows add
    assert spec.threat_boost_at(225) == pytest.approx(3.0)
    assert spec.threat_boost_at(250) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end through the environment
# ---------------------------------------------------------------------------


def pick_random(state, mask, rng):
    return Action(int(rng.choice(np.flatnonzero(mask))))


def pick_hybrid(state, mask, rng):
    """The always-hybrid villain: the policy the scarcity calibration
    is sized against, and the one SMARTKEYNET_BUILD_SPEC.md §S7 uses to
    prove a scenario has scarcity at all."""
    if mask[Action.SERVE_HYBRID]:
        return Action.SERVE_HYBRID
    return Action(int(np.flatnonzero(mask)[0]))


def run_episode(config: dict[str, Any], n_steps: int, seed: int = 0, policy=pick_random):
    """Drive an env with `policy`, returning per-step observations
    useful for asserting scenario effects."""
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    records = []
    for _ in range(n_steps):
        mask = info["action_mask"]
        action = policy(state, mask, rng)
        records.append(
            {
                "tick": env._step_count,
                "pool_fill": state["pool_fill"],
                "skr": state["skr"],
                "qber": state["qber"],
                "floor": state["policy_floor"],
                "posture": env._policy_table._ratcheted_posture,
                "regret": len(info["regret_events"]),
            }
        )
        state, reward, terminated, truncated, info = env.step(action)
    return records


def mean_skr(records, lo, hi):
    window = [record["skr"] for record in records if lo <= record["tick"] < hi]
    return sum(window) / max(1, len(window))


def test_s3_collapses_skr_in_the_degradation_window():
    """S3's mechanism, measured end-to-end through the env: refill in
    the peak-hold window falls below 30% of the pre-degradation level,
    while S1's stays flat."""
    base = {"scenario_steps": 900, "use_foresight": "off"}
    s1_records = run_episode(load_config({**base, "scenario": "S1"}), n_steps=800)
    s3_records = run_episode(load_config({**base, "scenario": "S3"}), n_steps=800)

    assert mean_skr(s3_records, 450, 600) < 0.3 * mean_skr(s3_records, 0, 300)
    assert mean_skr(s1_records, 450, 600) == pytest.approx(mean_skr(s1_records, 0, 300), rel=0.15)


def test_always_hybrid_exhausts_the_pool_on_s3_but_not_on_s1():
    """SMARTKEYNET_BUILD_SPEC.md §S7 test 2: "`regret_events > 0`. If
    this fails, your scenario has no scarcity."

    Measured against the always-hybrid villain, which is the policy the
    calibration is sized against (rho 1.14 on S1). A *random* policy
    correctly shows no regret on either scenario -- it spends about 0.2
    keys/step against 0.859 of refill -- so using one here would prove
    nothing about S3.
    """
    base = {"scenario_steps": 900, "use_foresight": "off"}
    s1_records = run_episode(
        load_config({**base, "scenario": "S1"}), n_steps=800, policy=pick_hybrid
    )
    s3_records = run_episode(
        load_config({**base, "scenario": "S3"}), n_steps=800, policy=pick_hybrid
    )

    s1_regret = sum(record["regret"] for record in s1_records)
    s3_regret = sum(record["regret"] for record in s3_records)

    assert s3_regret > 0, "S3 degradation produced no deferrals -- the scenario has no scarcity"
    assert s3_regret > s1_regret

    # and the pool visibly bottoms out during the degradation
    min_fill_during = min(r["pool_fill"] for r in s3_records if 450 <= r["tick"] < 600)
    assert min_fill_during < 0.1


def test_a_deliberate_policy_stays_within_budget_on_s1():
    """The other half of the designed contrast: a policy that spends
    the pool deliberately must not starve on the benign baseline. If
    this starts failing, the calibration has drifted too tight and S1
    is no longer benign.

    Measured against the tuned threshold rather than a random policy.
    Random is *not* frugal in this environment -- it returns REUSE only
    about one decision in five, so it re-establishes key material
    constantly and is profligate in exactly the way the scarcity
    calibration punishes (it records ~142 regret events here). That is
    correct behaviour, not a calibration failure, but it makes random
    the wrong probe for "is there enough budget for a sensible policy".
    """
    config = load_config({"scenario": "S1", "scenario_steps": 900, "use_foresight": "off"})
    deliberate = StaticThresholdPolicy(
        pool_fill_threshold=0.7,
        min_hybrid_class=2,
        rekey_age_frac=0.9,
        max_key_age=float(config["key_lifetime"]["max_key_age_steps"]),
    )
    records = run_episode(
        config, n_steps=800, policy=lambda state, mask, rng: deliberate.act(state, mask)
    )
    assert sum(record["regret"] for record in records) == 0


def test_s2_walks_calm_to_elevated_to_high_and_never_ratchets_back_down():
    """S2 steps the posture up one level per window, and
    `PolicyTable`'s sticky ratchet means it never falls back within the
    episode (Hard Rule 2)."""
    config = load_config({"scenario": "S2", "scenario_steps": 2500, "use_foresight": "ewma"})
    records = run_episode(config, n_steps=2300)

    postures = [int(record["posture"]) for record in records]
    assert postures == sorted(postures)  # monotonically non-decreasing: never ratchets down

    # all three levels are actually visited -- the full progression
    assert set(postures) == {
        int(ThreatPosture.CALM),
        int(ThreatPosture.ELEVATED),
        int(ThreatPosture.HIGH),
    }

    def posture_at(tick_lo, tick_hi):
        window = [r["posture"] for r in records if tick_lo <= r["tick"] < tick_hi]
        return max(int(p) for p in window)

    assert posture_at(0, 400) == int(ThreatPosture.CALM)  # benign before any window
    assert posture_at(900, 1100) == int(ThreatPosture.ELEVATED)  # first window
    assert posture_at(1900, 2100) == int(ThreatPosture.HIGH)  # second window

    # and the floors rise with it. Compared as means, not maxima: a
    # hybrid-mandatory request floors at SERVE_HYBRID even under CALM
    # (that is the point of the flag), so the *maximum* floor is 2 in
    # every window and discriminates nothing.
    floors_before = [r["floor"] for r in records if r["tick"] < 400]
    floors_after = [r["floor"] for r in records if r["tick"] > 1900]
    assert np.mean(floors_after) > np.mean(floors_before)


def test_s1_baseline_stays_calm():
    """Control for the test above. Before the 2026-08-15 fix to
    `_squash_non_negative`, a plain sigmoid over non-negative features
    could never read below 0.5, so even the benign baseline sat at
    ELEVATED from step one and S2 was measuring almost nothing."""
    config = load_config({"scenario": "S1", "scenario_steps": 2500, "use_foresight": "ewma"})
    records = run_episode(config, n_steps=1200)
    assert max(int(record["posture"]) for record in records) == int(ThreatPosture.CALM)


def test_s4_requires_the_graph_source():
    """S4's flood targets a tenant, which the plain Poisson stream has
    no concept of -- so it must fail loudly rather than quietly running
    an unflooded episode."""
    config = load_config({"scenario": "S4", "request_source": "random"})
    env = SmartKeyNetEnv(config)
    with pytest.raises(ValueError, match="request_source"):
        env.reset(seed=0)


def test_s4_on_the_graph_source_runs_and_raises_load():
    flooded = run_episode(
        load_config({"scenario": "S4", "request_source": "graph", "use_foresight": "off"}),
        n_steps=1500,
    )
    calm = run_episode(
        load_config({"scenario": "S1", "request_source": "graph", "use_foresight": "off"}),
        n_steps=1500,
    )
    # the flood window is steps 600-1200; under it the env spends many
    # more decisions per tick of simulated time, so the same number of
    # decisions covers far fewer ticks
    assert flooded[-1]["tick"] < calm[-1]["tick"]


# ---------------------------------------------------------------------------
# S6 migration wave (PLAN.md §5 S6; Hard Rules 2, 3, 8)
# ---------------------------------------------------------------------------


def test_s6_carries_a_migration_schedule():
    spec = build_scenario("S6", load_config(), episode_steps=2500)
    assert spec.eval_only is True
    assert spec.migration_schedule
    assert all(isinstance(change, FloorChange) for change in spec.migration_schedule)


def test_s6_floor_overrides_accumulate_and_only_rise():
    """Hard Rule 2 across time: as the episode advances, a cohort's
    scheduled floor can only go up."""
    spec = build_scenario("S6", load_config(), episode_steps=2500)

    previous: dict[str, int] = {}
    for step in range(0, 2600, 50):
        overrides = spec.floor_overrides_at(step)
        for cohort, floor in overrides.items():
            assert int(floor) >= previous.get(cohort, -1)
            previous[cohort] = int(floor)


def test_s6_schedule_is_empty_before_the_first_phase():
    spec = build_scenario("S6", load_config(), episode_steps=2500)
    first_step = min(change.step for change in spec.migration_schedule)
    assert spec.floor_overrides_at(first_step - 1) == {}
    assert spec.floor_overrides_at(first_step) != {}


def test_a_schedule_that_lowers_a_floor_is_rejected():
    """A downgrade dressed up as config is still a downgrade."""
    config = load_config(
        {
            "migration_schedule": [
                {"step": 100, "tenant_cohort": "hospital", "new_floor": "SERVE_HYBRID"},
                {"step": 200, "tenant_cohort": "hospital", "new_floor": "SERVE_PQC"},
            ]
        }
    )
    with pytest.raises(ScenarioError, match="ratchet up"):
        build_scenario("S6", config, episode_steps=2500)


def test_s6_raises_the_targeted_cohorts_floors_in_the_environment():
    """End-to-end: the scheduled cohorts see higher floors late in the
    episode than early, and untargeted cohorts do not."""
    config = load_config(
        {"scenario": "S6", "scenario_steps": 2500, "use_foresight": "off", "request_source": "graph"}
    )
    env = SmartKeyNetEnv(config)
    state, info = env.reset(seed=0)
    rng = np.random.default_rng(0)

    floors_by_tenant_early: dict[str, list[int]] = {}
    floors_by_tenant_late: dict[str, list[int]] = {}
    for _ in range(2400):
        mask = info["action_mask"]
        tenant = env._current_request["tenant"]
        bucket = floors_by_tenant_early if env._step_count < 400 else floors_by_tenant_late
        bucket.setdefault(tenant, []).append(int(state["policy_floor"]))
        state, reward, terminated, truncated, info = env.step(
            Action(int(rng.choice(np.flatnonzero(mask))))
        )

    for cohort in ("hospital", "fintech"):
        if cohort in floors_by_tenant_early and cohort in floors_by_tenant_late:
            assert max(floors_by_tenant_late[cohort]) >= max(floors_by_tenant_early[cohort])
    # the scheduled cohorts end up at the hybrid floor
    assert max(floors_by_tenant_late.get("hospital", [0])) == int(Action.SERVE_HYBRID)


def test_s6_cannot_be_trained_on():
    """Hard Rule 8 at the training entry point."""
    from experiments.train import train

    with pytest.raises(ScenarioError):
        train(
            full_config=load_config(),
            training_overrides={"total_steps": 10, "eval_every": 10, "eval_max_steps": 10},
            scenario="S6",
        )


def test_ramped_window_builds_gradually_so_escalation_is_forecastable():
    """The fix for the project's two null results.

    A rectangular window jumps to full intensity in one step, and since
    absolute episode time is excluded from the state, nothing observable
    predicts it -- the LSTM threat head scored exactly the
    majority-class rate because there was no signal to learn. A ramp
    makes the build-up observable.
    """
    window = ThreatWindow(500, 1100, 3.0, ramp_steps=120)

    assert window.intensity_at(499) == pytest.approx(0.0)
    assert window.intensity_at(500) == pytest.approx(0.0)
    assert window.intensity_at(560) == pytest.approx(1.5, rel=0.02)   # halfway up
    assert window.intensity_at(620) == pytest.approx(3.0)             # full intensity
    assert window.intensity_at(800) == pytest.approx(3.0)             # sustained
    assert window.intensity_at(1040) == pytest.approx(1.5, rel=0.02)  # ramping out
    assert window.intensity_at(1100) == pytest.approx(0.0)


def test_ramp_never_exceeds_intensity_or_goes_negative():
    """Hard Rule 2 is unchanged by the ramp: the contribution stays
    within [0, intensity] at every step."""
    window = ThreatWindow(200, 900, 8.0, ramp_steps=150)
    for step in range(0, 1200):
        assert 0.0 <= window.intensity_at(step) <= 8.0


def test_ramp_reaches_full_intensity_by_the_window_midpoint():
    """The ramp delays full protection; it must not delay it so far
    that a window never reaches its intended floor at all."""
    for ramp in (0, 50, 120):
        window = ThreatWindow(0, 600, 5.0, ramp_steps=ramp)
        assert window.intensity_at(300) == pytest.approx(5.0)


def test_ramped_signal_has_many_distinct_levels():
    """The property that makes it learnable: the rectangular version had
    two levels and four transitions per episode."""
    spec = build_scenario("S2", load_config(), episode_steps=2500)
    levels = {round(spec.threat_boost_at(step), 3) for step in range(2500)}
    assert len(levels) > 50
