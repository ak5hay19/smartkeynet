"""
experiments/harness.py

Comparison harness: runs any `Policy` (see `agents/baselines.py`'s
`Policy` protocol) across scenarios S1-S6 and records
`metrics/regret.py` output (PLAN.md §5 scenario grid; §10 kickoff step
6). Owned by Person C (split.md §1).

Build this *before* tuning the DQN (Hard Rule 7) -- baselines need to
be comparable from day one, not bolted on after the agent looks good.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.baselines import Policy
from env.contracts import Action, KeyType
from env.environment import _KEY_TYPE_TO_SERVE_ACTION, _LATENCY_UNITS, SmartKeyNetEnv
from metrics.regret import EpisodeMetrics, compute_episode_metrics

_DEFAULT_MAX_STEPS = 250
"""Episode length used when `config` doesn't already set `max_steps`
(env/environment.py's MDP has no natural terminal state -- `terminated`
is always False -- so a run needs an explicit truncation bound to be
"one full episode"). Mirrors the episode length split.md's Gate W2
tests already use. Callers that want a different length set
`config["max_steps"]` themselves; that value always wins (`setdefault`
below)."""

_TIER_ACTIONS = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)


@dataclass
class ScenarioResult:
    """One (policy, scenario, seed) run's outcome."""

    scenario: str
    seed: int
    episode_metrics: EpisodeMetrics
    p99_latency: float
    pool_exhaustion_events: int
    floor_violations: int  # must be 0 for any masked policy, by construction
    total_reward: float
    """Raw summed `env.step()` reward across the whole episode -- not
    weighted or normalized by anything (that's already baked into the
    reward formula's own `w_*` coefficients, see env/environment.py).
    Flagged as worth having in the 2026-08-10 epsilon-fix session:
    `p99_latency` is coarse (4 discrete values, floor-driven moments
    are environment- not policy-determined) and doesn't discriminate
    well between policies that behave very differently on rekey
    timing -- this is a less coarse per-policy comparison number."""


def _resolved_cost_action(action: Action, key_type_onehot: Any, floor: Action) -> Action:
    """Mirror `SmartKeyNetEnv._apply_action`'s `cost_action` resolution
    (env/environment.py design decision 4), computed here purely from
    what `step()`/`reset()` already hand back publicly -- the prior
    session's `key_type_onehot` (`StateDict`) and the floor for the
    request just decided (`StateDict["policy_floor"]`) -- since
    `step()`'s info dict doesn't itself surface the resolved key type
    or per-decision latency/hybrid-draw bit.

    `SERVE_CLASSICAL`/`SERVE_PQC`/`SERVE_HYBRID`/`REUSE` all cost
    against their own action directly; only `REKEY_NOW` needs
    resolving, to whichever tier it actually refreshes (the existing
    session's tier, or the floor's tier on a cold start).
    """
    if action is not Action.REKEY_NOW:
        return action
    for key_type in KeyType:
        if key_type_onehot[int(key_type)] == 1.0:
            return _KEY_TYPE_TO_SERVE_ACTION[key_type]
    return floor  # cold-start REKEY_NOW adopts the floor's tier (design decision 4)


def _active_key_tier(key_type_onehot: Any) -> Action | None:
    """Tier of the session's existing key, or `None` on a cold start.

    Reads the same public `StateDict["key_type_onehot"]` the mask is
    built from, so the violation check below sees exactly what the
    policy saw.
    """
    for key_type in KeyType:
        if key_type_onehot[int(key_type)] == 1.0:
            return _KEY_TYPE_TO_SERVE_ACTION[key_type]
    return None


def run_scenario(
    policy: Policy, scenario: str, config: dict[str, Any], seed: int
) -> ScenarioResult:
    """Run one episode of `scenario` with `policy`, return its
    `ScenarioResult`.

    `scenario` is threaded straight into the env config as
    `config["scenario"]` -- the right final interface (PLAN.md §5's
    S1-S6 grid) even though `env/environment.py` currently only
    dispatches S1 (a separate future session wires S2-S6; see
    PROGRESS.md).

    `pool_exhaustion_events` is reported as the count of `RegretEvent`s
    logged this episode: in the current environment every regret event
    *is* a pool-exhaustion event by construction (Hard Rule 9's
    pre-screen enqueues a request precisely when the pool can't cover
    its hybrid draw, or when masking leaves no legal action at all --
    see env/environment.py's module docstring, design decision 3 and
    the masking-gap-2 note) -- PLAN.md §6's demo beat describes them as
    the same on-screen moment ("red POOL EXHAUSTED event... its live
    regret counter ticks up"). Flagging this as this session's
    interpretation call, same as prior sessions' documented judgment
    calls on underspecified interfaces.
    """
    env_config = {**config, "scenario": scenario, "seed": seed}
    env_config.setdefault("max_steps", _DEFAULT_MAX_STEPS)
    env = SmartKeyNetEnv(env_config)
    state, info = env.reset(seed=seed)

    latencies: list[float] = []
    regret_events: list[Any] = list(info["regret_events"])
    deferred_steps: list[Any] = list(info["deferred_critical_steps"])
    forced_rekeys: list[Any] = []
    floor_violations = 0
    total_rekeys = 0
    total_requests = 0
    discretionary_hybrid_serves = 0
    total_reward = 0.0

    truncated = False
    while not truncated:
        mask = info["action_mask"]
        floor = Action(state["policy_floor"])
        key_type_onehot = state["key_type_onehot"]
        # env/environment.py doesn't expose `hybrid_mandatory` on
        # `StateDict` -- reading `env._current_request` here mirrors
        # tests/test_environment.py's own established precedent for
        # observability the public API doesn't (yet) surface.
        hybrid_mandatory = bool(env._current_request["hybrid_mandatory"])

        action = policy.act(state, mask)

        if action in _TIER_ACTIONS and int(action) < int(floor):
            floor_violations += 1  # should never fire -- the mask already forbids this

        # REUSE is a floor violation too, when the key being reused no
        # longer clears the floor. Counting only `_TIER_ACTIONS` is how
        # this metric reported a clean 0 right through 2026-08-15 while
        # `REUSE` was in fact bypassing floors 1,090 times in a
        # 3,000-step S2 episode -- the headline claim "floor violations:
        # 0, structurally guaranteed" was being satisfied by not
        # looking. `compute_mask` now forbids it (spec §S4 rule 4); this
        # measures the guarantee instead of assuming it.
        if action is Action.REUSE:
            active_tier = _active_key_tier(key_type_onehot)
            if active_tier is None or int(active_tier) < int(floor):
                floor_violations += 1

        cost_action = _resolved_cost_action(action, key_type_onehot, floor)
        latencies.append(_LATENCY_UNITS[cost_action])

        is_rekey = action is not Action.REUSE
        if is_rekey:
            total_rekeys += 1
            if cost_action is Action.SERVE_HYBRID and not hybrid_mandatory:
                discretionary_hybrid_serves += 1
        total_requests += 1

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        regret_events.extend(info["regret_events"])
        deferred_steps.extend(info["deferred_critical_steps"])
        if "forced_rekey" in info:
            forced_rekeys.append(info["forced_rekey"])

    episode_metrics = compute_episode_metrics(
        regret_events=regret_events,
        deferred_steps=deferred_steps,
        forced_rekeys=forced_rekeys,
        total_rekeys=total_rekeys,
        total_requests=total_requests,
        discretionary_hybrid_serves=discretionary_hybrid_serves,
    )
    p99_latency = float(np.percentile(latencies, 99)) if latencies else 0.0

    return ScenarioResult(
        scenario=scenario,
        seed=seed,
        episode_metrics=episode_metrics,
        p99_latency=p99_latency,
        pool_exhaustion_events=len(regret_events),
        floor_violations=floor_violations,
        total_reward=total_reward,
    )


def run_grid(
    policies: dict[str, Policy],
    scenarios: list[str],
    config: dict[str, Any],
    seeds: list[int],
) -> list[ScenarioResult]:
    """Run every (policy, scenario, seed) combination (PLAN.md §6
    closing table: Agent vs. always-PQC vs. always-hybrid vs.
    static-threshold vs. random, across S1-S4 + S6)."""
    results: list[ScenarioResult] = []
    for policy in policies.values():
        for scenario in scenarios:
            for seed in seeds:
                results.append(run_scenario(policy, scenario, config, seed))
    return results
