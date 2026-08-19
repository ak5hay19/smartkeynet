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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.baselines import Policy
from env.contracts import Action, KeyType
from env.environment import _KEY_TYPE_TO_SERVE_ACTION, _LATENCY_MS, SmartKeyNetEnv
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
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    pool_overflow_keys: int = 0
    mean_served_tier: float = 0.0
    served_tier_hist: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    reward_terms: dict[str, float] = field(default_factory=dict)
    """Raw summed `env.step()` reward across the whole episode -- not
    weighted or normalized by anything (that's already baked into the
    reward formula's own `w_*` coefficients, see env/environment.py).
    Flagged as worth having in the 2026-08-10 epsilon-fix session:
    `p99_latency` is coarse (4 discrete values, floor-driven moments
    are environment- not policy-determined) and doesn't discriminate
    well between policies that behave very differently on rekey
    timing -- this is a less coarse per-policy comparison number."""


_ACTION_TO_TIER: dict[Action, int] = {
    Action.SERVE_CLASSICAL: 0,  # T0
    Action.SERVE_PQC: 1,  # T1
    Action.SERVE_HYBRID: 2,  # T2
}
"""`Tier` index per serve action (env/contracts.py §4.1). T3
(hybrid-with-shortened-lifetime) shares T2's key material and is not
separately reachable in this environment, so the histogram's fourth bin
is always zero -- reported anyway so the column matches §3.3's width."""


def _served_tier(action: Action, cost_action: Action, key_type_onehot: Any) -> int | None:
    """Tier actually delivered to the request.

    `REUSE` is the case worth spelling out: it delivers whatever tier the
    existing key already has, so its tier comes from the session's key
    type rather than from the action. Charging it to a fixed tier is how a
    mean-served-tier metric ends up describing the action distribution
    instead of the protection actually delivered.
    """
    if action is Action.REUSE:
        active = _active_key_tier(key_type_onehot)
        return _ACTION_TO_TIER.get(active) if active is not None else None
    return _ACTION_TO_TIER.get(cost_action)


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
    active = _active_key_tier(key_type_onehot)
    if active is not None:
        return active
    return floor  # cold-start REKEY_NOW adopts the floor's tier (design decision 4)


def _active_key_tier(key_type_onehot: Any) -> Action | None:
    """Tier of the session's existing key, or `None` on a cold start.

    Reads the same public `StateDict["key_type_onehot"]` the mask is
    built from, so the violation check below sees exactly what the
    policy saw.
    """
    # 4-wide since 2026-08-18 (spec §4.2): {none, classical, pqc, hybrid}.
    # Slot 0 is the cold-start "no key" case, which the previous 3-wide
    # encoding could not represent -- it flattened to all-zeros, identical to
    # holding a classical key, and this function silently returned None for
    # both.
    if len(key_type_onehot) >= 4:
        if key_type_onehot[0] == 1.0:
            return None
        for key_type in KeyType:
            if key_type_onehot[int(key_type) + 1] == 1.0:
                return _KEY_TYPE_TO_SERVE_ACTION[key_type]
        return None
    for key_type in KeyType:  # legacy 3-wide
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
    # The perfect-foresight oracle needs the env to peek into; every other
    # policy ignores this (see agents/mpc_oracle.py for the single method
    # where the peek happens).
    if hasattr(policy, "bind"):
        policy.bind(env)
    state, info = env.reset(seed=seed)

    latencies: list[float] = []
    regret_events: list[Any] = list(info["regret_events"])
    deferred_steps: list[Any] = list(info["deferred_critical_steps"])
    forced_rekeys: list[Any] = []
    floor_violations = 0
    served_tier_counts = [0, 0, 0, 0]
    reward_terms: dict[str, float] = {}
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
        latencies.append(_LATENCY_MS[cost_action])
        served_tier = _served_tier(action, cost_action, key_type_onehot)
        if served_tier is not None:
            served_tier_counts[served_tier] += 1

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
        for term_name, term_value in info.get("reward_terms", {}).items():
            reward_terms[term_name] = reward_terms.get(term_name, 0.0) + term_value
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
    p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
    p99_latency = float(np.percentile(latencies, 99)) if latencies else 0.0

    total_served = sum(served_tier_counts)
    if total_served:
        served_tier_hist = tuple(count / total_served for count in served_tier_counts)
        mean_served_tier = sum(
            tier * count for tier, count in enumerate(served_tier_counts)
        ) / total_served
    else:
        served_tier_hist = (0.0, 0.0, 0.0, 0.0)
        mean_served_tier = 0.0

    return ScenarioResult(
        scenario=scenario,
        seed=seed,
        episode_metrics=episode_metrics,
        p99_latency=p99_latency,
        pool_exhaustion_events=len(regret_events),
        floor_violations=floor_violations,
        total_reward=total_reward,
        p50_latency_ms=p50_latency,
        p99_latency_ms=p99_latency,
        pool_overflow_keys=env.pool_overflow_keys,
        mean_served_tier=mean_served_tier,
        served_tier_hist=served_tier_hist,  # type: ignore[arg-type]
        reward_terms=reward_terms,
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
