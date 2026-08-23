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


def _existing_tier(key_type_onehot: Any) -> Action | None:
    """The tier the session's currently-established key delivers, read
    from the public `key_type_onehot` field, or `None` on a cold start
    (no key established yet)."""
    for key_type in KeyType:
        if key_type_onehot[int(key_type)] == 1.0:
            return _KEY_TYPE_TO_SERVE_ACTION[key_type]
    return None


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
    resolving, to whichever tier it actually refreshes -- `max(existing
    session tier, floor)` (2026-08-19 fix, mirrors
    `SmartKeyNetEnv._resulting_key_type`; never lower than floor, never
    a downgrade from an existing higher tier), or the floor's tier on a
    cold start.
    """
    if action is not Action.REKEY_NOW:
        return action
    existing = _existing_tier(key_type_onehot)
    if existing is None:
        return floor  # cold-start REKEY_NOW adopts the floor's tier (design decision 4)
    return Action(max(int(existing), int(floor)))


def _delivered_tier(action: Action, key_type_onehot: Any, floor: Action) -> Action:
    """The tier `action` actually delivers to the requester -- used only
    for the `floor_violations` check below, never for cost (see
    `_resolved_cost_action`, which keeps REUSE costing against
    `Action.REUSE` itself, not a tier).

    `SERVE_CLASSICAL`/`SERVE_PQC`/`SERVE_HYBRID` deliver themselves.
    `REUSE` delivers the existing session tier unchanged -- as of the
    2026-08-19 `env/masking.py` fix, `compute_mask` masks REUSE illegal
    whenever that would be below floor, so a real run should never
    reach this branch below floor; this function still reports the
    real delivered tier regardless, so `floor_violations` verifies that
    guarantee rather than assuming it. `REKEY_NOW` delivers
    `max(existing tier, floor)`, matching `_resolved_cost_action`
    above and `SmartKeyNetEnv._resulting_key_type`.
    """
    if action in _TIER_ACTIONS:
        return action
    existing = _existing_tier(key_type_onehot)
    if existing is None:
        return floor  # cold start: REKEY_NOW adopts floor; REUSE is illegal cold-start (unreachable here)
    if action is Action.REKEY_NOW:
        return Action(max(int(existing), int(floor)))
    return existing  # REUSE


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

        # 2026-08-19 fix: this used to only check `action in
        # _TIER_ACTIONS`, so a REUSE or REKEY_NOW that delivered a tier
        # below floor was never counted -- the metric claimed a
        # guarantee ("must be 0 -- by construction") it wasn't actually
        # checking for two of five actions. Now checks every action's
        # real *delivered* tier (`_delivered_tier`), which for the
        # three tier-serving actions is just the action itself, so this
        # is a strict superset of the old check, not a different one.
        if int(_delivered_tier(action, key_type_onehot, floor)) < int(floor):
            floor_violations += 1  # should never fire -- the mask already forbids this

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


@dataclass
class MultiSeedEvalResult:
    """Mean + spread of one policy's performance across several eval
    seeds on one scenario -- the fix at the *measurement* layer for
    what 2026-08-18's checkpoint-oscillation diagnostics found: a
    single fixed `eval_seed` (as `experiments/train.py`'s
    `evaluate_against_baseline` used) is not a reliable point estimate
    of a policy's real performance, even for a policy (like a trained
    DQN) that could itself swing checkpoint-to-checkpoint -- see
    SESSION_LOG.md's 2026-08-18/2026-08-19 entries. This is
    complementary to, not a substitute for, 2026-08-19's Hard Rule 2
    masking fix at the environment layer -- that fix reduced how much
    the *policy itself* swings between training checkpoints; this
    reduces how much any *one measurement* of a fixed policy's
    performance can mislead.

    `results` holds every real per-seed `ScenarioResult`, never
    discarded -- the summary statistics below are computed from it, not
    a separate source of truth, so a caller that wants to check the
    spread's shape (not just mean/std) always has the raw runs to look
    at.
    """

    scenario: str
    eval_seeds: list[int]
    results: list[ScenarioResult]
    p99_latency_mean: float
    p99_latency_std: float
    total_reward_mean: float
    total_reward_std: float
    forced_rekey_ratio_mean: float
    forced_rekey_ratio_std: float
    regret_events_mean: float
    pool_exhaustion_events_mean: float
    floor_violations_total: int
    """Summed, not averaged -- Hard Rule 2's guarantee is that this is
    always 0, for every seed; summing (rather than meaning) makes a
    single non-zero seed impossible to average away into a
    reassuring-looking small number."""


def evaluate_multi_seed(
    policy: Policy,
    scenario: str,
    config: dict[str, Any],
    eval_seeds: list[int],
) -> MultiSeedEvalResult:
    """Run `policy` on `scenario` across every seed in `eval_seeds` via
    `run_scenario`, and summarize the metrics PLAN.md's closing table
    (and Gate W3's comparison) care about as mean + std, never a bare
    single-seed point estimate.

    Policy-agnostic (works for `GreedyDQNPolicy`, any of
    `agents/baselines.py`'s policies, or anything else satisfying the
    `Policy` protocol) and scenario-agnostic -- lives here rather than
    in `experiments/train.py` because it's a generic evaluation
    primitive, matching this file's existing role ("runs any Policy"),
    not something DQN-specific.
    """
    if not eval_seeds:
        raise ValueError("eval_seeds must be non-empty")

    results = [run_scenario(policy, scenario, config, seed) for seed in eval_seeds]

    p99_latencies = [r.p99_latency for r in results]
    total_rewards = [r.total_reward for r in results]
    forced_rekey_ratios = [r.episode_metrics.forced_rekey_ratio for r in results]
    regret_events = [r.episode_metrics.regret_events for r in results]
    pool_exhaustion_events = [r.pool_exhaustion_events for r in results]

    return MultiSeedEvalResult(
        scenario=scenario,
        eval_seeds=list(eval_seeds),
        results=results,
        p99_latency_mean=float(np.mean(p99_latencies)),
        p99_latency_std=float(np.std(p99_latencies)),
        total_reward_mean=float(np.mean(total_rewards)),
        total_reward_std=float(np.std(total_rewards)),
        forced_rekey_ratio_mean=float(np.mean(forced_rekey_ratios)),
        forced_rekey_ratio_std=float(np.std(forced_rekey_ratios)),
        regret_events_mean=float(np.mean(regret_events)),
        pool_exhaustion_events_mean=float(np.mean(pool_exhaustion_events)),
        floor_violations_total=sum(r.floor_violations for r in results),
    )
