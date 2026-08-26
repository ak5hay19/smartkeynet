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
from attack.attacking_provider import AttackingForecastProvider
from env.contracts import Action, KeyType, SensitivityClass, ThreatPosture
from env.environment import _KEY_TYPE_TO_SERVE_ACTION, _LATENCY_UNITS, SmartKeyNetEnv
from env.forecast_provider import MovingAverageForecaster
from env.masking import PolicyTable
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
    timing -- this is a less coarse per-policy comparison number.

    2026-08-25 (diagnostic session) -- the coarseness above has an
    exact, confirmed mechanism, root-caused directly against real
    checkpoints (not just reasoned about): `latencies` only ever holds
    one of 4 discrete values (`_LATENCY_UNITS`: REUSE 0.2, SERVE_CLASSICAL
    1.0, SERVE_PQC 1.2, SERVE_HYBRID 1.5 -- SERVE_HYBRID's real, uncapped
    per-decision cost, not a ceiling). `np.percentile(latencies, 99)`
    uses linear interpolation; for a 250-decision episode this reads
    index `(250-1)*0.99 = 246.51`, i.e. it interpolates between the
    246th and 247th smallest values (0-indexed). Whenever at least
    `250 - 246 = 4` of the 250 decisions cost SERVE_HYBRID (>=1.6% of
    the episode), both of those values are `1.5`, so `p99_latency`
    reports exactly `1.5000` -- not a bug, a mechanical consequence of
    computing a percentile over a near-constant-above-threshold discrete
    series. Verified directly across 48 real eval episodes (masked DQN
    and soft-reward baseline, 3 training seeds x 8 eval seeds each, real
    S3 checkpoints): every episode with a SERVE_HYBRID count >= 4/250
    reported exactly `1.5000`; the one episode landing exactly at the
    boundary (SERVE_HYBRID count == 3/250) reported `1.3530`, matching
    the interpolation formula to 4 decimal places. Under S3 specifically,
    the scarcity-driven floor makes >=1.6% SERVE_HYBRID decisions typical
    for essentially any real trained policy, which is why this metric
    saturates for most policy/seed cells on S3 and is a low-information
    discriminator there -- `total_reward` and `below_floor_rate` (below,
    on `MultiSeedEvalResult`) are both already-computed, genuinely
    sharper alternatives for comparing policies on S3; report those
    instead of leaning on `p99_latency` for this kind of comparison."""


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
    """See `ScenarioResult.p99_latency`'s 2026-08-25 docstring note for
    the exact, confirmed saturation mechanism (a percentile-over-a-4-value
    -discrete-series artifact, not a cap) -- averaging across eval seeds
    here does not fix its low-information-content risk under scarcity
    scenarios (S3): most real policies there report `1.5000` for nearly
    every seed. Prefer `total_reward_mean`/`_std` or `below_floor_rate_mean`
    /`_std` below when comparing policies on S3."""
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
    below_floor_rate_mean: float
    below_floor_rate_std: float
    """2026-08-25 addition (masked-DQN-vs-soft-reward-baseline S3
    comparison session): `floor_violations / max_steps` per seed, then
    mean/std across seeds -- a RATE (PLAN.md's paper-draft "below-floor
    service rate", equation 4's numerator over its denominator), not
    the raw summed count `floor_violations_total` already reports.
    `max_steps` is read the same way `run_scenario` itself resolves it
    (`config.get("max_steps", _DEFAULT_MAX_STEPS)`) -- exactly the
    denominator each of this call's own `run_scenario` invocations
    actually used, not a separately-guessed number. Meaningful for any
    policy, masked or not: for a masked policy this is always `0.0` (the
    same Hard Rule 2 guarantee `floor_violations_total` already
    verifies, just rate-normalized); for a policy running under
    `security_masking: false` (see `env/environment.py`'s design
    decision 16), this is the direct, per-episode fraction of decisions
    served below the floor that would have applied -- computed exactly
    the same way `floor_violations` itself is (comparing each decision's
    real *delivered* tier against the real floor lookup, via
    `_delivered_tier`), never inferred from other metrics."""


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

    # Same max_steps resolution run_scenario itself used for every one of
    # the calls above (config.setdefault("max_steps", _DEFAULT_MAX_STEPS)
    # happens on run_scenario's own *copy* of config, so this file's
    # original `config` is unaffected either way -- reading it here with
    # the identical fallback is what keeps this the true denominator, not
    # a re-guessed one).
    effective_max_steps = config.get("max_steps", _DEFAULT_MAX_STEPS)
    below_floor_rates = [r.floor_violations / effective_max_steps for r in results]

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
        below_floor_rate_mean=float(np.mean(below_floor_rates)),
        below_floor_rate_std=float(np.std(below_floor_rates)),
    )


# ---------------------------------------------------------------------------
# S5 steering-attack dual-tracking measurement (PLAN.md §5; paper draft
# equation 4 -- V(pi), below-floor service rate measured against TRUE
# posture, not estimated/attacked posture). Built this session, on top of
# `attack/attacking_provider.py::AttackingForecastProvider` and
# `env/environment.py`'s new `forecast_provider_factory` injection point
# (design decision 17) -- see SESSION_LOG.md's newest entry.
# ---------------------------------------------------------------------------


@dataclass
class AttackScenarioResult:
    """One (policy, scenario, seed, alpha) run's outcome under the
    steering attack.

    `below_floor_rate_true` is the real V(pi) (equation 4): the
    fraction of decisions whose actually-DELIVERED tier fell below the
    floor the TRUE (unshaped) posture would have required at that same
    decision -- computed via a parallel, measurement-only shadow
    `MovingAverageForecaster` + `PolicyTable` fed the TRUE window every
    tick (see `AttackingForecastProvider`'s `shadow_provider`), never
    influencing the live episode itself. This is a DIFFERENT quantity
    from `ScenarioResult.floor_violations` / `MultiSeedEvalResult.
    below_floor_rate_*` above: those compare the delivered tier against
    the floor the AGENT ITSELF saw (i.e. under attack) -- trivially 0
    for any masked policy by construction, since the mask enforces
    exactly that (attacked) floor. `below_floor_rate_true` instead asks
    whether the attack succeeded at getting the agent to serve below
    what reality actually demanded -- the only quantity that can
    distinguish "the mask held against a lie it was told" (masked
    agent, expected V(pi)=0 always, Hard Rule 2) from "the mask held
    against the truth because there was no mask to fool" (soft-reward
    agent, expected V(pi) to rise with alpha).
    """

    scenario: str
    seed: int
    alpha: float
    below_floor_rate_true: float
    below_floor_true_count: int
    total_requests: int
    true_floor_log: list[Action]
    """Per-decision TRUE floor (equation 4's denominator side) -- also
    the raw material for a served-tier-vs-true-floor histogram."""
    delivered_tier_log: list[Action]
    """Per-decision actually-delivered tier -- same value
    `_delivered_tier` would compute, logged here so a caller can build
    a served-tier histogram without re-running the episode."""
    scenario_result: ScenarioResult
    """The standard metrics (p99_latency, total_reward, forced_rekey_ratio,
    regret/pool-exhaustion events, ATTACKED-floor floor_violations),
    computed identically to `run_scenario` -- for dose-response
    reporting alongside `below_floor_rate_true` above."""


def run_scenario_under_attack(
    policy: Policy, scenario: str, config: dict[str, Any], seed: int, alpha: float
) -> AttackScenarioResult:
    """Run one episode of `scenario` with `policy`, under the equation-7
    steering attack at strength `alpha`, and return its
    `AttackScenarioResult`.

    Mirrors `run_scenario` almost exactly (same episode loop, same
    metric bookkeeping) -- the two genuine differences are (1) the env
    is constructed with a `forecast_provider_factory` that substitutes
    `AttackingForecastProvider` for the real forecaster, so every
    decision the policy actually acts on is driven by the ATTACKED
    (shaped) window, and (2) a parallel, measurement-only shadow
    `PolicyTable` tracks what the floor would have been under the TRUE
    (unshaped) window at that same decision, via
    `AttackingForecastProvider.get_true_threat_forecast()` -- never
    influencing the live episode, only this function's own bookkeeping.
    Not written as a parameterized variant of `run_scenario` itself
    (e.g. an optional `alpha` there) because the shadow-tracking
    bookkeeping is genuinely new state with no analogue in the
    unattacked path, matching this file's own precedent of separate
    functions for genuinely different concerns (`run_scenario` vs.
    `evaluate_multi_seed`).
    """
    env_config = {**config, "scenario": scenario, "seed": seed}
    env_config.setdefault("max_steps", _DEFAULT_MAX_STEPS)

    shadow_provider = MovingAverageForecaster()
    attacking_provider = AttackingForecastProvider(
        base_provider=MovingAverageForecaster(), alpha=alpha, shadow_provider=shadow_provider
    )
    shadow_policy_table = PolicyTable()

    env = SmartKeyNetEnv(env_config, forecast_provider_factory=lambda _seed: attacking_provider)
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
    below_floor_true_count = 0
    true_floor_log: list[Action] = []
    delivered_tier_log: list[Action] = []

    truncated = False
    while not truncated:
        mask = info["action_mask"]
        floor = Action(state["policy_floor"])  # the ATTACKED floor -- what the agent itself saw
        key_type_onehot = state["key_type_onehot"]
        hybrid_mandatory = bool(env._current_request["hybrid_mandatory"])
        sensitivity_class = SensitivityClass(env._current_request["sensitivity_class"])

        # TRUE-posture-tracked floor for this exact decision -- read
        # BEFORE act()/step(), same relative timing env/environment.py's
        # own _prepare_decision uses for the attacked side (the
        # forecaster/policy_table state a decision is shown reflects
        # every tick's update() up to and including the one that
        # surfaced this decision, never a tick ahead).
        true_forecast = attacking_provider.get_true_threat_forecast()
        true_posture = ThreatPosture(int(np.argmax(true_forecast.posture_probs)))
        shadow_policy_table.ratchet_up(true_posture)
        true_floor = shadow_policy_table.floor(sensitivity_class, true_posture)

        action = policy.act(state, mask)

        delivered_tier = _delivered_tier(action, key_type_onehot, floor)
        if int(delivered_tier) < int(floor):
            floor_violations += 1  # against the ATTACKED floor -- same check run_scenario makes
        if int(delivered_tier) < int(true_floor):
            below_floor_true_count += 1  # equation 4's real V(pi) numerator
        true_floor_log.append(true_floor)
        delivered_tier_log.append(delivered_tier)

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

    scenario_result = ScenarioResult(
        scenario=scenario,
        seed=seed,
        episode_metrics=episode_metrics,
        p99_latency=p99_latency,
        pool_exhaustion_events=len(regret_events),
        floor_violations=floor_violations,
        total_reward=total_reward,
    )

    return AttackScenarioResult(
        scenario=scenario,
        seed=seed,
        alpha=alpha,
        below_floor_rate_true=below_floor_true_count / total_requests,
        below_floor_true_count=below_floor_true_count,
        total_requests=total_requests,
        true_floor_log=true_floor_log,
        delivered_tier_log=delivered_tier_log,
        scenario_result=scenario_result,
    )


@dataclass
class MultiSeedAttackEvalResult:
    """Mean + spread of one policy's `below_floor_rate_true` (and the
    standard scenario metrics) across several eval seeds, at one fixed
    `alpha` -- the `evaluate_multi_seed` analogue for the attacked
    path. `results` holds every real per-seed `AttackScenarioResult`,
    never discarded."""

    scenario: str
    alpha: float
    eval_seeds: list[int]
    results: list[AttackScenarioResult]
    below_floor_rate_true_mean: float
    below_floor_rate_true_std: float
    total_reward_mean: float
    total_reward_std: float
    forced_rekey_ratio_mean: float
    forced_rekey_ratio_std: float


def evaluate_attack_multi_seed(
    policy: Policy, scenario: str, config: dict[str, Any], eval_seeds: list[int], alpha: float
) -> MultiSeedAttackEvalResult:
    """Run `policy` on `scenario` under attack strength `alpha` across
    every seed in `eval_seeds` via `run_scenario_under_attack`, and
    summarize `below_floor_rate_true` (equation 4's real V(pi)) plus
    the standard metrics as mean + std -- never a bare single-seed
    point estimate, matching `evaluate_multi_seed`'s own convention.
    """
    if not eval_seeds:
        raise ValueError("eval_seeds must be non-empty")

    results = [run_scenario_under_attack(policy, scenario, config, seed, alpha) for seed in eval_seeds]

    below_floor_rates_true = [r.below_floor_rate_true for r in results]
    total_rewards = [r.scenario_result.total_reward for r in results]
    forced_rekey_ratios = [r.scenario_result.episode_metrics.forced_rekey_ratio for r in results]

    return MultiSeedAttackEvalResult(
        scenario=scenario,
        alpha=alpha,
        eval_seeds=list(eval_seeds),
        results=results,
        below_floor_rate_true_mean=float(np.mean(below_floor_rates_true)),
        below_floor_rate_true_std=float(np.std(below_floor_rates_true)),
        total_reward_mean=float(np.mean(total_rewards)),
        total_reward_std=float(np.std(total_rewards)),
        forced_rekey_ratio_mean=float(np.mean(forced_rekey_ratios)),
        forced_rekey_ratio_std=float(np.std(forced_rekey_ratios)),
    )
