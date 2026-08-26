"""
env/environment.py

Gymnasium-style environment: reset()/step()/state/mask (PLAN.md §4
architecture diagram; §10 kickoff step 5). Owned by Person B
(split.md §1) -- this is the spine; everyone else's work is
meaningless if this is wrong.

Wires together pool_sim, deferral_queue, masking, a
ForecastProvider, and the full reward formula (PLAN.md §4):

    r = - w_lat*latency - w_en*energy + w_fr*freshness
        - w_qkd*(pool bits consumed)
        - R_starve*(deferred_critical_steps)
        - c_rekey(load)*1[rekey]

    where c_rekey(load) = c0 * (1 + beta * load)

Hard Rule 1: no security term anywhere in this formula, ever -- not
even temporarily "to help training".

---------------------------------------------------------------------
Design decisions resolved this session (flagged in SESSION_LOG.md;
summarized here for anyone reading the code cold):

1. **One env.step() = one request decision.** `random_request_generator`
   is pulled one `Request` at a time into an internal pending-request
   deque; a future graph-based `RequestGenerator.step(step)->list[Request]`
   would feed the same deque from its per-tick batches without changing
   anything downstream (Hard Rule 3).
2. **Persistent per-(tenant, service) session key state**, lazily
   created on first sight. A cold-start session (no key yet) is
   initialized with `key_age = max_key_age` -- i.e. "as stale as
   possible" -- which is what makes `REUSE` correctly illegal for a
   session that has no key to reuse (`compute_mask`'s only lever over
   `REUSE` is the age comparison; there's no separate "no key yet"
   rule, so this is how that case is folded into the existing rule
   instead of inventing a new one). Every tracked session ages by one
   step on every internal tick, not just the one being decided.
3. **Hard Rule 9 pre-screening**: a pending request is checked
   (`hybrid_mandatory and not pool_sim.can_draw(bits_per_hybrid_draw)`)
   *before* it ever reaches `compute_mask`/the agent. If it fails that
   check it's routed straight to `deferral_queue.enqueue()` (emitting
   a `RegretEvent`) and the next pending request is tried instead,
   within the same internal tick. This is what makes "floor
   violations: 0" structural rather than aspirational.
4. **Action semantics** (not fully pinned down anywhere else, resolved
   here): `SERVE_CLASSICAL`/`SERVE_PQC`/`SERVE_HYBRID` always
   (re)establish fresh key material at that tier this step (always a
   rekey, whether or not `REUSE` was still legal -- this is what gives
   the agent a real "rekey wastefully vs. reuse" tradeoff to learn).
   `REUSE` never touches key state. `REKEY_NOW` refreshes the
   session's *current* tier (age reset, same `key_type`) without
   changing it; for a cold-start session with no tier yet, it adopts
   the request's policy floor tier (the only sensible default with no
   tier of its own to refresh). A rekey is logged as a `ForcedRekey`
   event (metrics only, not extra reward) iff `REUSE` was masked this
   decision because `key_age >= max_key_age`.
5. **Reward components**: `freshness = 1 - key_age/max_key_age`
   (post-action, clipped to [0,1]); `pool bits consumed` is the actual
   hybrid draw this step; per-tier `latency`/`energy` costs are a
   small, explicitly-labeled *placeholder* cost table (not security
   constants -- PLAN.md's real numbers are meant to come from
   published liboqs/pqm4 benchmarks later); `load` is
   `min(1, (pending + deferred) / _LOAD_REFERENCE_QUEUE_DEPTH)`.
   `deferred_critical_steps` counted toward `-R_starve*(...)` are the
   ones that accrue *after* the action is applied, during this same
   `step()` call's advance-to-next-decision phase (standard Gym
   per-transition reward semantics: the reward reflects what the
   environment did as a consequence of the action).
6. **`threat_features` placeholder**: no real RT-IoT2022 feature
   source is wired yet (Person A's future dataset-ingestion session).
   `[qber, load]` stands in so the forecaster pipeline is exercised
   end-to-end; it is not a real threat signal.
7. **`ratchet_up` wiring**: every decision, the environment computes
   the current instantaneous `ThreatPosture` (always `CALM` under
   `use_foresight: off`; `argmax(posture_probs)` under `ewma`/`lstm`)
   and calls `policy_table.ratchet_up(posture)` *before* calling
   `policy_table.floor(...)` -- this is what actually exercises
   `PolicyTable`'s sticky-ratchet design from last session; without
   this call the ratchet would never advance and floors could drift
   back down with a noisy instantaneous reading.
8. **Truncation**: episodes have no natural terminal state (`terminated`
   is always `False`); an optional `config["max_steps"]` truncates
   after that many *decisions* (`step()` calls), defaulting to `None`
   (never auto-truncates -- the caller manages episode length, as the
   gate test does).
9. **`config["load_spike"]` (diagnostic stub, NOT real S4)**: an
   optional block read here and threaded straight into
   `random_request_generator`'s `load_spike` kwarg (see that
   function's docstring for the exact shape and why it's periodic, not
   a one-off window). This exists to test a narrower question than S4
   itself -- does proactive rekeying emerge at all once arrival load
   genuinely varies over time -- without needing the real tenant graph
   that a genuine S4 (a specific low-sensitivity tenant flooding the
   system) requires. `config["load_spike"]` absent, `None`, or
   `{"enabled": False, ...}` all mean "no spike" -- byte-identical to
   this key not existing at all. Nothing else about `scenario`
   dispatch changes because of this: it is orthogonal to S1-S6 and
   layers on top of whichever scenario is active (today, only S1).
10. **Real S2/S3 scenario dispatch (2026-08-19)**: `config["scenario"]`
    now genuinely gates behavior for two of the six scenarios --
    `"S2"` and `"S3"` -- everything else (including `"S1"` and the
    still-undispatched `"S4"`/`"S5"`/`"S6"`) is unaffected and remains
    exactly as before. Both are pure *input* changes into existing,
    unmodified machinery -- neither touches `env/masking.py`'s floor
    table nor adds a second code path:
    - **S2 (HNDL posture)**: `config["threat_schedule"]` (required only
      when `scenario == "S2"`; `{elevate_at_step, elevated_signal}`)
      makes `_threat_features_placeholder()` return a scripted elevated
      signal from that internal tick onward instead of the ordinary
      `[qber, load]` placeholder -- see that method's docstring. The
      elevation still flows through the same forecaster-update ->
      policy-table -> `compute_mask` chain every scenario uses.
    - **S3 (QKD degradation)**: `config["qkd_degradation"]` (required
      only when `scenario == "S3"`; `{spike_start, spike_duration,
      spike_magnitude}`) is threaded straight into
      `SyntheticSKRQBERTrace`'s existing spike parameters at `reset()`
      time -- see `_qkd_degradation_trace_kwargs()`. `env/pool_sim.py`
      already implements this degradation window; nothing new was
      built there.
    - S4 (DDoS/noisy-neighbor) and S6 (migration wave) are deliberately
      NOT dispatched this session -- both need a "which tenant is this"
      concept `env/request_generator.py`'s current random stream
      doesn't have (Hard Rule 3: the agent must never need to know
      about a tenant graph, but *some* per-tenant identity has to exist
      for "protect this tenant's pool share" or "this tenant cohort's
      floor changed" to mean anything at all) -- bolting that on ad hoc
      here would be a worse decision than making it deliberately later.
11. **REUSE/REKEY_NOW floor-enforcement gap closed (2026-08-19, Hard
    Rule 2)**: found by independent review of this repo's own masking
    logic against Hard Rule 2's actual text, not by consulting any
    other branch -- and only empirically *observable* now that S2
    (design decision 10, same day) makes floors genuinely ratchet
    mid-episode. `compute_mask`'s original three rules gated REUSE on
    key *age* only, never on whether the session's already-established
    key material still meets the floor now in effect; `REKEY_NOW`
    (this file) always refreshed at the session's *existing* tier
    verbatim, same gap. A real S2 episode under `RandomPolicy` measured
    **64 of 279 REUSE/REKEY_NOW decisions (22.9%) delivering key
    material below the request's current floor** before this fix (32
    via each action) -- see `tests/test_environment.py`'s
    reproduction. Fixed on both sides at once, deliberately not by
    trading one Hard Rule requirement for another:
    - `env/masking.py::compute_mask` gained an opt-in
      `current_key_type` parameter and a fourth rule: REUSE is illegal
      if the tier that key material delivers is below `floor`.
    - `_resulting_key_type` (below) now resolves REKEY_NOW to
      `max(existing tier, floor)`, never lower -- guaranteeing it can
      never deliver below floor while still never downgrading a
      session that's already above floor (design decision 4's "never
      downgrade" behavior survives intact, verified by test).
    - Masking gap #1's `prospective_tier` computation (in
      `_prepare_decision`) now delegates to `_resulting_key_type`
      itself instead of a separate, previously-inconsistent copy of
      the same resolution -- required, not just tidier: post-fix,
      REKEY_NOW can newly resolve to a *higher* tier than before,
      which can newly need a pool draw the old duplicated check didn't
      account for.
    - `experiments/harness.py`'s `floor_violations` counter only ever
      checked the three tier-serving actions (`action in
      _TIER_ACTIONS`) -- REUSE and REKEY_NOW were silently excluded, a
      second bug: the metric claimed a guarantee ("must be 0... by
      construction") it wasn't actually checking for two of five
      actions. Fixed alongside the above to check every action's
      actually-*delivered* tier, not just the three whose action value
      already equals their delivered tier.
    - `dashboard/explain.py`'s step-4 mask-reason logic and step-5 cost
      resolution both had to be updated to stay consistent with the
      above (see that file's own comments) -- otherwise the Explain
      Decision panel would either crash on the new illegality reason or
      display a cost for the wrong (stale) delivered tier, a real Hard
      Rule 10 drift risk this fix would otherwise have introduced.
12. **Request generator made swappable (2026-08-23)**: `__init__`
    gained an optional `request_stream_factory` parameter (`episode_seed
    -> Iterator[Request]`, default `None`). `random_request_generator`
    was hardcoded inline in `reset()` before this session (confirmed by
    reading this file, not assumed) -- this is the narrow, additive
    exception needed for `env/request_generator.py::RequestGenerator`
    (the real NetworkX-graph-driven generator, this session's own new
    code) to be a genuine drop-in replacement, per Hard Rule 3's swap
    test. `None` (the default) reproduces prior behavior exactly; no
    other line in this file branches on which generator is in use.
13. **Real S4 (DDoS/noisy-neighbor) scenario dispatch (2026-08-24)**:
    `config["ddos"]` (required only under `scenario: S4`, same
    fail-fast convention as `threat_schedule`/`qkd_degradation`) picks
    a designated tenant (`tenant_index`, resolved against a tenant
    graph built once at construction time from `ddos.graph_seed` --
    see the `__init__` comment above the graph-build call for why that
    seed is deliberately decoupled from `episode_seed`) and floods it
    for the whole episode via `RequestGenerator`'s `flood_override`
    mechanism (`env/request_generator.py`'s class docstring has the
    full mechanism and why it's an *additive* second Poisson stream,
    not a `traffic_rate` multiply, given this codebase's actual
    weighted-multinomial-split sampling model). `reset()`'s
    request-stream selection gains one new `elif self._scenario ==
    "S4"` branch, parallel to the existing `if
    self._request_stream_factory is not None` branch -- same dispatch
    site, no special-casing anywhere else in this file, masking.py, or
    the reward calculation (Hard Rule 3): a flood request is a
    completely ordinary `Request`, indistinguishable in shape from any
    other, and the mask/reward/state-construction code downstream of
    `_prepare_decision` never learns which generator produced the
    `Request` it's holding.
14. **Real S6 (migration wave) scenario dispatch (2026-08-24)**:
    `config["migration_graph_seed"]` (required only under `scenario:
    S6`, same fail-fast convention as `ddos`) builds a real tenant
    graph once at construction time, same pattern as S4 -- a seed
    dedicated to graph structure, decoupled from `episode_seed`, so
    which tenant a schedule entry's `tenant_index` names is a fixed
    structural fact of the config. `config["migration_schedule"]`
    (already a top-level key in every config, default `[]` -- S2/S3/S4
    ignore it exactly as before since `self._migration_schedule` only
    ever populates from it under `scenario: S6`) is a small, explicit
    list of `{step, tenant_index, new_sensitivity_class}` entries --
    scripted and exogenous (Hard Rule 3: not generative, not seed-
    dependent, not something the environment invents). `reset()` under
    S6 constructs a `RequestGenerator` (no `flood_override`) and keeps
    a reference to the *instance* (`self._request_generator`, not just
    the `iter()`-wrapped stream S4 discards) so `_advance_to_next_decision`
    can call its new `set_tenant_sensitivity_class` at the scripted
    step. The dispatch site is a single, unconditional loop at the top
    of the internal tick loop (`for event in self._migration_schedule:
    if event["step"] == self._step_count: ...`) -- unconditional
    because `self._migration_schedule` is `[]` for every scenario but
    S6, so the loop body never executes elsewhere; no `if self._scenario
    == "S6"` guard is needed at the dispatch site itself. Entirely
    upstream, same standard as S4 (Hard Rule 3): a request emitted
    after a ratchet is a completely ordinary `Request` carrying the new
    `sensitivity_class`, and `env/masking.py`'s floor computation
    already reads that field fresh off every incoming request (verified
    by reading `_prepare_decision` below and `env/masking.py` before
    building this -- no masking.py change was needed or made). **Hard
    Rule 8 (train/eval split)**: enforced in `experiments/train.py`,
    not here -- `config["train_eligible"]` (defaults `True`, set
    `False` only in `configs/scenarios/s6_migration.yaml`) is checked
    by `train()` before any training proceeds; this file has no notion
    of "training" vs. "eval" and doesn't need one, since
    `experiments/harness.py`'s `run_scenario`/`run_grid` are
    legitimately meant to run any policy against any scenario,
    including S6, for held-out evaluation.
16. **`security_masking` config flag (2026-08-25, soft-reward baseline
    agent session)**: added, flagged, and signed off on before being
    built (see SESSION_LOG.md's 2026-08-25 entry) -- this session found
    that `step()`'s `IllegalActionError` (below) enforces the mask
    unconditionally, at the environment boundary, regardless of which
    agent is calling. This means `agents/soft_reward_baseline.py`'s
    Noetzold-style reproduction ("no action masking -- any action is
    always available") could NOT be achieved purely by that agent's own
    action-selection code ignoring the mask, the way its reward function
    could be made to differ purely by that agent's own code -- a genuine
    surprise this file's own "do not touch without flagging" convention
    exists to catch. `config["security_masking"]` (default `True` --
    every pre-existing config/caller is silently unaffected, proven by
    regression test) is a generic, documented capability, same shape as
    `use_foresight` -- not a branch on agent identity (Hard Rule 3).
    When `False`, `_prepare_decision` calls the *same*, unmodified
    `env/masking.py::compute_mask()` with `floor=Action.SERVE_CLASSICAL`
    and `current_key_type=None` instead of the real floor/session key
    type -- this makes the floor-based legality rule and the
    REUSE-below-floor rule both no-ops (every tier clears a floor of
    SERVE_CLASSICAL), while `pool_can_draw`/`key_age`/`max_key_age` are
    passed through unchanged, so the pool-exhaustion and key-age-cap
    feasibility rules still apply -- those are physical/protocol
    constraints (Hard Rule 9's deferral semantics, the SP 800-57 age
    cap), not the security-floor restriction this flag targets, and
    lifting them too would crash `pool_sim.draw()` or contradict Hard
    Rule 9. The real floor (from the real `PolicyTable`) is still what
    `state["policy_floor"]`/`self._current_floor` carry either way, so
    REKEY_NOW still resolves to at least that floor's tier (a property
    of what REKEY_NOW *means* via `_resulting_key_type`, not a masking
    rule) and the soft-reward agent's own reward function can still
    tell, after the fact, whether a decision landed below the floor it
    bypassed. Only `configs/soft_reward_baseline.yaml` sets this
    `False`. Zero `env/masking.py` changes.
17. **`forecast_provider_factory` constructor parameter (S5 steering-
    attack dose-response sweep session)**: `__init__` gained an
    optional `forecast_provider_factory` parameter, mirroring
    `request_stream_factory` (design decision 12) exactly -- same
    shape (`episode_seed -> T | None`, default `None`), same
    justification (a hardcoded internal construction, `self._forecaster
    = self._build_forecaster()`, needed to become swappable for a
    later session that could not otherwise work), same backward-
    compatibility guarantee (`None` reproduces `reset()`'s prior
    forecaster-construction behavior byte-for-byte -- proven via a
    real stashed-diff before/after comparison, not just passing tests;
    see SESSION_LOG.md's newest entry). Unlike `request_stream_factory`,
    this session found NO pre-existing injection point for the
    forecaster (`_build_forecaster()` was the only construction site,
    hardcoded to read `config["use_foresight"]` directly) -- flagged
    before being built, per this file's own "do not touch without
    flagging" convention, and signed off on explicitly. Exists so
    `attack/attacking_provider.py::AttackingForecastProvider` (PLAN.md
    §5 S5) can be substituted in place of the real forecaster for a
    single episode without this file needing any notion that an attack
    is happening -- a caller passes e.g. `lambda seed:
    AttackingForecastProvider(MovingAverageForecaster(), alpha)`; no
    other line in this file branches on which forecaster is in use.
---------------------------------------------------------------------
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import gymnasium as gym
import networkx as nx
import numpy as np

from env.contracts import (
    Action,
    ActionMask,
    DeferredCriticalStep,
    ForcedRekey,
    ForecastObservation,
    ForecastProvider,
    KeyType,
    RegretEvent,
    Request,
    SensitivityClass,
    StateDict,
    ThreatPosture,
)
from env.deferral_queue import DeferralQueue
from env.forecast_provider import MovingAverageForecaster
from env.masking import PolicyTable, compute_mask
from env.pool_sim import PoolSim, SyntheticSKRQBERTrace
from env.request_generator import RequestGenerator, build_tenant_graph, random_request_generator


class IllegalActionError(Exception):
    """Raised by `step()` if the given action is illegal under the mask
    returned by the previous decision.

    Same philosophy as `pool_sim.PoolExhaustedError`: a caller that
    ignores the mask is a bug, and this must never fail silently --
    Hard Rule 2's structural guarantee (floors enforced by masking)
    depends on illegal actions being loudly impossible, not just
    discouraged.
    """


@dataclass
class StepResult:
    """Named return shape mirroring Gymnasium's 5-tuple, so B/C don't
    have to remember positional order when passing results around."""

    state: StateDict
    mask: ActionMask
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class _SessionKeyState:
    """Persistent per-(tenant, service) session key bookkeeping
    (design decision 2). A cold-start session (`key_type is None`) is
    initialized with `key_age` already at the staleness cap -- see
    module docstring point 2."""

    key_age: float
    key_type: KeyType | None = None


# ---------------------------------------------------------------------------
# Placeholder per-tier operating-cost tables (design decision 5). These
# are performance/cost-model constants, not security constants (Hard
# Rule 4 is about security tiers/floors) -- but they are still
# invented pending real numbers, so they're flagged the same way:
# PLAN.md's tech stack calls for "real primitive latency/energy costs...
# published liboqs/pqm4 benchmarks... measured, not invented" -- that
# hasn't happened yet. REKEY_NOW is deliberately absent: its cost is
# always looked up via whichever tier it resolves to (see
# `_resulting_key_type`), never as its own entry.
_LATENCY_UNITS: dict[Action, float] = {
    Action.REUSE: 0.2,  # cache hit, no handshake
    Action.SERVE_CLASSICAL: 1.0,
    Action.SERVE_PQC: 1.2,
    Action.SERVE_HYBRID: 1.5,
}
_ENERGY_UNITS: dict[Action, float] = {
    Action.REUSE: 0.1,
    Action.SERVE_CLASSICAL: 1.0,
    Action.SERVE_PQC: 1.3,
    Action.SERVE_HYBRID: 1.6,
}
_ACTION_TO_KEY_TYPE: dict[Action, KeyType] = {
    Action.SERVE_CLASSICAL: KeyType.CLASSICAL,
    Action.SERVE_PQC: KeyType.PQC,
    Action.SERVE_HYBRID: KeyType.HYBRID,
}
_KEY_TYPE_TO_SERVE_ACTION: dict[KeyType, Action] = {v: k for k, v in _ACTION_TO_KEY_TYPE.items()}


class SmartKeyNetEnv(gym.Env):
    """The MDP (PLAN.md §4). One agent, one MDP (Hard Rule 3).

    `config` selects the scenario (S1-S6, PLAN.md §5) and the
    `use_foresight` flag (Addition A) that determines which
    `ForecastProvider` is constructed and how long the flattened state
    vector is.

    S1 (benign baseline), S2 (HNDL posture), S3 (QKD degradation), S4
    (DDoS/noisy-neighbor), and S6 (migration wave, held-out eval only --
    Hard Rule 8) scenario dispatch are implemented (see module
    docstring design decisions 10, 13, and 15). S5 (steering attack) is
    not yet dispatched -- `config["scenario"]` is read but has no
    effect for that value, same as before.
    """

    _TRACE_N_STEPS = 200_000
    """Generously large so a realistic episode never exhausts the
    trace; `SyntheticSKRQBERTrace` is cheap to construct at this size
    (see env/pool_sim.py)."""

    _LOAD_REFERENCE_QUEUE_DEPTH = 10.0
    """Placeholder normalization constant for `load` (pending +
    deferred requests / this = 1.0 "full load"). Not a security
    constant -- an operational-load proxy, pending calibration once
    the real bounded tenant graph exists."""

    _REGRET_RECENT_WINDOW_STEPS = 20
    """Window (in internal ticks) for `StateDict.regret_event_recent`
    -- "recent" is not otherwise defined anywhere in the codebase."""

    _MAX_INTERNAL_TICKS_PER_DECISION = 100_000
    """Safety bound on `_advance_to_next_decision`'s tick loop -- pure
    defensive guard against a runaway loop from a future bug; never
    hit in practice since `random_request_generator`'s arrival rate is
    fixed and nonzero."""

    def __init__(
        self,
        config: dict[str, Any],
        request_stream_factory: Callable[[int | None], Iterator[Request]] | None = None,
        forecast_provider_factory: Callable[[int | None], ForecastProvider | None] | None = None,
    ) -> None:
        """`request_stream_factory` (design decision 12) is the one
        injection point Hard Rule 3's swap test needs: an optional
        `episode_seed -> Iterator[Request]` callable, called fresh on
        every `reset()`. Defaults to `None`, which preserves the exact
        prior behavior (`random_request_generator(seed=..., load_spike=
        self._load_spike_cfg)`) byte-for-byte -- every existing caller
        that constructs `SmartKeyNetEnv(config)` with one positional
        argument is completely unaffected. A caller wanting the real
        graph-driven stream instead passes e.g. `lambda seed:
        iter(RequestGenerator(build_tenant_graph(seed=0), seed=seed))`
        -- no other line in this file branches on which generator is
        in use.

        `forecast_provider_factory` (design decision 17) mirrors
        `request_stream_factory` exactly, same shape and same
        justification: an optional `episode_seed -> ForecastProvider |
        None` callable, called fresh on every `reset()`. Defaults to
        `None`, which preserves the exact prior behavior
        (`self._build_forecaster()`, driven entirely by
        `config["use_foresight"]`) byte-for-byte -- every existing
        caller that constructs `SmartKeyNetEnv(config)` without this
        argument is completely unaffected. A caller wanting a
        wrapped/attacked forecaster instead passes e.g. `lambda seed:
        AttackingForecastProvider(MovingAverageForecaster(), alpha)` --
        no other line in this file branches on which forecaster is in
        use. Flagged and signed off on before being built (a session
        found no existing injection point for the forecaster, unlike
        the request stream -- see SESSION_LOG.md's S5 dose-response
        sweep entry); this is the second, narrowly-scoped instance of
        the identical pattern design decision 12 established, for the
        identical reason: a hardcoded internal construction needed to
        become swappable for a later session (the steering-attack
        sweep) that could not otherwise work without it."""
        self._config = config
        self._request_stream_factory = request_stream_factory
        self._forecast_provider_factory = forecast_provider_factory
        self._pool_capacity = float(config["pool"]["capacity_bits"])
        self._pool_initial_fill_frac = float(config["pool"]["initial_fill_frac"])
        self._bits_per_hybrid_draw = float(config["pool"]["bits_per_hybrid_draw"])
        self._max_key_age = float(config["key_lifetime"]["max_key_age_steps"])
        self._reward_cfg = config["reward"]
        self._use_foresight = config.get("use_foresight", "off")
        # 2026-08-25 addition (soft-reward baseline agent session):
        # `security_masking` (default True -- every pre-existing config/
        # caller is silently unaffected) controls whether `_prepare_decision`
        # builds this episode's `ActionMask` against the real policy-table
        # floor, or against a floor-free variant (see `_prepare_decision`'s
        # own comment for exactly what "floor-free" means and why it's a
        # parameterization of the *existing*, unmodified `compute_mask()`
        # rather than a masking.py change). This is a generic, documented,
        # config-driven capability -- not a branch on which *agent* is
        # running (Hard Rule 3) -- exactly the same shape as `use_foresight`
        # above. Only `configs/soft_reward_baseline.yaml` sets this `False`;
        # it exists because `agents/soft_reward_baseline.py`'s Noetzold-style
        # reproduction needs "any action always available" to be genuinely
        # true at the environment boundary, not just in how an agent
        # chooses among an already-legal set -- `step()` raises
        # `IllegalActionError` on any action outside the current mask
        # regardless of which agent is calling, so the agent's own
        # action-selection code cannot achieve this alone. See
        # SESSION_LOG.md's 2026-08-25 entry for the full investigation that
        # led here (this was flagged and signed off before being built, per
        # this file's own "do not touch without flagging first" convention).
        self._security_masking = bool(config.get("security_masking", True))
        self._seed = config.get("seed")
        self._max_steps = config.get("max_steps")
        self._load_spike_cfg = self._build_load_spike_cfg(config.get("load_spike"))

        # Real scenario dispatch (design decision 10 -- S2/S3; design
        # decision 13 -- S4, added 2026-08-24; see module docstring).
        # `config["threat_schedule"]`/`config["qkd_degradation"]`/
        # `config["ddos"]` are required *only* when the matching
        # scenario is selected -- a plain KeyError here is the same
        # fail-fast convention as `pool`/`key_lifetime`/`reward` above,
        # not a new pattern. Any scenario string other than "S2"/"S3"/
        # "S4" (including "S1" and the not-yet-dispatched "S5"/"S6")
        # behaves exactly as before.
        self._scenario = config.get("scenario", "S1")
        self._threat_schedule_cfg = config["threat_schedule"] if self._scenario == "S2" else None
        self._qkd_degradation_cfg = config["qkd_degradation"] if self._scenario == "S3" else None
        self._ddos_cfg = config["ddos"] if self._scenario == "S4" else None
        # S6 (migration wave, design decision 15): `migration_graph_seed`
        # is required only under scenario S6 (absent from every config
        # but s6_migration.yaml, same fail-fast convention as `ddos`).
        # `migration_schedule` is already a top-level key present (with
        # an empty-list default) in every config -- see configs/default.yaml
        # -- so it's read unconditionally, but only ever non-empty for
        # S6's own config; `self._migration_schedule` collapses to `[]`
        # for every other scenario so the dispatch loop in
        # `_advance_to_next_decision` is a guaranteed no-op there.
        self._migration_graph_seed = config["migration_graph_seed"] if self._scenario == "S6" else None
        self._migration_schedule: list[dict[str, Any]] = (
            list(config.get("migration_schedule", [])) if self._scenario == "S6" else []
        )
        # The tenant graph itself is built once, here, from a seed
        # dedicated to graph structure (`ddos.graph_seed`) --
        # deliberately decoupled from `episode_seed` (design decision
        # 13). "Which tenant `tenant_index` refers to" and that
        # tenant's own `sensitivity_class` must be a fixed structural
        # fact of this config for `configs/scenarios/s4_ddos.yaml`'s
        # `tenant_index` to mean anything stable across eval seeds --
        # if the graph were rebuilt fresh per `reset(seed=...)` call
        # (like the SKR/QBER trace is), "tenant_4" could denote a
        # different, possibly high-sensitivity tenant on a different
        # seed. Reused, unmodified, across every reset() this instance
        # runs -- analogous to how `pool`'s capacity is fixed
        # structural config, not a per-episode draw.
        self._tenant_graph: nx.Graph | None = None
        if self._scenario == "S4":
            self._tenant_graph = build_tenant_graph(
                n_nodes=config["tenant_graph"]["n_nodes"], seed=self._ddos_cfg["graph_seed"]
            )
        elif self._scenario == "S6":
            self._tenant_graph = build_tenant_graph(
                n_nodes=config["tenant_graph"]["n_nodes"], seed=self._migration_graph_seed
            )

        # Populated fresh by reset(); typed here for clarity.
        self._pool_sim: PoolSim | None = None
        self._deferral_queue: DeferralQueue | None = None
        self._policy_table: PolicyTable | None = None
        self._forecaster: ForecastProvider | None = None
        self._request_stream: Iterator[Request] | None = None
        self._request_generator: RequestGenerator | None = None
        """The live `RequestGenerator` instance backing `_request_stream`
        under S6 only (design decision 15) -- kept as an instance
        reference, not discarded via a bare `iter(...)` the way S4's
        branch does, because S6 needs to call
        `set_tenant_sensitivity_class` on it mid-episode.  `None` for
        every other scenario/dispatch path."""
        self._peeked_arrival: Request | None = None

        self._sessions: dict[tuple[str, str], _SessionKeyState] = {}
        self._pending_requests: deque[Request] = deque()
        self._regret_log: list[RegretEvent] = []
        self._deferred_step_log: list[DeferredCriticalStep] = []
        self._forced_rekey_log: list[ForcedRekey] = []

        self._step_count = -1
        self._decision_count = 0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._arrivals_total = 0
        self._ticks_total = 0
        self._arrivals_per_class_accum: list[int] = [0] * len(SensitivityClass)
        self._hybrid_serves_accum = 0
        self._last_regret_step: int | None = None
        self._last_pool_state = None

        self._current_request: Request | None = None
        self._current_floor: Action | None = None
        self._current_reuse_masked_due_to_age = False
        self._current_mask: ActionMask | None = None

    # -----------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[StateDict, dict[str, Any]]:
        """Gymnasium-standard reset. Also resets pool_sim,
        deferral_queue, the request stream, and the forecast
        provider's window."""
        episode_seed = seed if seed is not None else self._seed

        trace = SyntheticSKRQBERTrace(
            n_steps=self._TRACE_N_STEPS,
            seed=episode_seed if episode_seed is not None else 0,
            **self._qkd_degradation_trace_kwargs(),
        )
        self._pool_sim = PoolSim(
            capacity=self._pool_capacity,
            trace=trace,
            initial_fill_frac=self._pool_initial_fill_frac,
        )
        self._last_pool_state = self._pool_sim.reset()
        self._deferral_queue = DeferralQueue()
        self._policy_table = PolicyTable()  # fresh every episode -- sticky ratchet must not carry over
        self._forecaster = (
            self._forecast_provider_factory(episode_seed)
            if self._forecast_provider_factory is not None
            else self._build_forecaster()
        )
        self._request_generator = None
        if self._request_stream_factory is not None:
            self._request_stream = self._request_stream_factory(episode_seed)
        elif self._scenario == "S4":
            flood_override = {
                "tenant_id": f"tenant_{self._ddos_cfg['tenant_index']}",
                "extra_rate": self._ddos_cfg["extra_rate"],
            }
            self._request_stream = iter(
                RequestGenerator(self._tenant_graph, seed=episode_seed, flood_override=flood_override)
            )
        elif self._scenario == "S6":
            # design decision 15: no flood_override -- S6 needs the
            # instance kept around (not discarded into a bare iter())
            # so _advance_to_next_decision can mutate it mid-episode.
            self._request_generator = RequestGenerator(self._tenant_graph, seed=episode_seed)
            self._request_stream = iter(self._request_generator)
        else:
            self._request_stream = random_request_generator(seed=episode_seed, load_spike=self._load_spike_cfg)
        self._peeked_arrival = None

        self._sessions = {}
        self._pending_requests = deque()
        self._regret_log = []
        self._deferred_step_log = []
        self._forced_rekey_log = []

        self._step_count = -1
        self._decision_count = 0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._arrivals_total = 0
        self._ticks_total = 0
        self._arrivals_per_class_accum = [0] * len(SensitivityClass)
        self._hybrid_serves_accum = 0
        self._last_regret_step = None

        self._current_request = None
        self._current_floor = None
        self._current_reuse_masked_due_to_age = False
        self._current_mask = None

        state, mask, tick_info = self._advance_to_next_decision()
        self._current_mask = mask

        info = {
            "action_mask": mask,
            "regret_events": tick_info["regret_events"],
            "deferred_critical_steps": tick_info["deferred_critical_steps"],
        }
        return state, info

    def step(self, action: Action) -> tuple[StateDict, float, bool, bool, dict[str, Any]]:
        """Gymnasium-standard step.

        info must include the current action_mask (masked-env
        convention: the agent needs it for the next decision) and
        any event-log entries emitted this step (RegretEvent,
        DeferredCriticalStep, ForcedRekey -- see
        env/contracts.py)."""
        if self._current_mask is None:
            raise RuntimeError("step() called before reset()")

        action = Action(int(action))
        if not bool(self._current_mask[int(action)]):
            floor_name = self._current_floor.name if self._current_floor is not None else None
            raise IllegalActionError(
                f"action {action.name} is illegal under the current mask "
                f"{self._current_mask.tolist()} (policy_floor={floor_name})"
            )

        self._decision_count += 1
        reward, action_info = self._apply_action(action)

        state, mask, tick_info = self._advance_to_next_decision()
        self._current_mask = mask

        deferred_this_step = tick_info["deferred_critical_steps"]
        reward -= self._reward_cfg["r_starve"] * len(deferred_this_step)

        info: dict[str, Any] = {
            "action_mask": mask,
            "regret_events": tick_info["regret_events"],
            "deferred_critical_steps": deferred_this_step,
        }
        if "forced_rekey" in action_info:
            info["forced_rekey"] = action_info["forced_rekey"]

        terminated = False  # no natural terminal state in this MDP
        truncated = self._max_steps is not None and self._decision_count >= self._max_steps

        return state, reward, terminated, truncated, info

    def action_mask(self) -> ActionMask:
        """Current legal-action mask, per env/masking.py. Exposed
        separately so the agent can query it without stepping."""
        if self._current_mask is None:
            raise RuntimeError("action_mask() called before reset()")
        return self._current_mask

    # -----------------------------------------------------------------
    # Internal wiring
    # -----------------------------------------------------------------

    @staticmethod
    def _build_load_spike_cfg(raw: dict[str, Any] | None) -> dict[str, float] | None:
        """Normalize `config["load_spike"]` into the shape
        `random_request_generator` expects, or `None` for "no spike"
        (design decision 9 -- diagnostic stub, not real S4). Absent,
        `None`, or `enabled: False` all collapse to `None` here so
        `reset()` doesn't need to re-check `enabled` on every episode."""
        if not raw or not raw.get("enabled", False):
            return None
        return {
            "period_steps": raw["period_steps"],
            "spike_duration_steps": raw["spike_duration_steps"],
            "spike_rate_multiplier": raw["spike_rate_multiplier"],
            "low_rate_multiplier": raw["low_rate_multiplier"],
        }

    def _qkd_degradation_trace_kwargs(self) -> dict[str, Any]:
        """S3 dispatch (design decision 10): when `scenario: S3` is
        selected, feed `self._qkd_degradation_cfg`'s spike parameters
        straight into `SyntheticSKRQBERTrace` -- its constructor already
        implements exactly this "QBER up, SKR down" degradation window
        (see env/pool_sim.py's own docstring, "the dial-in hook for the
        S3 'QKD degradation' scenario"), so this is real reuse, not a
        new mechanism. Empty for every other scenario -- `reset()`'s
        trace construction is then byte-identical to before this
        session.

        **2026-08-24 (design decision 14, Gate W3 S3 recalibration)**:
        `spike_skr_multiplier` is read only if the S3 config's
        `qkd_degradation` block sets it -- `.get(...)`, not `[...]`,
        so every existing S3 config/test that doesn't set this key
        (none did before this session) still gets `None` here, which
        `SyntheticSKRQBERTrace` treats as "use the pre-existing
        `qber`-derived formula, unchanged" (see that module's
        docstring, step 4a). Only `configs/scenarios/s3_degradation.yaml`
        sets it now."""
        if self._qkd_degradation_cfg is None:
            return {}
        kwargs: dict[str, Any] = {
            "spike_start": int(self._qkd_degradation_cfg["spike_start"]),
            "spike_duration": int(self._qkd_degradation_cfg["spike_duration"]),
            "spike_magnitude": float(self._qkd_degradation_cfg["spike_magnitude"]),
        }
        if "spike_skr_multiplier" in self._qkd_degradation_cfg:
            kwargs["spike_skr_multiplier"] = float(self._qkd_degradation_cfg["spike_skr_multiplier"])
        return kwargs

    def _build_forecaster(self) -> ForecastProvider | None:
        if self._use_foresight == "off":
            return None
        if self._use_foresight == "ewma":
            return MovingAverageForecaster()
        if self._use_foresight == "lstm":
            raise NotImplementedError(
                "use_foresight='lstm' requires forecaster.model.LSTMForecastProvider, "
                "which doesn't exist yet (a future session's work)"
            )
        raise ValueError(f"unknown use_foresight value: {self._use_foresight!r}")

    def _advance_to_next_decision(self) -> tuple[StateDict, ActionMask, dict[str, Any]]:
        """Recommended step cycle, steps 1-7: advance the simulator one
        internal tick at a time until a request is ready for a
        decision (never showing a hybrid-mandatory-but-uncoverable
        request to the agent -- Hard Rule 9)."""
        regret_events_this_call: list[RegretEvent] = []
        deferred_steps_this_call: list[DeferredCriticalStep] = []

        for _ in range(self._MAX_INTERNAL_TICKS_PER_DECISION):
            # 1. advance the pool by one tick
            self._last_pool_state = self._pool_sim.step()
            self._step_count += 1

            # S6 (migration wave) dispatch (design decision 15): a
            # guaranteed no-op unless scenario == "S6" (the only case
            # self._migration_schedule is ever non-empty) -- see module
            # docstring point 15. Applying it here, before this tick's
            # arrivals are pulled (step 5 below), means any request
            # pulled this same tick already reflects a ratchet that
            # just fired.
            for event in self._migration_schedule:
                if event["step"] == self._step_count:
                    self._request_generator.set_tenant_sensitivity_class(
                        f"tenant_{event['tenant_index']}", event["new_sensitivity_class"]
                    )

            # age every tracked session by one tick, not just the one
            # being decided (design decision 2)
            for session in self._sessions.values():
                session.key_age += 1.0

            # 2. age waiting requests
            deferred_steps = self._deferral_queue.tick(self._step_count)
            deferred_steps_this_call.extend(deferred_steps)

            # 3. requests the pool can now cover rejoin the pending queue,
            #    with priority over brand-new arrivals this tick
            servable = self._deferral_queue.pop_servable(self._pool_sim.can_draw)
            for queued in servable:
                self._pending_requests.appendleft(queued.request)

            # 4. forecast update, using signal accumulated since the
            #    last observation (see module docstring point 6 for
            #    the threat_features placeholder)
            if self._forecaster is not None:
                observation = self._build_forecast_observation()
                self._forecaster.update(observation)
                self._arrivals_per_class_accum = [0] * len(SensitivityClass)
                self._hybrid_serves_accum = 0

            # 5. pull this tick's new arrivals
            n_new = self._pull_new_arrivals()
            self._arrivals_total += n_new
            self._ticks_total += 1

            # 6/7. try pending requests until one is ready for a decision
            while self._pending_requests:
                request = self._pending_requests.popleft()
                if request["hybrid_mandatory"] and not self._pool_sim.can_draw(self._bits_per_hybrid_draw):
                    event = self._deferral_queue.enqueue(
                        request, self._bits_per_hybrid_draw, self._step_count, self._pool_sim.fill
                    )
                    self._regret_log.append(event)
                    regret_events_this_call.append(event)
                    self._last_regret_step = self._step_count
                    continue

                result = self._prepare_decision(request)
                if not isinstance(result, tuple):
                    # masking gap #2 (see _prepare_decision) -- no
                    # legal action existed; it's already enqueued.
                    regret_events_this_call.append(result)
                    continue

                state, mask = result
                info = {
                    "regret_events": regret_events_this_call,
                    "deferred_critical_steps": deferred_steps_this_call,
                }
                return state, mask, info
            # nothing decidable this tick -- advance again

        raise RuntimeError(
            f"no decidable request found within {self._MAX_INTERNAL_TICKS_PER_DECISION} internal ticks "
            "-- almost certainly a bug (arrival rate should be nonzero)"
        )

    def _pull_new_arrivals(self) -> int:
        if self._peeked_arrival is None:
            self._peeked_arrival = next(self._request_stream, None)

        n_new = 0
        while self._peeked_arrival is not None and self._peeked_arrival["step"] <= self._step_count:
            self._pending_requests.append(self._peeked_arrival)
            self._arrivals_per_class_accum[self._peeked_arrival["sensitivity_class"]] += 1
            n_new += 1
            self._peeked_arrival = next(self._request_stream, None)

        return n_new

    def _build_forecast_observation(self) -> ForecastObservation:
        return ForecastObservation(
            qber=self._last_pool_state.qber,
            skr=self._last_pool_state.skr,
            pool_fill=self._pool_sim.fill / self._pool_sim.capacity,
            arrivals_per_class=list(self._arrivals_per_class_accum),
            hybrid_serves=self._hybrid_serves_accum,
            threat_features=self._threat_features_placeholder(),
        )

    def _threat_features_placeholder(self) -> list[float]:
        """No real RT-IoT2022 threat-feature source is wired yet
        (Person A's future dataset-ingestion session) -- see module
        docstring point 6. Not a real threat signal.

        S2 dispatch (design decision 10): from
        `self._threat_schedule_cfg["elevate_at_step"]` onward, this
        returns a scripted elevated signal instead of the ordinary
        `[qber, load]` placeholder -- the *only* change S2 makes. It
        still flows through the exact same `_build_forecast_observation`
        -> `self._forecaster.update(...)` path every other scenario
        uses, so posture elevation (and the resulting floor increase)
        happens entirely through the existing, unmodified
        `MovingAverageForecaster` -> `PolicyTable.ratchet_up`/`floor`
        -> `compute_mask` chain (Hard Rule 2) -- nothing about the
        (sensitivity_class, posture) -> floor table itself changes.
        `None` for every other scenario -- behavior is byte-identical
        to before this session."""
        if self._threat_schedule_cfg is not None and self._step_count >= self._threat_schedule_cfg["elevate_at_step"]:
            signal = float(self._threat_schedule_cfg["elevated_signal"])
            return [signal, signal]
        return [self._last_pool_state.qber, self._current_load()]

    def _current_load(self) -> float:
        backlog = len(self._pending_requests) + len(self._deferral_queue)
        return min(1.0, backlog / self._LOAD_REFERENCE_QUEUE_DEPTH)

    def _prepare_decision(self, request: Request) -> tuple[StateDict, ActionMask] | RegretEvent:
        """Compute the (state, mask) pair for one candidate request -- or,
        if it turns out no action is actually safe to offer, enqueue it
        to the deferral queue and return the resulting `RegretEvent`
        instead (see the two masking-gap notes below; the caller
        distinguishes the two return shapes via `isinstance(..., tuple)`).
        """
        tenant_service = (request["tenant"], request["service"])
        session = self._sessions.get(tenant_service)
        if session is None:
            session = _SessionKeyState(key_age=self._max_key_age, key_type=None)
            self._sessions[tenant_service] = session

        if self._forecaster is None:
            threat_score = 0.0
            threat_forecast_vec: list[float] = [0.0] * 5
            pool_level_hat: list[float] = [0.0] * 3
            skr_mean_hat: list[float] = [0.0] * 3
            hybrid_demand_hat: list[float] = [0.0] * 3
            current_posture = ThreatPosture.CALM
        else:
            threat_forecast = self._forecaster.get_threat_forecast()
            pool_forecast = self._forecaster.get_pool_forecast()
            threat_score = threat_forecast.threat_score
            threat_forecast_vec = list(threat_forecast.horizon_scores)
            pool_level_hat = list(pool_forecast.pool_level_hat)
            skr_mean_hat = list(pool_forecast.skr_mean_hat)
            hybrid_demand_hat = list(pool_forecast.hybrid_demand_hat)
            current_posture = ThreatPosture(int(np.argmax(threat_forecast.posture_probs)))

        # design decision 7: exercise the sticky ratchet every decision
        self._policy_table.ratchet_up(current_posture)
        floor = self._policy_table.floor(SensitivityClass(request["sensitivity_class"]), current_posture)

        pool_can_draw_hybrid = self._pool_sim.can_draw(self._bits_per_hybrid_draw)
        reuse_masked_due_to_age = session.key_age >= self._max_key_age
        # `security_masking` (design decision, 2026-08-25): when False,
        # the mask is built as if `floor` were always SERVE_CLASSICAL (the
        # lowest tier -- every action's own tier clears it, so
        # `compute_mask`'s floor rule becomes a no-op) and with
        # `current_key_type=None` (so the REUSE-below-floor rule, gated on
        # `current_key_type is not None`, is also a no-op). `floor` itself
        # (the real one, from the real `PolicyTable`) is untouched below --
        # it still drives `state["policy_floor"]` and `self._current_floor`
        # (used by `_apply_action`/`_resulting_key_type`) exactly as before,
        # so REKEY_NOW still resolves to at least the real floor's tier
        # (that resolution is a property of what REKEY_NOW *means*, not a
        # masking rule) and the soft-reward agent's own reward function can
        # still compare a decision against the real floor it bypassed.
        # `pool_can_draw`/`key_age`/`max_key_age` are passed through
        # unchanged either way -- these are physical/protocol feasibility
        # constraints (pool exhaustion, the SP 800-57 key-age cap), not the
        # security-floor restriction this flag targets, and lifting them
        # too would crash `_apply_action`/`pool_sim.draw()` or contradict
        # Hard Rule 9's deferral semantics. See `env/environment.py`'s
        # `__init__` and SESSION_LOG.md's 2026-08-25 entry for the full
        # reasoning and the investigation that led here.
        mask = compute_mask(
            request=request,
            floor=floor if self._security_masking else Action.SERVE_CLASSICAL,
            key_age=session.key_age,
            max_key_age=self._max_key_age,
            pool_can_draw=pool_can_draw_hybrid,
            current_key_type=session.key_type if self._security_masking else None,
        )

        # Masking gap #1 (discovered via testing, not anticipated by
        # compute_mask's rules): compute_mask only knows to gate
        # SERVE_HYBRID on pool_can_draw, because it has no visibility
        # into session key state. But REKEY_NOW (design decision 4)
        # can *also* resolve to a HYBRID draw -- refreshing an
        # existing HYBRID session, adopting a HYBRID floor on a
        # cold-start session, or (since the 2026-08-19 fix, design
        # decision 11) escalating a now-stale below-floor session up to
        # a HYBRID floor -- and compute_mask's rules never gate
        # REKEY_NOW on pool_can_draw at all. Left unpatched here, the
        # agent could legally pick REKEY_NOW and
        # `_apply_action`/`pool_sim.draw()` would raise
        # `PoolExhaustedError`, or worse, we'd have to silently
        # under-draw or downgrade -- both forbidden. This is an
        # environment-level augmentation on top of compute_mask's
        # output, not a change to masking.py's rules.
        #
        # `prospective_tier` delegates to `_resulting_key_type` -- the
        # exact function `_apply_action` will use -- rather than
        # duplicating its resolution logic. Before the 2026-08-19 fix
        # these were two separate copies of "what does REKEY_NOW
        # resolve to", and the duplication mattered: with the fix in
        # place, REKEY_NOW can now resolve to a *higher* tier than the
        # session's existing one (escalating a stale tier up to floor),
        # which can newly require a pool draw a pre-fix, un-refactored
        # copy of this check would have missed.
        if bool(mask[Action.REKEY_NOW]):
            prospective_tier = self._resulting_key_type(Action.REKEY_NOW, session, floor)
            if prospective_tier is KeyType.HYBRID and not pool_can_draw_hybrid:
                mask[Action.REKEY_NOW] = False

        # Masking gap #2: once gap #1 is closed, a narrow combination
        # can still leave *zero* legal actions -- a cold-start session
        # (REUSE forced illegal, design decision 2) or an aged-out
        # HYBRID session (REUSE illegal on age) whose floor is
        # SERVE_HYBRID while the pool can't cover it: SERVE_CLASSICAL/
        # PQC are below floor, SERVE_HYBRID and now REKEY_NOW are
        # pool-gated, and REUSE is age-gated. `hybrid_mandatory` is
        # only set on *some* such requests (it's an independent random
        # field on the synthetic stream, not derived from the floor),
        # so the Hard-Rule-9 pre-screen in `_advance_to_next_decision`
        # (which only checks `hybrid_mandatory`) doesn't catch this
        # case. Rather than enumerate every combination that can reach
        # a zero-legal-action mask, defer whenever the *computed* mask
        # ends up with nothing legal -- the general form of the same
        # guarantee Hard Rule 9 asks for: never offer the agent a
        # request it cannot legally serve.
        if not bool(mask.any()):
            event = self._deferral_queue.enqueue(
                request, self._bits_per_hybrid_draw, self._step_count, self._pool_sim.fill
            )
            self._regret_log.append(event)
            self._last_regret_step = self._step_count
            return event

        key_type_onehot = [0.0, 0.0, 0.0]
        if session.key_type is not None:
            key_type_onehot[int(session.key_type)] = 1.0

        avg_latency = self._latency_sum / self._latency_count if self._latency_count > 0 else 0.0
        arrival_rate = self._arrivals_total / self._ticks_total if self._ticks_total > 0 else 0.0
        regret_event_recent = (
            self._last_regret_step is not None
            and (self._step_count - self._last_regret_step) <= self._REGRET_RECENT_WINDOW_STEPS
        )

        state = StateDict(
            threat_score=threat_score,
            threat_forecast=threat_forecast_vec,
            qber=self._last_pool_state.qber,
            skr=self._last_pool_state.skr,
            pool_fill=self._pool_sim.fill / self._pool_sim.capacity,
            arrival_rate=arrival_rate,
            load=self._current_load(),
            avg_latency=avg_latency,
            key_age=session.key_age,
            key_type_onehot=key_type_onehot,
            sensitivity_class=request["sensitivity_class"],
            policy_floor=int(floor),
            pool_level_hat=pool_level_hat,
            skr_mean_hat=skr_mean_hat,
            hybrid_demand_hat=hybrid_demand_hat,
            regret_event_recent=regret_event_recent,
        )

        self._current_request = request
        self._current_floor = floor
        self._current_reuse_masked_due_to_age = reuse_masked_due_to_age

        return state, mask

    def _resulting_key_type(self, action: Action, session: _SessionKeyState, floor: Action) -> KeyType | None:
        if action in _ACTION_TO_KEY_TYPE:
            return _ACTION_TO_KEY_TYPE[action]
        if action is Action.REKEY_NOW:
            if session.key_type is None:
                return _ACTION_TO_KEY_TYPE[floor]  # cold start: adopt the floor's tier (design decision 4)
            # 2026-08-19 fix (Hard Rule 2, design decision 11): refresh at
            # the HIGHER of the existing tier and the current floor, never
            # lower -- this is what lets "REKEY_NOW never downgrades an
            # existing higher tier" (design decision 4) and Hard Rule 2
            # ("floors ... may only be raised") both hold at the same
            # time. Previously this always kept the existing tier
            # verbatim, so a session whose floor had since ratcheted up
            # (PolicyTable's sticky, one-way ratchet) could REKEY_NOW
            # straight back into its own now-stale, below-floor tier --
            # see env/masking.py's matching `compute_mask` fix (REUSE's
            # side of the same gap) for the full rationale and the real,
            # measured violation count this closes.
            existing_tier = _KEY_TYPE_TO_SERVE_ACTION[session.key_type]
            resolved_tier = Action(max(int(existing_tier), int(floor)))
            return _ACTION_TO_KEY_TYPE[resolved_tier]
        return session.key_type  # REUSE

    def _apply_action(self, action: Action) -> tuple[float, dict[str, Any]]:
        request = self._current_request
        floor = self._current_floor
        reuse_masked_due_to_age = self._current_reuse_masked_due_to_age
        tenant_service = (request["tenant"], request["service"])
        session = self._sessions[tenant_service]

        age_before_action = session.key_age
        is_rekey = action is not Action.REUSE
        new_key_type = self._resulting_key_type(action, session, floor)

        if is_rekey:
            session.key_type = new_key_type
            session.key_age = 0.0

        bits_consumed = 0.0
        if is_rekey and new_key_type is KeyType.HYBRID:
            bits_consumed = self._bits_per_hybrid_draw
            self._pool_sim.draw(bits_consumed)
            self._hybrid_serves_accum += 1

        cost_action = _KEY_TYPE_TO_SERVE_ACTION[new_key_type] if action is Action.REKEY_NOW else action
        latency = _LATENCY_UNITS[cost_action]
        energy = _ENERGY_UNITS[cost_action]
        freshness = 1.0 - min(1.0, max(0.0, session.key_age / self._max_key_age))

        self._latency_sum += latency
        self._latency_count += 1

        load = self._current_load()
        rekey_cost = 0.0
        if is_rekey:
            rekey_cost = self._reward_cfg["c_rekey_base"] * (1.0 + self._reward_cfg["c_rekey_load_beta"] * load)

        reward = (
            -self._reward_cfg["w_lat"] * latency
            - self._reward_cfg["w_en"] * energy
            + self._reward_cfg["w_fr"] * freshness
            - self._reward_cfg["w_qkd"] * bits_consumed
            - rekey_cost
        )
        # -R_starve*deferred_critical_steps is added by the caller
        # (step()), using this-call's *following* advance-to-next-
        # decision phase -- see module docstring point 5.

        info: dict[str, Any] = {}
        if is_rekey and reuse_masked_due_to_age:
            forced_event = ForcedRekey(
                step=self._step_count,
                request_id=request["request_id"],
                key_age_at_rekey=age_before_action,
                load_at_rekey=load,
                cost=rekey_cost,
            )
            self._forced_rekey_log.append(forced_event)
            info["forced_rekey"] = forced_event

        return reward, info
