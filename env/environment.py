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
   layers on top of whichever scenario is active.
10. **Scenario dispatch (2026-08-15)**: `config["scenario"]` is now
   resolved through `env/scenarios.py` into a frozen `ScenarioSpec`
   and applied through exactly three exogenous channels -- the SKR/QBER
   trace (S3's QBER drift), the request stream (S4's tenant flood), and
   the forecaster's threat-feature input (S2's elevation windows). No
   scenario adds a branch to `step()`, and none of them is visible to
   the agent as anything other than different numbers in the same
   state vector (Hard Rule 3). `config["request_source"]` selects the
   plain Poisson stream (`"random"`, the default and previous
   behaviour) or the NetworkX tenant graph (`"graph"`).
11. **The QKD scarcity price is per KEY, not per bit (2026-08-15 fix)**.
   PLAN.md §4's reward formula reads "- w_qkd*(pool bits consumed)"
   and this module took that literally, multiplying `w_qkd` by the raw
   256-bit draw. But `configs/default.yaml` documents `w_qkd` as a
   price per key, and SMARTKEYNET_BUILD_SPEC.md §3.2 states it
   outright: "w_qkd: 1.5  # per 256-bit key consumed". Charging per bit
   made the term 256x its intended size and produced a reward in which
   **starving was cheaper than spending**: one hybrid serve cost 256
   while deferring a critical request for ten steps cost only
   `r_starve * 10 = 100`. Spec §S5 test 5 names this exact inversion --
   "If this inequality fails, the agent learns to starve instead of
   spend, and your headline result inverts."

   The bug was unobservable before the same day's scarcity
   recalibration: with the pool pinned at 100% full, no policy ever had
   to trade a key against a deferral, so the relative size of the two
   terms never influenced a decision. Both fixes are needed for either
   to matter. `_assert_reward_weights_are_sane()` now enforces the
   spec's `r_starve >= 5 * w_qkd` inequality at construction so this
   cannot silently regress.
---------------------------------------------------------------------
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from env.constants import (
    assert_consistent_with_default_config,
    handshake_energy_mj,
    handshake_latency_ms,
    key_bits,
)
from env.contracts import (
    N_ACTIONS,
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
from env.masking import PolicyTable, compute_mask, effective_floor_for
from env.pool_sim import (
    PoolSim,
    PoolState,
    SKRQBERTrace,
    SyntheticSKRQBERTrace,
    TraceSKRQBERSource,
)
from env.request_generator import (
    RequestGenerator,
    build_tenant_graph,
    random_request_generator,
)
from env.reward import (
    RewardWeights,
    assert_weights_are_sane,
    compute_reward,
)
from env.scenarios import ScenarioSpec, build_scenario
from metrics.event_log import EventLog
from metrics.reward_inputs import RewardInputs

_ARRIVAL_RATE_REFERENCE = 1.0
"""Reference arrival rate for `arrival_rate_norm` (spec §4.2). Matches
`env/request_generator.py`'s `_ARRIVAL_RATE_PER_STEP`."""

_LATENCY_REFERENCE = 100.0
"""Spec §4.2 normalises average latency per 100 ms."""

_QUEUE_REFERENCE = 20.0
"""`queue_ref` for `queue_len_norm` (spec §4.2), sized against the deferral
queue depths actually observed under S3 scarcity."""


_TIER_ACTIONS_FOR_VIOLATIONS: tuple[Action, ...] = (
    Action.SERVE_CLASSICAL,
    Action.SERVE_PQC,
    Action.SERVE_HYBRID,
)
"""Actions that deliver a tier and can therefore under-protect a request.

`REUSE` and `REKEY_NOW` are excluded because their tier is resolved to a
concrete `SERVE_*` before this check runs -- counting both would double-count
the same serve, which is precisely how this metric reported a clean zero for
weeks while REUSE was bypassing floors."""


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
def _cost_table(per_tier: dict[str, float]) -> dict[Action, float]:
    """Map `constants.yaml`'s tier names onto the actions that incur them."""
    return {
        Action.REUSE: per_tier["reuse"],
        Action.SERVE_CLASSICAL: per_tier["classical"],
        Action.SERVE_PQC: per_tier["pqc"],
        Action.SERVE_HYBRID: per_tier["hybrid"],
    }


_LATENCY_MS: dict[Action, float] = _cost_table(handshake_latency_ms())
"""Per-tier handshake latency in ms, LOADED FROM configs/constants.yaml.

These were hardcoded here (as dimensionless 0.2/1.0/1.2/1.5 multipliers until
2026-08-19, then as ms literals) which meant the cost model was the one part
of the simulator with no cited source at all -- exactly what Hard Rule 4
exists to prevent. They now come from the citation-bearing file, whose entry
is explicit that this is an ordinal model and NOT measured on the evaluation
host.
"""

_ENERGY_MJ: dict[Action, float] = _cost_table(handshake_energy_mj())
"""Per-tier energy in mJ, from configs/constants.yaml. Ordinal, not measured."""
_ACTION_TO_KEY_TYPE: dict[Action, KeyType] = {
    Action.SERVE_CLASSICAL: KeyType.CLASSICAL,
    Action.SERVE_PQC: KeyType.PQC,
    Action.SERVE_HYBRID: KeyType.HYBRID,
}
_KEY_TYPE_TO_SERVE_ACTION: dict[KeyType, Action] = {v: k for k, v in _ACTION_TO_KEY_TYPE.items()}


class SmartKeyNetEnv(gym.Env[StateDict, int]):
    """The MDP (PLAN.md §4). One agent, one MDP (Hard Rule 3).

    `config` selects the scenario (S1-S6, PLAN.md §5) and the
    `use_foresight` flag (Addition A) that determines which
    `ForecastProvider` is constructed and how long the flattened state
    vector is.

    Scenario dispatch covers S1-S4 (see design decision 10). S5 and S6
    resolve to eval-only specs that currently carry no perturbations:
    S5's adversarial trace lives in `attack/` and S6's migration
    schedule wiring are both future sessions' work.
    """

    _EVAL_SPLIT: str = "eval"
    """Scenarios marked `eval_only` (S5, S6) read the held-out 30% of the
    trace. Tying the split to the scenario's own eval-only flag means the two
    cannot disagree."""

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

    _SCENARIO_REFERENCE_STEPS = 2_500
    """Default episode length that step-indexed scenario schedules are
    written against, overridable via `config["scenario_steps"]`.

    S3's QBER drift ramp occupies the middle third of *this* many
    steps, so it must not be confused with `config["max_steps"]` (which
    counts agent decisions and is often much shorter during evaluation).
    2500 steps is ~21 pool refill-from-empty times under the calibrated
    physics -- SMARTKEYNET_BUILD_SPEC.md §7.1 fix A asks for at least
    20x, "so a bad spend has visible consequences within the episode".
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._pool_capacity = float(config["pool"]["capacity_bits"])
        self._pool_initial_fill_frac = float(config["pool"]["initial_fill_frac"])
        self._bits_per_hybrid_draw = float(config["pool"]["bits_per_hybrid_draw"])
        # Whole keys per hybrid establishment -- the unit the pool and the
        # §4.4 event log both speak in (spec `hybrid_draw_keys`).
        # §S10: `masking.enabled: false` makes floors advisory so the
        # soft-reward victim can actually violate one. Default TRUE -- this is
        # the project's central guarantee, and it must take a deliberate config
        # change to switch off, never a default or an accident.
        self._masking_enabled = bool(config.get("masking", {}).get("enabled", True))
        self._floor_violations = 0

        queue_cfg = config.get("queue", {})
        self._head_reservation = str(queue_cfg.get("head_reservation", "none"))
        if self._head_reservation not in ("none", "strict"):
            raise ValueError(
                f"queue.head_reservation must be 'none' or 'strict', got {self._head_reservation!r}"
            )
        self._sla_max_steps = int(queue_cfg.get("sla_max_steps", 100))
        self._min_rekey_interval = float(
            config.get("key_lifetime", {}).get("min_rekey_interval_steps", 0.0)
        )

        self._keys_per_hybrid_draw = max(1, int(round(self._bits_per_hybrid_draw / key_bits())))
        self._max_key_age = float(config["key_lifetime"]["max_key_age_steps"])
        # Hard Rule 4: the cited constants must still describe the values in
        # use. Checked here rather than in review, because the two config
        # files duplicate four numbers between them.
        assert_consistent_with_default_config()

        self._reward_cfg = config["reward"]
        self._reward_weights = RewardWeights.from_config(self._reward_cfg)
        assert_weights_are_sane(self._reward_weights)
        self._use_foresight = config.get("use_foresight", "off")
        self._seed = config.get("seed")
        self._max_steps = config.get("max_steps")
        self._load_spike_cfg = self._build_load_spike_cfg(config.get("load_spike"))

        # --- scenario dispatch (design decision 10) ---
        self._qkd_cfg = config.get("qkd", {})
        self._request_source = config.get("request_source", "random")
        self._scenario_steps = int(config.get("scenario_steps", self._SCENARIO_REFERENCE_STEPS))
        self._scenario: ScenarioSpec = build_scenario(
            config.get("scenario", "S1"), config, self._scenario_steps
        )
        # S5 only. Injected by the attack experiment rather than read
        # from YAML, because it is a per-run adversary parameter, not a
        # deployment setting. `None` in every other scenario.
        self._steering_trace = config.get("steering_trace")

        # --- real threat features (RT-IoT2022) ---
        # `threat_source: rt_iot2022` replaces the synthetic placeholder with
        # real captured flow features. `threat_split` selects the train or
        # eval row pool so an agent is never evaluated on the same flows the
        # forecaster trained against.
        self._threat_source_name = config.get("threat_source", "synthetic")
        self._threat_split = config.get("threat_split", "train")
        self._threat_source = None
        self._threat_scorer = None
        if self._threat_source_name == "rt_iot2022":
            from env.threat_source import RTIoT2022ThreatSource, fit_graded_threat_scorer

            self._threat_source = RTIoT2022ThreatSource(split=self._threat_split, seed=self._seed)
            # Always fitted on the TRAIN pool, whichever split the episode
            # samples from -- fitting on eval flows would leak.
            self._threat_scorer = fit_graded_threat_scorer(
                RTIoT2022ThreatSource(split="train", seed=0)
            )
        elif self._threat_source_name != "synthetic":
            raise ValueError(
                f"unknown threat_source {self._threat_source_name!r} -- "
                "expected 'synthetic' or 'rt_iot2022'"
            )

        # Populated fresh by reset(). DECLARED, not assigned: these have no
        # meaningful pre-reset value, so binding them to None would force every
        # one of the ~40 use sites to narrow a type that is never actually
        # None in practice. Touching one before `reset()` now raises
        # AttributeError naming the attribute, which is a better error than an
        # `AttributeError: 'NoneType' object has no attribute ...` anyway.
        # `step()` and `action_mask()` still guard explicitly via
        # `_current_mask`, which IS legitimately None before the first reset.
        self._pool_sim: PoolSim
        self._deferral_queue: DeferralQueue
        self._policy_table: PolicyTable
        self._request_stream: Iterator[Request]
        self._last_pool_state: PoolState
        self._forecaster: ForecastProvider | None = None
        self._peeked_arrival: Request | None = None

        self._sessions: dict[tuple[str, str], _SessionKeyState] = {}
        self._pending_requests: deque[Request] = deque()
        self._regret_log: list[RegretEvent] = []
        self._deferred_step_log: list[DeferredCriticalStep] = []
        self._forced_rekey_log: list[ForcedRekey] = []

        self._step_count = -1
        self._decision_count = 0
        # §4.4 event log. One per episode, written to events.jsonl.gz by the
        # harness. The dashboard consumes ONLY this -- see dashboard/replay.py
        # -- so it can replay any recorded run offline and cannot slow
        # training down by reaching into the env.
        self._episode_index = getattr(self, "_episode_index", -1) + 1
        self._event_log = EventLog(episode=self._episode_index)
        self._pool_keys_before_refill: int | None = None
        self._resolved_waits: dict[str, int] = {}
        # One `floor_change` per cohort per episode, not one per request that
        # happens to hit the raised floor -- the event describes the schedule
        # firing, not each consequence of it.
        self._floor_changes_logged: set[str] = set()
        self._last_threat_score: float = 0.0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._latency_samples: list[float] = []
        self._served_tier_counts = [0] * N_ACTIONS
        self._reward_terms_accum = {
            "latency": 0.0,
            "energy": 0.0,
            "freshness": 0.0,
            "qkd": 0.0,
            "starve": 0.0,
            "rekey": 0.0,
        }
        self._arrivals_total = 0
        self._ticks_total = 0
        self._arrivals_per_class_accum: list[int] = [0] * len(SensitivityClass)
        self._hybrid_serves_accum = 0
        self._last_regret_step: int | None = None

        self._current_request: Request | None = None
        self._current_posture: ThreatPosture = ThreatPosture.CALM
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

        trace = self._build_trace(episode_seed if episode_seed is not None else 0)
        self._pool_sim = PoolSim(
            capacity=self._pool_capacity,
            trace=trace,
            initial_fill_frac=self._pool_initial_fill_frac,
        )
        self._last_pool_state = self._pool_sim.reset()
        self._deferral_queue = DeferralQueue()
        self._policy_table = (
            PolicyTable()
        )  # fresh every episode -- sticky ratchet must not carry over
        self._forecaster = self._build_forecaster()
        self._request_stream = self._build_request_stream(episode_seed)
        self._peeked_arrival = None

        self._sessions = {}
        self._pending_requests = deque()
        self._regret_log = []
        self._deferred_step_log = []
        self._forced_rekey_log = []

        self._step_count = -1
        self._decision_count = 0
        # §4.4 event log. One per episode, written to events.jsonl.gz by the
        # harness. The dashboard consumes ONLY this -- see dashboard/replay.py
        # -- so it can replay any recorded run offline and cannot slow
        # training down by reaching into the env.
        self._episode_index = getattr(self, "_episode_index", -1) + 1
        self._event_log = EventLog(episode=self._episode_index)
        self._pool_keys_before_refill = None
        self._resolved_waits = {}
        # One `floor_change` per cohort per episode, not one per request that
        # happens to hit the raised floor -- the event describes the schedule
        # firing, not each consequence of it.
        self._floor_changes_logged = set()
        self._last_threat_score = 0.0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._latency_samples = []
        self._served_tier_counts = [0] * N_ACTIONS
        self._reward_terms_accum = {
            "latency": 0.0,
            "energy": 0.0,
            "freshness": 0.0,
            "qkd": 0.0,
            "starve": 0.0,
            "rekey": 0.0,
        }
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

    def step(  # type: ignore[override]  # narrows gym.Env's `int` to `Action`
        self, action: Action
    ) -> tuple[StateDict, float, bool, bool, dict[str, Any]]:
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
        outcome, action_info = self._apply_action(action)

        state, mask, tick_info = self._advance_to_next_decision()
        self._current_mask = mask

        deferred_this_step = tick_info["deferred_critical_steps"]

        # Hard Rule 1 lives here: this is the complete set of things the
        # reward is allowed to see. `RewardInputs` is frozen and carries no
        # security field, so the reward physically cannot condition on
        # threat, posture, floor or tier -- it is not a matter of this call
        # site being careful.
        reward_inputs = RewardInputs(
            latency_ms=outcome["latency_ms"],
            energy_mj=outcome["energy_mj"],
            key_age_steps=outcome["key_age_steps"],
            key_lifetime_cap_steps=int(self._max_key_age),
            qkd_keys_consumed=outcome["qkd_keys_consumed"],
            deferred_critical_steps=len(deferred_this_step),
            did_rekey=outcome["did_rekey"],
            normalised_load=outcome["normalised_load"],
        )
        reward, reward_terms = compute_reward(reward_inputs, self._reward_weights)
        for term_name, term_value in reward_terms.items():
            self._reward_terms_accum[term_name] += term_value

        info: dict[str, Any] = {
            "action_mask": mask,
            "regret_events": tick_info["regret_events"],
            "deferred_critical_steps": deferred_this_step,
            "reward_terms": reward_terms,
        }
        if "forced_rekey" in action_info:
            info["forced_rekey"] = action_info["forced_rekey"]

        terminated = False  # no natural terminal state in this MDP
        truncated = self._max_steps is not None and self._decision_count >= self._max_steps

        return state, reward, terminated, truncated, info

    def _emit_defer_onset(self, event: RegretEvent, request: Request) -> None:
        """§4.4 `defer_onset` -- which **is** the regret event.

        Emitted once per request at deferral onset, never once per waiting
        step: §S2 test 2 calls that miscount "the single most common", and it
        would inflate the headline regret figure by the queue's dwell time.
        """
        self._event_log.emit(
            "defer_onset",
            event["step"],
            request_id=event["request_id"],
            tenant=event["tenant"],
            sensitivity_class=int(event["sensitivity_class"]),
            floor=int(event["policy_floor"]),
            keys_required=int(self._keys_per_hybrid_draw),
            pool_keys=int(self._pool_sim.level),
            queue_position=len(self._deferral_queue),
        )

    @property
    def event_log(self) -> EventLog:
        """This episode's §4.4 event log.

        The dashboard reads the written log, never this object or any other
        env attribute (§S13). Exposed so the harness can persist it.
        """
        return self._event_log

    def write_event_log(self, path: str | Path) -> Path:
        """Persist the episode's events to gzipped JSONL."""
        return self._event_log.write(path)

    @property
    def floor_violations(self) -> int:
        """Serves that went out below the request's policy floor.

        Structurally ZERO whenever masking is enabled -- the action never
        reaches the environment, because it was never in the agent's action
        set. Non-zero only under §S10's advisory-floor mode, which exists so
        the soft-reward victim can demonstrate the failure our architecture
        makes unreachable.
        """
        return self._floor_violations

    @property
    def masking_enabled(self) -> bool:
        return self._masking_enabled

    @property
    def pool_overflow_keys(self) -> int:
        """Keys the QKD link produced this episode that the pool was too
        full to hold (§3.3 `pool_overflow_keys`).

        Reported because it is a second, independent axis of evidence:
        an always-PQC policy never draws, so it wastes the entire link
        output and shows huge overflow with zero regret, while
        always-hybrid shows zero overflow and heavy regret. A policy
        doing real inventory control should show little of either, and
        no single-number metric can say that.
        """
        return self._pool_sim.overflow_keys_total

    @property
    def reward_terms_total(self) -> dict[str, float]:
        """Per-term reward totals for the episode so far (§3.3
        `reward_terms`). Spec §S5 point 1: log every term separately,
        because reading which term dominates is most of §7's debugging."""
        return dict(self._reward_terms_accum)

    @property
    def latency_samples_ms(self) -> list[float]:
        """Per-decision realised latency in ms, for the p50/p99 columns."""
        return list(self._latency_samples)

    def action_mask(self) -> ActionMask:
        """Current legal-action mask, per env/masking.py. Exposed
        separately so the agent can query it without stepping."""
        if self._current_mask is None:
            raise RuntimeError("action_mask() called before reset()")
        return self._current_mask

    # -----------------------------------------------------------------
    # Internal wiring
    # -----------------------------------------------------------------

    def _unused_assert_reward_weights_are_sane(self) -> None:
        """Load-time guard on the reward's internal balance.

        SMARTKEYNET_BUILD_SPEC.md §S5 test 5: "the reward of 'defer one
        critical request for 10 steps' must be worse than 'spend 1 key'
        for any config satisfying `r_starve >= 5 * w_qkd`. Assert the
        config satisfies it at load time. **If this inequality fails,
        the agent learns to starve instead of spend, and your headline
        result inverts.**"

        This is not a security constraint and does not touch Hard Rule
        1 -- it is an ordering constraint between two purely
        operational costs (a resource price and an availability
        penalty). See design decision 11 for the units bug that made
        this guard necessary.
        """
        r_starve = float(self._reward_cfg["r_starve"])
        w_qkd = float(self._reward_cfg["w_qkd"])
        if r_starve < 5.0 * w_qkd:
            raise ValueError(
                f"reward.r_starve ({r_starve}) must be at least 5 x reward.w_qkd "
                f"({w_qkd}) -- otherwise deferring a critical request is cheaper than "
                "spending a key, and the agent learns to starve instead of spend "
                "(SMARTKEYNET_BUILD_SPEC.md §S5 test 5)"
            )

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

    def _build_trace(self, episode_seed: int) -> SKRQBERTrace:
        """Construct the SKR/QBER trace for this episode, applying the
        scenario's QBER drift schedule if it has one (S3).

        SKR/QBER process parameters come from `config["qkd"]` -- see
        that block in `configs/default.yaml` for the scarcity
        calibration they encode. Missing keys fall back to
        `SyntheticSKRQBERTrace`'s own defaults, which are the same
        calibrated values, so a slimmed-down test config still runs.
        """
        # `qkd.source: trace` loads a CSV instead, with the split §S1 demands:
        # training scenarios see the first 70% and evaluation the last 30%, so
        # "reusing the same trace segment for train and eval" -- which the spec
        # calls "a silent leak that a reviewer will find" -- cannot happen by
        # forgetting, only by deliberately passing the wrong split.
        if str(self._qkd_cfg.get("source", "synthetic")) == "trace":
            return TraceSKRQBERSource(
                trace_path=self._qkd_cfg["trace_path"],
                n_steps=self._TRACE_N_STEPS,
                split=self._EVAL_SPLIT if self._scenario.eval_only else "train",
                train_fraction=float(self._qkd_cfg.get("train_fraction", 0.70)),
                seed=episode_seed,
            )

        defaults = SyntheticSKRQBERTrace(n_steps=1)
        return SyntheticSKRQBERTrace(
            n_steps=self._TRACE_N_STEPS,
            mean_skr_kbps=float(self._qkd_cfg.get("mean_skr_kbps", defaults.mean_skr_kbps)),
            skr_noise_frac=float(self._qkd_cfg.get("skr_noise_frac", defaults.skr_noise_frac)),
            baseline_qber=float(self._qkd_cfg.get("baseline_qber", defaults.baseline_qber)),
            qber_noise_std=float(self._qkd_cfg.get("qber_noise_std", defaults.qber_noise_std)),
            qber_abort=float(self._qkd_cfg.get("qber_abort", defaults.qber_abort)),
            gate_kappa=float(self._qkd_cfg.get("gate_kappa", defaults.gate_kappa)),
            skr_ou_theta=float(self._qkd_cfg.get("skr_ou_theta", defaults.skr_ou_theta)),
            skr_ou_sigma=float(self._qkd_cfg.get("skr_ou_sigma", defaults.skr_ou_sigma)),
            drift=self._scenario.qber_drift,
            seed=episode_seed,
        )

    def _build_request_stream(self, episode_seed: int | None) -> Iterator[Request]:
        """Construct this episode's request stream.

        Two interchangeable sources, selected by
        `config["request_source"]`:

          - `"random"` (default): `random_request_generator`, the plain
            stationary Poisson process. Unchanged behaviour.
          - `"graph"`: the NetworkX tenant graph sampled by
            `RequestGenerator`, with tenant-level MMPP bursts and
            tenant-conditioned sensitivity classes.

        The two are drop-in substitutes behind the same
        `Iterator[Request]` contract -- that substitutability *is* the
        Hard Rule 3 test ("deleting the NetworkX graph and replacing it
        with a plain arrival process must not change one line of agent
        code"). Note the direction: the plain process is the
        substitute, the graph is the real source.

        S4's tenant flood is only meaningful on the graph source, since
        the plain process has no tenant structure to target; requesting
        S4 with `request_source: random` therefore raises rather than
        silently running an unflooded episode.
        """
        if self._request_source == "random":
            if self._scenario.tenant_flood is not None:
                raise ValueError(
                    f"scenario {self._scenario.name} needs a per-tenant flood, which "
                    "request_source='random' cannot express (it has no tenant structure). "
                    "Set request_source: graph."
                )
            return random_request_generator(seed=episode_seed, load_spike=self._load_spike_cfg)

        if self._request_source == "graph":
            graph = build_tenant_graph(
                n_nodes=int(self._config.get("tenant_graph", {}).get("n_nodes", 50)),
                seed=episode_seed,
            )
            generator = RequestGenerator(
                graph, seed=episode_seed, tenant_flood=self._scenario.tenant_flood
            )
            return generator.as_stream()

        raise ValueError(
            f"unknown request_source {self._request_source!r} -- expected 'random' or 'graph'"
        )

    def _build_forecaster(self) -> ForecastProvider | None:
        if self._use_foresight == "off":
            return None
        if self._use_foresight == "ewma":
            return MovingAverageForecaster()
        if self._use_foresight == "lstm":
            # Imported lazily: `forecaster/` pulls in the agents package
            # for rollout collection, and importing it at module scope
            # would make `env/` depend on `agents/` -- the wrong
            # direction, and one the Hard Rule 3 graph-agnosticism test
            # would rightly object to.
            from forecaster.model import LSTMForecastProvider

            checkpoint = self._config.get("forecaster_checkpoint", "checkpoints/forecaster.pt")
            if not Path(checkpoint).exists():
                raise FileNotFoundError(
                    f"use_foresight='lstm' needs a trained forecaster at {checkpoint}. "
                    "Train one with:  .venv/bin/python -m forecaster.train"
                )
            return LSTMForecastProvider.from_checkpoint(checkpoint)
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
            self._event_log.emit(
                "pool_refill",
                self._step_count,
                keys_added=int(self._last_pool_state.keys - self._pool_keys_before_refill)
                if self._pool_keys_before_refill is not None
                else int(self._last_pool_state.keys),
                skr_kbps=float(self._last_pool_state.skr),
                qber=float(self._last_pool_state.qber),
                pool_keys_after=int(self._last_pool_state.keys),
                overflow_keys=int(self._last_pool_state.overflow_keys),
                # Additive beyond §4.4's listed payload: without capacity the
                # dashboard cannot turn `pool_keys_after` into a fill fraction
                # without importing the env, which §S13 forbids.
                pool_capacity_keys=int(self._last_pool_state.capacity_keys),
            )
            self._pool_keys_before_refill = int(self._last_pool_state.keys)
            if self._last_pool_state.keys == 0:
                self._event_log.emit(
                    "pool_exhausted",
                    self._step_count,
                    pool_keys=0,
                    pending_hybrid_mandatory=len(self._deferral_queue),
                )

            # age every tracked session by one tick, not just the one
            # being decided (design decision 2)
            for session in self._sessions.values():
                session.key_age += 1.0

            # 2. age waiting requests
            deferred_steps = self._deferral_queue.tick(self._step_count)
            deferred_steps_this_call.extend(deferred_steps)
            for deferred in deferred_steps:
                self._event_log.emit(
                    "defer_step",
                    self._step_count,
                    request_id=deferred["request_id"],
                    wait_steps_so_far=int(deferred["steps_waited"]),
                )

            # 3. requests the pool can now cover rejoin the pending queue,
            #    with priority over brand-new arrivals this tick
            servable = self._deferral_queue.pop_servable(self._pool_sim.can_draw)
            for queued in servable:
                self._pending_requests.appendleft(queued.request)
                self._resolved_waits[queued.request["request_id"]] = int(
                    self._step_count - queued.step_enqueued
                )
                self._event_log.emit(
                    "defer_resolved",
                    self._step_count,
                    request_id=queued.request["request_id"],
                    total_wait_steps=int(self._step_count - queued.step_enqueued),
                    keys_drawn=int(self._keys_per_hybrid_draw),
                )

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
                if request["hybrid_mandatory"] and not self._pool_sim.can_draw(
                    self._bits_per_hybrid_draw
                ):
                    event = self._deferral_queue.enqueue(
                        request, self._bits_per_hybrid_draw, self._step_count, self._pool_sim.fill
                    )
                    self._regret_log.append(event)
                    regret_events_this_call.append(event)
                    self._last_regret_step = self._step_count
                    self._emit_defer_onset(event, request)
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

        The scenario's threat boost (S2's elevation windows) is
        appended as a third feature rather than added into the existing
        two, so the exogenous signal stays separable from the
        environment's own observations. It is non-negative by
        construction (`env/scenarios.py` validates this), so it can
        only ever raise the derived posture -- Hard Rule 2.
        """
        if self._threat_source is not None:
            features = self._real_threat_features()
        else:
            # Last element is the scalar threat summary in [0, 1] -- see
            # `_real_threat_features` for the convention. The synthetic boost
            # is normalised so both sources agree on that contract.
            features = [
                self._last_pool_state.qber,
                self._current_load(),
                min(
                    1.0,
                    self._scenario.threat_boost_at(self._step_count) / self._MAX_THREAT_BOOST,
                ),
            ]
        if self._steering_trace is not None:
            # S5: the adversary's influence enters HERE and nowhere else
            # -- on the telemetry the forecaster reads. It never touches
            # the policy table, the mask, or the pool. That containment
            # is the whole experiment: the attack gets exactly the
            # access a real adversary would plausibly have, and the
            # question is what it can do with it.
            features = self._steering_trace.apply(features, self._step_count)
        return features

    _MAX_THREAT_BOOST = 8.0
    """Largest intensity any scenario threat window reaches (env/scenarios.py
    `_S2_THREAT_WINDOWS`). Used only to normalise the boost into the [0, 1]
    mixing intensity below."""

    def _real_threat_features(self) -> list[float]:
        """Real RT-IoT2022 flow features for this step.

        The scenario says *how escalated* things are; the dataset says *what
        that looks like on the wire*. The scenario's threat boost is
        normalised to an intensity in [0, 1] and turned into mixing weights
        over the three class pools:

            intensity 0.0  ->  all benign IoT traffic
            intensity 0.5  ->  benign mixed with reconnaissance scans
            intensity 1.0  ->  predominantly active attack traffic

        Mixing rather than switching is deliberate. A real escalation does not
        replace normal traffic; attack flows appear *alongside* it and grow as
        a share. That gradual change is exactly the structure the forecaster
        has to pick up on, and it is what makes reconnaissance a usable
        leading indicator for the attack that follows.

        The returned vector is the 8 standardised flow features plus the
        scalar discriminant score. The scalar is appended because the EWMA
        fallback averages this vector, and the raw features are not monotone
        in threat -- see `ThreatScorer` for the measurement.
        """
        intensity = min(
            1.0, self._scenario.threat_boost_at(self._step_count) / self._MAX_THREAT_BOOST
        )
        calm_weight = max(0.0, 1.0 - 2.0 * intensity)
        elevated_weight = 1.0 - abs(2.0 * intensity - 1.0)
        high_weight = max(0.0, 2.0 * intensity - 1.0)

        if self._threat_source is None or self._threat_scorer is None:
            raise RuntimeError(
                "threat_source: rt_iot2022 was selected but the source failed to load"
            )
        sampled = self._threat_source.sample_mixture(calm_weight, elevated_weight, high_weight)
        threat_score = self._threat_scorer.score(sampled)
        return [float(value) for value in sampled] + [threat_score]

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
            posture_probs_vec = [1.0] + [0.0] * (len(ThreatPosture))
            skr_trend = 0.0
        else:
            threat_forecast = self._forecaster.get_threat_forecast()
            pool_forecast = self._forecaster.get_pool_forecast()
            threat_score = threat_forecast.threat_score
            threat_forecast_vec = list(threat_forecast.horizon_scores)
            pool_level_hat = list(pool_forecast.pool_level_hat)
            skr_mean_hat = list(pool_forecast.skr_mean_hat)
            hybrid_demand_hat = list(pool_forecast.hybrid_demand_hat)
            current_posture = ThreatPosture(int(np.argmax(threat_forecast.posture_probs)))
            # 4-wide per spec {normal, elevated, high, critical}; this
            # environment has three postures, so the fourth slot is always
            # zero. Padded rather than narrowed, to match the frozen contract.
            posture_probs_vec = list(threat_forecast.posture_probs) + [0.0]
            # signed normalised slope of the SKR forecast
            skr_mean_hat_list = list(pool_forecast.skr_mean_hat)
            skr_trend = float(
                (skr_mean_hat_list[-1] - skr_mean_hat_list[0])
                / max(1e-9, abs(skr_mean_hat_list[0]) + 1e-9)
            )
            skr_trend = float(np.clip(skr_trend, -1.0, 1.0))

        # Instantaneous (pre-ratchet) posture, exposed for the
        # forecaster's supervised targets. The *ratcheted* posture is
        # monotone and sticky, so labelling it produces a target that is
        # "the same as now" 99.9% of the time and teaches a forecaster
        # nothing; the raw reading is the quantity with something to
        # predict. See forecaster/dataset.py.
        self._current_posture = current_posture

        # design decision 7: exercise the sticky ratchet every decision
        posture_before_ratchet = self._policy_table._ratcheted_posture
        self._policy_table.ratchet_up(current_posture)
        if self._policy_table._ratcheted_posture is not posture_before_ratchet:
            # §4.4 requires `raised` to be true whenever source == "forecast".
            # It is unconditionally true here because `ratchet_up` is a
            # one-way door -- which is the machine-checked half of Hard Rule 2.
            self._event_log.emit(
                "posture_change",
                self._step_count,
                old=int(posture_before_ratchet),
                new=int(self._policy_table._ratcheted_posture),
                source="forecast" if self._forecaster is not None else "current_step",
                raised=int(self._policy_table._ratcheted_posture) > int(posture_before_ratchet),
            )
        floor = self._policy_table.floor(
            SensitivityClass(request["sensitivity_class"]),
            current_posture,
            pqc_capable=request["pqc_capable"],
        )
        # S6: the scripted migration schedule ratchets a tenant cohort's
        # floor at a fixed step. Exogenous config, never agent-chosen
        # (Hard Rule 3), and validated at build time to only ever raise
        # a floor (Hard Rule 2), so `max` here is belt-and-braces rather
        # than the guarantee itself.
        cohort_floor = self._scenario.floor_overrides_at(self._step_count).get(request["tenant"])
        if cohort_floor is not None:
            raised_floor = Action(max(int(floor), int(cohort_floor)))
            if raised_floor is not floor and request["tenant"] not in self._floor_changes_logged:
                self._event_log.emit(
                    "floor_change",
                    self._step_count,
                    tenant_cohort=str(request["tenant"]),
                    old_floor=int(floor),
                    new_floor=int(raised_floor),
                    source="migration_schedule",
                )
                self._floor_changes_logged.add(request["tenant"])
            floor = raised_floor

        # `REKEY_NOW` on a cold-start session adopts the floor's tier, and
        # masking-gap-1 below gates it on the pool -- both must use the
        # floor that is actually enforced, not the raw table value.
        floor = effective_floor_for(request, floor)

        pool_can_draw_hybrid = self._pool_sim.can_draw(self._bits_per_hybrid_draw)
        reuse_masked_due_to_age = session.key_age >= self._max_key_age
        mask = compute_mask(
            request=request,
            floor=floor,
            key_age=session.key_age,
            max_key_age=self._max_key_age,
            pool_can_draw=pool_can_draw_hybrid,
            active_key_tier=(
                _KEY_TYPE_TO_SERVE_ACTION[session.key_type]
                if session.key_type is not None
                else None
            ),
            steps_since_rekey=session.key_age,
            min_rekey_interval=self._min_rekey_interval,
            queue_non_empty=len(self._deferral_queue) > 0,
            head_reservation=self._head_reservation,
            request_is_hybrid_mandatory=bool(request["hybrid_mandatory"]),
            enforce_floor=self._masking_enabled,
        )

        # Masking gap #1 (discovered via testing, not anticipated by
        # compute_mask's three rules): compute_mask only knows to gate
        # SERVE_HYBRID on pool_can_draw, because it has no visibility
        # into session key state. But REKEY_NOW (design decision 4)
        # can *also* resolve to a HYBRID draw -- refreshing an
        # existing HYBRID session, or adopting a HYBRID floor on a
        # cold-start session -- and compute_mask's frozen rules never
        # gate REKEY_NOW on pool_can_draw at all. Left unpatched here,
        # the agent could legally pick REKEY_NOW and
        # `_apply_action`/`pool_sim.draw()` would raise
        # `PoolExhaustedError`, or worse, we'd have to silently
        # under-draw or downgrade -- both forbidden. This is an
        # environment-level augmentation on top of compute_mask's
        # output, not a change to masking.py's three rules.
        if bool(mask[Action.REKEY_NOW]):
            prospective_tier = _ACTION_TO_KEY_TYPE[self._rekey_tier(session, floor)]
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
            # Masking gap #2 is the *second* place a request can be deferred,
            # and it was missing its event until the golden fixture caught the
            # mismatch (34 regret-log entries against 20 `defer_onset` events).
            # A log that under-reports regret would have quietly understated
            # the headline metric everywhere the dashboard or any log-based
            # analysis reads it.
            self._emit_defer_onset(event, request)
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

        # --- spec §4.2 additions: normalised scalars and one-hots -------
        # Every quantity the agent sees is now scale-free. Raw units were a
        # real defect, not a cosmetic one: `key_age` reached 500 while other
        # features sat at or below 3, and the first layer of an unnormalised
        # MLP was dominated by it. Observation normalisation (agents/dqn.py)
        # patched the symptom; normalising at source fixes the cause and makes
        # the state readable.
        qber_abort = float(self._qkd_cfg.get("qber_abort", 0.11))
        skr_mean = float(self._qkd_cfg.get("mean_skr_kbps", 0.025))

        # 4-wide, per spec: {none, classical, pqc, hybrid}. The "none" slot is
        # what distinguishes a cold-start session from one holding a classical
        # key -- previously indistinguishable, since both flattened to zeros.
        key_type_onehot_4 = [0.0, 0.0, 0.0, 0.0]
        if session.key_type is None:
            key_type_onehot_4[0] = 1.0
        else:
            key_type_onehot_4[int(session.key_type) + 1] = 1.0

        request_class_onehot = [0.0] * len(SensitivityClass)
        request_class_onehot[int(request["sensitivity_class"])] = 1.0

        # 4-wide {T0..T3}; this environment uses three tiers, so T3 is always
        # zero. Kept 4-wide to match the frozen contract rather than silently
        # narrowing it.
        floor_onehot = [0.0, 0.0, 0.0, 0.0]
        floor_onehot[min(3, int(floor))] = 1.0

        queue_head_wait = 0.0
        if len(self._deferral_queue) > 0:
            queue_head_wait = float(
                max((entry.wait_steps for entry in self._deferral_queue._heap), default=0.0)
                if hasattr(self._deferral_queue, "_heap")
                else 0.0
            )

        # Cached for the §4.4 serve event; the dashboard reconstructs the
        # threat/floor panel from the log alone (§S13).
        self._last_threat_score = float(threat_score)
        state = StateDict(
            threat_score=threat_score,
            threat_forecast=threat_forecast_vec,
            posture_probs=list(posture_probs_vec),
            qber=self._last_pool_state.qber / max(1e-9, qber_abort),
            skr=self._last_pool_state.skr / max(1e-9, skr_mean),
            pool_fill=self._pool_sim.fill / self._pool_sim.capacity,
            arrival_rate=arrival_rate / max(1e-9, _ARRIVAL_RATE_REFERENCE),
            load=self._current_load(),
            avg_latency=avg_latency / _LATENCY_REFERENCE,
            key_age=session.key_age / max(1e-9, self._max_key_age),
            key_type_onehot=key_type_onehot_4,
            request_class_onehot=request_class_onehot,
            floor_onehot=floor_onehot,
            pqc_capable=1.0 if request["pqc_capable"] else 0.0,
            queue_len_norm=min(1.0, len(self._deferral_queue) / _QUEUE_REFERENCE),
            queue_head_wait_norm=min(1.0, queue_head_wait / 100.0),
            steps_since_rekey_norm=min(1.0, session.key_age / max(1e-9, self._max_key_age)),
            sensitivity_class=request["sensitivity_class"],
            policy_floor=int(floor),
            pool_level_hat=pool_level_hat,
            skr_mean_hat=skr_mean_hat,
            skr_trend=skr_trend,
            hybrid_demand_hat=hybrid_demand_hat,
            regret_event_recent=regret_event_recent,
        )

        self._current_request = request
        self._current_floor = floor
        self._current_reuse_masked_due_to_age = reuse_masked_due_to_age

        return state, mask

    def _resulting_key_type(
        self, action: Action, session: _SessionKeyState, floor: Action
    ) -> KeyType | None:
        if action in _ACTION_TO_KEY_TYPE:
            return _ACTION_TO_KEY_TYPE[action]
        if action is Action.REKEY_NOW:
            return _ACTION_TO_KEY_TYPE[self._rekey_tier(session, floor)]
        return session.key_type  # REUSE

    @staticmethod
    def _rekey_tier(session: _SessionKeyState, floor: Action) -> Action:
        """Tier `REKEY_NOW` re-establishes at: the session's current
        tier, but never below the floor.

        FIXED 2026-08-15 -- this was a Hard Rule 2 violation. The
        original semantics (design decision 4) were "refresh the
        session's *current* tier without changing it", which is wrong
        whenever the floor has ratcheted up since that key was
        established: a session holding a PQC key under a hybrid floor
        would re-establish at PQC, one tier below the floor the mask had
        just enforced. Measured on an S2 episode, 461 of 1,500
        decisions did exactly this.

        SMARTKEYNET_BUILD_SPEC.md §4.1 specifies the correct behaviour
        outright: "REKEY_NOW is a *meta* action: it re-establishes at
        the **lowest legal tier >= floor**".

        This is the third floor hole found in the same session, all of
        them in actions whose tier is state-dependent rather than named
        by the action itself -- `hybrid_mandatory` not reaching the
        mask, `REUSE` ignoring the active key's tier, and now this. That
        is not a coincidence, and the spec flags exactly why:
        "`REUSE`/`REKEY_NOW` are the only actions whose tier is
        state-dependent, which is why masking must be recomputed from
        the *current* key's tier every step". Any future action with a
        state-dependent tier needs the same scrutiny.
        """
        if session.key_type is None:
            return floor  # cold start: adopt the floor's tier
        current_tier = _KEY_TYPE_TO_SERVE_ACTION[session.key_type]
        return Action(max(int(current_tier), int(floor)))

    def _apply_action(self, action: Action) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply one action and report its PHYSICAL outcome.

        Returns `(outcome, info)`, not `(reward, info)` -- the annotation said
        `tuple[float, ...]` until 2026-08-19, left stale by the Hard Rule 1
        refactor that moved reward arithmetic out of this method. `step()`
        turns the outcome into a `RewardInputs` and calls `env/reward.py`.
        """
        request = self._current_request
        if request is None:
            raise RuntimeError("_apply_action called with no request in flight")
        floor = self._current_floor
        if floor is None:
            raise RuntimeError("_apply_action called with no floor resolved for the request")
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

        # The scarcity price is charged per 256-bit ETSI KEY, not per bit
        # (design decision 11).
        keys_consumed = bits_consumed / self._bits_per_hybrid_draw

        if action is Action.REKEY_NOW:
            # REKEY_NOW is a meta action: it costs whatever tier it actually
            # re-established at, which `_resulting_key_type` has just decided.
            if new_key_type is None:
                raise RuntimeError("REKEY_NOW resolved to no key type")
            cost_action = _KEY_TYPE_TO_SERVE_ACTION[new_key_type]
        else:
            cost_action = action
        latency_ms = _LATENCY_MS[cost_action]
        energy_mj = _ENERGY_MJ[cost_action]

        self._latency_sum += latency_ms
        self._latency_count += 1
        self._latency_samples.append(latency_ms)
        self._served_tier_counts[int(cost_action)] += 1

        # §S10 advisory-floor mode: count what masking would have prevented.
        # With masking ON this branch is unreachable -- `step()` has already
        # rejected any action outside the mask, and the mask excluded every
        # sub-floor tier before the agent ever saw it. That unreachability is
        # the guarantee; this counter is how the victim demonstrates its
        # absence.
        if not self._masking_enabled and cost_action in _TIER_ACTIONS_FOR_VIOLATIONS:
            if int(cost_action) < int(floor):
                self._floor_violations += 1
                self._event_log.emit(
                    "serve",
                    self._step_count,
                    request_id=request["request_id"],
                    tenant=request["tenant"],
                    sensitivity_class=int(request["sensitivity_class"]),
                    floor=int(floor),
                    action=int(action),
                    tier_served=int(cost_action),
                    latency_ms=float(latency_ms),
                    energy_mj=float(energy_mj),
                    keys_drawn=0,
                    was_deferred=False,
                    wait_steps=0,
                    floor_violation=True,
                )

        load = self._current_load()

        # NOTE: no reward arithmetic happens here. This method has the
        # request, the floor and the policy table in scope, so computing the
        # reward here is precisely what Hard Rule 1 forbids. It assembles the
        # *physical* outcome; `step()` turns that into a `RewardInputs` and
        # hands it to `env/reward.py`, which cannot see security state at all.
        outcome: dict[str, Any] = {
            "latency_ms": latency_ms,
            "energy_mj": energy_mj,
            "key_age_steps": int(session.key_age),
            "qkd_keys_consumed": int(keys_consumed),
            "did_rekey": bool(is_rekey),
            "normalised_load": load,
        }

        wait_steps = self._resolved_waits.pop(request["request_id"], 0)
        self._event_log.emit(
            "serve",
            self._step_count,
            request_id=request["request_id"],
            tenant=request["tenant"],
            sensitivity_class=int(request["sensitivity_class"]),
            floor=int(floor) if floor is not None else -1,
            action=int(action),
            tier_served=int(cost_action),
            latency_ms=float(latency_ms),
            energy_mj=float(energy_mj),
            keys_drawn=int(keys_consumed),
            was_deferred=bool(wait_steps > 0),
            wait_steps=int(wait_steps),
            # Additive: the dashboard's Beat 1 panel plots the threat signal
            # against the floor it produced, and neither is otherwise
            # recoverable from the log. Logging a security signal is not a
            # Hard Rule 1 concern -- HR1 constrains the *reward*, and the
            # reward cannot see this file.
            threat_score=float(self._last_threat_score),
            posture=int(self._policy_table._ratcheted_posture),
            queue_depth=len(self._deferral_queue),
            regret_events_total=len(self._regret_log),
        )

        info: dict[str, Any] = {}
        if is_rekey and reuse_masked_due_to_age:
            forced_event = ForcedRekey(
                step=self._step_count,
                request_id=request["request_id"],
                key_age_at_rekey=age_before_action,
                load_at_rekey=load,
                cost=self._reward_weights.c_rekey_base
                * (1.0 + self._reward_weights.c_rekey_load_beta * load),
            )
            self._forced_rekey_log.append(forced_event)
            self._event_log.emit(
                "forced_rekey",
                self._step_count,
                request_id=request["request_id"],
                age_at_force=float(age_before_action),
                lifetime_cap=float(self._max_key_age),
                cost=float(forced_event["cost"]),
            )
            info["forced_rekey"] = forced_event

        return outcome, info
