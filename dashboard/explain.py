"""
dashboard/explain.py

Explain Decision panel backend (PLAN2.md §7.3, Addition D; Hard Rule
10). Owned by Person D (dashboard/attack/API/paper, per HANDOFF_D.md).

Assembles the six-step decision trace PLAN2.md §7.3 specifies from the
real, already-computed values at decision time. This module never
generates a narrative -- every field is either a real computed value
or a sentence templated deterministically from those values (Hard Rule
10). It is policy-agnostic: nothing here depends on `agents/dqn.py` or
any specific policy, only on the environment/masking objects a
decision was actually made against -- consistent with `experiments/
harness.py`'s "any policy x any scenario" design.

Step 4 (the action mask) calls `env.masking.compute_mask()` directly,
using the exact same inputs it uses (`floor`, `key_age`/`max_key_age`,
`pool_can_draw`) -- this module's legal/illegal flags and reasons are
therefore structurally unable to drift from the masking layer's real
behavior, since they come from the same function call, not a parallel
reimplementation. The per-action reason text is derived by replaying
`compute_mask()`'s own three legality rules, in the same order, on the
same inputs -- see `_mask_entries` below.

Scope note: this deliberately reflects only `env/masking.py`'s three
legality rules. `env/environment.py`'s `_prepare_decision` layers two
additional environment-level mask augmentations on top of
`compute_mask()`'s output (REKEY_NOW's own pool-draw gap, and
deferring a request entirely if zero actions end up legal) -- both are
explicitly documented there as augmentations *on top of* masking.py,
not part of masking.py's three rules, so they are out of scope here
per PLAN2.md's Hard Rule 10 (this panel's ground truth is what the
masking layer itself computed). In practice this only matters for the
rare decision where the pool can't cover a HYBRID-resolving REKEY_NOW;
see `explain_decision_from_env`'s docstring for how that shows up.

Step 5's cost numbers are read directly from `env/environment.py`'s
real `_LATENCY_UNITS`/`_ENERGY_UNITS` tables and `_KEY_TYPE_TO_SERVE_ACTION`
mapping (imported, not re-typed) -- the same constants `_apply_action`
uses to compute reward, and the same ones `experiments/harness.py`'s
`_resolved_cost_action` already mirrors for REKEY_NOW's tier
resolution (this module's `_resolved_cost_action` follows the same
precedent, from the same public `StateDict` field, `key_type_onehot`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from env.contracts import Action, Request, SensitivityClass, StateDict, ThreatPosture
from env.environment import _ENERGY_UNITS, _KEY_TYPE_TO_SERVE_ACTION, _LATENCY_UNITS
from env.masking import _PLACEHOLDER_FLOOR_TABLE, compute_mask
from env.contracts import KeyType

if TYPE_CHECKING:  # pragma: no cover
    from env.environment import SmartKeyNetEnv


@dataclass(frozen=True)
class MaskEntry:
    """One action's legality + reason (PLAN2.md §7.3 step 4)."""

    action: Action
    legal: bool
    reason: str


@dataclass(frozen=True)
class CostEntry:
    """One legal action's real per-tier cost (PLAN2.md §7.3 step 5)."""

    action: Action
    latency: float
    energy: float
    chosen: bool


@dataclass(frozen=True)
class DecisionTrace:
    """The full six-step trace PLAN2.md §7.3 describes."""

    # step 1 -- threat signal ingested
    threat_score: float
    threat_source: str

    # step 2 -- posture classified
    posture_probs: dict[str, float] | None
    resolved_posture: ThreatPosture

    # step 3 -- policy floor lookup
    sensitivity_class: SensitivityClass
    floor_cell: tuple[SensitivityClass, ThreatPosture]
    floor: Action
    floor_table: dict[tuple[SensitivityClass, ThreatPosture], Action]

    # step 4 -- action mask computed
    mask: list[MaskEntry]

    # step 5 -- cost comparison among legal actions
    costs: list[CostEntry]
    cost_note: str | None

    # step 6 -- final decision
    chosen_action: Action
    final_text: str


def _mask_entries(
    request: Request,
    floor: Action,
    key_age: float,
    max_key_age: float,
    pool_can_draw: bool,
    has_existing_key: bool,
) -> list[MaskEntry]:
    """Build all five actions' legal/reason pairs.

    `mask` itself is `compute_mask()`'s real output -- never
    recomputed by hand. The per-action reason below replays that same
    function's three checks, in the same order, on the same inputs, so
    a reason can only be wrong if `compute_mask()`'s own behavior
    changes underneath it (in which case the reason changes too, since
    it re-derives from the same inputs every call).
    """
    mask = compute_mask(request=request, floor=floor, key_age=key_age, max_key_age=max_key_age, pool_can_draw=pool_can_draw)

    entries: list[MaskEntry] = []
    for action in Action:
        legal = bool(mask[int(action)])
        if not legal:
            if int(action) < int(floor):
                reason = f"below floor (requires >= {floor.name})"
            elif action is Action.SERVE_HYBRID and not pool_can_draw:
                reason = "pool cannot cover the draw"
            elif action is Action.REUSE and key_age >= max_key_age:
                if not has_existing_key:
                    reason = "no existing key yet (cold start)"
                else:
                    reason = f"key age ({key_age:g}) exceeded its cap ({max_key_age:g})"
            else:  # pragma: no cover -- unreachable: compute_mask's only 3 rules are covered above
                raise AssertionError(f"compute_mask marked {action.name} illegal for an unrecognized reason")
        else:
            if action is Action.REUSE:
                reason = f"key age ({key_age:g}) within cap ({max_key_age:g})"
            else:
                reason = f"clears floor (requires >= {floor.name})"
        entries.append(MaskEntry(action=action, legal=legal, reason=reason))
    return entries


def _resolved_cost_action(action: Action, key_type_onehot: Sequence[float], floor: Action) -> Action:
    """Mirror `experiments/harness.py`'s `_resolved_cost_action`, which
    itself mirrors `SmartKeyNetEnv._apply_action`'s `cost_action`
    resolution (env/environment.py design decision 4) -- REKEY_NOW
    costs against whichever tier it actually refreshes (the existing
    session's tier, read off the public `key_type_onehot` field, or
    the floor's tier on a cold start), every other action costs
    against itself directly.
    """
    if action is not Action.REKEY_NOW:
        return action
    for key_type in KeyType:
        if key_type_onehot[int(key_type)] == 1.0:
            return _KEY_TYPE_TO_SERVE_ACTION[key_type]
    return floor  # cold-start REKEY_NOW adopts the floor's tier


def _cost_entries(
    mask_entries: list[MaskEntry],
    floor: Action,
    key_type_onehot: Sequence[float],
    chosen_action: Action,
) -> tuple[list[CostEntry], str | None]:
    legal_actions = [e.action for e in mask_entries if e.legal]
    entries: list[CostEntry] = []
    for action in legal_actions:
        cost_action = _resolved_cost_action(action, key_type_onehot, floor)
        entries.append(
            CostEntry(
                action=action,
                latency=_LATENCY_UNITS[cost_action],
                energy=_ENERGY_UNITS[cost_action],
                chosen=(action is chosen_action),
            )
        )
    entries.sort(key=lambda e: e.latency + e.energy)

    note = None
    if len(entries) <= 1:
        note = "Only one legal action existed here -- no cost tradeoff to make."
    return entries, note


def _final_text(
    sensitivity_class: SensitivityClass,
    resolved_posture: ThreatPosture,
    floor: Action,
    cost_entries: list[CostEntry],
    chosen_action: Action,
) -> str:
    floor_clause = f"Floor requires >= {floor.name} at {sensitivity_class.name} + {resolved_posture.name} posture."

    if len(cost_entries) <= 1:
        return (
            f"{floor_clause} {chosen_action.name} was the only legal action here -- "
            "chosen because it was required, not preferred."
        )

    cheapest = cost_entries[0]
    chosen_entry = next(e for e in cost_entries if e.action is chosen_action)

    if chosen_entry.action is cheapest.action:
        return (
            f"{floor_clause} {chosen_action.name} was the cheapest legal action that cleared it "
            f"(latency {chosen_entry.latency:g}, energy {chosen_entry.energy:g})."
        )

    return (
        f"{floor_clause} {cheapest.action.name} was legal and cheaper "
        f"(latency {cheapest.latency:g}, energy {cheapest.energy:g}), but {chosen_action.name} "
        f"(latency {chosen_entry.latency:g}, energy {chosen_entry.energy:g}) was chosen instead -- "
        "the policy's own preference among legal options, not a rule the masking layer enforces."
    )


def explain_decision(
    *,
    request: Request,
    threat_score: float,
    threat_source: str,
    posture_probs: Sequence[float] | None,
    floor: Action,
    key_age: float,
    max_key_age: float,
    pool_can_draw: bool,
    key_type_onehot: Sequence[float],
    chosen_action: Action,
) -> DecisionTrace:
    """Assemble one decision's six-step trace from real computed values.

    Args mirror what's actually available at decision time (PLAN2.md
    §8's "decision-trace assembler... reads the same six values
    directly off the environment/masking/agent objects"):

    - `request`: the `Request` being decided (gives tenant/service/
      sensitivity_class -- `StateDict` doesn't carry tenant/service).
    - `threat_score`/`posture_probs`: the forecaster's real
      `ThreatForecast` output (`posture_probs=None` for
      `use_foresight: off`, matching `env/environment.py`'s own
      `current_posture = ThreatPosture.CALM` fallback when there is no
      forecaster).
    - `floor`: the real `PolicyTable.floor(...)` return value already
      used to build this decision's mask (`StateDict["policy_floor"]`).
    - `key_age`/`max_key_age`/`pool_can_draw`: the exact inputs
      `compute_mask()` was called with for this decision.
    - `key_type_onehot`: `StateDict["key_type_onehot"]`, used only to
      resolve REKEY_NOW's real cost tier (step 5).
    - `chosen_action`: the action the policy actually picked; must be
      legal under this decision's mask (raises `ValueError` otherwise,
      same philosophy as `SmartKeyNetEnv.step`'s `IllegalActionError`).
    """
    sensitivity_class = SensitivityClass(request["sensitivity_class"])
    has_existing_key = any(v == 1.0 for v in key_type_onehot)

    if posture_probs is None:
        resolved_posture = ThreatPosture.CALM
        probs_dict: dict[str, float] | None = None
    else:
        probs_list = [float(p) for p in posture_probs]
        resolved_posture = ThreatPosture(int(np.argmax(probs_list)))
        probs_dict = {ThreatPosture(i).name: probs_list[i] for i in range(len(probs_list))}

    mask_entries = _mask_entries(
        request=request,
        floor=floor,
        key_age=key_age,
        max_key_age=max_key_age,
        pool_can_draw=pool_can_draw,
        has_existing_key=has_existing_key,
    )
    legal_actions = {e.action for e in mask_entries if e.legal}
    if chosen_action not in legal_actions:
        raise ValueError(f"chosen_action {chosen_action.name} is not legal under this decision's real mask")

    cost_entries, cost_note = _cost_entries(mask_entries, floor, key_type_onehot, chosen_action)
    final_text = _final_text(sensitivity_class, resolved_posture, floor, cost_entries, chosen_action)

    return DecisionTrace(
        threat_score=threat_score,
        threat_source=threat_source,
        posture_probs=probs_dict,
        resolved_posture=resolved_posture,
        sensitivity_class=sensitivity_class,
        floor_cell=(sensitivity_class, resolved_posture),
        floor=floor,
        floor_table=dict(_PLACEHOLDER_FLOOR_TABLE),
        mask=mask_entries,
        costs=cost_entries,
        cost_note=cost_note,
        chosen_action=chosen_action,
        final_text=final_text,
    )


def explain_decision_from_env(env: "SmartKeyNetEnv", state: StateDict, chosen_action: Action) -> DecisionTrace:
    """Convenience wrapper: pull the six inputs `explain_decision`
    needs directly off a live `SmartKeyNetEnv` and its just-returned
    `state` (PLAN2.md §8's "decision-trace assembler... no separate
    model, no separate source of truth").

    Call this after `env.reset()`/`env.step()` returned `state`, before
    the *next* `env.step()` call advances past this decision (mirrors
    `experiments/harness.py`'s `run_scenario` loop shape: `state, mask
    = ...; action = policy.act(...); trace =
    explain_decision_from_env(env, state, action); state, ... =
    env.step(action)`).

    Reaches into a few of `env`'s private attributes
    (`_current_request`, `_pool_sim`, `_bits_per_hybrid_draw`,
    `_forecaster`) because the public Gym API doesn't yet surface a
    request's `hybrid_mandatory`/pool-draw-eligibility/forecaster
    state -- `experiments/harness.py`'s `run_scenario` already
    establishes this exact precedent (see its `env._current_request`
    access and docstring).

    Note (see module docstring's Scope note): the mask this produces
    is `compute_mask()`'s pure three-rule output, not necessarily
    byte-identical to the real mask `env` used for this decision on
    the rare step where the pool can't cover a HYBRID-resolving
    REKEY_NOW (`env/environment.py`'s masking-gap-#1 augmentation).
    `chosen_action` came from the real mask either way, so this never
    raises -- REKEY_NOW would just show as legal here even on that
    rare step where the real environment had additionally forbidden
    it.
    """
    request = env._current_request
    floor = Action(state["policy_floor"])
    pool_can_draw = env._pool_sim.can_draw(env._bits_per_hybrid_draw)

    if env._forecaster is None:
        threat_source = "off (no forecaster configured)"
        posture_probs = None
    else:
        threat_source = type(env._forecaster).__name__
        posture_probs = env._forecaster.get_threat_forecast().posture_probs

    return explain_decision(
        request=request,
        threat_score=state["threat_score"],
        threat_source=threat_source,
        posture_probs=posture_probs,
        floor=floor,
        key_age=state["key_age"],
        max_key_age=env._max_key_age,
        pool_can_draw=pool_can_draw,
        key_type_onehot=state["key_type_onehot"],
        chosen_action=chosen_action,
    )
