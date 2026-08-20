"""
env/decision_trace.py

The six-step decision trace behind Dashboard v2's Explain Decision
panel (PLAN2 §7.3, Addition D in §8).

Hard Rule 10, which governs this module entirely:

    "The Explain Decision panel may only display values the pipeline
    actually computed -- threat_score, posture_probs, the resolved
    policy floor, the action mask, per-action cost lookups, and the
    final chosen action. It must never synthesize an explanation via a
    generative model standing in for these values. Any human-readable
    sentence shown must be templated deterministically from the values
    above -- swap in the real numbers, don't invent new reasoning."

So this module contains no model, no heuristics and no narration. It
reads the six values **directly off the live environment, policy table
and mask objects** at decision time and formats them. PLAN2 §8 requires
exactly one source of truth for these numbers, which is why the
assembler lives next to the environment rather than inside `api/` or
`dashboard/` -- both of those import it, neither reimplements it.

It also degrades honestly (PLAN2 §7.3's closing requirement): where
only one tier was ever legal, step 5 says there was no cost tradeoff
rather than presenting a comparison that did not happen; and where
several legal options existed and the policy's own learned preference
picked among them, step 6 says that, rather than claiming a rule
decided it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from env.contracts import (
    Action,
    ActionMask,
    KeyType,
    Request,
    SensitivityClass,
    StateDict,
    ThreatPosture,
)

_TIER_ACTIONS = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)

_ILLEGAL_REASONS = {
    "below_floor": "below the policy floor",
    "pool": "pool cannot cover the 256-bit draw",
    "key_age": "key age exceeded its cap (no existing key counts as maximally stale)",
    "stale_tier": "existing key material is below the floor",
}
"""The reasons `env/masking.py` can make an action illegal, spelled for
display. These map one-to-one onto its five legality rules.

There is deliberately no separate "no existing key yet" reason, because
`compute_mask` has no such rule: `env/environment.py` cold-starts a
session at `key_age = max_key_age` (its design decision 2) precisely so
that the age rule covers that case. Spelling it as its own reason here
made the trace disagree with the mask for
`current_key_type=None, key_age=0` -- a combination the environment
never constructs but `compute_mask` accepts -- and a displayed reason
that disagrees with the mask actually applied is exactly the "trust me"
storytelling Hard Rule 10 forbids. The mask is authoritative; this
module describes it."""


@dataclass
class TraceStep:
    """One inspectable computation. `values` holds the raw numbers; the
    UI renders them, it does not re-derive them."""

    index: int
    title: str
    values: dict[str, Any]
    summary: str


@dataclass
class DecisionTrace:
    """The full six-step trace for one decision."""

    request_id: str
    step: int
    tenant: str
    service: str
    sensitivity_class: int
    threat_score: float
    posture_probs: list[float]
    resolved_posture: str
    policy_floor: str
    mask: list[bool]
    illegal_reasons: dict[str, str]
    action_costs: dict[str, dict[str, float]]
    chosen_action: str
    delivered_tier: str
    cost_tradeoff_existed: bool
    decided_by: str
    steps: list[TraceStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "step": self.step,
            "tenant": self.tenant,
            "service": self.service,
            "sensitivity_class": self.sensitivity_class,
            "threat_score": self.threat_score,
            "posture_probs": self.posture_probs,
            "resolved_posture": self.resolved_posture,
            "policy_floor": self.policy_floor,
            "mask": self.mask,
            "illegal_reasons": self.illegal_reasons,
            "action_costs": self.action_costs,
            "chosen_action": self.chosen_action,
            "delivered_tier": self.delivered_tier,
            "cost_tradeoff_existed": self.cost_tradeoff_existed,
            "decided_by": self.decided_by,
            "steps": [
                {"index": s.index, "title": s.title, "values": s.values, "summary": s.summary}
                for s in self.steps
            ],
        }


def illegal_reason(
    action: Action,
    floor: Action,
    key_age: float,
    max_key_age: float,
    pool_can_draw: bool,
    current_key_type: KeyType | None,
) -> str | None:
    """Why `env/masking.py` would rule `action` illegal, or None if legal.

    Deliberately re-derived from the *same inputs* `compute_mask` is
    given rather than guessed from its output, so a displayed reason
    can never disagree with the mask that was actually applied.
    `tests/test_decision_trace.py` asserts the two agree on every
    combination.
    """
    if int(action) < int(floor):
        return _ILLEGAL_REASONS["below_floor"]
    if action is Action.SERVE_HYBRID and not pool_can_draw:
        return _ILLEGAL_REASONS["pool"]
    if action is Action.REUSE:
        if key_age >= max_key_age:
            return _ILLEGAL_REASONS["key_age"]
        if current_key_type is not None and int(_tier_of(current_key_type)) < int(floor):
            return _ILLEGAL_REASONS["stale_tier"]
    if action is Action.REKEY_NOW and current_key_type is not None:
        if int(_tier_of(current_key_type)) < int(floor):
            return _ILLEGAL_REASONS["stale_tier"]
    return None


def _tier_of(key_type: KeyType) -> Action:
    return {
        KeyType.CLASSICAL: Action.SERVE_CLASSICAL,
        KeyType.PQC: Action.SERVE_PQC,
        KeyType.HYBRID: Action.SERVE_HYBRID,
    }[KeyType(int(key_type))]


def build_decision_trace(
    env: Any,
    state: StateDict,
    mask: ActionMask,
    chosen_action: Action,
    q_values: Sequence[float] | None = None,
) -> DecisionTrace:
    """Assemble the six-step trace for the decision `env` is currently
    presenting.

    Every field is read from the live objects: `state` is what the agent
    was handed, `mask` is what `compute_mask` produced, the floor is
    `env._current_floor` (the value `PolicyTable.floor()` actually
    returned this decision), and the cost tables are
    `env/environment.py`'s own. Nothing is recomputed by a different
    route, because two routes are two chances to disagree.
    """
    from env.environment import _ENERGY_UNITS, _LATENCY_UNITS

    request: Request = env._current_request
    floor: Action = env._current_floor
    session = env._sessions.get((request["tenant"], request["service"]))
    current_key_type = getattr(session, "key_type", None)
    key_age = float(getattr(session, "key_age", 0.0))
    max_key_age = float(env._max_key_age)
    pool_can_draw = bool(env._pool_sim.can_draw(env._bits_per_hybrid_draw))

    forecast = env._forecaster.get_threat_forecast() if env._forecaster is not None else None
    posture_probs = list(forecast.posture_probs) if forecast else [1.0, 0.0, 0.0]
    resolved_posture = ThreatPosture(int(max(range(len(posture_probs)), key=posture_probs.__getitem__)))

    reasons = {
        action.name: reason
        for action in Action
        if (
            reason := illegal_reason(
                action, floor, key_age, max_key_age, pool_can_draw, current_key_type
            )
        )
        is not None
    }

    legal = [action for action in Action if mask[int(action)]]
    costs = {
        action.name: {
            "latency": float(_LATENCY_UNITS[_cost_action(action, current_key_type, floor)]),
            "energy": float(_ENERGY_UNITS[_cost_action(action, current_key_type, floor)]),
            "pool_bits": float(
                env._bits_per_hybrid_draw
                if _cost_action(action, current_key_type, floor) is Action.SERVE_HYBRID
                and action is not Action.REUSE
                else 0.0
            ),
        }
        for action in legal
    }

    delivered = _cost_action(chosen_action, current_key_type, floor)
    legal_tiers = [a for a in legal if a in _TIER_ACTIONS]
    cost_tradeoff_existed = len(legal) > 1

    cheapest = (
        min(legal, key=lambda a: costs[a.name]["latency"] + costs[a.name]["energy"])
        if legal
        else chosen_action
    )
    if not cost_tradeoff_existed:
        decided_by = "floor"
    elif chosen_action is cheapest:
        decided_by = "cost"
    else:
        decided_by = "policy_preference"

    trace = DecisionTrace(
        request_id=request["request_id"],
        step=int(env._step_count),
        tenant=request["tenant"],
        service=request["service"],
        sensitivity_class=int(request["sensitivity_class"]),
        threat_score=float(state["threat_score"]),
        posture_probs=posture_probs,
        resolved_posture=resolved_posture.name,
        policy_floor=floor.name,
        mask=[bool(mask[int(a)]) for a in Action],
        illegal_reasons=reasons,
        action_costs=costs,
        chosen_action=chosen_action.name,
        delivered_tier=delivered.name,
        cost_tradeoff_existed=cost_tradeoff_existed,
        decided_by=decided_by,
    )
    trace.steps = _build_steps(trace, legal, legal_tiers, cheapest, q_values)
    return trace


def _cost_action(action: Action, current_key_type: KeyType | None, floor: Action) -> Action:
    """Mirrors `env/environment.py`'s `cost_action` resolution."""
    if action in _TIER_ACTIONS:
        return action
    if action is Action.REKEY_NOW:
        return _tier_of(current_key_type) if current_key_type is not None else floor
    return _tier_of(current_key_type) if current_key_type is not None else floor


def _build_steps(
    trace: DecisionTrace,
    legal: list[Action],
    legal_tiers: list[Action],
    cheapest: Action,
    q_values: Sequence[float] | None,
) -> list[TraceStep]:
    """The six steps of PLAN2 §7.3, each templated deterministically
    from `trace`'s already-computed values. No sentence here contains a
    value that is not in one of those fields (Hard Rule 10)."""
    sensitivity = SensitivityClass(trace.sensitivity_class).name

    steps = [
        TraceStep(
            1,
            "Threat signal ingested",
            {"threat_score": trace.threat_score},
            f"threat_score = {trace.threat_score:.3f}",
        ),
        TraceStep(
            2,
            "Posture classified",
            {"posture_probs": trace.posture_probs, "resolved": trace.resolved_posture},
            (
                f"posture = {trace.resolved_posture} "
                f"(p = {max(trace.posture_probs):.3f})"
            ),
        ),
        TraceStep(
            3,
            "Policy floor lookup",
            {
                "sensitivity_class": sensitivity,
                "posture": trace.resolved_posture,
                "floor": trace.policy_floor,
            },
            f"({sensitivity} x {trace.resolved_posture}) -> floor = {trace.policy_floor}",
        ),
        TraceStep(
            4,
            "Action mask computed",
            {"mask": trace.mask, "illegal_reasons": trace.illegal_reasons},
            (
                f"{len(legal)} of {len(Action)} actions legal: "
                + ", ".join(a.name for a in legal)
            ),
        ),
    ]

    if trace.cost_tradeoff_existed:
        step5_summary = (
            f"cheapest legal option was {cheapest.name} "
            f"(latency {trace.action_costs[cheapest.name]['latency']:.2f}, "
            f"energy {trace.action_costs[cheapest.name]['energy']:.2f})"
        )
    else:
        # PLAN2 §7.3's explicit requirement: degrade honestly rather than
        # present a comparison that did not happen.
        step5_summary = "no cost tradeoff existed here -- only one action was legal"
    steps.append(
        TraceStep(
            5,
            "Cost comparison among legal actions",
            {"costs": trace.action_costs, "cheapest": cheapest.name},
            step5_summary,
        )
    )

    if trace.decided_by == "floor":
        final = (
            f"floor required {trace.policy_floor}; {trace.chosen_action} was the only legal option"
        )
    elif trace.decided_by == "cost":
        final = (
            f"floor required {trace.policy_floor}; {trace.chosen_action} was the cheapest legal "
            "option that cleared it"
        )
    else:
        # Honest about the limit of the explanation: several options were
        # legal and the policy's own learned preference picked among them.
        final = (
            f"floor required {trace.policy_floor}; several options cleared it and the policy's "
            f"own learned preference selected {trace.chosen_action} over the cheapest "
            f"({cheapest.name})"
        )

    step6_values: dict[str, Any] = {
        "chosen_action": trace.chosen_action,
        "delivered_tier": trace.delivered_tier,
        "decided_by": trace.decided_by,
    }
    if q_values is not None:
        step6_values["q_values"] = {
            action.name: float(q_values[int(action)]) for action in Action
        }
    steps.append(TraceStep(6, "Final decision", step6_values, final))
    return steps
