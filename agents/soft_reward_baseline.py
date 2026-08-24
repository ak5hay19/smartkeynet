"""
agents/soft_reward_baseline.py

Soft-reward baseline agent -- a deliberate, contained reproduction of a
flawed prior-work design (the "Noetzold" reward structure, PLAN.md §5
S5 / PLAN2.md §7.5's steering-attack target), built specifically so its
flaw (security as a soft reward term, not a hard floor) can be measured
and attacked in a later session. This is NOT a stealth violation of Hard
Rule 1 -- it's the reference artifact Hard Rule 1's argument is tested
against.

**Hard Rule 1 tension, resolved explicitly (read this before touching
either agent):** Hard Rule 1 ("no security term in the reward, ever")
governs `agents/dqn.py` (the masked agent) and `env/environment.py`'s own
`_apply_action` reward computation, absolutely and without exception --
neither file gains a security term, here or ever, and both are verified
clean of `security_score`/any equivalent by a direct grep (see
tests/test_soft_reward_baseline.py's Hard Rule 1 boundary test). THIS
module is the opposite case: for the steering-attack thesis to mean
anything ("security isn't in our reward, so it isn't for sale"), there
must exist a real, honestly-built agent whose reward genuinely does
contain a security term, so the two agents' behavior under an
adversarial threat trace (a future session's S5) can be honestly
compared. A security term bolted onto our own full reward formula
would NOT be a genuine reproduction of Noetzold's design -- it would
just be our own architecture with an add-on. So this agent's reward is
the reproduced design's OWN, simpler formula (latency, energy, security
-- no freshness/pool-scarcity/rekey-cost/starvation terms), not "our
formula plus a security term". `compute_soft_reward` below is the entire
reward this agent optimizes; `env/environment.py`'s own `_apply_action`
reward is still computed every step (SmartKeyNetEnv can't skip its own
internal accounting) but is never read by this agent's training loop
(see `experiments/train.py::train_soft_reward_baseline`, which discards
`env.step()`'s own returned reward and calls `compute_soft_reward`
instead) -- zero-drift by construction, not by convention.

**No action masking, resolved via environment config, not agent code**
(see `env/environment.py`'s module docstring, design decision 16, and
SESSION_LOG.md's 2026-08-25 entry for the full investigation): this
agent's action-selection logic does not differ from `agents.dqn.DQNAgent`
at all -- `DQNAgent` is reused directly, unmodified, not subclassed.
"No masking" is instead a property of the ENVIRONMENT this agent trains
against: `configs/soft_reward_baseline.yaml` sets `security_masking:
false`, which makes `env/environment.py::_prepare_decision` build the
`ActionMask` with the floor-based rule turned into a no-op (pool-
exhaustion/key-age feasibility rules still apply -- those are physical,
not security-floor, constraints). Because the mask itself is what's
different, not how any agent reads it, `agents.dqn.DQNAgent` (and
`experiments/train.py::GreedyDQNPolicy` for greedy eval) both work
correctly, unmodified, for this agent too -- there is no new agent
*class* in this file, only the reward function and its config.

**security_score provenance (Hard Rule 4):** `classical=0.2, pqc=0.6,
hybrid=1.0` come directly from the reproduced design's own spec (the
HANDOFF_C soft-reward-baseline spec this module reproduces) -- not
invented fresh for this project. `REUSE`/`REKEY_NOW` are not raw actions
with their own score; both resolve to whatever tier they actually
deliver (see `delivered_tier` below), and are scored via that tier's own
entry in the same three-value table -- grounded in what the request is
actually served with, not a fourth/fifth invented constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from env.contracts import Action, KeyType
from env.environment import _ENERGY_UNITS, _KEY_TYPE_TO_SERVE_ACTION, _LATENCY_UNITS

_TIER_ACTIONS = (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID)

_TIER_SECURITY_SCORE: dict[Action, float] = {
    Action.SERVE_CLASSICAL: 0.2,
    Action.SERVE_PQC: 0.6,
    Action.SERVE_HYBRID: 1.0,
}
"""Noetzold-style security_score(tier) -- see module docstring's
"security_score provenance" note. Keyed by the tier an action actually
*delivers* (`delivered_tier` below resolves REUSE/REKEY_NOW to one of
these three before lookup), never by the raw `Action` value directly."""


@dataclass
class SoftRewardConfig:
    """Hyperparameters for `compute_soft_reward` (values live in
    `configs/soft_reward_baseline.yaml`'s own `soft_reward:` block --
    deliberately NOT `configs/default.yaml`'s `reward:` section, per this
    module's Hard Rule 1 tension note above)."""

    w_lat: float = 1.0
    w_en: float = 0.1
    w_sec: float = 1.0


def load_soft_reward_config(path: str | Path | None = None) -> SoftRewardConfig:
    """Read the `soft_reward:` block out of `configs/soft_reward_baseline.yaml`
    into a `SoftRewardConfig` -- mirrors `agents.dqn.load_dqn_config`'s
    existing convention."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "soft_reward_baseline.yaml"
    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return SoftRewardConfig(**config["soft_reward"])


def _existing_tier(key_type_onehot: Sequence[float]) -> Action | None:
    """The tier the session's currently-established key delivers, read
    from the public `key_type_onehot` `StateDict` field -- `None` on a
    cold start (no key established yet). Mirrors
    `experiments/harness.py::_existing_tier` exactly (not imported from
    there: `agents/` must not depend on `experiments/`, the reverse of
    this repo's existing layering -- `experiments/train.py` already
    imports from `agents/dqn.py`, never the other way around)."""
    for key_type in KeyType:
        if key_type_onehot[int(key_type)] == 1.0:
            return _KEY_TYPE_TO_SERVE_ACTION[key_type]
    return None


def resolved_cost_action(action: Action, key_type_onehot: Sequence[float], floor: Action) -> Action:
    """The action this agent's latency/energy cost is actually charged
    against -- mirrors `env/environment.py::SmartKeyNetEnv._apply_action`'s
    own `cost_action` resolution (design decision 4) and
    `experiments/harness.py::_resolved_cost_action`, computed here purely
    from a `StateDict`'s own public fields (`key_type_onehot`,
    `policy_floor`) plus the chosen `action` -- the same information
    `experiments/harness.py` already reconstructs this from, since
    `env/environment.py`'s own reward path is off-limits to read from
    directly (Hard Rule 1: this agent's reward must be computed
    independently, not by reaching into the masked agent's private
    `_apply_action` internals).

    `SERVE_CLASSICAL`/`SERVE_PQC`/`SERVE_HYBRID`/`REUSE` all cost against
    themselves directly; only `REKEY_NOW` resolves, to
    `max(existing session tier, floor)` (or `floor` on a cold start) --
    note this uses the REAL `floor` regardless of `security_masking`,
    since REKEY_NOW's resolution is a property of what REKEY_NOW *means*
    in this environment, not a masking rule (see
    `env/environment.py`'s design decision 16 docstring).
    """
    if action is not Action.REKEY_NOW:
        return action
    existing = _existing_tier(key_type_onehot)
    if existing is None:
        return floor
    return Action(max(int(existing), int(floor)))


def delivered_tier(action: Action, key_type_onehot: Sequence[float], floor: Action) -> Action:
    """The tier `action` actually delivers -- used for `security_score`
    lookup (never for cost; see `resolved_cost_action`, which keeps
    REUSE costing against `Action.REUSE` itself, not a tier). Mirrors
    `experiments/harness.py::_delivered_tier` for the same
    not-importing-across-layers reason as `_existing_tier` above.

    `SERVE_CLASSICAL`/`SERVE_PQC`/`SERVE_HYBRID` deliver themselves.
    `REUSE` delivers the existing session tier unchanged -- with
    `security_masking: false`, this can genuinely be a tier below the
    real floor (unlike the masked agent, where `compute_mask` forbids
    REUSE-ing a stale below-floor key). `REKEY_NOW` delivers
    `max(existing tier, floor)`, matching `resolved_cost_action` above.
    """
    if action in _TIER_ACTIONS:
        return action
    existing = _existing_tier(key_type_onehot)
    if existing is None:
        return floor  # cold start: REKEY_NOW adopts floor; REUSE is illegal cold-start (unreachable here)
    if action is Action.REKEY_NOW:
        return Action(max(int(existing), int(floor)))
    return existing  # REUSE


def security_score(tier: Action) -> float:
    """`_TIER_SECURITY_SCORE[tier]` -- a thin, named wrapper so call
    sites read as "the security_score of a tier" (HANDOFF_C's own
    terminology), not a bare dict lookup."""
    return _TIER_SECURITY_SCORE[tier]


def compute_soft_reward(state: dict[str, Any], action: Action, cfg: SoftRewardConfig) -> float:
    """The Noetzold-style soft reward this agent actually trains against:

        r = -w_lat * latency - w_en * energy + w_sec * security_score(tier)

    Computed entirely from `state` (the `StateDict` `agent.act()` was
    just given -- i.e. the state *before* `action` is applied) and
    `action` itself -- never from `env/environment.py`'s own
    `_apply_action`/reward computation, which stays completely
    untouched and is never consulted here (see this module's Hard Rule 1
    tension note). `state` is typed as a plain dict rather than
    `StateDict` because at a training loop's call site it's often the
    literal dict `env.reset()`/`env.step()` hand back, which satisfies
    the `StateDict` TypedDict's shape without the type checker being
    able to prove it structurally in every caller; runtime behavior is
    identical either way.
    """
    floor = Action(int(state["policy_floor"]))
    key_type_onehot = state["key_type_onehot"]

    cost_action = resolved_cost_action(action, key_type_onehot, floor)
    tier = delivered_tier(action, key_type_onehot, floor)

    latency = _LATENCY_UNITS[cost_action]
    energy = _ENERGY_UNITS[cost_action]
    sec = security_score(tier)

    return -cfg.w_lat * latency - cfg.w_en * energy + cfg.w_sec * sec
