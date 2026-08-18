"""
agents/baselines.py

Tuned non-RL baselines: always-PQC, always-hybrid, static-threshold
(grid-searched), random (PLAN.md Hard Rule 7). Owned by Person C
(split.md §1).

Build these *before* tuning the DQN (Hard Rule 7; split.md Week 3).
Disqualification rule (PLAN.md §2): if a simple threshold policy
matches the DQN in evaluation, the project premise fails.

Reuse-awareness (2026-08-19)
----------------------------
Every tier policy here reuses the existing session key whenever REUSE
is legal, and only picks a tier when key material actually has to be
established. Until 2026-08-19 they did not: each one re-established
fresh key material on *every* request, which meant

  * they paid a full rekey cost, serve latency and (for hybrid) a fresh
    256-bit pool draw on requests that a real KMS would have served
    from the existing session key -- measured on S1/seed 0, all three
    tier policies took a tier action on 250/250 decisions while REUSE
    was legal on 244 of them;
  * and consequently the DQN's headline advantage over them was mostly
    "the DQN discovered REUSE", not "the DQN budgets a scarce resource
    better" -- a 10x reward gap that evaporates once the baselines
    stop rekeying wastefully.

A baseline that hands the agent a free 10x is not a tuned baseline, and
Hard Rule 7 asks for tuned ones. It is also the physically wrong model:
under ETSI GS QKD 014 key material is consumed when a key is
*established*, not on every request that a live session key already
covers, so charging a fresh QKD draw per cache hit was never right.

`RandomPolicy` is deliberately excluded from this -- it is the
uniform-random control, and biasing it toward REUSE would stop it
being one.
"""

from __future__ import annotations

import random
from typing import Callable, Protocol

from env.contracts import Action, ActionMask, StateDict


class Policy(Protocol):
    """Common shape for every baseline plus the DQN, so
    `experiments/harness.py` can run any of them identically."""

    def act(self, state: StateDict, mask: ActionMask) -> Action: ...


def _lowest_legal_action(mask: ActionMask) -> Action:
    """First legal action in `Action`'s fixed enum order (SERVE_CLASSICAL
    < SERVE_PQC < SERVE_HYBRID < REUSE < REKEY_NOW).

    `compute_mask` (env/masking.py) already excludes every action below
    the current policy floor, so the first legal *tier* action here is
    exactly the cheapest tier that still clears the floor -- this is
    the "lowest currently-legal tier" fallback every non-random policy
    in this module uses when its preferred action is masked out. Holds
    for any mask with at least one legal entry, however contrived (a
    mask where only REUSE or only REKEY_NOW is legal still resolves
    correctly, since those are simply later in the same fixed order).
    """
    for action in Action:
        if mask[int(action)]:
            return action
    raise ValueError("no legal action in mask -- a valid mask must have at least one True entry")


def _reuse_is_legal(mask: ActionMask) -> bool:
    """Whether the existing session key can still serve this request.

    `env/masking.py` makes REUSE illegal exactly when `key_age >=
    max_key_age` (the SP 800-57-derived cap L) or when there is no key
    yet -- so "REUSE is legal" is precisely "a live, in-cryptoperiod
    session key already covers this request". Every tier policy in this
    module checks this first: see the module docstring's
    reuse-awareness note.
    """
    return bool(mask[int(Action.REUSE)])


class AlwaysPQCPolicy:
    """Reuses the live session key when it can; otherwise establishes
    SERVE_PQC whenever legal, else the lowest legal tier.

    The "never voluntarily spends pool" baseline."""

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        if _reuse_is_legal(mask):
            return Action.REUSE
        if mask[int(Action.SERVE_PQC)]:
            return Action.SERVE_PQC
        return _lowest_legal_action(mask)


class AlwaysHybridPolicy:
    """Reuses the live session key when it can; otherwise establishes
    SERVE_HYBRID whenever legal, else the lowest legal tier.

    The "drains the pool" baseline (PLAN.md §6 Demo Beat 2): it spends
    QKD material at every key establishment it is allowed to, which is
    the maximal honest drain rate -- a policy that also drew on cache
    hits would be draining the pool for key material nobody asked for.
    """

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        if _reuse_is_legal(mask):
            return Action.REUSE
        if mask[int(Action.SERVE_HYBRID)]:
            return Action.SERVE_HYBRID
        return _lowest_legal_action(mask)


class StaticThresholdPolicy:
    """Reuses the live session key when it can; otherwise establishes
    SERVE_HYBRID iff `pool_fill` exceeds a fixed threshold, else
    SERVE_PQC. Threshold is grid-searched, not hand-picked (Hard
    Rule 7).

    This is the baseline the project's premise is measured against
    (PLAN2 §3.2's disqualification rule), so it gets the same
    reuse-awareness as the others -- a threshold policy that rekeyed on
    every request would be a strawman, and beating a strawman proves
    nothing about budgeting.
    """

    def __init__(self, pool_fill_threshold: float) -> None:
        self.pool_fill_threshold = pool_fill_threshold

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        if _reuse_is_legal(mask):
            return Action.REUSE
        if state["pool_fill"] > self.pool_fill_threshold and mask[int(Action.SERVE_HYBRID)]:
            return Action.SERVE_HYBRID
        if mask[int(Action.SERVE_PQC)]:
            return Action.SERVE_PQC
        return _lowest_legal_action(mask)

    @staticmethod
    def grid_search(
        candidate_thresholds: list[float], eval_fn: Callable[["StaticThresholdPolicy"], float]
    ) -> "StaticThresholdPolicy":
        """Pick the threshold maximizing `eval_fn(StaticThresholdPolicy(t))`."""
        if not candidate_thresholds:
            raise ValueError("candidate_thresholds must be non-empty")

        best_policy: StaticThresholdPolicy | None = None
        best_score = float("-inf")
        for threshold in candidate_thresholds:
            candidate = StaticThresholdPolicy(threshold)
            score = eval_fn(candidate)
            if score > best_score:
                best_score = score
                best_policy = candidate
        return best_policy  # type: ignore[return-value]  # candidate_thresholds is non-empty, so this is always set


class RandomPolicy:
    """Uniform-random choice among currently-legal actions."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        legal = [action for action in Action if mask[int(action)]]
        if not legal:
            raise ValueError("no legal action in mask -- a valid mask must have at least one True entry")
        return self._rng.choice(legal)
