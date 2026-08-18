"""
agents/baselines.py

Tuned non-RL baselines: always-PQC, always-hybrid, static-threshold
(grid-searched), random (PLAN.md Hard Rule 7). Owned by Person C
(split.md §1).

Build these *before* tuning the DQN (Hard Rule 7; split.md Week 3).
Disqualification rule (PLAN.md §2): if a simple threshold policy
matches the DQN in evaluation, the project premise fails.
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


class AlwaysPQCPolicy:
    """Serves SERVE_PQC whenever legal, else the lowest legal tier."""

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        if mask[int(Action.SERVE_PQC)]:
            return Action.SERVE_PQC
        return _lowest_legal_action(mask)


class AlwaysHybridPolicy:
    """Serves SERVE_HYBRID whenever legal, else the lowest legal tier.

    The "drains the pool" baseline (PLAN.md §6 Demo Beat 2).
    """

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        if mask[int(Action.SERVE_HYBRID)]:
            return Action.SERVE_HYBRID
        return _lowest_legal_action(mask)


class StaticThresholdPolicy:
    """Serves SERVE_HYBRID iff `pool_fill` exceeds a fixed threshold,
    else SERVE_PQC. Threshold is grid-searched, not hand-picked (Hard
    Rule 7)."""

    def __init__(self, pool_fill_threshold: float) -> None:
        self.pool_fill_threshold = pool_fill_threshold

    def act(self, state: StateDict, mask: ActionMask) -> Action:
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
