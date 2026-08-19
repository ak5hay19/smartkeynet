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
from collections.abc import Callable
from typing import Protocol

from env.contracts import Action, ActionMask, StateDict
from env.masking import load_key_lifetime_config


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
    """The tuned non-RL baseline the whole project is measured against
    (Hard Rule 7; PLAN.md's disqualification rule).

    Implements SMARTKEYNET_BUILD_SPEC.md §S7's three-parameter rule in
    full:

        serve hybrid iff `pool_fill >= tau` AND `class >= c_min`,
        else the lowest legal action >= floor;
        rekey iff `key_age >= rho * L`  (otherwise REUSE).

    **The `rho` / REUSE half of that rule was missing until 2026-08-15,
    and its absence made this baseline a strawman.** Every `SERVE_*`
    action in this environment re-establishes key material (see
    `env/environment.py` design decision 4), so a policy that never
    returns `REUSE` pays `c_rekey(load)` on *every single decision*. A
    first Gate W3 run against the old version had the DQN winning by an
    order of magnitude (-276 vs -3957) purely by discovering `REUSE` --
    a result about key-lifetime management that says nothing whatsoever
    about pool budgeting, and one a reviewer would dismiss on sight.

    A baseline has to be given every lever the agent has, or beating it
    proves nothing. Tuning it lazily is, in the spec's words, "the
    fastest way for a reviewer to dismiss the paper".
    """

    def __init__(
        self,
        pool_fill_threshold: float,
        min_hybrid_class: int = 0,
        rekey_age_frac: float = 0.9,
        max_key_age: float | None = None,
    ) -> None:
        self.pool_fill_threshold = pool_fill_threshold
        self.min_hybrid_class = int(min_hybrid_class)
        self.rekey_age_frac = float(rekey_age_frac)
        self.max_key_age = (
            float(max_key_age)
            if max_key_age is not None
            else float(load_key_lifetime_config()["max_key_age_steps"])
        )

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        # 1. key-lifetime rule: hold the existing key until it is `rho` of the
        #    way to the SP 800-57-derived cap `L`.
        #
        #    `state["key_age"]` is NORMALISED by `L` (spec §4.2), so the
        #    comparison is directly against `rho`. It used to be raw steps;
        #    comparing a 0..1 value against `rho * L` would make this branch
        #    always true and the policy would never rekey at all.
        if float(state["key_age"]) < self.rekey_age_frac and mask[int(Action.REUSE)]:
            return Action.REUSE

        # 2. tier rule: spend the pool only above both thresholds.
        spends_pool = (
            state["pool_fill"] >= self.pool_fill_threshold
            and int(state["sensitivity_class"]) >= self.min_hybrid_class
        )
        if spends_pool and mask[int(Action.SERVE_HYBRID)]:
            return Action.SERVE_HYBRID
        if mask[int(Action.SERVE_PQC)]:
            return Action.SERVE_PQC
        return _lowest_legal_action(mask)

    def __repr__(self) -> str:
        return (
            f"StaticThresholdPolicy(tau={self.pool_fill_threshold}, "
            f"c_min={self.min_hybrid_class}, rho={self.rekey_age_frac})"
        )

    @staticmethod
    def grid_search(
        candidate_thresholds: list[float], eval_fn: Callable[[StaticThresholdPolicy], float]
    ) -> StaticThresholdPolicy:
        """Pick the threshold maximizing `eval_fn(StaticThresholdPolicy(t))`.

        Single-parameter sweep over `tau` only, kept for existing
        callers. The full three-parameter search the spec asks for lives
        in `experiments/gate_w3.py::tuned_threshold_for`, because it
        needs per-scenario evaluation across multiple seeds and that is
        the harness's job, not this class's.
        """
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


class GreedyRecommenderPolicy:
    """Per-request-optimal, ignoring the pool's future
    (SMARTKEYNET_BUILD_SPEC.md §S7 diagnostic 6).

    This is the "isn't this just a recommender system?" objection from
    PLAN.md §8, turned into a number. A recommender optimises each
    request in isolation; this policy does exactly that -- it takes the
    legal action with the best *immediate* reward, with no regard for
    the fact that spending a key now removes an option later.

    Immediate reward in this environment orders the actions strictly:
    `REUSE` is cheapest (no rekey cost, lowest latency and energy), then
    classical, then PQC, and hybrid last because it additionally pays
    the `w_qkd` scarcity price. Myopic optimality is therefore just "the
    cheapest legal action", encoded directly here rather than by
    importing the env's private cost tables.

    The gap between this and the DQN is the value of *coupling*
    decisions through the pool. If that gap is zero, the problem really
    is a recommender problem and the project's premise fails.
    """

    _CHEAPEST_FIRST: tuple[Action, ...] = (
        Action.REUSE,  # no rekey cost, lowest latency/energy
        Action.SERVE_CLASSICAL,
        Action.SERVE_PQC,
        Action.REKEY_NOW,  # resolves to the current tier; never cheaper than serving it directly
        Action.SERVE_HYBRID,  # additionally pays the QKD scarcity price
    )

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        for action in self._CHEAPEST_FIRST:
            if mask[int(action)]:
                return action
        return _lowest_legal_action(mask)


class RandomPolicy:
    """Uniform-random choice among currently-legal actions."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, state: StateDict, mask: ActionMask) -> Action:
        legal = [action for action in Action if mask[int(action)]]
        if not legal:
            raise ValueError(
                "no legal action in mask -- a valid mask must have at least one True entry"
            )
        return self._rng.choice(legal)
