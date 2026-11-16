"""Behavioral tests for `agents.reuse_first.ReuseFirstPolicy`.

The property that matters, tested first and adversarially (same
discipline as `tests/test_baselines.py`): the policy can never return an
action outside the mask it was given, however contrived the mask -- and
then, on a real environment episode, that every action it picks is legal
under the mask the env itself produced and delivers zero below-floor
service (Hard Rule 2 is inherited from the mask, not re-implemented).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from agents.reuse_first import ReuseFirstPolicy
from env.contracts import N_ACTIONS, Action
from experiments.harness import run_scenario
from experiments.train import load_full_config


def _mask(*legal_actions: Action) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for action in legal_actions:
        mask[int(action)] = True
    return mask


def _all_nonempty_masks() -> list[np.ndarray]:
    masks = []
    for r in range(1, N_ACTIONS + 1):
        for combo in itertools.combinations(list(Action), r):
            masks.append(_mask(*combo))
    return masks


def test_never_returns_an_action_outside_any_nonempty_mask() -> None:
    policy = ReuseFirstPolicy()
    masks = _all_nonempty_masks()
    assert len(masks) == 2**N_ACTIONS - 1
    for mask in masks:
        action = policy.act({}, mask)  # type: ignore[arg-type]
        assert mask[int(action)], f"returned illegal {action!r} for mask {mask}"


def test_takes_reuse_whenever_it_is_legal() -> None:
    policy = ReuseFirstPolicy()
    for mask in _all_nonempty_masks():
        if mask[int(Action.REUSE)]:
            assert policy.act({}, mask) is Action.REUSE  # type: ignore[arg-type]


def test_falls_back_to_the_lowest_legal_action_when_reuse_is_masked() -> None:
    policy = ReuseFirstPolicy()
    for mask in _all_nonempty_masks():
        if mask[int(Action.REUSE)]:
            continue
        expected = next(a for a in Action if mask[int(a)])
        assert policy.act({}, mask) is expected  # type: ignore[arg-type]


def test_specific_fallbacks() -> None:
    policy = ReuseFirstPolicy()
    # Floor at PQC: classical already masked -> cheapest legal tier is PQC.
    assert policy.act({}, _mask(Action.SERVE_PQC, Action.SERVE_HYBRID, Action.REKEY_NOW)) is Action.SERVE_PQC  # type: ignore[arg-type]
    # Hybrid-mandatory, pool can draw, REUSE aged out.
    assert policy.act({}, _mask(Action.SERVE_HYBRID, Action.REKEY_NOW)) is Action.SERVE_HYBRID  # type: ignore[arg-type]
    # Contrived: only REKEY_NOW legal.
    assert policy.act({}, _mask(Action.REKEY_NOW)) is Action.REKEY_NOW  # type: ignore[arg-type]


def test_empty_mask_raises() -> None:
    with pytest.raises(ValueError):
        ReuseFirstPolicy().act({}, np.zeros(N_ACTIONS, dtype=bool))  # type: ignore[arg-type]


class _LegalityRecordingPolicy:
    """Wraps a policy and records every decision against the REAL mask
    the env handed it, so an illegal choice cannot hide inside a
    summary metric."""

    def __init__(self, inner: ReuseFirstPolicy) -> None:
        self.inner = inner
        self.n_decisions = 0
        self.n_illegal = 0
        self.n_reuse = 0

    def act(self, state, mask):  # type: ignore[no-untyped-def]
        action = self.inner.act(state, mask)
        self.n_decisions += 1
        if not mask[int(action)]:
            self.n_illegal += 1
        if action is Action.REUSE:
            self.n_reuse += 1
        return action


@pytest.mark.parametrize("scenario,config_path", [("S1", None), ("S3", "configs/scenarios/s3_degradation.yaml")])
def test_real_episode_is_always_legal_and_never_below_floor(scenario: str, config_path: str | None) -> None:
    config = load_full_config(config_path) if config_path else load_full_config()
    policy = _LegalityRecordingPolicy(ReuseFirstPolicy())
    result = run_scenario(policy, scenario, config, seed=900)
    assert policy.n_decisions > 0
    assert policy.n_illegal == 0
    assert result.floor_violations == 0
    # It is a *reuse-first* policy: on a real episode it must actually reuse.
    assert policy.n_reuse > 0
