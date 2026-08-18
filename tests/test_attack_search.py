"""Tests for `attack.search` -- rung 2 of the attack ladder (§S11)."""

from __future__ import annotations

import numpy as np
import pytest

from attack.search import BOUNDS, TraceParameters, search


def test_search_finds_a_known_optimum():
    """A synthetic objective peaking at dose=1.0 must be found."""
    result = search(
        objective=lambda params, seeds: params.dose,
        seeds=[0, 1, 2],
        n_evaluations=120,
        method="random",
        rng_seed=0,
    )
    assert result.best.dose > 0.9
    assert result.best_score > 0.9


def test_search_uses_common_random_numbers():
    """The spec's named failure mode: evaluating candidates on DIFFERENT
    episodes makes the optimiser maximise seed noise. Every candidate must
    see the identical seed list."""
    seen: list[tuple[int, ...]] = []

    def objective(params, seeds):
        seen.append(tuple(seeds))
        return params.dose

    search(objective, seeds=[7, 8, 9], n_evaluations=30, method="random", rng_seed=0)
    assert len(set(seen)) == 1, "candidates were scored on different seed lists"
    assert seen[0] == (7, 8, 9)


def test_parameters_stay_inside_the_plausibility_box():
    """§S11: 'project theta after each step'. Nothing the search returns may
    sit outside the budget."""
    result = search(
        objective=lambda params, seeds: -abs(params.dose - 5.0),  # pulls out of range
        seeds=[0],
        n_evaluations=60,
        method="random",
        rng_seed=0,
    )
    assert BOUNDS[0][0] <= result.best.dose <= BOUNDS[0][1]
    assert result.best.ramp_steps >= 0


def test_produced_trace_is_valid_and_bounded():
    params = TraceParameters(dose=0.6, ramp_steps=80, start_fraction=0.25)
    trace = params.to_trace(episode_steps=2000)
    assert trace.start_step == 500
    for step in range(0, 2200, 13):
        assert 0.0 <= trace.multiplier_at(step) <= 1.0


def test_search_is_deterministic_under_seed():
    objective = lambda params, seeds: params.dose * (1 - params.start_fraction)  # noqa: E731
    a = search(objective, [0], n_evaluations=40, method="random", rng_seed=3)
    b = search(objective, [0], n_evaluations=40, method="random", rng_seed=3)
    assert a.best == b.best


def test_reports_which_method_ran():
    result = search(lambda p, s: p.dose, [0], n_evaluations=10, method="random")
    assert result.method == "random"
    assert result.n_evaluations == 10
