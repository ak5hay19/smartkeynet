"""
attack/search.py

Rung 2 of the attack ladder -- black-box optimisation over the trace family
(SMARTKEYNET_BUILD_SPEC.md §S11).

Rung 1 (a hand-crafted suppression trace) is in `attack/steering_trace.py`.
This module widens the parameter set and searches it, so the reported attack
is the strongest one available within the plausibility budget rather than the
first one that happened to work.

---------------------------------------------------------------------
Common random numbers are not optional here
---------------------------------------------------------------------
The spec flags the failure mode by name: evaluating each candidate on
*different* episodes means the optimiser maximises seed noise rather than
attack strength, and the result looks strong and reproduces at chance. Every
candidate in `search` is scored on the identical seed list, and
`test_search_uses_common_random_numbers` asserts it.

CMA-ES is used when the optional `cma` package is present; otherwise the
search falls back to seeded random search over the same space. Both are
reported, per the spec, because a random-search result that matches CMA-ES is
evidence the space is easy rather than that the optimiser is good.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from attack.steering_trace import SuppressionTrace


@dataclass(frozen=True)
class TraceParameters:
    """The searchable parameterisation of a suppression trace.

    Deliberately small and interpretable. A 50-parameter per-feature gain
    vector would search a larger space, but every extra dimension is another
    thing a reviewer must be convinced is physically realisable, and the
    marginal attack strength here is not worth that.
    """

    dose: float
    ramp_steps: int
    start_fraction: float

    def to_trace(self, episode_steps: int) -> SuppressionTrace:
        start = int(self.start_fraction * episode_steps)
        return SuppressionTrace(
            start_step=start,
            end_step=episode_steps,
            dose=float(np.clip(self.dose, 0.0, 1.0)),
            ramp_steps=int(max(0, self.ramp_steps)),
        )


BOUNDS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),  # dose -- the plausibility budget, epsilon in the spec's terms
    (0.0, 300.0),  # ramp_steps -- smoothness; 0 is a detectable step change
    (0.05, 0.8),  # start_fraction -- when in the episode the attack begins
)


def _project(vector: np.ndarray) -> np.ndarray:
    """Clip into the plausibility box (§S11: 'project theta after each step')."""
    return np.array(
        [np.clip(value, low, high) for value, (low, high) in zip(vector, BOUNDS)]
    )


def _to_parameters(vector: np.ndarray) -> TraceParameters:
    projected = _project(vector)
    return TraceParameters(
        dose=float(projected[0]),
        ramp_steps=int(projected[1]),
        start_fraction=float(projected[2]),
    )


@dataclass
class SearchResult:
    best: TraceParameters
    best_score: float
    method: str
    n_evaluations: int
    history: list[float]
    seeds_used: list[int]


def search(
    objective: Callable[[TraceParameters, Sequence[int]], float],
    seeds: Sequence[int],
    n_evaluations: int = 200,
    method: str = "auto",
    rng_seed: int = 0,
) -> SearchResult:
    """Maximise `objective(parameters, seeds)` over the trace family.

    `objective` is passed the SAME `seeds` on every call -- that is the common
    random numbers guarantee, and it is why the seed list is a parameter here
    rather than something the objective chooses.
    """
    if method == "auto":
        try:
            import cma  # noqa: F401

            method = "cma"
        except ImportError:
            method = "random"

    if method == "cma":
        return _search_cma(objective, seeds, n_evaluations, rng_seed)
    return _search_random(objective, seeds, n_evaluations, rng_seed)


def _search_random(
    objective: Callable[[TraceParameters, Sequence[int]], float],
    seeds: Sequence[int],
    n_evaluations: int,
    rng_seed: int,
) -> SearchResult:
    rng = np.random.default_rng(rng_seed)
    best_vector, best_score, history = None, -np.inf, []

    for _ in range(n_evaluations):
        candidate = np.array([rng.uniform(low, high) for low, high in BOUNDS])
        score = objective(_to_parameters(candidate), seeds)
        history.append(score)
        if score > best_score:
            best_score, best_vector = score, candidate

    return SearchResult(
        best=_to_parameters(best_vector if best_vector is not None else np.zeros(len(BOUNDS))),
        best_score=float(best_score),
        method="random",
        n_evaluations=n_evaluations,
        history=history,
        seeds_used=list(seeds),
    )


def _search_cma(
    objective: Callable[[TraceParameters, Sequence[int]], float],
    seeds: Sequence[int],
    n_evaluations: int,
    rng_seed: int,
) -> SearchResult:
    import cma

    centre = [np.mean(bound) for bound in BOUNDS]
    sigma = float(np.mean([(high - low) / 4.0 for low, high in BOUNDS]))
    strategy = cma.CMAEvolutionStrategy(
        centre, sigma, {"popsize": 8, "seed": rng_seed + 1, "verbose": -9}
    )

    best_vector, best_score, history, evaluations = None, -np.inf, [], 0
    while evaluations < n_evaluations and not strategy.stop():
        population = strategy.ask()
        # CMA-ES minimises, so negate the objective we want maximised.
        scores = [objective(_to_parameters(np.array(v)), seeds) for v in population]
        strategy.tell(population, [-s for s in scores])
        evaluations += len(population)
        history.extend(scores)
        for vector, score in zip(population, scores):
            if score > best_score:
                best_score, best_vector = score, np.array(vector)

    return SearchResult(
        best=_to_parameters(best_vector if best_vector is not None else np.array(centre)),
        best_score=float(best_score),
        method="cma",
        n_evaluations=evaluations,
        history=history,
        seeds_used=list(seeds),
    )
