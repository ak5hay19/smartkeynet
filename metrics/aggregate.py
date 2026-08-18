"""
metrics/aggregate.py

The statistical protocol from SMARTKEYNET_BUILD_SPEC.md §9. Every number
that appears in the report should come through here.

The three rules that matter, and why:

  1. **The unit of analysis is the seed, not the episode.** Episodes within
     one seed share an arrival stream and an SKR trace, so treating them as
     independent samples inflates n and shrinks every interval.
  2. **IQM, not the mean.** The interquartile mean discards the top and
     bottom quartile across seeds. With RL that matters concretely: one
     diverged seed can move a mean by an order of magnitude, and this project
     has measured per-seed spreads from -1,326 to -3,015,813 on the same
     configuration.
  3. **Paired comparisons.** Policies are run on shared seeds under common
     random numbers, so the per-seed *difference* has far less variance than
     either policy's own spread. Bootstrapping the difference is what makes a
     claim like "A beats B" defensible from five seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap confidence interval."""

    point: float
    low: float
    high: float
    n_seeds: int

    def excludes(self, value: float) -> bool:
        """Whether `value` lies outside the interval -- the actual test for
        "is this difference real?" when applied to a paired difference with
        `value=0`."""
        return value < self.low or value > self.high

    def __str__(self) -> str:
        return f"{self.point:.1f} [{self.low:.1f}, {self.high:.1f}] (n={self.n_seeds})"


def interquartile_mean(values: np.ndarray | list[float]) -> float:
    """Mean of the middle 50% of the sample (§9 rule 2).

    Falls back to the plain mean below four samples, where trimming a quartile
    from each end would leave nothing.
    """
    array = np.sort(np.asarray(values, dtype=float))
    if array.size < 4:
        return float(array.mean()) if array.size else float("nan")

    low = int(np.floor(array.size * 0.25))
    high = int(np.ceil(array.size * 0.75))
    return float(array[low:high].mean())


def bootstrap_ci(
    values: np.ndarray | list[float],
    statistic=interquartile_mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI over seeds (§9 rule 3)."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    resampled = [
        statistic(rng.choice(array, size=array.size, replace=True))
        for _ in range(n_resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    return Estimate(
        point=statistic(array),
        low=float(np.percentile(resampled, 100 * tail)),
        high=float(np.percentile(resampled, 100 * (1 - tail))),
        n_seeds=int(array.size),
    )


def paired_difference(
    values_a: np.ndarray | list[float],
    values_b: np.ndarray | list[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Bootstrap CI of the per-seed difference `a - b` (§9 rule 4).

    Requires the two sequences to be aligned by seed -- that is the whole
    point, and it is why the harness uses common random numbers. An interval
    that excludes zero is the claim; a p-value is optional garnish.
    """
    array_a = np.asarray(values_a, dtype=float)
    array_b = np.asarray(values_b, dtype=float)
    if array_a.shape != array_b.shape:
        raise ValueError(
            f"paired comparison needs matching seed counts, got {array_a.shape} and {array_b.shape}"
        )
    return bootstrap_ci(array_a - array_b, n_resamples=n_resamples, confidence=confidence, seed=seed)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction (§9 rule 5).

    Returns per-comparison reject/accept in the caller's original order. With
    ~7 policies x 5 scenarios the family is large enough that uncorrected
    comparisons would be an easy criticism; one sentence of correction removes
    it.
    """
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    rejected = [False] * n
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (n - rank):
            rejected[index] = True
        else:
            break  # step-down: once one fails, all larger p-values fail
    return rejected
