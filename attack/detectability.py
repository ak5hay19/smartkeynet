"""
attack/detectability.py

How visible is the steering trace to a defender who is simply watching?

---------------------------------------------------------------------
Why this exists
---------------------------------------------------------------------
SMARTKEYNET_BUILD_SPEC.md §S11 asks for this explicitly, and gives the reason:
it "pre-empts the obvious objection". The objection is that an attack which
suppresses threat indicators hard enough to steer a victim is also suppressing
them hard enough to look obviously wrong, so the whole scenario is unrealistic.

Answering it needs one extra axis. With detectability measured, the paper can
say both halves of a much stronger claim:

    "the trace that meaningfully steers the victim is detectable at AUC X,
     and even an undetectable trace cannot move our agent at all."

The second half is what the masking architecture buys, and it is only
interesting once the first half is quantified.

---------------------------------------------------------------------
The detector
---------------------------------------------------------------------
Deliberately simple, and deliberately *not* tuned against the attack: an EWMA
control chart on the threat signal, which is the kind of monitoring a real
operations team already has. Tuning a bespoke detector against this specific
trace family would measure the detector's overfitting rather than the attack's
stealth, and would flatter the defence.

The control chart is fitted on honest traffic only. Its statistic is the
standardised deviation of the EWMA from the honest mean; a point is flagged
when that exceeds `n_sigma`. AUC is computed over the flagged-score
distribution rather than at a single threshold, so the number does not depend
on where the alarm line happens to sit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorFit:
    """An EWMA control chart fitted on honest traffic."""

    mean: float
    std: float
    span: int

    @property
    def alpha(self) -> float:
        return 2.0 / (self.span + 1.0)


def fit_control_chart(honest_signal: np.ndarray, span: int = 25) -> DetectorFit:
    """Fit on honest traffic only.

    Fitting on a mixture of honest and attacked traffic would be the classic
    leak: the detector would learn the attack's own statistics and report a
    detection rate no real defender could achieve, because a real defender
    never has the attacked distribution in advance.
    """
    signal = np.asarray(honest_signal, dtype=np.float64)
    if signal.size == 0:
        raise ValueError("cannot fit a control chart on an empty signal")
    smoothed = _ewma(signal, span)
    return DetectorFit(
        mean=float(smoothed.mean()),
        std=float(max(smoothed.std(), 1e-9)),
        span=span,
    )


def _ewma(signal: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    smoothed = np.empty_like(signal, dtype=np.float64)
    running = float(signal[0])
    for index, value in enumerate(signal):
        running = alpha * float(value) + (1.0 - alpha) * running
        smoothed[index] = running
    return smoothed


def anomaly_scores(fit: DetectorFit, signal: np.ndarray) -> np.ndarray:
    """Standardised absolute deviation of the EWMA from the honest mean.

    Absolute, because *suppression* is what this attack does -- a detector that
    only alarmed on the signal being unusually HIGH would be blind to it by
    construction, which would be a rigged comparison.
    """
    smoothed = _ewma(np.asarray(signal, dtype=np.float64), fit.span)
    return np.abs(smoothed - fit.mean) / fit.std


def detection_rate(fit: DetectorFit, signal: np.ndarray, n_sigma: float = 3.0) -> float:
    """Fraction of steps the control chart alarms on."""
    return float((anomaly_scores(fit, signal) > n_sigma).mean())


def detector_auc(fit: DetectorFit, honest: np.ndarray, attacked: np.ndarray) -> float:
    """AUC of the detector separating honest from attacked traces.

    Computed by the Mann-Whitney U identity rather than by sweeping thresholds,
    which makes it exact and threshold-free. 0.5 means the detector cannot tell
    the two apart at all; 1.0 means perfect separation.
    """
    honest_scores = anomaly_scores(fit, honest)
    attacked_scores = anomaly_scores(fit, attacked)
    if honest_scores.size == 0 or attacked_scores.size == 0:
        raise ValueError("both score sets must be non-empty")

    combined = np.concatenate([honest_scores, attacked_scores])
    ranks = combined.argsort().argsort().astype(np.float64) + 1.0
    attacked_rank_sum = ranks[honest_scores.size :].sum()
    n_attacked = float(attacked_scores.size)
    n_honest = float(honest_scores.size)
    u_statistic = attacked_rank_sum - n_attacked * (n_attacked + 1.0) / 2.0
    return float(u_statistic / (n_attacked * n_honest))


def dose_response_detectability(
    honest_signal: np.ndarray,
    attacked_signals: dict[float, np.ndarray],
    span: int = 25,
    n_sigma: float = 3.0,
) -> dict[str, list[float]]:
    """Detectability at each attack dose.

    Returns parallel lists so the plot script and the results JSON read the
    same shape as the rest of the dose-response output.
    """
    fit = fit_control_chart(honest_signal, span=span)
    doses = sorted(attacked_signals)
    return {
        "dose": [float(dose) for dose in doses],
        "detection_rate": [detection_rate(fit, attacked_signals[dose], n_sigma) for dose in doses],
        "auc": [detector_auc(fit, honest_signal, attacked_signals[dose]) for dose in doses],
    }
