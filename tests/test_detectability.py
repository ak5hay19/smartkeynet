"""Tests for `attack.detectability` (SMARTKEYNET_BUILD_SPEC.md §S11)."""

from __future__ import annotations

import numpy as np
import pytest

from attack.detectability import (
    anomaly_scores,
    detection_rate,
    detector_auc,
    dose_response_detectability,
    fit_control_chart,
)


def _honest(n: int = 500, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.30, 0.05, size=n)


def test_honest_traffic_scores_near_the_baseline():
    honest = _honest()
    fit = fit_control_chart(honest)
    assert detection_rate(fit, honest) < 0.05


def test_auc_is_chance_when_comparing_honest_against_honest():
    """The floor case. If a detector 'separates' two honest traces, its AUC on
    an attacked one means nothing."""
    fit = fit_control_chart(_honest(seed=0))
    auc = detector_auc(fit, _honest(seed=1), _honest(seed=2))
    assert 0.35 < auc < 0.65


def test_suppression_is_detected():
    """The attack suppresses the threat signal, so the detector must alarm on
    a *downward* excursion -- one that only flagged unusually high values would
    be blind to this attack by construction, which would rig the comparison."""
    honest = _honest()
    fit = fit_control_chart(honest)
    suppressed = np.full(500, 0.02)
    assert detection_rate(fit, suppressed) > 0.8
    assert detector_auc(fit, honest, suppressed) > 0.9


def test_detectability_rises_with_dose():
    """The core relationship: a harder attack is a more visible one. This is
    what lets the report state the trade-off rather than just the attack."""
    honest = _honest()
    attacked = {dose: honest * (1.0 - dose) + 0.02 * dose for dose in (0.0, 0.25, 0.5, 0.75, 1.0)}
    result = dose_response_detectability(honest, attacked)

    assert result["dose"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(result["auc"]) == len(result["dose"])
    # Monotone non-decreasing in dose, allowing for sampling noise at the ends.
    assert result["auc"][-1] > result["auc"][0]
    assert result["detection_rate"][-1] >= result["detection_rate"][0]


def test_zero_dose_is_indistinguishable_from_honest():
    honest = _honest()
    fit = fit_control_chart(honest)
    assert detector_auc(fit, honest, honest) == pytest.approx(0.5, abs=0.02)


def test_fit_rejects_an_empty_signal():
    with pytest.raises(ValueError, match="empty"):
        fit_control_chart(np.array([]))


def test_scores_are_finite_on_a_constant_signal():
    """A constant signal has zero EWMA variance; the fit floors the standard
    deviation so scores stay finite rather than becoming inf/NaN."""
    fit = fit_control_chart(np.full(100, 0.3))
    scores = anomaly_scores(fit, np.full(100, 0.3))
    assert np.isfinite(scores).all()
