"""
forecaster/calibrate.py

Calibration for the threat head (SMARTKEYNET_BUILD_SPEC.md §S9).

Why this matters operationally rather than academically: the threat head
drives floors, and `PolicyTable`'s ratchet is one-way. An overconfident head
raises a floor that never comes back down, so overconfidence is paid for in
deferrals for the rest of the episode -- measured on this project's own S1
baseline, a single false-positive step held the floor up for the remaining
1,199 steps.

So the number that belongs in the report is not ECE. It is the **floor
over-raise rate**: how often the derived posture exceeded the true one, and
what that cost. ECE is the diagnostic; over-raise is the consequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CalibrationReport:
    ece: float
    """Expected calibration error, 15 bins."""

    temperature: float
    """Fitted scaling. >1 means the head was overconfident."""

    ece_after: float
    nll_before: float
    nll_after: float
    over_raise_rate: float
    """Fraction of steps where the predicted posture EXCEEDED the true one.

    Asymmetric on purpose: under-raising is a security failure and
    over-raising is an availability failure, and this architecture converts
    all of the former into the latter. This measures the price."""

    under_raise_rate: float

    def __str__(self) -> str:
        return (
            f"ECE {self.ece:.4f} -> {self.ece_after:.4f} (T={self.temperature:.3f}), "
            f"over-raise {self.over_raise_rate:.3f}, under-raise {self.under_raise_rate:.3f}"
        )


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Standard ECE: |confidence - accuracy| averaged over confidence bins,
    weighted by bin population."""
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (confidences > low) & (confidences <= high)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(error)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Single-scalar temperature scaling, fitted by minimising NLL.

    Fitted on the VALIDATION split and applied to test -- fitting on test
    would be calibrating to the thing being measured.
    """
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits / torch.exp(log_temperature)
        loss = torch.nn.functional.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).item())


def calibrate_threat_head(
    model,
    validation_windows: torch.Tensor,
    validation_targets: torch.Tensor,
    test_windows: torch.Tensor,
    test_targets: torch.Tensor,
) -> CalibrationReport:
    """Fit temperature on validation, report ECE and over-raise on test."""
    model.eval()
    with torch.no_grad():
        val_logits, _ = model(validation_windows)
        test_logits, _ = model(test_windows)

    # flatten the k-step horizon into independent classification decisions
    val_flat = val_logits.reshape(-1, val_logits.shape[-1])
    val_labels = validation_targets.reshape(-1)
    test_flat = test_logits.reshape(-1, test_logits.shape[-1])
    test_labels = test_targets.reshape(-1)

    temperature = fit_temperature(val_flat, val_labels)

    before = torch.softmax(test_flat, dim=-1).numpy()
    after = torch.softmax(test_flat / temperature, dim=-1).numpy()
    labels = test_labels.numpy()

    predictions = after.argmax(axis=1)
    return CalibrationReport(
        ece=expected_calibration_error(before, labels),
        ece_after=expected_calibration_error(after, labels),
        temperature=temperature,
        nll_before=float(torch.nn.functional.cross_entropy(test_flat, test_labels).item()),
        nll_after=float(
            torch.nn.functional.cross_entropy(test_flat / temperature, test_labels).item()
        ),
        over_raise_rate=float((predictions > labels).mean()),
        under_raise_rate=float((predictions < labels).mean()),
    )
