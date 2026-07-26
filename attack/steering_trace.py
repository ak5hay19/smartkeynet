"""
attack/steering_trace.py

Adversarial threat-trace generator for the steering attack (PLAN.md
§2 research problem; §5 scenario S5 -- the headline contribution).
Owned by Person D (split.md §1).

Feeds an adversarially shaped threat signal to both
`agents/soft_reward_baseline.py` (expected to bend: served-tier
histogram slides toward weaker keys) and the masked DQN (expected to
be immune: floor line can only step up). Never cut (split.md §2.1,
"NEVER cut: ... the steering attack").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SteeringTraceConfig:
    """Dose-response sweep parameters (PLAN.md §6 Demo Beat 3)."""

    dose: float  # 0 = no manipulation, 1 = max adversarial shaping
    duration_steps: int
    seed: int | None = None


def generate_steering_trace(config: SteeringTraceConfig) -> list[float]:
    """Generate a sequence of adversarially shaped threat-score values
    designed to talk a soft-reward agent into serving weaker keys."""
    raise NotImplementedError


def dose_response_sweep(
    doses: list[float], config: SteeringTraceConfig
) -> dict[float, list[float]]:
    """Generate one trace per dose level for the dose-response sweep.

    Cut line #4 in split.md §2.1 -- if cut, keep the single-point
    steering result from `generate_steering_trace` alone.
    """
    raise NotImplementedError
