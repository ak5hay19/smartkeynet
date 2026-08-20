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

Threat model
------------
The adversary's capability is deliberately narrow and deliberately
realistic: it can **shape the threat signal**, and nothing else. It
cannot touch the reward, the policy table, the mask, the pool, the
agent's weights, or the traffic the tenants actually send. It is an
adversary that has compromised or spoofed the *sensor*, not the system
-- the "plays soothing music so the guard relaxes" adversary of PLAN2
§2's second analogy.

That is exactly the capability a soft-reward design is vulnerable to,
because in such a design the observed threat score is a *coefficient on
a preference*: drive it toward zero and the security term goes to zero
with it, leaving a cost-minimizing agent that picks the cheapest, and
therefore weakest, action. The attack does not need to be clever. It
needs to be quiet.

Why suppression rather than amplification
-----------------------------------------
An adversary who *inflates* the threat signal makes both agents serve
stronger key material -- costly, but not a security failure. The
interesting direction, and the one Hard Rule 2 is written against, is
suppression: convincing the system that things are calmer than they
are. `dose` is how hard the adversary pushes in that direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_CALM_LEVEL = 0.0
"""The standardized threat level of genuinely benign traffic -- the
value the adversary is trying to make the system believe."""

_PLAUSIBILITY_JITTER = 0.15
"""A perfectly flat signal is a detectable signal. A real suppression
attack has to keep the trace looking like a measurement, so the
generator preserves a small amount of noise around the suppressed
level. This is presentation, not power: it does not change the mean the
agent's reward is priced against, and setting it to 0 changes no
result reported here."""


@dataclass
class SteeringTraceConfig:
    """Dose-response sweep parameters (PLAN.md §6 Demo Beat 3)."""

    dose: float  # 0 = no manipulation, 1 = max adversarial shaping
    duration_steps: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.dose <= 1.0:
            raise ValueError(f"dose must be in [0, 1], got {self.dose}")
        if self.duration_steps <= 0:
            raise ValueError(f"duration_steps must be positive, got {self.duration_steps}")


def generate_steering_trace(
    config: SteeringTraceConfig,
    honest_trace: Sequence[float] | None = None,
) -> list[float]:
    """Generate a sequence of adversarially shaped threat-score values
    designed to talk a soft-reward agent into serving weaker keys.

    `honest_trace` is what the sensor *would* have reported -- the
    scenario's true threat level per step. When omitted, a default
    elevated trace is used so the generator is usable standalone (an
    attack on an already-calm signal has nothing to suppress and would
    make the dose axis meaningless).

    The shaping is a convex blend toward calm:

        attacked[t] = (1 - dose) * honest[t] + dose * CALM + jitter

    so `dose = 0` reproduces the honest trace exactly (the control arm
    of the dose-response sweep, which must be a genuine no-op) and
    `dose = 1` reports calm no matter what is actually happening.

    The output is clipped at `CALM` from below: an adversary gains
    nothing by driving the reported signal *below* benign, and allowing
    it would let the attack manufacture negative threat levels that no
    real sensor could produce.
    """
    rng = np.random.default_rng(config.seed)
    honest = (
        list(honest_trace)
        if honest_trace is not None
        else default_honest_trace(config.duration_steps)
    )
    if len(honest) < config.duration_steps:
        honest = honest + [honest[-1]] * (config.duration_steps - len(honest))

    trace: list[float] = []
    for step in range(config.duration_steps):
        suppressed = (1.0 - config.dose) * honest[step] + config.dose * _CALM_LEVEL
        if config.dose > 0.0:
            suppressed += float(rng.normal(0.0, _PLAUSIBILITY_JITTER * config.dose))
        trace.append(float(max(_CALM_LEVEL, suppressed)))
    return trace


def default_honest_trace(duration_steps: int, peak_level: float = 3.2) -> list[float]:
    """The honest signal the attack is run against when none is given:
    calm, then a ramp to a genuinely elevated level, then sustained.

    Mirrors the S2 HNDL ramp in `configs/default.yaml` so the standalone
    trace and the in-environment scenario tell the same story.
    """
    start = max(1, duration_steps // 10)
    ramp = max(1, duration_steps // 10)
    trace: list[float] = []
    for step in range(duration_steps):
        if step < start:
            trace.append(0.0)
        else:
            trace.append(peak_level * min(1.0, (step - start) / ramp))
    return trace


def dose_response_sweep(
    doses: list[float], config: SteeringTraceConfig
) -> dict[float, list[float]]:
    """Generate one trace per dose level for the dose-response sweep.

    Cut line #4 in split.md §2.1 -- if cut, keep the single-point
    steering result from `generate_steering_trace` alone.

    Every dose uses the *same* honest trace and the same seed, so the
    only thing varying across the sweep is the adversary's strength.
    """
    honest = default_honest_trace(config.duration_steps)
    return {
        float(dose): generate_steering_trace(
            SteeringTraceConfig(
                dose=float(dose), duration_steps=config.duration_steps, seed=config.seed
            ),
            honest_trace=honest,
        )
        for dose in doses
    }


def suppression_ratio(honest: Sequence[float], attacked: Sequence[float]) -> float:
    """How much of the honest signal's magnitude the attack removed.

    Reported alongside the dose so the sweep's x-axis has a measured
    meaning ("this dose actually suppressed 74% of the signal") rather
    than only a nominal one.
    """
    honest_total = float(np.sum(np.abs(honest)))
    if honest_total <= 0.0:
        return 0.0
    attacked_total = float(np.sum(np.abs(attacked[: len(honest)])))
    return max(0.0, 1.0 - attacked_total / honest_total)
