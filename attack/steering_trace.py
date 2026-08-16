"""
attack/steering_trace.py

Adversarial threat-trace generator for the steering attack (PLAN.md
§2 research problem; §5 scenario S5 -- the headline contribution;
§6 Demo Beat 3). PLAN.md §7 lists this as the one thing never to cut.

---------------------------------------------------------------------
The claim being tested
---------------------------------------------------------------------
Prior RL-for-crypto designs place security in the *reward*. PLAN.md §2
argues that this is an attack surface: an adversary who can shape the
threat signal can talk such an agent into serving weaker key material
without touching any cryptography. Our design puts security in a
*constraint* (action masking driven by a policy table), where the same
manipulated signal can only ever raise floors.

This module builds the adversarially shaped trace;
`experiments/steering_attack.py` drives both agents through it on the
same environment, same seeds and same arrival stream, so the only
difference between them is the architecture.

---------------------------------------------------------------------
The attack: suppression, not injection
---------------------------------------------------------------------
The attacker wants *less* protection, so the attack **suppresses** the
threat signal rather than amplifying it. `SuppressionTrace` scales the
threat features the forecaster observes toward zero during a window --
modelling an adversary with partial influence over telemetry, quietening
indicators rather than fabricating them.

`dose` in [0, 1] is the suppression strength: 0 is no attack (the
control), 1 fully zeroes the signal. Sweeping it produces the
dose-response curve PLAN.md §7 allows to be cut back to a single point.

Two properties of our design make the attack structurally futile, and
the tests assert them rather than arguing them:

  1. **The threat signal only reaches the policy table**, where the
     posture is the argmax of a distribution that can only move the
     floor up, never down.
  2. **`PolicyTable`'s sticky ratchet** holds a posture once reached
     for the rest of the episode. Suppression therefore cannot undo
     protection already triggered -- at most it delays the trigger, and
     only while sustained, which is what makes the attack detectable
     (`detectability_score`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SuppressionTrace:
    """An adversarially shaped threat trace.

    Multiplies the threat features the environment reports by
    `1 - suppression` inside `[start_step, end_step)`, leaving them
    untouched elsewhere. `dose=0.0` reproduces the unattacked signal
    exactly, which is what makes the control run a genuine control
    rather than a separately-configured episode.

    Ramping in and out over `ramp_steps` models an adversary avoiding
    the obvious step change a first-difference detector would trip on
    instantly. `ramp_steps=0` gives the blunt rectangular version.
    """

    start_step: int
    end_step: int
    dose: float
    ramp_steps: int = 50

    def __post_init__(self) -> None:
        if not 0.0 <= self.dose <= 1.0:
            raise ValueError(f"dose must be in [0, 1], got {self.dose}")
        if self.end_step <= self.start_step:
            raise ValueError(f"empty attack window: [{self.start_step}, {self.end_step})")
        if self.ramp_steps < 0:
            raise ValueError(f"ramp_steps must be non-negative, got {self.ramp_steps}")

    def suppression_at(self, step: int) -> float:
        """Fraction of the threat signal removed at `step`, in [0, 1]."""
        if step < self.start_step or step >= self.end_step:
            return 0.0
        if self.ramp_steps == 0:
            return self.dose

        into_window = step - self.start_step
        out_of_window = self.end_step - step
        ramp_progress = min(into_window, out_of_window) / self.ramp_steps
        return self.dose * min(1.0, ramp_progress)

    def multiplier_at(self, step: int) -> float:
        """Factor applied to the observed threat features at `step`."""
        return 1.0 - self.suppression_at(step)

    def apply(self, threat_features: list[float], step: int) -> list[float]:
        """Scale one step's threat-feature vector."""
        multiplier = self.multiplier_at(step)
        return [float(value) * multiplier for value in threat_features]


def detectability_score(trace: SuppressionTrace, episode_steps: int) -> dict[str, float]:
    """How visible is this attack to a simple anomaly detector?

    SMARTKEYNET_BUILD_SPEC.md §S4 makes this worth measuring: the policy
    table's ratchet means "even a successful suppression takes
    `posture_hold_steps` to have any effect, so the attack has to be
    sustained and therefore detectable". An attack that is cheap to spot
    is a weak attack, and reporting that cost is what keeps the claim
    honest rather than triumphalist.

    Two statistics a defender would plausibly have:
      - `duty_cycle` -- fraction of the episode under suppression at
        all. Sustained suppression is conspicuous.
      - `max_first_difference` -- largest step-to-step change in the
        multiplier. A rectangular attack trips this immediately;
        ramping reduces it, at the cost of a longer duty cycle. The
        attacker cannot minimise both.
    """
    multipliers = np.array([trace.multiplier_at(step) for step in range(episode_steps)])
    suppressed_steps = int((multipliers < 1.0).sum())
    first_differences = np.abs(np.diff(multipliers)) if episode_steps > 1 else np.array([0.0])

    return {
        "duty_cycle": suppressed_steps / max(1, episode_steps),
        "max_first_difference": float(first_differences.max()),
        "mean_suppression": float(1.0 - multipliers.mean()),
    }


def dose_response_traces(
    start_step: int,
    end_step: int,
    doses: list[float],
    ramp_steps: int = 50,
) -> list[SuppressionTrace]:
    """One trace per dose, all sharing the same window and ramp.

    The dose-0 trace is included deliberately: it is the control, and it
    must be produced by the same code path as the attacked runs so that
    any difference between them is attributable to the dose alone.
    """
    return [
        SuppressionTrace(start_step=start_step, end_step=end_step, dose=dose, ramp_steps=ramp_steps)
        for dose in doses
    ]
