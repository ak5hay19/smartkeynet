"""
metrics/reward_inputs.py

Hard Rule 1, enforced by TYPE rather than by vigilance --
SMARTKEYNET_BUILD_SPEC.md §2.1.

The spec's reasoning, which is worth keeping in front of anyone editing this
file: "Rules that live only in a review checklist get violated at 2 a.m. in
week 5. Make them structural."

`RewardInputs` is the complete set of things the reward function is permitted
to see. Because it is a frozen dataclass with an exact, tested field list, a
reward term that wanted to read the threat score would have to widen this type
first -- a visible, reviewable, test-breaking change rather than a quiet one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardInputs:
    """The ONLY thing the reward function may see.

    Deliberately excludes: threat_score, threat_forecast, posture, floor,
    tier_of_action, sensitivity_class, security_level. Hard Rule 1.
    Adding any security-flavoured field here is a thesis-level defect.
    """

    latency_ms: float  # realised latency of this step
    energy_mj: float  # primitive + handshake energy
    key_age_steps: int  # for the freshness bonus
    key_lifetime_cap_steps: int  # SP 800-57 derived cap L
    qkd_keys_consumed: int  # 256-bit units drawn this step
    deferred_critical_steps: int  # waiting steps accrued this step
    did_rekey: bool
    normalised_load: float  # 0..1, for load-scaled rekey cost


FORBIDDEN_FIELD_SUBSTRINGS: frozenset[str] = frozenset(
    {"threat", "posture", "floor", "tier", "sensitivity", "security", "risk", "confidential"}
)
"""Vocabulary that must never appear in a `RewardInputs` field name (§2.1)."""
