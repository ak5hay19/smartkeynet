"""
env/reward.py

The reward function, and the only place reward arithmetic happens.

---------------------------------------------------------------------
Why this module exists at all (Hard Rule 1)
---------------------------------------------------------------------
PLAN.md Hard Rule 1 and SMARTKEYNET_BUILD_SPEC.md §2.1 require that the
reward function be *physically unable* to read security state -- threat
score, posture, floor, tier, sensitivity class. The spec's mechanism for
that is a type: the reward takes a `RewardInputs` and nothing else, and
`RewardInputs` has no security-flavoured field.

Until 2026-08-19 this repo had the type but not the mechanism. The
reward was computed inline inside `SmartKeyNetEnv._apply_action`, which
has the request, the floor, the posture and the policy table all in
scope; `metrics/reward_inputs.py` existed and was imported by nothing.
What actually guarded the rule was a substring scan of that one method
for the words "threat"/"posture"/"security"/"risk" -- a real check, but
one that an alias (`t = state["threat_score"]`) defeats, and one that
says nothing about the twenty other lines of that method.

Hard Rule 1 is the project's central claim: the paper argues that
keeping security out of the objective and into the constraint is what
makes the agent unsteerable. A claim that rests on a grep is not a
claim. So the reward now lives here, behind the type the spec chose,
and the environment's job is reduced to *assembling* a `RewardInputs`.

Two machine checks back this up, both in tests/test_hard_rules.py:
  - `test_reward_inputs_has_no_security_fields` pins the field set;
  - `test_reward_module_imports_no_security_state` AST-parses this file
    and asserts it imports nothing from `env.policy_table`,
    `env.masking`, `env.contracts`' posture types, or `forecaster`.

---------------------------------------------------------------------
Normalisation (spec §S5 point 2)
---------------------------------------------------------------------
Every term is normalised *before* weighting -- latency per 100 ms,
energy per reference mJ, QKD per whole 256-bit key. The spec's reason is
worth restating: if you weight raw quantities, the weights silently
encode unit conversions and become impossible to reason about. A reader
asking "is w_latency=1.0 big?" can only answer it if 1.0 multiplies a
dimensionless number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from metrics.reward_inputs import RewardInputs

LATENCY_REFERENCE_MS: float = 100.0
"""Latency normalisation divisor (spec §S5: "latency per 100 ms")."""

ENERGY_REFERENCE_MJ: float = 1.0
"""Energy normalisation divisor, in mJ (spec §S5: "energy per reference mJ")."""


@dataclass(frozen=True)
class RewardWeights:
    """The tunable half of the reward. Frozen after Gate W3 per §7.4 --
    changing weights once results exist invalidates every comparison.
    """

    w_latency: float
    w_energy: float
    w_freshness: float
    w_qkd: float
    r_starve: float
    c_rekey_base: float
    c_rekey_load_beta: float

    @classmethod
    def from_config(cls, reward_config: dict[str, Any]) -> "RewardWeights":
        """Build from `configs/default.yaml`'s `reward:` block.

        The config's short names (`w_lat`, `w_en`, `w_fr`) predate this
        module and are kept so existing configs and every recorded run
        stay loadable; the mapping is spelled out here rather than
        duplicated at each call site.
        """
        return cls(
            w_latency=float(reward_config["w_lat"]),
            w_energy=float(reward_config["w_en"]),
            w_freshness=float(reward_config["w_fr"]),
            w_qkd=float(reward_config["w_qkd"]),
            r_starve=float(reward_config["r_starve"]),
            c_rekey_base=float(reward_config["c_rekey_base"]),
            c_rekey_load_beta=float(reward_config["c_rekey_load_beta"]),
        )


def compute_reward(
    inputs: RewardInputs, weights: RewardWeights
) -> tuple[float, dict[str, float]]:
    """Total reward for one step, plus the per-term breakdown.

    The breakdown is not optional instrumentation. Spec §S5 point 1 and
    the whole of §7 are built on being able to read which term dominates
    -- "ninety percent of RL debugging in section 7 is reading which term
    dominates" -- and §3.3 puts `reward_terms` in the episode row for
    exactly that reason. Returning it here means no caller can compute a
    reward without also being handed the explanation.

    Signs, each asserted individually by tests/test_reward.py:
      latency, energy, qkd, starve, rekey are costs (<= 0);
      freshness is the only bonus (>= 0).
    """
    lifetime_cap = max(1, int(inputs.key_lifetime_cap_steps))
    age_fraction = min(1.0, max(0.0, inputs.key_age_steps / lifetime_cap))
    freshness = 1.0 - age_fraction

    rekey_cost = 0.0
    if inputs.did_rekey:
        # Rekeying during a busy period costs more than rekeying during a
        # quiet one, which is what makes "rekey early at a cheap moment"
        # a learnable skill rather than a fixed schedule.
        rekey_cost = weights.c_rekey_base * (
            1.0 + weights.c_rekey_load_beta * inputs.normalised_load
        )

    terms: dict[str, float] = {
        "latency": -weights.w_latency * (inputs.latency_ms / LATENCY_REFERENCE_MS),
        "energy": -weights.w_energy * (inputs.energy_mj / ENERGY_REFERENCE_MJ),
        "freshness": weights.w_freshness * freshness,
        "qkd": -weights.w_qkd * inputs.qkd_keys_consumed,
        "starve": -weights.r_starve * inputs.deferred_critical_steps,
        "rekey": -rekey_cost,
    }
    return sum(terms.values()), terms


def assert_weights_are_sane(weights: RewardWeights) -> None:
    """Guard the one inequality that can invert the headline result.

    Spec §S5 test 5: `r_starve >= 5 * w_qkd`. If starving is cheaper than
    spending a key, the agent learns to starve high-sensitivity requests
    rather than serve them, and the paper's central claim reverses. This
    is checked at construction, not in review.
    """
    if weights.r_starve < 5.0 * weights.w_qkd:
        raise ValueError(
            f"r_starve ({weights.r_starve}) must be >= 5 * w_qkd ({5.0 * weights.w_qkd}) "
            "or the agent learns to starve instead of spend "
            "(SMARTKEYNET_BUILD_SPEC.md §S5 test 5)"
        )
