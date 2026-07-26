"""
env/masking.py

Policy table (sensitivity class x threat posture -> tier floor) and
structural action masking (PLAN.md §4 architecture diagram, Hard Rule
2). Owned by Person B (split.md §1).

Hard Rule 2: floors are enforced by action masking in the environment,
not by reward penalties. Threat signals may only *raise* floors --
this module must never accept a signal path that could lower one.
"""

from __future__ import annotations

import numpy as np

from env.contracts import (
    N_ACTIONS,
    Action,
    ActionMask,
    Request,
    SensitivityClass,
    ThreatPosture,
)


class PolicyTable:
    """Maps (sensitivity_class, threat_posture) -> minimum tier (`Action` floor).

    Calibrated against Q-OPSEC's `synthetic_context_dataset` (PLAN.md
    "Datasets & Provenance" -> "Policy-table calibration" row). Tiers
    map only to citable artifacts (Hard Rule 4): NIST PQC categories,
    SP 800-57, CNSA 2.0, BSI/ANSSI, ETSI GS QKD 014.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def floor(self, sensitivity_class: SensitivityClass, threat_posture: ThreatPosture) -> Action:
        """Return the minimum legal tier (as an `Action`) for this
        (class, posture) pair."""
        raise NotImplementedError

    def ratchet_up(self, threat_posture: ThreatPosture) -> None:
        """Apply a threat-driven floor increase.

        Never call this to lower a floor (Hard Rule 2) -- there is
        deliberately no symmetric `ratchet_down`.
        """
        raise NotImplementedError


def compute_mask(
    request: Request,
    floor: Action,
    key_age: float,
    max_key_age: float,
    pool_can_draw: bool,
) -> ActionMask:
    """Build the boolean action mask for one request (Hard Rules 2, 5, 9).

    Legality rules (PLAN.md Hard Rules):
      - Actions below `floor` are illegal.
      - `SERVE_HYBRID` is illegal if `not pool_can_draw` -- pool
        exhaustion routes the request to `env/deferral_queue.py`
        instead of masking in a downgrade (Hard Rule 9).
      - `REUSE` is illegal if `key_age >= max_key_age` (the SP
        800-57-derived cap `L`); this triggers a forced rekey instead,
        logged via `contracts.ForcedRekey` (Addition C).
      - Key-type changes only happen at rekey boundaries (Hard Rule
        5); this function only encodes *this step's* legality, not
        mid-session switching.

    Returns an `ActionMask` of shape (N_ACTIONS,), aligned to `Action`.
    """
    raise NotImplementedError
