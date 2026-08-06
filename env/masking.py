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

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from env.contracts import (
    N_ACTIONS,
    Action,
    ActionMask,
    Request,
    SensitivityClass,
    ThreatPosture,
)


def load_key_lifetime_config(path: str | Path | None = None) -> dict[str, float]:
    """Read the `key_lifetime:` block out of `configs/default.yaml`.

    `max_key_age_steps` (the SP 800-57-derived cap `L`) belongs in
    config, not as a literal in `env/environment.py` or test code --
    mirrors `env.pool_sim.load_pool_config`.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config["key_lifetime"]


# ---------------------------------------------------------------------------
# Placeholder (class x posture) -> floor table (Hard Rule 4: citable
# artifacts only, no invented constants). Q-OPSEC's `synthetic_context_dataset`
# calibration (PLAN.md "Datasets & Provenance" -> "Policy-table
# calibration" row) hasn't happened yet -- that's Person A's future
# job -- so this is a documented placeholder, not a final table.
#
# Grounding (no numeric thresholds invented -- only relative ordering,
# which is what's citable at placeholder stage):
#   - SERVE_CLASSICAL (T0, X25519/AES-256-GCM): floor for data with no
#     confidentiality exposure to Harvest-Now-Decrypt-Later (S0 public/
#     non-sensitive telemetry) under CALM/ELEVATED posture.
#   - SERVE_PQC (T1, ML-KEM-768, NIST PGC Category 3): the default
#     floor once either the data has any real confidentiality need
#     (S1+) or the posture is HIGH -- PQC is the "free" workhorse
#     (PLAN.md), so raising this floor costs nothing but interop
#     with non-PQC-capable legacy endpoints (handled by `pqc_capable`
#     at the request-generator level, not here).
#   - SERVE_HYBRID (T2/T3, ML-KEM-768 XOR QKD via HKDF): reserved for
#     where CNSA 2.0 / SP 800-57's "protect highest-sensitivity data
#     soonest, longest-lived data strongest" posture and ETSI GS QKD
#     014-delivered key material are actually warranted -- long-lived,
#     highly regulated data (S2/S3) once threat posture is elevated,
#     and S3 (patient-record-grade, decades-long lifetime) even before
#     any threat elevation, per instruction: never below SERVE_PQC at
#     CALM, escalating to SERVE_HYBRID under ELEVATED/HIGH.
#
# The one invariant that isn't a placeholder and must never regress:
# floor is monotonically non-decreasing in both sensitivity_class and
# threat_posture (verified by `PolicyTable.floor`'s docstring contract
# and covered by tests/test_masking.py's monotonicity tests).
_PLACEHOLDER_FLOOR_TABLE: dict[tuple[SensitivityClass, ThreatPosture], Action] = {
    (SensitivityClass.S0, ThreatPosture.CALM): Action.SERVE_CLASSICAL,
    (SensitivityClass.S0, ThreatPosture.ELEVATED): Action.SERVE_CLASSICAL,
    (SensitivityClass.S0, ThreatPosture.HIGH): Action.SERVE_PQC,
    (SensitivityClass.S1, ThreatPosture.CALM): Action.SERVE_CLASSICAL,
    (SensitivityClass.S1, ThreatPosture.ELEVATED): Action.SERVE_PQC,
    (SensitivityClass.S1, ThreatPosture.HIGH): Action.SERVE_PQC,
    (SensitivityClass.S2, ThreatPosture.CALM): Action.SERVE_PQC,
    (SensitivityClass.S2, ThreatPosture.ELEVATED): Action.SERVE_PQC,
    (SensitivityClass.S2, ThreatPosture.HIGH): Action.SERVE_HYBRID,
    (SensitivityClass.S3, ThreatPosture.CALM): Action.SERVE_PQC,
    (SensitivityClass.S3, ThreatPosture.ELEVATED): Action.SERVE_HYBRID,
    (SensitivityClass.S3, ThreatPosture.HIGH): Action.SERVE_HYBRID,
}


class PolicyTable:
    """Maps (sensitivity_class, threat_posture) -> minimum tier (`Action` floor).

    Calibrated against Q-OPSEC's `synthetic_context_dataset` (PLAN.md
    "Datasets & Provenance" -> "Policy-table calibration" row). Tiers
    map only to citable artifacts (Hard Rule 4): NIST PQC categories,
    SP 800-57, CNSA 2.0, BSI/ANSSI, ETSI GS QKD 014. The lookup table
    itself is currently `_PLACEHOLDER_FLOOR_TABLE` above -- documented
    placeholder, not the final calibrated mapping.

    `ratchet_up` interpretation (flagged in SESSION_LOG -- the stub's
    docstring establishes *that* threat signals may only raise floors,
    not *how* that interacts with `floor()`'s own `threat_posture`
    argument): this class keeps a sticky "ratcheted posture" floor, on
    top of whichever `threat_posture` is passed into `floor()` each
    call. `floor()` always resolves against
    `max(passed_in_posture, ratcheted_posture)`. This means once
    `ratchet_up(HIGH)` has been called, a *later* `floor()` call with
    `threat_posture=CALM` (e.g. the raw forecaster reading has since
    dropped back down) still returns at least the HIGH-posture floor
    -- a transient threat signal can raise the floor permanently for
    the life of this `PolicyTable` instance, never silently relax it
    back down when the raw signal drops (there is deliberately no
    `ratchet_down`; PLAN.md Hard Rule 2: "Threat signals may only
    *raise* floors"). Callers that want the ratchet reset (e.g. a new
    episode) should construct a new `PolicyTable`.
    """

    def __init__(self) -> None:
        self._table = _PLACEHOLDER_FLOOR_TABLE
        self._ratcheted_posture: ThreatPosture = ThreatPosture.CALM

    def floor(self, sensitivity_class: SensitivityClass, threat_posture: ThreatPosture) -> Action:
        """Return the minimum legal tier (as an `Action`) for this
        (class, posture) pair.

        Resolved against `max(threat_posture, ratcheted_posture)` --
        see the class docstring's `ratchet_up` interpretation.
        """
        effective_class = SensitivityClass(int(sensitivity_class))
        effective_posture = ThreatPosture(max(int(threat_posture), int(self._ratcheted_posture)))
        return self._table[(effective_class, effective_posture)]

    def ratchet_up(self, threat_posture: ThreatPosture) -> None:
        """Apply a threat-driven floor increase.

        Never call this to lower a floor (Hard Rule 2) -- there is
        deliberately no symmetric `ratchet_down`. A no-op if
        `threat_posture` isn't higher than the current ratchet level.
        """
        if int(threat_posture) > int(self._ratcheted_posture):
            self._ratcheted_posture = ThreatPosture(int(threat_posture))


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

    `request` is accepted for signature completeness (masking is
    always computed per-request) but isn't consulted by any of the
    three rules below -- only `floor`, `key_age`/`max_key_age`, and
    `pool_can_draw` are. (`floor` is expected to already be the output
    of `PolicyTable.floor(request['sensitivity_class'], ...)`.)
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for action in Action:
        legal = int(action) >= int(floor)
        if action is Action.SERVE_HYBRID and not pool_can_draw:
            legal = False
        if action is Action.REUSE and key_age >= max_key_age:
            legal = False
        mask[action] = legal
    return mask
