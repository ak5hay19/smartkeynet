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
    KeyType,
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
# 2026-08-19 CORRECTION -- (S3, CALM): SERVE_PQC -> SERVE_HYBRID.
#
# The comment block above already states the intent: "S3 (patient-record-
# grade, decades-long lifetime) [gets SERVE_HYBRID] even before any threat
# elevation". The table did not implement it -- it gave S3 SERVE_PQC at
# CALM, contradicting the paragraph directly above it, and the following
# clause ("never below SERVE_PQC at CALM") reads as the weaker rule the
# table actually encoded. Two readings of the same placeholder, and the
# code shipped the one that was never argued for.
#
# The consequence was not cosmetic. S1, S3 and S4 all run at CALM posture,
# and with no CALM-row hybrid floor anywhere in the table, **nothing in
# those scenarios ever mandated a QKD draw at all**. Since Hard Rule 1
# keeps security out of the reward, an unmandated hybrid serve is pure
# cost -- so "never spend the pool" was optimal by construction, the
# scarce resource the project is about was decorative, and
# `AlwaysPQCPolicy` was unbeatable on reward for structural reasons rather
# than because it budgets well.
#
# Grounding for the correction (Hard Rule 4 -- citable artifacts only):
# S3 is the decades-long-confidentiality class, which is precisely the
# Harvest-Now-Decrypt-Later target (PLAN2 §3.1). SP 800-57's cryptoperiod
# guidance and CNSA 2.0's migration posture both say protect the
# longest-lived data strongest and soonest, and hybrid (ML-KEM-768 XOR
# ETSI GS QKD 014 key material) is the strongest tier available here. A
# floor that waits for a *current* threat elevation before protecting data
# whose exposure window is measured in decades has the HNDL threat model
# backwards -- the whole point is that the adversary records now and
# decrypts later, so present-tense calm is not evidence of safety.
#
# This is a floor RAISE, so Hard Rule 2 is unaffected in direction, and it
# makes the environment strictly harder for every policy including the
# agent (always-PQC must now serve hybrid for S3 traffic and can no longer
# coast). It was made after Gate W3's first run came back negative, for
# transparency -- Hard Rule 7 directs exactly this ("investigate
# environment design first"), and both the before and after results are
# reported. It is not a change made to help the agent win: see
# SESSION_LOG.md 2026-08-19 for the re-run and its outcome.
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
    (SensitivityClass.S3, ThreatPosture.CALM): Action.SERVE_HYBRID,
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


_KEY_TYPE_TO_TIER_ACTION: dict[KeyType, Action] = {
    KeyType.CLASSICAL: Action.SERVE_CLASSICAL,
    KeyType.PQC: Action.SERVE_PQC,
    KeyType.HYBRID: Action.SERVE_HYBRID,
}
"""Which tier a session's existing key material actually delivers.
Needed by rule 4 below; mirrors `env/environment.py`'s
`_KEY_TYPE_TO_SERVE_ACTION` (kept local so masking.py doesn't import
the environment)."""


def _delivers_below_floor(current_key_type: KeyType | None, floor: Action) -> bool:
    """Would this session's existing key material serve below `floor`?

    `None` (no key established yet) is never below the floor -- there is
    nothing to deliver, and both actions that would consult it are
    already illegal or resolve to the floor's own tier.
    """
    if current_key_type is None:
        return False
    return int(_KEY_TYPE_TO_TIER_ACTION[current_key_type]) < int(floor)


def compute_mask(
    request: Request,
    floor: Action,
    key_age: float,
    max_key_age: float,
    pool_can_draw: bool,
    current_key_type: KeyType | None = None,
) -> ActionMask:
    """Build the boolean action mask for one request (Hard Rules 2, 5, 9).

    Legality rules (PLAN.md Hard Rules):
      1. Actions below `floor` are illegal.
      2. `SERVE_HYBRID` is illegal if `not pool_can_draw` -- pool
         exhaustion routes the request to `env/deferral_queue.py`
         instead of masking in a downgrade (Hard Rule 9).
      3. `REUSE` is illegal if `key_age >= max_key_age` (the SP
         800-57-derived cap `L`); this triggers a forced rekey instead,
         logged via `contracts.ForcedRekey` (Addition C).
      4. `REUSE` is illegal if the session's **existing key material is
         below the current floor** -- see below.
      5. `REKEY_NOW` is illegal on the same grounds: it refreshes the
         session's *existing* tier in place (env/environment.py design
         decision 4), so if that tier is below the floor it would
         re-establish below-floor key material.
      - Key-type changes only happen at rekey boundaries (Hard Rule
        5); this function only encodes *this step's* legality, not
        mid-session switching.

    Rule 4 (added 2026-08-19) closes a real Hard Rule 2 hole. Rules 1-3
    gate REUSE on *age* only, never on the tier the existing key
    actually delivers. Because `PolicyTable`'s ratchet is one-way, a
    session that established a PQC key under CALM posture kept being
    allowed to REUSE it after the floor ratcheted to SERVE_HYBRID --
    i.e. it went on serving PQC-grade key material to a request whose
    floor said hybrid. Measured before the fix on S2 (2,000 steps,
    seed 0, always-PQC): **275 of 1,788 REUSE decisions -- 15.4% --
    delivered key material below the request's current floor.** It was
    invisible because `experiments/harness.py`'s `floor_violations`
    counter only inspected the three tier actions, so it dutifully
    reported 0. The project's headline structural guarantee (PLAN2
    §7.7: floor violations "0 -- structural") was therefore true of
    tier actions and untested for REUSE.

    Rule 5 closes the same hole on the other delivery path, and it is
    reachable even with a flat CALM posture: sessions are keyed on
    (tenant, service) while the floor is a function of the *request's*
    sensitivity class, so two requests on one session can carry
    different floors. A session established CLASSICAL for an S0 request
    and then refreshed via REKEY_NOW for an S2 request was
    re-establishing classical key material under a PQC floor -- caught
    on S1/seed 1 as 2 violations once `experiments/harness.py` started
    counting delivered tier rather than chosen action.

    Neither masked-out action downgrades anything: the request falls
    through to a rekey at or above the floor, exactly as a
    forced-rekey-on-age does, and SERVE_PQC/SERVE_HYBRID remain legal
    by rule 1. If the floor is SERVE_HYBRID and the pool cannot cover
    it, rule 2 leaves nothing legal and `env/environment.py` defers the
    request (Hard Rule 9) rather than serving it weak.

    `current_key_type=None` means "no key established yet", for which
    REUSE is already illegal via rule 3 (`env/environment.py`
    cold-starts a session at `key_age = max_key_age` precisely so that
    rule covers it) -- so the default preserves every pre-existing call
    shape exactly.

    Returns an `ActionMask` of shape (N_ACTIONS,), aligned to `Action`.

    `request` is accepted for signature completeness (masking is
    always computed per-request) but isn't consulted by any rule --
    only `floor`, `key_age`/`max_key_age`, `pool_can_draw` and
    `current_key_type` are. (`floor` is expected to already be the
    output of `PolicyTable.floor(request['sensitivity_class'], ...)`.)
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for action in Action:
        legal = int(action) >= int(floor)
        if action is Action.SERVE_HYBRID and not pool_can_draw:
            legal = False
        if action is Action.REUSE:
            if key_age >= max_key_age:
                legal = False
            elif _delivers_below_floor(current_key_type, floor):
                legal = False  # rule 4
        if action is Action.REKEY_NOW and _delivers_below_floor(current_key_type, floor):
            legal = False  # rule 5
        mask[action] = legal
    return mask
