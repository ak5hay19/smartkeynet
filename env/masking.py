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

from dataclasses import dataclass
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
# (class x posture) -> floor table.
#
# RECALIBRATED 2026-09-14 (prior placeholder values replaced with values
# grounded in real, citable standards -- a capstone review had flagged
# the previous cell values as arbitrary/hardcoded with no justification).
# The table's STRUCTURE (this dict shape, the lookup mechanism in
# `PolicyTable.floor` below, and the monotonicity guarantee) is
# unchanged by this recalibration -- only the 12 cell values and their
# documentation changed. Full per-cell reasoning lives in
# `_FLOOR_TABLE_CALIBRATION` immediately below this dict (one entry per
# cell, each explicitly labeled "citation_with_judgment", "citation", or
# "judgment" -- see that dict's own docstring) and in PROGRESS.md's
# 2026-09-14 session entry (before/after table + full writeup).
#
# Honesty note, load-bearing for how to read `_FLOOR_TABLE_CALIBRATION`:
# none of the four standards cited anywhere in this module (NIST FIPS
# 203; NIST SP 800-57 Part 1 Rev. 5; NSA/CNSA 2.0; ETSI GS QKD 014
# V1.1.1) literally defines a 4-sensitivity-class x 3-posture -> tier
# grid -- that structure is this project's own design. A citation
# attached to a cell means "this standard directly dictates or strongly
# implies this cell's specific value"; most cells instead carry a
# "judgment" label, meaning "a defensible engineering choice informed
# by, but not required by, the cited standard(s)." Both labels are
# honest outcomes here -- do not read a "judgment" label as a defect.
#
# Only ONE cell's VALUE actually changed in this recalibration:
# (S3, CALM): SERVE_PQC -> SERVE_HYBRID (see that cell's
# "citation_with_judgment" entry below for the full CNSA 2.0-grounded
# reasoning). Every other cell's value is unchanged from the prior
# placeholder -- re-examined against the four standards and found to
# already be a defensible engineering judgment, not in need of a
# numeric change, just honest labeling.
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


@dataclass(frozen=True)
class FloorCalibrationBasis:
    """One `_PLACEHOLDER_FLOOR_TABLE` cell's calibration justification.

    `kind` is one of:
      - `"citation"`: a named standard directly dictates or strongly
        implies this exact cell value.
      - `"judgment"`: a defensible engineering choice informed by, but
        not required by, any of the four standards this module cites
        -- labeled as such plainly rather than forcing a citation that
        wouldn't hold up if someone read the source.
      - `"citation_with_judgment"`: part of the value traces directly
        to a cited standard and part is an additional engineering
        choice layered on top -- `reasoning` spells out exactly where
        the citation's support ends, so the two aren't blurred
        together as if the standard mandated the whole value.

    `reasoning` must name the specific standard(s) precisely (title +
    date, no loose paraphrasing) whenever `kind != "judgment"`.
    """

    kind: str
    reasoning: str


# Per-cell calibration record (2026-09-14 recalibration session). Every
# one of the 12 real `_PLACEHOLDER_FLOOR_TABLE` keys must appear here
# exactly once -- enforced directly by
# tests/test_masking.py::test_floor_table_calibration_covers_every_real_cell_exactly,
# so a future edit to either dict without updating the other fails
# loudly instead of silently drifting out of sync.
_FLOOR_TABLE_CALIBRATION: dict[tuple[SensitivityClass, ThreatPosture], FloorCalibrationBasis] = {
    (SensitivityClass.S0, ThreatPosture.CALM): FloorCalibrationBasis(
        kind="judgment",
        reasoning=(
            "S0 is public/non-sensitive telemetry (env/contracts.py::SensitivityClass) with no "
            "Harvest-Now-Decrypt-Later-relevant confidentiality exposure. None of the four standards "
            "this module cites (NIST FIPS 203; NIST SP 800-57 Part 1 Rev. 5; NSA/CNSA 2.0; ETSI GS "
            "QKD 014 V1.1.1) impose a protection floor on data with no confidentiality value to "
            "protect -- SERVE_CLASSICAL absent any detected threat is a design default, not "
            "something any of them require."
        ),
    ),
    (SensitivityClass.S0, ThreatPosture.ELEVATED): FloorCalibrationBasis(
        kind="judgment",
        reasoning=(
            "Same reasoning as (S0, CALM): an ELEVATED-but-not-HIGH signal doesn't create "
            "confidentiality exposure S0 didn't already lack. Held at SERVE_CLASSICAL by the same "
            "design choice, not escalated a step early."
        ),
    ),
    (SensitivityClass.S0, ThreatPosture.HIGH): FloorCalibrationBasis(
        kind="judgment",
        reasoning=(
            "Explicitly NOT standards-required -- S0 has no confidentiality exposure for any of the "
            "four cited standards to protect, at any posture. Stepping to SERVE_PQC only at the "
            "highest discrete threat posture is a deliberate, low-cost engineering choice (SERVE_PQC "
            "is this codebase's 'free' workhorse tier -- no pool draw, no legacy-interop cost beyond "
            "what request-generator-level `pqc_capable` already handles) to reduce the diversity of "
            "still-classical-only crypto pathways while the system-wide threat signal is at its worst "
            "-- operational defense-in-depth, not a confidentiality requirement."
        ),
    ),
    (SensitivityClass.S1, ThreatPosture.CALM): FloorCalibrationBasis(
        kind="judgment",
        reasoning=(
            "S1 is routine business data (env/contracts.py::SensitivityClass) with no acute HNDL "
            "exposure. NIST SP 800-57 Part 1 Rev. 5's general graduated-protection posture (protection "
            "intensity should track data sensitivity and expected lifetime) is consistent with leaving "
            "S1 at SERVE_CLASSICAL absent any detected threat, but SP 800-57 does not name this "
            "project's sensitivity classes or dictate this specific boundary -- informed by, not "
            "required by, that standard."
        ),
    ),
    (SensitivityClass.S1, ThreatPosture.ELEVATED): FloorCalibrationBasis(
        kind="judgment",
        reasoning=(
            "Informed by NIST FIPS 203, \"Module-Lattice-Based Key-Encapsulation Mechanism Standard\" "
            "(Aug 2024): FIPS 203 standardizes ML-KEM as a no-extra-cost quantum-resistant option "
            "(this codebase's SERVE_PQC tier), so once any elevated threat signal is detected there is "
            "no reason to keep serving S1 on a classically-secure-only scheme. FIPS 203 defines the "
            "algorithm and its security categories; it does not itself say *when* a system should "
            "invoke it -- the escalation trigger (posture >= ELEVATED) is this project's judgment."
        ),
    ),
    (SensitivityClass.S1, ThreatPosture.HIGH): FloorCalibrationBasis(
        kind="judgment",
        reasoning=(
            "Capped at SERVE_PQC even at HIGH: S1 does not carry the long-lived/regulated exposure "
            "that NIST SP 800-57 Part 1 Rev. 5's cryptoperiod-by-sensitivity framing associates with "
            "needing the strongest available tier. Escalating S1 all the way to SERVE_HYBRID at any "
            "posture is not grounded in any of the four cited standards, so it deliberately isn't done "
            "here -- an engineering judgment about where to cap this class, not a citation."
        ),
    ),
    (SensitivityClass.S2, ThreatPosture.CALM): FloorCalibrationBasis(
        kind="judgment",
        reasoning=(
            "S2 is regulated/financial data (env/contracts.py::SensitivityClass). NIST SP 800-57 Part "
            "1 Rev. 5's general risk-graduated protection posture (higher-impact data warrants "
            "stronger controls) supports defaulting S2 to SERVE_PQC even absent a detected threat -- "
            "but SP 800-57 does not name 'regulated/financial data' as a class or mandate SERVE_PQC "
            "specifically for it, so this is engineering judgment informed by, not dictated by, that "
            "standard."
        ),
    ),
    (SensitivityClass.S2, ThreatPosture.ELEVATED): FloorCalibrationBasis(
        kind="judgment",
        reasoning="Same basis as (S2, CALM); already at the SERVE_PQC floor, so ELEVATED changes nothing.",
    ),
    (SensitivityClass.S2, ThreatPosture.HIGH): FloorCalibrationBasis(
        kind="judgment",
        reasoning=(
            "Informed by NIST SP 800-57 Part 1 Rev. 5's graduated-response posture and ETSI GS QKD "
            "014 V1.1.1 (Feb 2019, the QKD key-delivery API spec -- establishes what a 'QKD-backed' "
            "SERVE_HYBRID tier assumes technically). Under a confirmed HIGH threat signal, the "
            "second-most-sensitive class escalates to this system's strongest tier -- consistent with "
            "SP 800-57's escalation philosophy, not a literal requirement of it or of ETSI GS QKD 014."
        ),
    ),
    (SensitivityClass.S3, ThreatPosture.CALM): FloorCalibrationBasis(
        kind="citation_with_judgment",
        reasoning=(
            "CHANGED this recalibration (was SERVE_PQC). Direct, strong basis: NSA/CNSA 2.0 "
            "(\"Commercial National Security Algorithm Suite 2.0\", Sept 2022, updated Dec 2024) sets "
            "a 2030 deadline for National Security Systems to be FULLY quantum-resistant. S3 is "
            "patient-record-grade data with a decades-long confidentiality lifetime "
            "(env/contracts.py::SensitivityClass) -- exactly the long-lived, high-value data that "
            "CNSA 2.0's harvest-now-decrypt-later-driven urgency targets. That gives a strong, direct "
            "basis for making S3's floor NOT depend on the currently observed threat posture at all: "
            "gating the single highest-sensitivity class behind reactive threat detection would defeat "
            "the entire preemptive point of a hard transition deadline, so the floor is set the same at "
            "CALM as at HIGH rather than escalating gradually. Explicit judgment layered on top, "
            "stated to avoid overclaiming: CNSA 2.0 itself only mandates quantum-resistance, which "
            "SERVE_PQC alone already satisfies -- it does not mandate the additional QKD-derived "
            "SERVE_HYBRID layer. That the unconditional floor is specifically SERVE_HYBRID (not "
            "'merely' SERVE_PQC) is this project's own engineering judgment about how much margin the "
            "top-sensitivity class deserves, not something CNSA 2.0 itself requires."
        ),
    ),
    (SensitivityClass.S3, ThreatPosture.ELEVATED): FloorCalibrationBasis(
        kind="citation_with_judgment",
        reasoning=(
            "Same value as before this recalibration. Same CNSA 2.0-grounded basis as (S3, CALM) -- "
            "see that entry for the full reasoning and its explicit scope limit; S3 is already at its "
            "unconditional SERVE_HYBRID floor, so ELEVATED changes nothing."
        ),
    ),
    (SensitivityClass.S3, ThreatPosture.HIGH): FloorCalibrationBasis(
        kind="citation_with_judgment",
        reasoning=(
            "Same value as before this recalibration. Same CNSA 2.0-grounded basis as (S3, CALM) -- "
            "already at the unconditional SERVE_HYBRID floor, the system's strongest tier, so there is "
            "no further cell to escalate to."
        ),
    ),
}


class PolicyTable:
    """Maps (sensitivity_class, threat_posture) -> minimum tier (`Action` floor).

    Recalibrated 2026-09-14 against four citable standards (NIST FIPS
    203; NIST SP 800-57 Part 1 Rev. 5; NSA/CNSA 2.0; ETSI GS QKD 014
    V1.1.1) -- see `_FLOOR_TABLE_CALIBRATION` above for the per-cell
    citation-or-judgment reasoning and PROGRESS.md's 2026-09-14 entry
    for the full before/after writeup. The lookup table itself is
    `_PLACEHOLDER_FLOOR_TABLE` above; Q-OPSEC's `synthetic_context_dataset`
    (PLAN.md "Datasets & Provenance" -> "Policy-table calibration" row)
    still hasn't been folded in -- this recalibration is against public
    standards, not that dataset, so the name is kept but this is no
    longer an unjustified/arbitrary placeholder.

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
    current_key_type: KeyType | None = None,
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
      - `REUSE` is illegal if `current_key_type` is given and the tier
        it actually delivers is below `floor` (added 2026-08-19 --
        closes a real Hard Rule 2 gap, see below).
      - Key-type changes only happen at rekey boundaries (Hard Rule
        5); this function only encodes *this step's* legality, not
        mid-session switching.

    2026-08-19 fix -- what the gap was and why it was invisible until
    now: the original three rules gated REUSE on *age* only, never on
    whether the session's *already-established* key material still
    meets the floor now in effect. Because `PolicyTable`'s ratchet is
    deliberately one-way (Hard Rule 2: "threat signals may only raise
    floors"), a session that established e.g. a PQC key under CALM
    posture could keep legally REUSE-ing that same PQC key forever
    after the floor later ratcheted up to SERVE_HYBRID mid-episode --
    Hard Rule 2's floor was being enforced at key *establishment* time
    only, not for the life of the session. This was invisible under S1
    (posture barely moves past what the first few decisions already
    see) and only became empirically observable once S2's scripted,
    mid-episode ratchet existed (2026-08-19) -- a real S2 episode under
    `RandomPolicy` measured **64 of 279 REUSE/REKEY_NOW decisions
    (22.9%) delivering key material below the request's current
    floor** before this fix (32 via each action) -- see
    `tests/test_environment.py`'s and `tests/test_masking.py`'s
    reproductions.

    `current_key_type=None` (the default) preserves every pre-existing
    call shape exactly -- this rule is a no-op unless a caller supplies
    the session's actual key type. `KeyType` and `Action`'s tier
    members share the same ordinal values by construction
    (CLASSICAL/SERVE_CLASSICAL=0, PQC/SERVE_PQC=1,
    HYBRID/SERVE_HYBRID=2 -- see env/contracts.py), so
    `Action(int(current_key_type))` is exactly the tier that key
    material actually delivers; no separate mapping table is
    introduced here.

    REKEY_NOW is deliberately NOT masked by this or any rule here: a
    stale existing tier doesn't make REKEY_NOW illegal, it changes what
    REKEY_NOW *resolves to* -- see
    `env/environment.py::SmartKeyNetEnv._resulting_key_type`'s matching
    2026-08-19 fix, which guarantees REKEY_NOW's resolved tier is
    always `>= floor` (while never downgrading a session that's already
    above floor). Masking REKEY_NOW here instead would remove the
    guaranteed escape hatch `tests/test_masking.py::
    test_at_least_one_action_always_legal` relies on; fixing what it
    resolves to, rather than masking it, is what lets Hard Rule 2 and
    the "REKEY_NOW never downgrades an existing higher tier" design
    decision both hold at the same time.

    Returns an `ActionMask` of shape (N_ACTIONS,), aligned to `Action`.

    `request` is accepted for signature completeness (masking is
    always computed per-request) but isn't consulted by any of the
    rules below -- only `floor`, `key_age`/`max_key_age`,
    `pool_can_draw`, and (optionally) `current_key_type` are. (`floor`
    is expected to already be the output of
    `PolicyTable.floor(request['sensitivity_class'], ...)`.)
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for action in Action:
        legal = int(action) >= int(floor)
        if action is Action.SERVE_HYBRID and not pool_can_draw:
            legal = False
        if action is Action.REUSE and key_age >= max_key_age:
            legal = False
        if action is Action.REUSE and current_key_type is not None and int(Action(int(current_key_type))) < int(floor):
            legal = False
        mask[action] = legal
    return mask
