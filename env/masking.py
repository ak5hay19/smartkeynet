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


_LEGACY_ENDPOINT_FLOOR: Action = Action.SERVE_CLASSICAL
"""Floor applied to flows terminating on non-PQC-capable endpoints.

WHY THIS EXISTS AND WHY IT IS NOT A HARD RULE 2 VIOLATION. Read this
before touching it.

SMARTKEYNET_BUILD_SPEC.md §S4's masking rule 2 says a legacy endpoint
masks out `SERVE_PQC` and `SERVE_HYBRID`, because classical is the only
thing it can negotiate. But the floor table sends even class S0 to
`SERVE_PQC` under `ThreatPosture.HIGH`. Put those together without an
exemption and a legacy S0 flow at HIGH posture has **no legal action at
all**: everything at or above its floor is un-negotiable, and
everything it can negotiate is below its floor. The masking layer would
then have to choose between Hard Rule 2 (never serve below floor) and
liveness (serve something) -- and the spec explicitly refuses to let it
make that choice.

Three ways out were considered:

  1. **Defer forever.** Most consistent with Hard Rule 9's "never
     downgrade, defer instead", and arguably realistic -- a flow you
     cannot protect to policy genuinely should not be served. Rejected
     because under a sustained HIGH posture it silently converts a
     fixed share of all traffic into permanent deferrals, which swamps
     the regret metric and confounds S2's measurement with an artifact
     that has nothing to do with pool budgeting.
  2. **Raise `pqc_capable` flows' floors only.** Does not help: the
     problem is the legacy flows, not the capable ones.
  3. **An explicit, static policy exemption** (this). The operator's
     key policy enumerates legacy flows and permits classical for them,
     which is exactly how real cloud KMS key policies carve out
     exceptions for systems that cannot yet meet a standard.

This does not violate Hard Rule 2, which constrains **threat signals**:
"Threat signals may only *raise* floors." `pqc_capable` is not a threat
signal. It is a static, exogenous capability fact about an endpoint,
fixed at graph-generation time, invisible to the forecaster, and
unreachable by the agent. No posture change, forecast, or adversarial
trace can flip it -- which is precisely what
`test_legacy_exemption_cannot_be_triggered_by_any_threat_posture`
asserts.

What it *does* mean, and what belongs in the report: legacy endpoints
are a standing residual risk that policy cannot fix, only migration
can. That is the honest reading, and it is exactly what scenario S6
models when it flips `pqc_capable` to `True` cohort by cohort. The
count of legacy flows served classical under an elevated posture is
worth reporting as its own number -- it is the migration backlog,
measured.

`env/request_generator.py` additionally guarantees legacy edges only
ever carry class S0 (`_LEGACY_MAX_CLASS`), so this exemption never
applies to genuinely sensitive data.
"""


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

    def floor(
        self,
        sensitivity_class: SensitivityClass,
        threat_posture: ThreatPosture,
        pqc_capable: bool = True,
    ) -> Action:
        """Return the minimum legal tier (as an `Action`) for this
        (class, posture) pair.

        Resolved against `max(threat_posture, ratcheted_posture)` --
        see the class docstring's `ratchet_up` interpretation.

        `pqc_capable=False` applies the **legacy-endpoint exemption**
        (see `_LEGACY_ENDPOINT_FLOOR` below).
        """
        effective_class = SensitivityClass(int(sensitivity_class))
        effective_posture = ThreatPosture(max(int(threat_posture), int(self._ratcheted_posture)))
        table_floor = self._table[(effective_class, effective_posture)]

        if not pqc_capable:
            return _LEGACY_ENDPOINT_FLOOR
        return table_floor

    def ratchet_up(self, threat_posture: ThreatPosture) -> None:
        """Apply a threat-driven floor increase.

        Never call this to lower a floor (Hard Rule 2) -- there is
        deliberately no symmetric `ratchet_down`. A no-op if
        `threat_posture` isn't higher than the current ratchet level.
        """
        if int(threat_posture) > int(self._ratcheted_posture):
            self._ratcheted_posture = ThreatPosture(int(threat_posture))


def effective_floor_for(request: Request, floor: Action) -> Action:
    """The floor actually enforced for `request`, given the policy
    table's `floor` for its (class, posture).

    Currently this raises the table floor to `SERVE_HYBRID` for a
    `hybrid_mandatory` request. It is `max`-only by construction, so it
    can never lower a floor.

    Exists as a separate function because more than one caller needs
    the same answer and they must not disagree: `compute_mask` uses it
    to build the mask, and `env/environment.py` uses it to resolve
    `REKEY_NOW`'s tier on a cold-start session. Before this was
    factored out, the environment resolved that tier against the *raw*
    table floor, so a `REKEY_NOW` on a hybrid-mandatory request with no
    existing key would have established key material one tier below the
    floor the mask had just enforced -- a floor violation reached
    through the one action whose tier is state-dependent.
    """
    if request["hybrid_mandatory"]:
        return Action(max(int(floor), int(Action.SERVE_HYBRID)))
    return floor


def compute_mask(
    request: Request,
    floor: Action,
    key_age: float,
    max_key_age: float,
    pool_can_draw: bool,
    active_key_tier: Action | None = None,
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

    `floor` is expected to already be the output of
    `PolicyTable.floor(request['sensitivity_class'], posture,
    request['pqc_capable'])`.

    Two request fields ARE consulted, both added 2026-08-15:

      - **`hybrid_mandatory`** raises the effective floor to
        `SERVE_HYBRID`. Previously this flag only triggered the
        environment's Hard Rule 9 deferral pre-screen and was invisible
        to masking, which meant that whenever the pool *could* cover
        such a request nothing forced a hybrid serve -- `always_pqc`
        served hybrid-mandatory requests at PQC and recorded zero
        violations, because the mask never said otherwise. The flag is
        a floor statement ("this data's lifetime demands QKD-backed
        material regardless of current posture" -- the Harvest Now,
        Decrypt Later argument), so folding it in via `max` is both the
        correct semantics and raise-only, per Hard Rule 2.
      - **`pqc_capable`** masks `SERVE_PQC` and `SERVE_HYBRID` when
        false: a legacy endpoint cannot negotiate either
        (SMARTKEYNET_BUILD_SPEC.md §S4 masking rule 2). See
        `_LEGACY_ENDPOINT_FLOOR` for how liveness is preserved.

    A request that is both `hybrid_mandatory` and not `pqc_capable` is
    contradictory -- it must be served hybrid and cannot negotiate
    hybrid -- and yields an all-false mask, which the environment
    routes to the deferral queue rather than resolving by downgrade.
    Both request sources are constrained so they cannot emit one; see
    `env/request_generator.py`.

    `active_key_tier` is the tier of the session's existing key, or
    `None` for a cold-start session with no key. **REUSE is masked when
    that tier is below the effective floor** -- spec §S4 masking rule 4,
    implemented 2026-08-15.

    THIS CLOSED A HARD RULE 2 HOLE, and it is worth stating plainly
    because the whole thesis rests on it. Until this rule existed,
    `REUSE` was legal regardless of what tier of key was being reused,
    so a session holding a classical key could keep right on reusing it
    after the floor ratcheted to hybrid. Measured on an S2 episode:
    **1,090 of 3,000 REUSE actions kept a key below the floor the
    environment had just enforced.** The `floor_violations` metric
    reported zero throughout, because the harness only ever inspected
    *serve* actions -- so the headline claim "floor violations: 0,
    structurally guaranteed" was being satisfied by not looking.

    Fixing it also supplies the environment's genuine anticipation
    problem, which is why this matters beyond correctness. With the
    rule in place, a key provisioned at a high tier while the pool is
    healthy can be REUSEd for free after floors rise, whereas a
    cheaply-provisioned one forces a rekey -- and a pool draw -- at
    exactly the moment the pool may be depleted. That is the coupling
    PLAN.md §8 describes ("serving hybrid now removes an option ten
    minutes from now") and the reason `REKEY_NOW` exists as an action
    at all ("force refresh ahead of a forecast threat spike"). Before
    the fix a purely myopic policy was optimal, and the
    `GreedyRecommenderPolicy` diagnostic tied the DQN exactly.
    """
    effective_floor = effective_floor_for(request, floor)

    mask = np.zeros(N_ACTIONS, dtype=bool)
    for action in Action:
        legal = int(action) >= int(effective_floor)
        if action is Action.SERVE_HYBRID and not pool_can_draw:
            legal = False
        if action is Action.REUSE and key_age >= max_key_age:
            legal = False
        if action is Action.REUSE and (
            active_key_tier is None or int(active_key_tier) < int(effective_floor)
        ):
            # spec §S4 rule 4: cannot reuse a key that no longer clears
            # the floor, and cannot reuse a key that does not exist
            legal = False
        if action in (Action.SERVE_PQC, Action.SERVE_HYBRID) and not request["pqc_capable"]:
            legal = False  # legacy endpoint: classical is the only interoperable option
        mask[action] = legal
    return mask
