"""
api/main.py

AWS-KMS-flavored REST facade (PLAN.md §4 "API surface"; §3 cloud
framing). Owned by Person D (split.md §1).

Conceptually backed by ETSI GS QKD 014 key delivery. Endpoints mirror
AWS KMS naming (`GenerateDataKey`, key policies, tenants) so the demo
reads as "what AWS KMS looks like ~2030" (PLAN.md §3).

This facade calls into env/agents at runtime for the live demo; it is
not itself part of the MDP (Hard Rule 3 -- no agent or environment
logic lives here, only routing, session bookkeeping, and real primitive
calls for authenticity). Deleting this module changes no result.

Cryptographic honesty
---------------------
PLAN.md asks for "real primitive calls ... used where cheap", and this
module is explicit about which calls are real and which are not,
because a demo that quietly fakes a primitive is worse than one that
says so:

  * **classical (T0)** -- REAL. X25519 ephemeral key agreement via
    `cryptography`, HKDF-SHA256 to a 256-bit data key, AES-256-GCM
    wrap. Every byte returned is genuinely computed.
  * **PQC (T1, ML-KEM-768)** -- **SIMULATED**. `liboqs-python` is
    listed as an optional dependency in requirements.txt and is not
    installed here, so there is no real ML-KEM encapsulation. The
    response carries `primitive_real: false` and an explicit
    `simulation_note`, and the key material is derived from a
    domain-separated HKDF so it is at least distinct from the classical
    path rather than a relabelled copy of it. It is not
    quantum-resistant and is not claimed to be.
  * **hybrid (T2/T3)** -- **PARTIALLY REAL**. The combiner is real:
    HKDF-SHA256 over `classical || pqc || qkd` material, which is the
    standard hybrid construction. The QKD side is where the honesty
    matters most: `env/pool_sim.py` models key *availability* (a
    documented synthetic SKR/QBER process driving a finite pool that
    really is drawn down by this request), but the pool holds a bit
    *count*, not bytes -- so the 256 bits mixed in are locally
    generated. The scarcity is simulated faithfully; the quantum
    material is not quantum.

`GET /Health` reports this matrix, so nobody has to read this docstring
to find out what is real.
"""

from __future__ import annotations

import base64
import os
import secrets
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.baselines import AlwaysPQCPolicy, Policy
from env.contracts import Action, KeyType
from env.decision_trace import DecisionTrace, build_decision_trace
from env.environment import SmartKeyNetEnv
from experiments.train import load_full_config

app = FastAPI(
    title="SmartKeyNet KMS Facade",
    description=(
        "AWS-KMS-flavoured facade over the SmartKeyNet decision layer. "
        "Every decision is made by the real environment + policy; see GET /Health "
        "for exactly which cryptographic primitives are real and which are simulated."
    ),
)

_TIER_NAMES = {
    Action.SERVE_CLASSICAL: "classical",
    Action.SERVE_PQC: "pqc",
    Action.SERVE_HYBRID: "hybrid",
}

_PRIMITIVE_MATRIX = {
    "classical": {
        "real": True,
        "detail": "X25519 ECDH + HKDF-SHA256 + AES-256-GCM via `cryptography`",
    },
    "pqc": {
        "real": False,
        "detail": (
            "ML-KEM-768 is SIMULATED -- liboqs-python is an optional dependency and is not "
            "installed. Key material is domain-separated HKDF output, not a real KEM "
            "encapsulation, and is not quantum-resistant."
        ),
    },
    "hybrid": {
        "real": "partial",
        "detail": (
            "The HKDF-SHA256 combiner over (classical secret || pqc secret || QKD material) is "
            "REAL and is the standard hybrid construction. What is simulated: env/pool_sim.py "
            "models QKD key AVAILABILITY (a documented synthetic SKR/QBER process driving a "
            "finite pool, which really is drawn down by this request), but it holds a bit "
            "COUNT, not bytes -- so the 256 bits mixed in here are locally generated, not "
            "delivered by a CV-QKD link. The ML-KEM half is simulated as above."
        ),
    },
}


# ---------------------------------------------------------------------------
# Session: one live environment + policy behind the facade
# ---------------------------------------------------------------------------


@dataclass
class KmsSession:
    """One live `SmartKeyNetEnv` advanced by one decision per request.

    Deliberately thin. The facade owns no decision logic: it asks the
    policy, steps the environment, and formats. Hard Rule 3's test --
    "deleting the tenant graph must not change one line of agent code"
    -- extends here: deleting this module changes no result either.
    """

    scenario: str = "S1"
    seed: int = 0
    config: dict[str, Any] = field(default_factory=load_full_config)
    policy: Policy = field(default_factory=AlwaysPQCPolicy)
    _env: SmartKeyNetEnv | None = None
    _state: Any = None
    _info: Any = None
    _traces: dict[str, DecisionTrace] = field(default_factory=dict)
    _trace_order: list[str] = field(default_factory=list)

    _MAX_RETAINED_TRACES = 500

    def reset(self) -> None:
        env_config = {**self.config, "scenario": self.scenario, "seed": self.seed}
        env_config.pop("max_steps", None)  # the facade is a live stream, not an episode
        self._env = SmartKeyNetEnv(env_config)
        self._state, self._info = self._env.reset(seed=self.seed)
        self._traces.clear()
        self._trace_order.clear()

    @property
    def env(self) -> SmartKeyNetEnv:
        if self._env is None:
            self.reset()
        assert self._env is not None
        return self._env

    def decide(self) -> DecisionTrace:
        """Advance one decision and return its full six-step trace."""
        env = self.env
        mask = self._info["action_mask"]
        action = self.policy.act(self._state, mask)
        trace = build_decision_trace(env, self._state, mask, action)
        self._state, _reward, _terminated, _truncated, self._info = env.step(action)

        self._traces[trace.request_id] = trace
        self._trace_order.append(trace.request_id)
        while len(self._trace_order) > self._MAX_RETAINED_TRACES:
            self._traces.pop(self._trace_order.pop(0), None)
        return trace

    def trace(self, request_id: str) -> DecisionTrace | None:
        return self._traces.get(request_id)

    def recent(self, limit: int) -> list[DecisionTrace]:
        return [self._traces[rid] for rid in self._trace_order[-limit:]]


_SESSION = KmsSession()


def get_session() -> KmsSession:
    """Seam for tests and for the dashboard to inject its own session."""
    return _SESSION


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


def _hkdf(secret: bytes, info: bytes, length: int = 32) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(), length=length, salt=None, info=info
    ).derive(secret)


def derive_data_key(tier: str, qkd_bits: bytes | None = None) -> tuple[bytes, dict[str, Any]]:
    """Derive a 256-bit data key for `tier`.

    Returns `(key, provenance)` where `provenance` records exactly which
    steps were real -- see this module's docstring. The provenance
    travels with every response so a viewer never has to assume.
    """
    private = X25519PrivateKey.generate()
    peer = X25519PrivateKey.generate().public_key()
    classical_secret = private.exchange(peer)  # real X25519 ECDH

    if tier == "classical":
        key = _hkdf(classical_secret, b"smartkeynet/t0/classical")
        return key, {"tier": "classical", **_PRIMITIVE_MATRIX["classical"]}

    if tier == "pqc":
        # SIMULATED ML-KEM-768: domain-separated so it is at least a
        # distinct key rather than a relabelled classical one, but this
        # is HKDF output, not a KEM encapsulation.
        simulated_kem_secret = secrets.token_bytes(32)
        key = _hkdf(
            classical_secret + simulated_kem_secret, b"smartkeynet/t1/ml-kem-768-SIMULATED"
        )
        return key, {"tier": "pqc", **_PRIMITIVE_MATRIX["pqc"]}

    if tier == "hybrid":
        simulated_kem_secret = secrets.token_bytes(32)
        qkd = qkd_bits if qkd_bits is not None else secrets.token_bytes(32)
        # REAL combiner: HKDF over (classical || pqc || qkd). This is the
        # part of the hybrid construction that genuinely is what it says.
        key = _hkdf(
            classical_secret + simulated_kem_secret + qkd,
            b"smartkeynet/t2/hybrid-hkdf",
        )
        return key, {"tier": "hybrid", **_PRIMITIVE_MATRIX["hybrid"]}

    raise ValueError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GenerateDataKeyRequest(BaseModel):
    tenant: str = Field(default="", description="Informational only -- the live request stream "
                                                "is generated by the tenant graph, not by callers.")
    service: str = ""
    sensitivity_class: int = 0


class GenerateDataKeyResponse(BaseModel):
    key_id: str
    key_type: str  # "classical" | "pqc" | "hybrid"
    ciphertext_blob: str
    tenant: str
    service: str
    sensitivity_class: int
    policy_floor: str
    action: str
    primitive_real: Any
    simulation_note: str
    request_id: str


class ExplainResponse(BaseModel):
    trace: dict[str, Any]


class PoolStatusResponse(BaseModel):
    fill_bits: float
    capacity_bits: float
    fill_fraction: float
    skr_kbps: float
    qber: float
    bits_per_hybrid_draw: float
    deferral_queue_depth: int
    regret_events: int


class ThreatStatusResponse(BaseModel):
    source: str
    threat_score: float
    posture_probs: list[float]
    resolved_posture: str
    horizon_scores: list[float]
    foresight_mode: str
    frozen_during_training: bool
    raise_only: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/GenerateDataKey", response_model=GenerateDataKeyResponse)
def generate_data_key(request: GenerateDataKeyRequest) -> GenerateDataKeyResponse:
    """AWS-KMS-style `GenerateDataKey`.

    Routes the request through the live agent/env for a demo decision,
    then returns real primitive output for the chosen key type -- with
    `primitive_real` and `simulation_note` stating exactly how real.

    The caller's `tenant`/`service`/`sensitivity_class` are accepted for
    API shape but are informational: the live request stream comes from
    the tenant graph inside the environment (Hard Rule 3 -- the graph
    shapes which requests arrive), so the decision returned is for
    whichever request the environment is currently presenting.
    """
    session = get_session()
    trace = session.decide()

    tier = _TIER_NAMES.get(Action[trace.delivered_tier], "classical")
    qkd_bits = secrets.token_bytes(32) if tier == "hybrid" else None
    key, provenance = derive_data_key(tier, qkd_bits)

    aes = AESGCM(key)
    nonce = os.urandom(12)
    blob = aes.encrypt(nonce, b"smartkeynet-data-key", None)

    return GenerateDataKeyResponse(
        key_id=f"skn-{trace.request_id}",
        key_type=tier,
        ciphertext_blob=base64.b64encode(nonce + blob).decode(),
        tenant=trace.tenant,
        service=trace.service,
        sensitivity_class=trace.sensitivity_class,
        policy_floor=trace.policy_floor,
        action=trace.chosen_action,
        primitive_real=provenance["real"],
        simulation_note=provenance["detail"],
        request_id=trace.request_id,
    )


@app.get("/ExplainDecision/{request_id}", response_model=ExplainResponse)
def explain_decision(request_id: str) -> ExplainResponse:
    """The six-step trace for one decision (PLAN2 §7.3).

    Hard Rule 10: every value here was computed by the pipeline. There
    is no generative step anywhere in this path -- see
    `env/decision_trace.py`.
    """
    trace = get_session().trace(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"no retained trace for {request_id!r}")
    return ExplainResponse(trace=trace.to_dict())


@app.get("/Decisions")
def recent_decisions(limit: int = 20) -> dict[str, Any]:
    """Recent decisions, newest last -- Panel 2's "recent decisions" list."""
    return {
        "decisions": [
            {
                "request_id": t.request_id,
                "step": t.step,
                "tenant": t.tenant,
                "service": t.service,
                "sensitivity_class": t.sensitivity_class,
                "policy_floor": t.policy_floor,
                "action": t.chosen_action,
                "delivered_tier": t.delivered_tier,
                "reason": t.steps[-1].summary,
            }
            for t in get_session().recent(limit)
        ]
    }


@app.get("/PoolStatus", response_model=PoolStatusResponse)
def pool_status() -> PoolStatusResponse:
    """ETSI GS QKD 014-style key-store status -- Panel 2's pool gauge."""
    env = get_session().env
    pool = env._pool_sim
    return PoolStatusResponse(
        fill_bits=float(pool.fill),
        capacity_bits=float(pool.capacity),
        fill_fraction=float(pool.fill / pool.capacity),
        skr_kbps=float(env._last_pool_state.skr),
        qber=float(env._last_pool_state.qber),
        bits_per_hybrid_draw=float(env._bits_per_hybrid_draw),
        deferral_queue_depth=len(env._deferral_queue),
        regret_events=len(env._regret_log),
    )


@app.get("/ThreatStatus", response_model=ThreatStatusResponse)
def threat_status() -> ThreatStatusResponse:
    """Panel 1's threat-signal readout, plus the two invariants that
    panel is required to state (PLAN2 §7.1): the signal is frozen during
    agent training, and it may only raise floors."""
    session = get_session()
    env = session.env
    forecaster = env._forecaster
    if forecaster is None:
        return ThreatStatusResponse(
            source=env._threat_source_name,
            threat_score=0.0,
            posture_probs=[1.0, 0.0, 0.0],
            resolved_posture="CALM",
            horizon_scores=[0.0] * 5,
            foresight_mode="off",
            frozen_during_training=True,
            raise_only=True,
        )

    forecast = forecaster.get_threat_forecast()
    probs = list(forecast.posture_probs)
    from env.contracts import ThreatPosture

    resolved = ThreatPosture(int(max(range(len(probs)), key=probs.__getitem__)))
    return ThreatStatusResponse(
        source=env._threat_source_name,
        threat_score=float(forecast.threat_score),
        posture_probs=probs,
        resolved_posture=resolved.name,
        horizon_scores=[float(v) for v in forecast.horizon_scores],
        foresight_mode=str(env._use_foresight),
        frozen_during_training=True,
        raise_only=True,
    )


@app.post("/Scenario/{scenario}")
def set_scenario(scenario: str) -> dict[str, str]:
    """Switch the live session's scenario (Panel 4's scenario selector)."""
    session = get_session()
    try:
        session.scenario = scenario
        session.reset()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"scenario": scenario, "status": "reset"}


@app.get("/Health")
def health() -> dict[str, Any]:
    """Liveness plus the cryptographic-honesty matrix.

    The matrix is on the health endpoint on purpose: which primitives
    are real is not a footnote, and nobody should have to read source to
    find out.
    """
    return {"status": "ok", "primitives": _PRIMITIVE_MATRIX}
