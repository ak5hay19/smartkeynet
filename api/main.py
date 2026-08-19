"""
api/main.py

AWS-KMS-flavored REST facade (PLAN.md §4 "API surface"; §3 cloud
framing). Owned by Person D (split.md §1).

Conceptually backed by ETSI GS QKD 014 key delivery. Endpoints mirror
AWS KMS naming (`GenerateDataKey`, `Encrypt`, key policies, tenants) so
the demo reads as "what AWS KMS looks like ~2030" (PLAN.md §3).

This facade calls into env/agents at runtime for the live demo; it is
not itself part of the MDP (Hard Rule 3 -- no agent or environment
logic lives here, only routing, plus real primitive calls for
authenticity).

---------------------------------------------------------------------
What is real and what is simulated -- state this in the viva
---------------------------------------------------------------------
  * **Real:** the key-derivation arithmetic. `GenerateDataKey` performs
    an actual HKDF-SHA256 derivation via the `cryptography` package,
    and hybrid keys are a real HKDF over concatenated PQC-and-QKD
    secret material. PLAN.md §4: "Real primitive calls ... used where
    cheap, for authenticity -- the crypto payloads are real even though
    the network is simulated."
  * **Simulated:** the network, the QKD link, the pool. There is no ML-KEM
    implementation here and none is claimed; the PQC contribution to a
    hybrid key is a placeholder secret of the right size, clearly named
    as such. Installing `liboqs-python` would let `_pqc_secret` become a
    genuine ML-KEM-768 encapsulation without changing anything else.
  * **Not a security product.** This facade exists to make the demo
    legible. It has no authentication, and the keys it returns must
    never be used to protect anything.

Run it:

    .venv/bin/python -m uvicorn api.main:app --reload
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.baselines import StaticThresholdPolicy
from env.contracts import Action, KeyType
from env.environment import SmartKeyNetEnv
from experiments.train import load_full_config

app = FastAPI(
    title="SmartKeyNet KMS Facade",
    description=(
        "AWS-KMS-flavoured facade over a simulated multi-tenant KMS with a QKD "
        "backhaul. Demo instrument, not a security product: no authentication, "
        "and the keys it returns must not protect anything real."
    ),
    version="1.0.0",
)

_KEY_BYTES = 32
"""256-bit data keys -- ETSI GS QKD 014 key delivery size, matching
`configs/default.yaml`'s `pool.bits_per_hybrid_draw`."""

_TIER_NAMES: dict[Action, str] = {
    Action.SERVE_CLASSICAL: "CLASSICAL_X25519_AES256GCM",
    Action.SERVE_PQC: "PQC_ML_KEM_768",
    Action.SERVE_HYBRID: "HYBRID_ML_KEM_768_XOR_QKD",
}

_KEY_TYPE_NAMES: dict[KeyType, str] = {
    KeyType.CLASSICAL: "CLASSICAL_X25519_AES256GCM",
    KeyType.PQC: "PQC_ML_KEM_768",
    KeyType.HYBRID: "HYBRID_ML_KEM_768_XOR_QKD",
}


# ---------------------------------------------------------------------------
# Key derivation (real arithmetic, simulated transport)
# ---------------------------------------------------------------------------


def _pqc_secret() -> bytes:
    """Stand-in for an ML-KEM-768 shared secret.

    Deliberately a named placeholder rather than a silent random blob:
    with `liboqs-python` installed this becomes a genuine encapsulation
    and nothing else in this file changes. Claiming ML-KEM here without
    running ML-KEM would be exactly the kind of unearned authenticity
    Hard Rule 4 exists to prevent.
    """
    return os.urandom(_KEY_BYTES)


def _qkd_secret() -> bytes:
    """Stand-in for key material delivered by the QKD link.

    In a real deployment this is an ETSI GS QKD 014 `GET /key` response
    from the trusted node; here the pool simulator decides *whether* a
    key is available, and this supplies the bytes once it has.
    """
    return os.urandom(_KEY_BYTES)


def derive_data_key(tier: Action, context: str) -> bytes:
    """Derive a data key at `tier` via HKDF-SHA256.

    This is real: hybrid genuinely combines two independent secrets
    through a KDF, so compromise of either input alone does not yield
    the output. That is the actual security argument for hybrid key
    establishment, and it is worth having the demo perform it rather
    than assert it.
    """
    if tier is Action.SERVE_HYBRID:
        input_material = _pqc_secret() + _qkd_secret()
    elif tier is Action.SERVE_PQC:
        input_material = _pqc_secret()
    else:
        input_material = os.urandom(_KEY_BYTES)

    return HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_BYTES,
        salt=None,
        info=f"smartkeynet/{context}".encode(),
    ).derive(input_material)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class GenerateDataKeyRequest(BaseModel):
    tenant: str = Field(..., examples=["hospital"])
    service: str = Field(..., examples=["records"])
    sensitivity_class: int = Field(0, ge=0, le=3)
    pqc_capable: bool = True
    hybrid_mandatory: bool = False


class GenerateDataKeyResponse(BaseModel):
    key_id: str
    tenant: str
    service: str
    tier_served: str
    policy_floor: str
    plaintext_key_b64: str
    pool_fill: float
    deferred: bool
    reason: str


class EncryptRequest(BaseModel):
    key_id: str
    plaintext: str


class EncryptResponse(BaseModel):
    key_id: str
    ciphertext_b64: str
    algorithm: str


# ---------------------------------------------------------------------------
# Live environment behind the facade
# ---------------------------------------------------------------------------


@dataclass
class _KmsState:
    """One shared simulated KMS instance behind the API.

    A single process-wide environment, deliberately: the point of the
    facade is that successive requests see a *shared, depleting* pool,
    which is the whole subject of the project. Per-request environments
    would make every call look identical and would hide the resource
    contention entirely.
    """

    env: SmartKeyNetEnv | None = None
    policy: Any = None
    state: dict[str, Any] | None = None
    info: dict[str, Any] | None = None
    issued_keys: dict[str, bytes] = field(default_factory=dict)
    issued_tiers: dict[str, str] = field(default_factory=dict)
    request_count: int = 0


_kms = _KmsState()


def _ensure_started() -> _KmsState:
    if _kms.env is None:
        config = load_full_config()
        config = {**config, "scenario": "S1", "seed": 0, "scenario_steps": 100_000}
        _kms.env = SmartKeyNetEnv(config)
        _kms.state, _kms.info = _kms.env.reset(seed=0)
        _kms.policy = StaticThresholdPolicy(
            pool_fill_threshold=0.7,
            min_hybrid_class=2,
            rekey_age_frac=0.9,
            max_key_age=float(config["key_lifetime"]["max_key_age_steps"]),
        )
    return _kms


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "SmartKeyNet KMS Facade",
        "backend": "simulated multi-tenant KMS with QKD backhaul (ETSI GS QKD 014 style)",
        "warning": "demo instrument -- unauthenticated, keys must not protect real data",
        "endpoints": ["/GenerateDataKey", "/Encrypt", "/DescribeKeyPolicy", "/PoolStatus"],
    }


@app.get("/PoolStatus")
def pool_status() -> dict[str, Any]:
    """The QKD pool gauge behind Demo Beat 1."""
    kms = _ensure_started()
    assert kms.env is not None and kms.state is not None
    return {
        "pool_fill_fraction": float(kms.state["pool_fill"]),
        "pool_keys_available": int(kms.env._pool_sim.fill // kms.env._bits_per_hybrid_draw),
        "skr_kbps": float(kms.state["skr"]),
        "qber": float(kms.state["qber"]),
        "deferral_queue_depth": len(kms.env._deferral_queue),
        "regret_events_total": len(kms.env._regret_log),
        "keys_issued": kms.request_count,
    }


@app.get("/DescribeKeyPolicy")
def describe_key_policy(sensitivity_class: int = 0, pqc_capable: bool = True) -> dict[str, Any]:
    """The per-class floor, the way a cloud KMS exposes a key policy.

    Reports the floor at every posture so a caller can see that the
    policy only ever ratchets *up* with threat -- the property the
    steering attack exists to demonstrate.
    """
    from env.contracts import SensitivityClass, ThreatPosture
    from env.masking import PolicyTable

    table = PolicyTable()
    return {
        "sensitivity_class": sensitivity_class,
        "pqc_capable": pqc_capable,
        "minimum_tier_by_posture": {
            posture.name: _TIER_NAMES[
                table.floor(SensitivityClass(sensitivity_class), posture, pqc_capable)
            ]
            for posture in ThreatPosture
        },
        "note": "floors are monotone non-decreasing in threat posture (Hard Rule 2)",
    }


@app.post("/GenerateDataKey", response_model=GenerateDataKeyResponse)
def generate_data_key(request: GenerateDataKeyRequest) -> GenerateDataKeyResponse:
    """AWS-KMS-style data key generation, served through the live agent.

    The tier is decided by the same masked policy the experiments use,
    so an operator hitting this endpoint sees exactly the behaviour the
    results tables describe -- including deferral when the pool cannot
    cover a hybrid-mandatory request, which returns HTTP 503 rather than
    quietly downgrading (Hard Rule 9).
    """
    kms = _ensure_started()
    assert kms.env is not None and kms.state is not None and kms.info is not None

    mask = kms.info["action_mask"]
    floor = Action(int(kms.state["policy_floor"]))
    action = kms.policy.act(kms.state, mask)

    kms.state, _reward, _terminated, _truncated, kms.info = kms.env.step(action)

    if kms.info["regret_events"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "Request deferred: the QKD pool cannot currently cover a "
                "hybrid-mandatory key. It is queued and will be served at its "
                "required tier once key material is available. It is never "
                "downgraded (Hard Rule 9)."
            ),
        )

    # NOTE: the served tier is derived from the action taken, not from the
    # session snapshot -- a session lookup here was dead code (assigned, never
    # read), and reading it would have been wrong anyway: `_apply_action` has
    # already mutated the session, so it reflects the post-serve state.
    tier = (
        Action.SERVE_HYBRID
        if action is Action.SERVE_HYBRID
        else (action if action in _TIER_NAMES else floor)
    )

    kms.request_count += 1
    key_id = f"skn-{kms.request_count:08d}"
    key_bytes = derive_data_key(tier, context=f"{request.tenant}/{request.service}")
    kms.issued_keys[key_id] = key_bytes
    kms.issued_tiers[key_id] = _TIER_NAMES[tier]

    return GenerateDataKeyResponse(
        key_id=key_id,
        tenant=request.tenant,
        service=request.service,
        tier_served=_TIER_NAMES[tier],
        policy_floor=_TIER_NAMES[floor],
        plaintext_key_b64=base64.b64encode(key_bytes).decode(),
        pool_fill=float(kms.state["pool_fill"]),
        deferred=False,
        reason=f"agent chose {action.name} at floor {floor.name}",
    )


@app.post("/Encrypt", response_model=EncryptResponse)
def encrypt(request: EncryptRequest) -> EncryptResponse:
    """Encrypt under a previously issued data key (AES-256-GCM).

    Real AEAD, not a stub -- the ciphertext genuinely decrypts under the
    issued key.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    kms = _ensure_started()
    key_bytes = kms.issued_keys.get(request.key_id)
    if key_bytes is None:
        raise HTTPException(status_code=404, detail=f"unknown key_id {request.key_id}")

    nonce = os.urandom(12)
    ciphertext = AESGCM(key_bytes).encrypt(nonce, request.plaintext.encode(), None)
    return EncryptResponse(
        key_id=request.key_id,
        ciphertext_b64=base64.b64encode(nonce + ciphertext).decode(),
        algorithm=f"AES-256-GCM under {kms.issued_tiers[request.key_id]}",
    )


def reset_for_testing() -> None:
    """Drop the shared environment so tests start from a clean pool."""
    _kms.env = None
    _kms.policy = None
    _kms.state = None
    _kms.info = None
    _kms.issued_keys.clear()
    _kms.issued_tiers.clear()
    _kms.request_count = 0
