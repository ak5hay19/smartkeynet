"""
api/main.py

AWS-KMS-flavored REST facade (PLAN.md §4 "API surface"; §3 cloud
framing). Owned by Person D (split.md §1).

Conceptually backed by ETSI GS QKD 014 key delivery. Endpoints mirror
AWS KMS naming (`GenerateDataKey`, `Encrypt`, key policies, tenants) so
the demo reads as "what AWS KMS looks like ~2030" (PLAN.md §3).

This facade calls into env/agents at runtime for the live demo; it is
not itself part of the MDP (Hard Rule 3 -- no agent/environment logic
lives here, only routing + real primitive calls for authenticity).
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SmartKeyNet KMS Facade")


class GenerateDataKeyRequest(BaseModel):
    tenant: str
    service: str
    sensitivity_class: int


class GenerateDataKeyResponse(BaseModel):
    key_id: str
    key_type: str  # "classical" | "pqc" | "hybrid"
    ciphertext_blob: str


@app.post("/GenerateDataKey", response_model=GenerateDataKeyResponse)
def generate_data_key(request: GenerateDataKeyRequest) -> GenerateDataKeyResponse:
    """AWS-KMS-style `GenerateDataKey`.

    Routes the request through the live agent/env for a demo decision,
    then returns real primitive output (liboqs / `cryptography`) for
    the chosen key type.
    """
    raise NotImplementedError


@app.get("/health")
def health() -> dict[str, str]:
    """Trivial liveness check -- not part of the KMS facade proper."""
    return {"status": "ok"}
