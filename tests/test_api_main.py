"""Behavioral tests for `api.main` -- the AWS-KMS-flavoured facade
(PLAN.md §4 API surface; PLAN2 §7.3's Explain Decision endpoint).

Two things matter here beyond "the endpoints respond": the facade owns
no decision logic (Hard Rule 3), and it never overstates what its
cryptography is (the primitive-honesty matrix).
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from api.main import KmsSession, _PRIMITIVE_MATRIX, app, derive_data_key, get_session
from env.contracts import Action


@pytest.fixture()
def client() -> TestClient:
    get_session().scenario = "S1"
    get_session().reset()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Primitive honesty
# ---------------------------------------------------------------------------


def test_health_publishes_the_primitive_honesty_matrix():
    """Which primitives are real is not a footnote, and nobody should
    have to read source to find out."""
    payload = TestClient(app).get("/Health").json()
    assert payload["status"] == "ok"
    assert set(payload["primitives"]) == {"classical", "pqc", "hybrid"}
    assert payload["primitives"]["classical"]["real"] is True
    assert payload["primitives"]["pqc"]["real"] is False
    assert payload["primitives"]["hybrid"]["real"] == "partial"


def test_pqc_is_never_advertised_as_real():
    """liboqs is an optional dependency and is not installed, so there is
    no ML-KEM encapsulation here. Claiming otherwise would be the single
    most misleading thing this demo could do."""
    _key, provenance = derive_data_key("pqc")
    assert provenance["real"] is False
    assert "SIMULATED" in provenance["detail"]
    assert "not quantum-resistant" in provenance["detail"]


def test_hybrid_states_that_the_pool_holds_a_count_not_bytes():
    _key, provenance = derive_data_key("hybrid")
    assert provenance["real"] == "partial"
    assert "count" in provenance["detail"].lower()


def test_every_tier_derives_a_distinct_256_bit_key():
    keys = {tier: derive_data_key(tier)[0] for tier in ("classical", "pqc", "hybrid")}
    assert all(len(key) == 32 for key in keys.values())
    assert len(set(keys.values())) == 3


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        derive_data_key("quantum-magic")


# ---------------------------------------------------------------------------
# GenerateDataKey
# ---------------------------------------------------------------------------


def test_generate_data_key_returns_a_real_decryptable_blob(client):
    response = client.post("/GenerateDataKey", json={}).json()
    assert response["key_type"] in {"classical", "pqc", "hybrid"}
    blob = base64.b64decode(response["ciphertext_blob"])
    assert len(blob) > 12  # nonce + ciphertext + tag
    assert response["simulation_note"]


def test_generated_key_type_never_falls_below_the_policy_floor(client):
    """Hard Rule 2 end to end, through the API rather than the harness."""
    tier_index = {"classical": 0, "pqc": 1, "hybrid": 2}
    for _ in range(60):
        response = client.post("/GenerateDataKey", json={}).json()
        floor = Action[response["policy_floor"]]
        assert tier_index[response["key_type"]] >= int(floor)


def test_caller_supplied_tenant_does_not_steer_the_decision(client):
    """Hard Rule 3: the request stream comes from the tenant graph
    inside the environment. A caller naming a tenant must not be able to
    conjure a decision for it -- the API is a facade, not an input to
    the MDP."""
    response = client.post(
        "/GenerateDataKey",
        json={"tenant": "attacker-controlled", "service": "x", "sensitivity_class": 3},
    ).json()
    assert response["tenant"] != "attacker-controlled"


# ---------------------------------------------------------------------------
# ExplainDecision (Hard Rule 10)
# ---------------------------------------------------------------------------


def test_explain_decision_returns_the_six_step_trace(client):
    generated = client.post("/GenerateDataKey", json={}).json()
    trace = client.get(f"/ExplainDecision/{generated['request_id']}").json()["trace"]

    assert [step["index"] for step in trace["steps"]] == [1, 2, 3, 4, 5, 6]
    assert trace["policy_floor"] == generated["policy_floor"]
    assert trace["chosen_action"] == generated["action"]


def test_explain_decision_404s_for_an_unknown_request(client):
    assert client.get("/ExplainDecision/not-a-real-id").status_code == 404


def test_recent_decisions_are_returned_newest_last(client):
    for _ in range(5):
        client.post("/GenerateDataKey", json={})
    decisions = client.get("/Decisions?limit=5").json()["decisions"]
    assert len(decisions) == 5
    assert [d["step"] for d in decisions] == sorted(d["step"] for d in decisions)
    assert all(d["reason"] for d in decisions)


# ---------------------------------------------------------------------------
# Status endpoints
# ---------------------------------------------------------------------------


def test_pool_status_reports_the_live_pool(client):
    before = client.get("/PoolStatus").json()
    for _ in range(30):
        client.post("/GenerateDataKey", json={})
    after = client.get("/PoolStatus").json()

    assert before["capacity_bits"] == after["capacity_bits"]
    assert 0.0 <= after["fill_fraction"] <= 1.0
    assert after["bits_per_hybrid_draw"] == 256.0


def test_threat_status_states_the_two_invariants_panel_one_must_show(client):
    """PLAN2 §7.1 requires the Threat Input panel to say that the signal
    feeds the floor computation "only in the raise direction" and is
    "frozen during agent training"."""
    payload = client.get("/ThreatStatus").json()
    assert payload["frozen_during_training"] is True
    assert payload["raise_only"] is True
    assert payload["resolved_posture"] in {"CALM", "ELEVATED", "HIGH"}
    assert len(payload["posture_probs"]) == 3
    assert sum(payload["posture_probs"]) == pytest.approx(1.0)


def test_scenario_switch_resets_the_session(client):
    assert client.post("/Scenario/S3").json()["scenario"] == "S3"
    assert get_session().scenario == "S3"
    assert client.post("/Scenario/S99").status_code == 400


def test_session_retains_a_bounded_number_of_traces():
    """A live demo runs indefinitely; unbounded trace retention is a
    memory leak with a countdown."""
    session = KmsSession()
    session.reset()
    for _ in range(session._MAX_RETAINED_TRACES + 25):
        session.decide()
    assert len(session._traces) == session._MAX_RETAINED_TRACES
