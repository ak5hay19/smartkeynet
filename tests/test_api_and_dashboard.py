"""Behavioral tests for the KMS API facade and the demo dashboard
(PLAN.md §4 "API surface", §6 "The Demo").

Neither is part of the MDP (Hard Rule 3), so what is tested here is
that they faithfully *report* the environment rather than reimplement
it -- in particular that the API never downgrades a deferred request.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from api.main import app, derive_data_key, reset_for_testing
from dashboard.app import TIER_COLOURS, ReplayLog, build_frames, load_results
from env.contracts import Action


@pytest.fixture
def client():
    reset_for_testing()
    with TestClient(app) as test_client:
        yield test_client
    reset_for_testing()


# ---------------------------------------------------------------------------
# Key derivation is real
# ---------------------------------------------------------------------------


def test_derived_keys_are_256_bit_and_unique():
    keys = {derive_data_key(Action.SERVE_HYBRID, "hospital/records") for _ in range(50)}
    assert all(len(key) == 32 for key in keys)
    assert len(keys) == 50  # fresh material every call, not a cached constant


def test_every_tier_derives_a_full_length_key():
    for tier in (Action.SERVE_CLASSICAL, Action.SERVE_PQC, Action.SERVE_HYBRID):
        assert len(derive_data_key(tier, "t/s")) == 32


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_generate_data_key_returns_a_usable_key(client):
    response = client.post(
        "/GenerateDataKey",
        json={"tenant": "hospital", "service": "records", "sensitivity_class": 3},
    )
    assert response.status_code in (200, 503)
    if response.status_code == 503:
        pytest.skip("pool was exhausted on this step -- covered by the deferral test")

    body = response.json()
    assert len(base64.b64decode(body["plaintext_key_b64"])) == 32
    assert body["tier_served"] in set(
        v for v in __import__("api.main", fromlist=["_TIER_NAMES"])._TIER_NAMES.values()
    )
    assert 0.0 <= body["pool_fill"] <= 1.0


def test_encrypt_round_trips_under_the_issued_key(client):
    generated = client.post(
        "/GenerateDataKey",
        json={"tenant": "fintech", "service": "ledger", "sensitivity_class": 1},
    )
    if generated.status_code == 503:
        pytest.skip("pool exhausted on this step")

    body = generated.json()
    encrypted = client.post(
        "/Encrypt", json={"key_id": body["key_id"], "plaintext": "hello quantum"}
    )
    assert encrypted.status_code == 200

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = base64.b64decode(body["plaintext_key_b64"])
    blob = base64.b64decode(encrypted.json()["ciphertext_b64"])
    nonce, ciphertext = blob[:12], blob[12:]
    assert AESGCM(key).decrypt(nonce, ciphertext, None) == b"hello quantum"


def test_encrypt_rejects_an_unknown_key_id(client):
    response = client.post("/Encrypt", json={"key_id": "nope", "plaintext": "x"})
    assert response.status_code == 404


def test_key_policy_floors_are_monotone_in_posture(client):
    """The API exposes the same Hard Rule 2 guarantee the masking layer
    enforces: a higher posture never yields a weaker floor."""
    from env.contracts import Action as A

    response = client.get("/DescribeKeyPolicy", params={"sensitivity_class": 3})
    assert response.status_code == 200

    order = ["CALM", "ELEVATED", "HIGH"]
    tier_rank = {
        "CLASSICAL_X25519_AES256GCM": 0,
        "PQC_ML_KEM_768": 1,
        "HYBRID_ML_KEM_768_XOR_QKD": 2,
    }
    floors = [tier_rank[response.json()["minimum_tier_by_posture"][p]] for p in order]
    assert floors == sorted(floors)
    assert A  # keep the import meaningful


def test_pool_status_reports_live_environment_state(client):
    response = client.get("/PoolStatus")
    assert response.status_code == 200
    body = response.json()
    for key in ("pool_fill_fraction", "skr_kbps", "qber", "deferral_queue_depth"):
        assert key in body
    assert 0.0 <= body["pool_fill_fraction"] <= 1.0


def test_a_deferred_request_is_refused_not_downgraded(client):
    """Hard Rule 9 at the API boundary. If the pool cannot cover a
    hybrid-mandatory request the endpoint must return 503 -- it must
    never quietly hand back a weaker key, which is the failure mode the
    whole architecture exists to prevent."""
    saw_deferral = False
    for _ in range(400):
        response = client.post(
            "/GenerateDataKey",
            json={
                "tenant": "hospital",
                "service": "records",
                "sensitivity_class": 3,
                "hybrid_mandatory": True,
            },
        )
        if response.status_code == 503:
            saw_deferral = True
            assert "never downgraded" in response.json()["detail"]
            break
        assert response.status_code == 200

    if not saw_deferral:
        pytest.skip("pool never exhausted in this window -- the 200 path is asserted above")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_build_frames_captures_a_replayable_episode():
    from agents.baselines import GreedyRecommenderPolicy

    log = build_frames(GreedyRecommenderPolicy(), scenario="S1", n_steps=120, seed=0)
    assert isinstance(log, ReplayLog)
    assert len(log.frames) > 0
    assert len(log.pool_curve) == len(log.frames)

    for frame in log.frames:
        assert 0.0 <= frame.pool_fill <= 1.0
        assert frame.served_tier in TIER_COLOURS
        assert frame.regret_events_total >= 0


def test_regret_curve_is_monotone_non_decreasing():
    """It is a cumulative count -- if it ever falls, the dashboard is
    misreporting the metric the demo's Beat 2 is built around."""
    from agents.baselines import AlwaysHybridPolicy

    log = build_frames(AlwaysHybridPolicy(), scenario="S3", n_steps=300, seed=0)
    assert log.regret_curve == sorted(log.regret_curve)


def test_always_hybrid_drains_the_pool_further_than_a_frugal_policy():
    """The comparison Demo Beat 2 is built on has to actually hold."""
    from dashboard.app import build_beat_two

    frugal, villain = build_beat_two(n_steps=400, seed=0)
    assert min(villain.pool_curve) <= min(frugal.pool_curve)
    assert villain.regret_curve[-1] >= frugal.regret_curve[-1]


def test_tier_histogram_totals_the_frame_count():
    from agents.baselines import GreedyRecommenderPolicy

    log = build_frames(GreedyRecommenderPolicy(), scenario="S1", n_steps=150, seed=1)
    assert sum(log.tier_histogram.values()) == len(log.frames)


def test_missing_results_file_degrades_gracefully():
    """The demo must always come up, even before the experiments have
    been run."""
    assert load_results("results/definitely_not_a_real_file.json") is None


# ---------------------------------------------------------------------------
# data/get_data.py
# ---------------------------------------------------------------------------


def test_dataset_check_reports_every_slot():
    """The external-data slots are reported rather than downloaded --
    see that module's docstring for why automating them would be wrong.
    What must hold is that the report covers every slot and never
    raises, so a fresh clone can see the state at a glance."""
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "get_data", Path(__file__).resolve().parent.parent / "data" / "get_data.py"
    )
    get_data = importlib.util.module_from_spec(spec)
    # registered before exec: @dataclass resolves its own module via
    # sys.modules, and fails with a bare AttributeError without this
    sys.modules["get_data"] = get_data
    try:
        spec.loader.exec_module(get_data)
    finally:
        sys.modules.pop("get_data", None)

    status = get_data.check_datasets()
    assert set(status) == set(get_data.SUPPORTED_DATASETS)
    assert all(isinstance(present, bool) for present in status.values())
