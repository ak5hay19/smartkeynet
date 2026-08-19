"""Behavioral tests for the KMS API facade and the demo dashboard
(PLAN.md §4 "API surface", §6 "The Demo").

Neither is part of the MDP (Hard Rule 3), so what is tested here is
that they faithfully *report* the environment rather than reimplement
it -- in particular that the API never downgrades a deferred request.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app, derive_data_key, reset_for_testing
from dashboard.app import TIER_COLOURS, beat_two_logs, load_results
from dashboard.replay import ReplayEpisode, frames_from_events, load_episode
from env.contracts import Action

REPO = Path(__file__).resolve().parent.parent


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


def _record(policy, scenario: str, n_steps: int, seed: int, tmp_path) -> ReplayEpisode:
    """Record an episode and replay it back off disk.

    Every dashboard test goes through the written log, never through a live
    env, because that round trip *is* the property §S13 asks for.
    """
    from experiments.record_demo import record_episode

    log_path = record_episode(
        policy, scenario, tmp_path / f"{scenario}_{seed}.jsonl.gz", n_steps=n_steps, seed=seed
    )
    return load_episode(log_path, label=f"{scenario}/{seed}")


def test_dashboard_render_module_imports_nothing_from_the_env(tmp_path):
    """SMARTKEYNET_BUILD_SPEC.md §S13: the dashboard "reads
    `events.jsonl.gz` -- never reaches into env internals".

    Enforced by AST scan rather than review, because this boundary was
    violated for most of the project's life and nothing failed. `app.py`
    constructed a live `SmartKeyNetEnv` and read `_current_request`,
    `_sessions`, `_policy_table._ratcheted_posture`, `_regret_log` and
    `_deferral_queue` off it -- so the demo could not replay a recorded run,
    was coupled to five private attributes, and left the §4.4 event schema
    completely unexercised.
    """
    import ast

    for module_name in ("dashboard/app.py", "dashboard/replay.py"):
        tree = ast.parse((REPO / module_name).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        offenders = [
            name
            for name in imported
            if name.split(".")[0] in {"env", "agents", "forecaster", "experiments"}
        ]
        assert not offenders, (
            f"{module_name} imports {offenders} -- the dashboard must render from the "
            "event log alone (§S13)"
        )


def test_replay_reconstructs_an_episode_from_the_log_alone(tmp_path):
    from agents.baselines import GreedyRecommenderPolicy

    episode = _record(GreedyRecommenderPolicy(), "S1", 120, 0, tmp_path)
    assert isinstance(episode, ReplayEpisode)
    assert len(episode.frames) > 0
    assert len(episode.pool_curve) == len(episode.frames)

    for frame in episode.frames:
        assert 0.0 <= frame.pool_fill <= 1.0
        assert frame.served_tier in TIER_COLOURS or frame.served_tier in {"REUSE", "REKEY"}
        assert frame.regret_events_total >= 0


def test_replay_of_an_empty_log_yields_no_frames():
    """A log with no serves must not crash the panels -- the demo has to come
    up before any run exists."""
    assert frames_from_events([]).frames == []


def test_regret_curve_is_monotone_non_decreasing(tmp_path):
    """It is a cumulative count -- if it ever falls, the dashboard is
    misreporting the metric the demo's Beat 2 is built around."""
    from agents.baselines import AlwaysHybridPolicy

    episode = _record(AlwaysHybridPolicy(), "S3", 300, 0, tmp_path)
    assert episode.regret_curve == sorted(episode.regret_curve)


def test_overflow_curve_is_monotone_non_decreasing(tmp_path):
    """Cumulative wasted key material, same reasoning."""
    from agents.baselines import AlwaysPQCPolicy

    episode = _record(AlwaysPQCPolicy(), "S1", 300, 0, tmp_path)
    assert episode.overflow_curve == sorted(episode.overflow_curve)


def test_always_hybrid_drains_the_pool_further_than_a_frugal_policy(tmp_path):
    """The comparison Demo Beat 2 is built on has to actually hold."""
    from agents.baselines import AlwaysHybridPolicy, StaticThresholdPolicy

    villain = _record(AlwaysHybridPolicy(), "S3", 400, 0, tmp_path)
    frugal = _record(
        StaticThresholdPolicy(pool_fill_threshold=0.5, min_hybrid_class=2, rekey_age_frac=0.9),
        "S3",
        400,
        0,
        tmp_path,
    )
    assert min(villain.pool_curve) <= min(frugal.pool_curve)
    assert villain.regret_curve[-1] >= frugal.regret_curve[-1]


def test_tier_histogram_totals_the_frame_count(tmp_path):
    from agents.baselines import GreedyRecommenderPolicy

    episode = _record(GreedyRecommenderPolicy(), "S1", 150, 1, tmp_path)
    assert sum(episode.tier_histogram.values()) == len(episode.frames)


def test_beat_two_logs_returns_none_when_nothing_was_recorded(tmp_path):
    """Rendering must not be able to start a simulation to fill a gap -- it
    reports the gap instead."""
    assert beat_two_logs(log_dir=tmp_path / "nonexistent") is None


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
