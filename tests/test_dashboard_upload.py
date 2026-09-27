"""Tests for the v2 Phase 2 upload-and-run demo
(`dashboard/upload_demo.py`, `dashboard/render_upload.py`, and
`dashboard/app.py`'s `/upload` route). Mirrors
`tests/test_pcap_extract_provider.py`'s real-tshark/real-checkpoint
skip markers exactly -- these are real, end-to-end tests (real tshark
subprocess, real checkpoints, a real live `SmartKeyNetEnv` episode
under a real trained masked-DQN checkpoint), not mocked.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dashboard.app import resolve_route, resolve_upload_post
from dashboard.render_upload import render_upload_error_html, render_upload_form_html, render_upload_result_html
from dashboard.upload_demo import (
    DecisionRow,
    DEMO_TENANT_CAVEAT_TEXT,
    InvalidPcapError,
    UPLOAD_CAVEAT_TEXT,
    UploadDemoResult,
    run_pcap_upload_demo,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_LSTM_CHECKPOINT = _REPO_ROOT / "checkpoints" / "lstm_threat_forecaster.pt"
REAL_DQN_CHECKPOINT = _REPO_ROOT / "checkpoints" / "dqn_s1.pt"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pcap_flow_features_fixture.pcap"

requires_tshark = pytest.mark.skipif(shutil.which("tshark") is None, reason="tshark not found on PATH")
requires_real_checkpoints = pytest.mark.skipif(
    not (REAL_LSTM_CHECKPOINT.exists() and REAL_DQN_CHECKPOINT.exists()),
    reason="lstm_threat_forecaster.pt / dqn_s1.pt not present",
)


def test_upload_form_route_returns_200_html() -> None:
    status, content_type, body = resolve_route("/upload")
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"pcapFile" in body  # the real file-input element id


def test_upload_form_html_is_standalone_pure_function() -> None:
    html_text = render_upload_form_html()
    assert "<input type=\"file\"" in html_text
    assert "/upload" in html_text  # posts to the real route


@requires_tshark
@requires_real_checkpoints
def test_valid_pcap_upload_produces_real_decisions() -> None:
    pcap_bytes = FIXTURE.read_bytes()
    result = run_pcap_upload_demo(pcap_bytes, "pcap_flow_features_fixture.pcap", max_decisions=10)

    assert result.n_flows == 10  # matches test_pcap_extract_provider.py's own real fixture count
    assert result.n_packets_total > 0
    assert len(result.decisions) == 10

    for row in result.decisions:
        assert 0.0 <= row.threat_score <= 1.0
        assert row.chosen_action in row.legal_actions
        assert 0 <= row.pcap_flow_index < result.n_flows
        assert row.final_text  # never blank -- real cheapest-legal reasoning from dashboard/explain.py


@requires_tshark
@requires_real_checkpoints
def test_valid_pcap_upload_route_renders_decisions_and_the_honesty_caveat() -> None:
    pcap_bytes = FIXTURE.read_bytes()
    status, content_type, body = resolve_upload_post("/upload", pcap_bytes, "pcap_flow_features_fixture.pcap")

    assert status == 200
    assert content_type.startswith("text/html")
    text = body.decode("utf-8")
    assert "pcap_flow_features_fixture.pcap" in text
    assert "threat_score" in text
    # Step 3: the honest domain-mismatch caveat must be visible in the
    # rendered output itself, not just in docs.
    assert "may differ from the network security context" in text
    assert "not a validated detection claim on arbitrary traffic" in text


def test_invalid_pcap_bytes_raise_invalid_pcap_error_not_a_fabricated_result() -> None:
    garbage = b"this is not a pcap file, just plain text bytes" * 10
    with pytest.raises(InvalidPcapError):
        run_pcap_upload_demo(garbage, "not_a_pcap.txt")


def test_empty_upload_raises_invalid_pcap_error() -> None:
    with pytest.raises(InvalidPcapError):
        run_pcap_upload_demo(b"", "empty.pcap")


def test_invalid_pcap_upload_route_returns_honest_400_error_page() -> None:
    garbage = b"this is not a pcap file, just plain text bytes" * 10
    status, content_type, body = resolve_upload_post("/upload", garbage, "not_a_pcap.txt")

    assert status == 400
    assert content_type.startswith("text/html")
    text = body.decode("utf-8")
    assert "not_a_pcap.txt" in text
    assert "Could not process" in text
    # Must not silently claim success -- no decisions table on an error page.
    assert "threat_score" not in text


def test_unknown_post_route_returns_404() -> None:
    status, _content_type, _body = resolve_upload_post("/not-a-real-route", b"x", None)
    assert status == 404


def test_error_page_render_is_a_standalone_pure_function() -> None:
    text = render_upload_error_html("bad.pcap", "some real error message")
    assert "bad.pcap" in text
    assert "some real error message" in text
    assert "threat_score" not in text


def _synthetic_result() -> UploadDemoResult:
    # A hand-built UploadDemoResult -- unlike the pipeline tests above,
    # this doesn't need tshark/checkpoints, so it always runs. Only used
    # to check the renderer's own honesty-caveat wiring in isolation.
    row = DecisionRow(
        decision_num=1,
        demo_tenant="tenant_0",
        demo_service="svc",
        demo_sensitivity_class="S0",
        pcap_flow_index=0,
        pcap_flow_proto="tcp",
        pcap_flow_service="ssl",
        threat_score=0.1234,
        posture="CALM",
        floor="SERVE_CLASSICAL",
        legal_actions=("SERVE_CLASSICAL", "REKEY_NOW"),
        chosen_action="SERVE_CLASSICAL",
        final_text="SERVE_CLASSICAL was the cheapest legal action that cleared it.",
    )
    return UploadDemoResult(
        pcap_filename="synthetic.pcap",
        n_packets_total=5,
        n_packets_used=5,
        n_packets_skipped_non_ipv4=0,
        n_flows=1,
        service_counts={"ssl": 1},
        checkpoint_name="checkpoints/dqn_s1.pt",
        config_name="configs/default.yaml (S1)",
        decisions=(row,),
    )


def test_upload_caveat_text_constant_appears_verbatim_in_the_rendered_result_page() -> None:
    # Guards against the rendered page's caveat copy silently drifting
    # from the module-level constant other code might reuse.
    text = render_upload_result_html(_synthetic_result())
    assert "live demonstration of the real pipeline" in UPLOAD_CAVEAT_TEXT
    assert "live demonstration of the real pipeline" in text
    assert "synthetic.pcap" in text
    assert "0.1234" in text


def test_demo_tenant_caveat_text_constant_appears_verbatim_in_the_rendered_result_page() -> None:
    # Disclosure-only note: tenant/service/sensitivity class come from an
    # independent synthetic policy stream, not the real captured traffic --
    # guards against this copy drifting from the module-level constant.
    text = render_upload_result_html(_synthetic_result())
    assert "independent synthetic policy stream" in DEMO_TENANT_CAVEAT_TEXT
    assert "independent synthetic policy stream" in text
    assert "Demo tenant note" in text
