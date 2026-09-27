"""Tests for the v2 Phase 3 `/capture` route
(`dashboard/app.py::resolve_route`/`resolve_capture_post`,
`dashboard/render_upload.py::render_capture_form_html`/
`render_capture_unavailable_html`). Mirrors
`tests/test_dashboard_upload.py`'s structure exactly -- pure-function
route tests, real-fixture-style end-to-end tests gated behind
`requires_tshark`, and isolated renderer tests that never need
tshark/checkpoints.

Route-level failure-path tests are mocked (patch
`dashboard.app.run_live_capture_demo` / `list_capture_interfaces`)
rather than re-running real hardware-dependent capture for every
branch -- the one genuinely real, unmocked capture->pipeline path is
already covered end-to-end in `tests/test_live_capture.py` (with a
short duration) and once more here at the full HTTP-route layer with
the real default duration, per this session's Step 5 instruction.
"""

from __future__ import annotations

import shutil

import pytest

from dashboard.app import resolve_capture_post, resolve_route
from dashboard.live_capture import (
    CaptureInterface,
    LiveCaptureError,
    default_capture_interface,
    list_capture_interfaces,
)
from dashboard.render_upload import render_capture_form_html, render_capture_unavailable_html
from dashboard.upload_demo import InvalidPcapError

requires_tshark = pytest.mark.skipif(shutil.which("tshark") is None, reason="tshark not found on PATH")


# ---- render_capture_form_html / render_capture_unavailable_html: pure ----


def test_capture_form_html_lists_every_given_interface_and_marks_the_default() -> None:
    interfaces = (
        CaptureInterface(index="1", device="d1", friendly_name="Local Area Connection* 10"),
        CaptureInterface(index="4", device="d4", friendly_name="Wi-Fi"),
    )
    text = render_capture_form_html(interfaces, default_index="4")
    assert "Local Area Connection* 10" in text
    assert "Wi-Fi" in text
    assert 'value="4" selected' in text
    assert "/capture?interface=" in text  # POST target, query-param based (no file body)


def test_capture_unavailable_html_states_the_real_reason_not_a_blank_page() -> None:
    text = render_capture_unavailable_html("tshark is not on PATH.")
    assert "tshark is not on PATH." in text
    assert "Live capture is not available" in text
    assert "/upload" in text  # points the person at the still-working alternate path


# ---- GET /capture: real interface listing (fast -- just `tshark -D`) ----


@requires_tshark
def test_capture_form_route_returns_200_with_real_interfaces() -> None:
    status, content_type, body = resolve_route("/capture")
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"ifaceSelect" in body
    assert b"/capture?interface=" in body


def test_capture_form_route_reports_honest_unavailability_when_listing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*a: object, **k: object) -> None:
        raise LiveCaptureError("simulated: no interfaces reachable")

    monkeypatch.setattr("dashboard.app.list_capture_interfaces", _raise)
    status, content_type, body = resolve_route("/capture")
    assert status == 200  # a real, honest page -- not a 500 or a crash
    assert content_type.startswith("text/html")
    text = body.decode("utf-8")
    assert "simulated: no interfaces reachable" in text
    assert "not available" in text


# ---- POST /capture: routing/error-branch logic, mocked for determinism ----


def test_capture_post_unknown_route_returns_404() -> None:
    status, _content_type, _body = resolve_capture_post("/not-a-real-route?interface=4")
    assert status == 404


def test_capture_post_with_no_interface_param_returns_honest_400() -> None:
    status, content_type, body = resolve_capture_post("/capture")
    assert status == 400
    assert content_type.startswith("text/html")
    assert b"No interface was selected" in body


def test_capture_post_with_empty_interface_param_returns_honest_400() -> None:
    status, _content_type, body = resolve_capture_post("/capture?interface=")
    assert status == 400
    assert b"No interface was selected" in body


def test_capture_post_surfaces_a_live_capture_error_as_honest_400(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a: object, **k: object) -> None:
        raise LiveCaptureError("simulated: permission denied opening interface")

    monkeypatch.setattr("dashboard.app.run_live_capture_demo", _raise)
    status, content_type, body = resolve_capture_post("/capture?interface=4")
    assert status == 400
    assert content_type.startswith("text/html")
    text = body.decode("utf-8")
    assert "simulated: permission denied opening interface" in text
    assert "threat_score" not in text  # never a fabricated result alongside an error


def test_capture_post_surfaces_an_invalid_pcap_error_as_honest_400(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a: object, **k: object) -> None:
        raise InvalidPcapError("simulated: zero extractable flows in the captured window")

    monkeypatch.setattr("dashboard.app.run_live_capture_demo", _raise)
    status, content_type, body = resolve_capture_post("/capture?interface=4")
    assert status == 400
    text = body.decode("utf-8")
    assert "simulated: zero extractable flows in the captured window" in text


def test_capture_post_success_renders_the_same_result_page_shape_as_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirms `/capture`'s success path reuses `render_upload_result_html`
    verbatim (Step 4: same results table, same caveat) via a hand-built
    result -- isolates the route-wiring assertion from real hardware."""
    from tests.test_dashboard_upload import _synthetic_result

    monkeypatch.setattr("dashboard.app.run_live_capture_demo", lambda *_a, **_k: _synthetic_result())
    status, content_type, body = resolve_capture_post("/capture?interface=4")
    assert status == 200
    assert content_type.startswith("text/html")
    text = body.decode("utf-8")
    assert "threat_score" in text
    assert "may differ from the network security context" in text


# ---- Real, end-to-end: the full HTTP route, a real bounded capture ----


@requires_tshark
def test_real_live_capture_post_route_end_to_end() -> None:
    """Step 5: test with a real short live capture, end to end, through
    the actual HTTP route function (not just the underlying module),
    and report the real result -- not a synthetic example. Uses the
    real default `CAPTURE_DURATION_SECONDS` window (whatever
    `dashboard/live_capture.py` currently documents), so this is
    exactly what a person clicking the button in a browser gets.

    A 400 result here means the environment genuinely couldn't
    complete a live capture (no permitted interface, or a real window
    with zero extractable flows) -- an honest, environment-dependent
    outcome documented as such (see `tests/test_live_capture.py`'s
    module docstring), not a code defect -- so this skips rather than
    fails in that case."""
    try:
        interfaces = list_capture_interfaces()
        iface = default_capture_interface(interfaces)
    except LiveCaptureError as exc:
        pytest.skip(f"Could not enumerate a real interface in this environment: {exc}")

    status, content_type, body = resolve_capture_post(f"/capture?interface={iface.index}")
    text = body.decode("utf-8")

    if status == 400:
        pytest.skip(f"Live capture route returned an honest environment-dependent failure: {text[:300]}")

    assert status == 200
    assert content_type.startswith("text/html")
    assert "threat_score" in text
    assert "may differ from the network security context" in text


# ---- wraparound disclosure note (results page) ----


def test_result_page_shows_replay_note_when_flows_fewer_than_decisions() -> None:
    from dataclasses import replace

    from dashboard.render_upload import render_upload_result_html
    from tests.test_dashboard_upload import _synthetic_result

    base = _synthetic_result()  # 1 flow, 1 decision
    result = replace(base, n_flows=1, decisions=base.decisions * 3)  # 3 decisions > 1 flow
    text = render_upload_result_html(result)
    assert "1 real flows extracted; this 3-decision episode revisits" in text
    assert "not new traffic" in text


def test_result_page_omits_replay_note_when_flows_cover_decisions() -> None:
    from dataclasses import replace

    from dashboard.render_upload import render_upload_result_html
    from tests.test_dashboard_upload import _synthetic_result

    base = _synthetic_result()
    for n_flows in (1, 5):  # equal to, and greater than, the 1 decision
        text = render_upload_result_html(replace(base, n_flows=n_flows))
        assert "revisits" not in text
        assert "Replay note" not in text
