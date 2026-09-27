"""Tests for `dashboard/live_capture.py` (v2 Phase 3 -- an alternate
TRIGGER for Phase 1/2's already-tested pipeline, docs/v2_gameplan.md).

Split into two groups, per this session's own instruction ("add tests
where testable; where live-capture behavior genuinely can't be unit-
tested, document what was verified manually instead, and why"):

  - Pure/mockable logic (interface-listing parsing, the default-pick
    heuristic, subprocess-failure handling) -- fully unit-testable,
    real coverage, no hardware/permissions dependency.
  - A small number of REAL end-to-end tests that actually invoke
    tshark against a real local interface for a short window. These
    are inherently environment-dependent (they need a real, permitted
    network interface -- confirmed present and capturable in this
    session's own environment, see PROGRESS.md/SESSION_LOG.md's Phase
    3 entry) -- on a machine where live capture genuinely isn't
    reachable (no Npcap driver access, no interface, CI sandbox with
    no network), these tests SKIP with the real `LiveCaptureError`
    message as the reason, rather than failing the whole suite for an
    environment property this module is explicitly designed to report
    honestly, not crash on.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from dashboard.live_capture import (
    CaptureInterface,
    LiveCaptureError,
    default_capture_interface,
    list_capture_interfaces,
    run_live_capture,
    run_live_capture_demo,
)

requires_tshark = pytest.mark.skipif(shutil.which("tshark") is None, reason="tshark not found on PATH")

# A real `tshark -D` output shape, copied verbatim from this session's
# own terminal output (see SESSION_LOG.md) -- not a hypothetical format.
_REAL_TSHARK_D_OUTPUT = (
    "1. \\Device\\NPF_{C7E6E469-76DE-4A29-B4BE-157C05F81744} (Local Area Connection* 10)\n"
    "2. \\Device\\NPF_{536BBA82-FAD5-4816-9D42-C96292CACEEC} (Local Area Connection* 9)\n"
    "4. \\Device\\NPF_{80D76280-9EFD-4E82-B83C-EC3E0DA6A608} (Wi-Fi)\n"
    "7. \\Device\\NPF_{99B46AF7-E714-4565-BA10-D3C272C56ACF} (Ethernet 2)\n"
    "8. \\Device\\NPF_{AD75DBAF-4E45-4B98-99DC-DB86DABDB768} (vEthernet (WSL (Hyper-V firewall)))\n"
    "9. \\Device\\NPF_Loopback (Adapter for loopback traffic capture)\n"
    "10. etwdump (Event Tracing for Windows (ETW) reader)\n"
)


class _FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---- list_capture_interfaces / default_capture_interface: pure, mocked ----


def test_list_capture_interfaces_parses_real_tshark_dash_d_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dashboard.live_capture.shutil.which", lambda _name: "/usr/bin/tshark")
    monkeypatch.setattr(
        "dashboard.live_capture.subprocess.run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout=_REAL_TSHARK_D_OUTPUT),
    )
    interfaces = list_capture_interfaces()
    assert [i.friendly_name for i in interfaces] == [
        "Local Area Connection* 10",
        "Local Area Connection* 9",
        "Wi-Fi",
        "Ethernet 2",
        "vEthernet (WSL (Hyper-V firewall))",
        "Adapter for loopback traffic capture",
        "Event Tracing for Windows (ETW) reader",
    ]
    wifi = next(i for i in interfaces if i.friendly_name == "Wi-Fi")
    assert wifi.index == "4"
    assert wifi.device == "\\Device\\NPF_{80D76280-9EFD-4E82-B83C-EC3E0DA6A608}"


def test_list_capture_interfaces_raises_honestly_when_tshark_not_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dashboard.live_capture.shutil.which", lambda _name: None)
    with pytest.raises(LiveCaptureError, match="not on PATH"):
        list_capture_interfaces()


def test_list_capture_interfaces_raises_honestly_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dashboard.live_capture.shutil.which", lambda _name: "/usr/bin/tshark")
    monkeypatch.setattr(
        "dashboard.live_capture.subprocess.run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="permission denied"),
    )
    with pytest.raises(LiveCaptureError, match="permission denied"):
        list_capture_interfaces()


def test_list_capture_interfaces_raises_when_output_has_no_parseable_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dashboard.live_capture.shutil.which", lambda _name: "/usr/bin/tshark")
    monkeypatch.setattr(
        "dashboard.live_capture.subprocess.run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout="not a real interface listing\n"),
    )
    with pytest.raises(LiveCaptureError, match="no usable interfaces"):
        list_capture_interfaces()


def test_list_capture_interfaces_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dashboard.live_capture.shutil.which", lambda _name: "/usr/bin/tshark")

    def _raise_timeout(*a: object, **k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="tshark -D", timeout=10)

    monkeypatch.setattr("dashboard.live_capture.subprocess.run", _raise_timeout)
    with pytest.raises(LiveCaptureError, match="timed out"):
        list_capture_interfaces()


def test_default_capture_interface_prefers_wifi_over_pseudo_and_loopback_adapters() -> None:
    interfaces = [
        CaptureInterface(index="1", device="d1", friendly_name="Local Area Connection* 10"),
        CaptureInterface(index="4", device="d4", friendly_name="Wi-Fi"),
        CaptureInterface(index="9", device="d9", friendly_name="Adapter for loopback traffic capture"),
    ]
    assert default_capture_interface(interfaces).friendly_name == "Wi-Fi"


def test_default_capture_interface_falls_back_to_first_when_no_hint_matches() -> None:
    interfaces = [
        CaptureInterface(index="1", device="d1", friendly_name="Local Area Connection* 10"),
        CaptureInterface(index="9", device="d9", friendly_name="Adapter for loopback traffic capture"),
    ]
    assert default_capture_interface(interfaces).index == "1"


def test_default_capture_interface_never_picks_wsl_hyper_v_pseudo_adapter() -> None:
    interfaces = [
        CaptureInterface(index="8", device="d8", friendly_name="vEthernet (WSL (Hyper-V firewall))"),
        CaptureInterface(index="7", device="d7", friendly_name="Ethernet 2"),
    ]
    assert default_capture_interface(interfaces).friendly_name == "Ethernet 2"


# ---- run_live_capture: failure paths, mocked (no real hardware needed) ----


def test_run_live_capture_raises_honestly_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_timeout(*a: object, **k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="tshark", timeout=5)

    monkeypatch.setattr("dashboard.live_capture.subprocess.run", _raise_timeout)
    with pytest.raises(LiveCaptureError, match="did not finish within"):
        run_live_capture("4", duration_seconds=5)


def test_run_live_capture_raises_honestly_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dashboard.live_capture.subprocess.run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="the interface index is not valid"),
    )
    with pytest.raises(LiveCaptureError, match="interface index is not valid"):
        run_live_capture("999", duration_seconds=5)


def test_run_live_capture_raises_honestly_when_no_output_file_was_written(monkeypatch: pytest.MonkeyPatch) -> None:
    # tshark exits 0 but (in this simulated case) never wrote/populated
    # the capture file -- the real "zero packets captured" case.
    monkeypatch.setattr(
        "dashboard.live_capture.subprocess.run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout="0 packets captured"),
    )
    with pytest.raises(LiveCaptureError, match="no packets were captured"):
        run_live_capture("4", duration_seconds=5)


# ---- Real, end-to-end: actual tshark, actual local interface ----
#
# These are the only genuinely non-mockable behaviors in this module
# (whether Npcap driver access is actually granted, whether a real
# interface is actually capturable from wherever the test process
# runs). Confirmed working in this session's own environment (see
# PROGRESS.md's Phase 3 entry: a real 5s capture on the Wi-Fi
# interface produced 101 real packets, 9 real flows, correct 39-dim
# feature output). Kept short (5-8s, not the UI's 20s default) to keep
# the suite fast.


@requires_tshark
def test_real_short_live_capture_produces_a_readable_pcap_file() -> None:
    try:
        interfaces = list_capture_interfaces()
        iface = default_capture_interface(interfaces)
        pcap_bytes = run_live_capture(iface.index, duration_seconds=5)
    except LiveCaptureError as exc:
        pytest.skip(f"Live capture not reachable in this environment: {exc}")

    assert len(pcap_bytes) > 0
    # pcap/pcapng magic bytes -- confirms tshark wrote a real capture
    # file, not an empty/corrupt one.
    assert pcap_bytes[:4] in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x0a\x0d\x0d\x0a")


@requires_tshark
def test_real_short_live_capture_feeds_the_unmodified_phase1_2_pipeline() -> None:
    """End-to-end: real capture -> the SAME `run_pcap_upload_demo` Phase
    2 already validated -- confirms Phase 3 is genuinely just an
    alternate trigger, not a parallel pipeline."""
    from dashboard.upload_demo import InvalidPcapError

    try:
        interfaces = list_capture_interfaces()
        iface = default_capture_interface(interfaces)
        result = run_live_capture_demo(iface.index, duration_seconds=8)
    except LiveCaptureError as exc:
        pytest.skip(f"Live capture not reachable in this environment: {exc}")
    except InvalidPcapError as exc:
        pytest.skip(f"Captured window had no extractable flows (quiet network): {exc}")

    assert result.n_packets_total > 0
    assert result.n_flows > 0
    assert len(result.decisions) > 0
    for row in result.decisions:
        assert 0.0 <= row.threat_score <= 1.0
        assert row.chosen_action in row.legal_actions
