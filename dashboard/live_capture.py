"""
dashboard/live_capture.py

v2 Phase 3 (docs/v2_gameplan.md) -- an alternate TRIGGER for the exact
same pipeline Phase 1/2 already built and validated (tshark extraction
-> `PCAPExtractProvider` -> LSTM -> masked DQN), not a new pipeline.
This module's only job is: capture a short, bounded live packet stream
from a real interface on this machine, then hand the resulting bytes
to `dashboard.upload_demo.run_pcap_upload_demo` UNCHANGED -- the exact
same function Phase 2's uploaded-file path already calls. No
extraction/provider/masking/DQN logic is duplicated or forked here.

Aligns with PLAN2.md Hard Rule 11 (SS5.5): true live capture is
explicitly scoped as a stretch item that must never compete for time
against the project's guaranteed deliverables -- it is not those. It
is built here only because (a) the guaranteed deliverables (base
project, Addition E Phase 4b, Gate W3) are already complete and frozen
on `main`/`v2`, and (b) `docs/v2_gameplan.md`'s own Phase 3 already
scoped this as optional, time-boxed, stretch work on a separate branch
(`v3-live-capture`) -- consistent with, not a violation of, Hard Rule
11's framing.

CAPTURE_DURATION_SECONDS documents this session's explicit bounded-
duration choice (short and FIXED, not open-ended): 20 seconds is long
enough to gather a handful of real flows (Phase 1/2's own reference
pcap produced ~30 flows from ~10s of ordinary browsing; 20s gives
comfortable margin over that) while staying well inside a demo/
rehearsal attention span. The whole capture runs to completion before
anything is shown -- there is no incremental/streaming result display,
matching Phase 2's own synchronous request-response result page.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from dashboard.upload_demo import UploadDemoResult, run_pcap_upload_demo

CAPTURE_DURATION_SECONDS = 20
"""Bounded, fixed capture window -- see module docstring for the
reasoning. Not user-adjustable in the UI, by design (Step 2: a short,
FIXED duration, not open-ended)."""

_TSHARK_BINARY = "tshark"
_INTERFACE_LIST_TIMEOUT_SECONDS = 10
# Margin over the capture duration itself, so a genuinely hung tshark
# process (not just a slow one) still bounds how long a request blocks.
_SUBPROCESS_TIMEOUT_MARGIN_SECONDS = 15


class LiveCaptureError(Exception):
    """A real, user-facing failure at the CAPTURE step (tshark
    missing, no interfaces available, permission denied, the capture
    window produced literally zero packets) -- distinct from
    `dashboard.upload_demo.InvalidPcapError`, which is raised by the
    (unmodified) EXTRACTION step this module hands its output to.
    Never caught silently into a fabricated result."""


@dataclass(frozen=True)
class CaptureInterface:
    """One row of `tshark -D`'s real interface listing."""

    index: str  # tshark's own `-D` numbering, e.g. "4" -- what `-i` accepts
    device: str  # e.g. \\Device\\NPF_{...}
    friendly_name: str  # e.g. "Wi-Fi"


# tshark -D output looks like: "4. \Device\NPF_{...} (Wi-Fi)". The
# friendly name itself can contain nested parens (a real, confirmed
# case on this session's own machine: "vEthernet (WSL (Hyper-V
# firewall))") -- `.*` (greedy, anchored to end-of-line) rather than
# `[^)]*` is required so the LAST `)` closes the group, not the first.
_INTERFACE_LINE_RE = re.compile(r"^(\d+)\.\s+(\S+)\s+\((.*)\)\s*$")


def list_capture_interfaces() -> list[CaptureInterface]:
    """Real interface enumeration via `tshark -D` -- never a
    hardcoded/guessed list (Step 3). Raises `LiveCaptureError` plainly
    if tshark isn't callable, or lists no interfaces at all, rather
    than returning an empty list that would silently read as "no
    interfaces exist"."""
    if shutil.which(_TSHARK_BINARY) is None:
        raise LiveCaptureError("tshark is not on PATH -- live capture is not available in this environment.")

    try:
        proc = subprocess.run(
            [_TSHARK_BINARY, "-D"],
            capture_output=True,
            text=True,
            timeout=_INTERFACE_LIST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise LiveCaptureError("tshark -D timed out while listing interfaces.") from exc
    except OSError as exc:
        raise LiveCaptureError(f"Could not run tshark -D: {exc}") from exc

    if proc.returncode != 0:
        raise LiveCaptureError(
            f"tshark -D failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    interfaces = [
        CaptureInterface(index=m.group(1), device=m.group(2), friendly_name=m.group(3))
        for line in proc.stdout.splitlines()
        if (m := _INTERFACE_LINE_RE.match(line.strip()))
    ]
    if not interfaces:
        raise LiveCaptureError("tshark -D returned no usable interfaces on this machine.")
    return interfaces


# Best-effort, name-only heuristic (Step 3: "or default to whichever
# looks like the active one"). This environment has no reliable
# cross-platform way to query which adapter currently carries traffic
# without adding a new dependency -- a real limitation, stated here
# rather than glossed over. The person can always override the pick.
_LIKELY_ACTIVE_NAME_HINTS = ("wi-fi", "wifi", "ethernet")
_LIKELY_INACTIVE_NAME_HINTS = ("loopback", "wsl", "hyper-v", "npcap", "etw", "local area connection")


def default_capture_interface(interfaces: list[CaptureInterface]) -> CaptureInterface:
    """Best-effort default selection, never a guess presented as
    certainty -- the UI always shows the full real list so the person
    can override this. Raises nothing; falls back to the first real
    interface if no name-based hint matches."""
    for iface in interfaces:
        name = iface.friendly_name.lower()
        if any(hint in name for hint in _LIKELY_ACTIVE_NAME_HINTS) and not any(
            hint in name for hint in _LIKELY_INACTIVE_NAME_HINTS
        ):
            return iface
    return interfaces[0]


def run_live_capture(interface_index: str, duration_seconds: int = CAPTURE_DURATION_SECONDS) -> bytes:
    """Real, bounded live capture -- blocks for exactly
    `duration_seconds` (tshark's own `-a duration:N` autostop does the
    bounding, not a wrapper-level timer/kill), then returns the
    captured bytes. Raises `LiveCaptureError` on any real failure (bad
    interface, permission denied, zero packets captured, a hung
    process) rather than returning empty/fabricated bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "live_capture.pcap"
        try:
            proc = subprocess.run(
                [
                    _TSHARK_BINARY,
                    "-i", interface_index,
                    "-a", f"duration:{duration_seconds}",
                    "-w", str(tmp_path),
                ],
                capture_output=True,
                text=True,
                timeout=duration_seconds + _SUBPROCESS_TIMEOUT_MARGIN_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise LiveCaptureError(
                f"Live capture on interface {interface_index!r} did not finish within "
                f"{duration_seconds + _SUBPROCESS_TIMEOUT_MARGIN_SECONDS}s -- aborted rather than left blocking."
            ) from exc
        except OSError as exc:
            raise LiveCaptureError(f"Could not start tshark for live capture: {exc}") from exc

        if proc.returncode != 0:
            raise LiveCaptureError(
                f"tshark could not capture on interface {interface_index!r} (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise LiveCaptureError(
                f"Capture on interface {interface_index!r} produced no output file -- "
                "no packets were captured in the window."
            )

        return tmp_path.read_bytes()


def run_live_capture_demo(
    interface_index: str,
    duration_seconds: int = CAPTURE_DURATION_SECONDS,
    max_decisions: int | None = None,
) -> UploadDemoResult:
    """The full Phase 3 flow: real bounded live capture -> the exact
    same `run_pcap_upload_demo` Phase 2's uploaded-file path calls,
    unmodified. Raises `LiveCaptureError` (this module) for a capture-
    step failure, or `dashboard.upload_demo.InvalidPcapError`
    (unchanged, Phase 1/2) if the capture succeeded but yielded zero
    extractable flows -- the caller is expected to handle both as
    honest errors, never falling back to fabricated output."""
    pcap_bytes = run_live_capture(interface_index, duration_seconds)
    filename = f"live-capture-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pcap"
    kwargs = {} if max_decisions is None else {"max_decisions": max_decisions}
    return run_pcap_upload_demo(pcap_bytes, filename, **kwargs)
