"""Tests for `dashboard.render_posture_validation`/`dashboard
.render_posture_validation_demo` -- the Posture Validation panel added
2026-09-23, turning the already-real, already-committed
posture-variance/floor-monotonicity validation result (SESSION_LOG.md/
PROGRESS.md's matching entry, commit `83f23ba`) into a visible
dashboard panel.

The central honesty guard here (Hard Rule 7 discipline, same as every
other panel's own exact-match tests): the committed `dashboard/samples/
posture_validation_data.json` must show EXACTLY the same real numbers
already reported in SESSION_LOG.md/PROGRESS.md (100 total decisions,
CALM 66 / ELEVATED 4 / HIGH 30, 0 floor-monotonicity violations) -- so
this panel can never silently drift from the real validation result it
claims to visualize. Values below are copied verbatim from that
committed JSON, not retyped from memory or re-derived.
"""

from __future__ import annotations

import json
from pathlib import Path

from dashboard.render_posture_validation import (
    FloorsByClass,
    PostureValidationData,
    SourceInfo,
    ThreatScoreExample,
    render_posture_validation_html,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_PATH = _REPO_ROOT / "dashboard" / "samples" / "posture_validation_data.json"
_HTML_PATH = _REPO_ROOT / "dashboard" / "samples" / "posture_validation.html"


def _load_committed_data() -> dict:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def test_committed_data_file_exists() -> None:
    assert _DATA_PATH.is_file(), "dashboard/render_posture_validation_demo.py must be run to produce the committed data file"


def test_committed_data_matches_the_real_reported_validation_numbers() -> None:
    # Exact numbers from SESSION_LOG.md/PROGRESS.md's 2026-09-23 entry
    # (commit 83f23ba) -- the whole point of this test is that these
    # can never silently drift.
    data = _load_committed_data()
    assert data["total_decisions"] == 100
    assert data["posture_counts"] == {"CALM": 66, "ELEVATED": 4, "HIGH": 30}
    assert data["floor_violations"] == 0
    assert data["action_counts"] == {
        "SERVE_CLASSICAL": 34,
        "REKEY_NOW": 35,
        "SERVE_PQC": 15,
        "REUSE": 16,
    }
    assert sum(s["n_decisions"] for s in data["sources"]) == 100


def test_committed_data_floor_by_class_is_monotonic_and_matches_session_log() -> None:
    data = _load_committed_data()
    tier_order = {"SERVE_CLASSICAL": 0, "SERVE_PQC": 1, "SERVE_HYBRID": 2}

    by_posture: dict[str, dict[str, set[str]]] = {}
    for row in data["floor_by_class"]:
        by_posture.setdefault(row["posture"], {})[row["sensitivity_class"]] = set(row["floors"])

    # Exact table reported in SESSION_LOG.md.
    assert by_posture["CALM"]["S0"] == {"SERVE_CLASSICAL"}
    assert by_posture["CALM"]["S1"] == {"SERVE_CLASSICAL", "SERVE_PQC"}
    assert by_posture["CALM"]["S2"] == {"SERVE_PQC"}
    assert by_posture["CALM"]["S3"] == {"SERVE_HYBRID"}
    assert by_posture["HIGH"]["S0"] == {"SERVE_PQC"}
    assert by_posture["HIGH"]["S1"] == {"SERVE_PQC"}
    assert by_posture["HIGH"]["S2"] == {"SERVE_HYBRID"}
    assert by_posture["HIGH"]["S3"] == {"SERVE_HYBRID"}

    for posture, by_class in by_posture.items():
        classes = sorted(by_class.keys())
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                max_lo = max(tier_order[f] for f in by_class[classes[i]])
                min_hi = min(tier_order[f] for f in by_class[classes[j]])
                assert min_hi >= max_lo, f"monotonicity violated at {posture}: {classes[i]} vs {classes[j]}"


def test_elevated_transient_note_is_present_and_honest() -> None:
    data = _load_committed_data()
    note = data["elevated_transient_note"]
    assert "transient" in note.lower() or "transition" in note.lower()
    assert "4" in note  # the real ELEVATED count this note explains


def test_rendered_html_shows_the_real_posture_counts_and_zero_violations() -> None:
    raw = _load_committed_data()
    data = PostureValidationData(
        total_decisions=raw["total_decisions"],
        posture_counts=raw["posture_counts"],
        floor_violations=raw["floor_violations"],
        floor_by_class=tuple(FloorsByClass(posture=f["posture"], sensitivity_class=f["sensitivity_class"], floors=tuple(f["floors"])) for f in raw["floor_by_class"]),
        action_counts=raw["action_counts"],
        sources=tuple(SourceInfo(**s) for s in raw["sources"]),
        elevated_transient_note=raw["elevated_transient_note"],
        examples=tuple(ThreatScoreExample(**e) for e in raw["examples"]),
    )
    text = render_posture_validation_html(data)

    assert ">66<" in text
    assert ">4<" in text
    assert ">30<" in text
    assert "0 -- no violations found" in text
    assert "0.0689" in text
    assert "0.1376" in text
    assert "transient" in text.lower() or "transition" in text.lower()
    # Honesty: the panel must explicitly say the per-posture action
    # breakdown isn't available, never fabricate one.
    assert "was not computed" in text


def test_rendered_html_does_not_fabricate_a_per_posture_action_breakdown() -> None:
    # The per-posture action distribution was never computed by the
    # original validation session -- this panel must not invent one.
    raw = _load_committed_data()
    assert "posture_action_counts" not in raw
    assert "action_by_posture" not in raw


def test_committed_html_sample_exists_and_references_real_numbers() -> None:
    assert _HTML_PATH.is_file()
    text = _HTML_PATH.read_text(encoding="utf-8")
    assert "66" in text and "30" in text
    assert "no violations found" in text


def test_panel_is_registered_in_the_dashboard() -> None:
    from dashboard.app import PANELS, resolve_route

    titles = [p.title for p in PANELS]
    assert "Posture Validation" in titles

    status, _content_type, body = resolve_route("/samples/posture_validation.html")
    assert status == 200
    assert b"66" in body
