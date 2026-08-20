"""Behavioral tests for `dashboard.data` and `dashboard.app` (PLAN2 §7,
Dashboard v2's seven panels).

The property that matters most is provenance: `mock.html` is layout
truth and nothing else, and PLAN2's header states that every number in
it was hand-authored. So these tests check that the panels carry real
values from real runs, and that an ungenerated artefact produces an
explicit "not yet run" rather than a plausible-looking placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from agents.baselines import AlwaysPQCPolicy
from dashboard import data as ddata
from dashboard.app import _PANEL_TITLES, build_app, export_html, render_html
from dashboard.data import (
    build_dashboard_payload,
    panel_explain_decision,
    panel_living_system,
    panel_migration_wave,
    panel_results,
    panel_steering_attack,
    record_replay,
)
from env.contracts import Action


def load_test_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


@pytest.fixture(scope="module")
def replay():
    return record_replay(load_test_config(), "S1", AlwaysPQCPolicy(), seed=0, steps=200)


@pytest.fixture(scope="module")
def payload():
    return build_dashboard_payload(
        load_test_config(), replay_steps=200, include_slow_panels=False
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_replay_records_real_decisions_from_the_real_environment(replay):
    assert len(replay.decisions) > 50
    for decision in replay.decisions:
        assert decision["delivered_tier"] in {a.name for a in Action}
        # Hard Rule 2, visible in the panel's own data
        assert int(Action[decision["delivered_tier"]]) >= int(Action[decision["policy_floor"]])


def test_replay_graph_is_the_environments_own_tenant_graph(replay):
    assert len(replay.graph["nodes"]) == load_test_config()["tenant_graph"]["n_nodes"]
    assert len(replay.graph["edges"]) > 0
    for edge in replay.graph["edges"]:
        assert edge["traffic_rate"] > 0


def test_pool_and_threat_series_track_the_live_simulator(replay):
    assert len(replay.pool_series) == len(replay.decisions)
    assert all(0.0 <= p["fill_fraction"] <= 1.0 for p in replay.pool_series)
    assert all(0.0 <= t["threat_score"] <= 1.0 for t in replay.threat_series)


def test_missing_artifacts_render_as_not_yet_run_never_as_a_placeholder(tmp_path, monkeypatch):
    """The single most damaging thing this dashboard could do is show a
    plausible number that no experiment produced."""
    monkeypatch.setattr(ddata, "RESULTS_DIR", tmp_path)
    for panel in (panel_steering_attack(), panel_results(load_test_config())):
        assert panel["available"] is False
        assert "run `python -m" in panel["reason"]


def test_a_corrupt_artifact_is_treated_as_missing_rather_than_half_rendered(tmp_path, monkeypatch):
    monkeypatch.setattr(ddata, "RESULTS_DIR", tmp_path)
    (tmp_path / "closing_table.json").write_text("{not json")
    assert panel_results(load_test_config())["available"] is False


# ---------------------------------------------------------------------------
# The committed artefacts, and the panels they populate
#
# These are the other half of the placeholder contract: the two tests above
# assert that a MISSING artefact never becomes a plausible-looking number,
# and these assert that a PRESENT one is actually rendered. Both experiment
# runs completed 2026-08-20 and their outputs are committed, so a failure
# here means an artefact was deleted or a regenerating run did not finish --
# which is exactly what the message should say.
# ---------------------------------------------------------------------------


def test_the_experiment_artifacts_are_committed():
    for name in ("steering_dose_response.json", "closing_table.json"):
        path = ddata.RESULTS_DIR / name
        assert path.exists(), (
            f"{path} is missing -- regenerate it "
            "(`python -m attack.run_attack` / `python -m experiments.results_table`) "
            "and commit it; the dashboard renders 'not yet run' without it"
        )


def test_panel_seven_renders_the_real_closing_table():
    panel = panel_results(load_test_config())
    assert panel["available"] is True
    assert panel["scenarios"] == ["S1", "S2", "S3", "S4", "S6"]
    assert "masked DQN" in panel["policies"]
    # one cell per (scenario, policy)
    assert len(panel["cells"]) == len(panel["scenarios"]) * len(panel["policies"])


def test_panel_seven_reports_zero_floor_violations_in_every_cell():
    """The one column the architecture actually promises. If this ever
    reads non-zero, the "structural" label in the panel and in
    docs/report.md §5.4 is a lie and the masking rules regressed."""
    panel = panel_results(load_test_config())
    violations = {
        key: cell["floor_violations"]["mean"] for key, cell in panel["cells"].items()
    }
    assert set(violations.values()) == {0.0}, {
        k: v for k, v in violations.items() if v != 0.0
    }


def test_panel_five_renders_the_real_dose_response():
    panel = panel_steering_attack()
    assert panel["available"] is True
    masked = next(runs for name, runs in panel["policies"].items() if "masked" in name)
    # the headline: flat at zero at every dose, structurally
    assert {run["below_class_floor_share"] for run in masked} == {0.0}


def test_the_shipped_dashboard_export_contains_no_placeholder():
    """The strongest form of the placeholder contract: check the file that
    actually ships. `dashboard/index.html` is regenerated by
    `python -m dashboard.app` from a payload with every panel enabled, so
    "Not yet run" appearing in it means a panel silently lost its data.

    Asserted against the committed export rather than a freshly built
    payload because building one with the slow panels enabled runs two full
    scenario episodes -- and because the committed file is what a reader
    opens."""
    export = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    assert export.exists(), "run `python -m dashboard.app` to regenerate dashboard/index.html"
    markup = export.read_text()
    assert "Not yet run" not in markup
    for key in _PANEL_TITLES:
        assert f'id="{key}"' in markup
    # and it carries the real structural guarantee, all 25 cells
    assert markup.count("0 &mdash; structural") == 25


# ---------------------------------------------------------------------------
# Panel contents
# ---------------------------------------------------------------------------


def test_panel_one_labels_the_pipeline_for_what_was_actually_built(payload):
    """The mockup depicts autoencoder + XGBoost + fusion. The panel must
    name the real mechanisms and state the divergence, not adopt the
    mockup's labels for a different model."""
    panel = payload["p1_threat_input"]
    labels = [stage["label"] for stage in panel["pipeline_stages"]]
    assert "LSTM threat head" in labels
    assert all("XGBoost" not in label for label in labels)
    assert any(stage["mockup_label"] == "XGBoost classifier" for stage in panel["pipeline_stages"])
    assert "mockup depicts" in panel["divergence_note"]


def test_panel_one_states_the_two_invariants_plan2_requires(payload):
    invariants = " ".join(payload["p1_threat_input"]["invariants"])
    assert "raise direction" in invariants
    assert "Frozen during DQN training" in invariants


def test_panel_one_never_claims_an_unbuilt_source_is_available(payload):
    modes = {m["id"]: m for m in payload["p1_threat_input"]["source_modes"]}
    assert modes["pcap_replay"]["implemented"] is False
    assert modes["pcap_replay"]["active"] is False


def test_panel_three_floor_table_is_the_real_masking_table(payload):
    from env.masking import _PLACEHOLDER_FLOOR_TABLE
    from env.contracts import SensitivityClass, ThreatPosture

    grid = {row["sensitivity_class"]: row["floors"] for row in payload["p3_explain_decision"]["floor_table"]}
    for (sensitivity, posture), floor in _PLACEHOLDER_FLOOR_TABLE.items():
        assert grid[sensitivity.name][posture.name] == floor.name


def test_panel_three_trace_has_the_six_steps(payload):
    trace = payload["p3_explain_decision"]["trace"]
    assert [step["index"] for step in trace["steps"]] == [1, 2, 3, 4, 5, 6]


def test_panel_three_prefers_a_decision_where_a_tradeoff_actually_existed(replay):
    panel = panel_explain_decision(replay)
    if any(d["trace"]["cost_tradeoff_existed"] for d in replay.decisions):
        assert panel["trace"]["cost_tradeoff_existed"] is True


def test_panel_six_states_both_hard_rules_it_depends_on():
    panel = panel_migration_wave(load_test_config(), seed=0, steps=600)
    assert panel["held_out"] is True
    assert "Hard Rule 8" in panel["held_out_note"]
    assert "Hard Rule 3" in panel["exogenous_note"]
    assert len(panel["phases"]) == 3


def test_recent_decisions_carry_a_reason_templated_from_the_trace(replay):
    panel = panel_living_system(replay)
    for decision in panel["recent_decisions"]:
        assert decision["reason"]
        assert decision["policy_floor"] in decision["reason"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_static_html_contains_every_panel(payload):
    markup = render_html(payload)
    for key in _PANEL_TITLES:
        assert f'id="{key}"' in markup


def test_static_html_is_self_contained(payload):
    """No external stylesheet, script or image -- the export has to be
    openable from a filesystem with no network."""
    markup = render_html(payload)
    assert "<style>" in markup
    assert "http://" not in markup and "https://" not in markup
    assert "<script" not in markup


def test_static_html_escapes_values_from_the_run(payload):
    hostile = json.loads(json.dumps(payload))
    hostile["generated_from"] = "<script>alert(1)</script>"
    markup = render_html(hostile)
    assert "<script>alert(1)</script>" not in markup
    assert "&lt;script&gt;" in markup


def test_export_writes_a_file(tmp_path, payload):
    path = export_html(tmp_path / "index.html", payload=payload)
    assert path.exists()
    assert path.read_text().startswith("<!doctype html>")


def test_build_app_returns_an_app_without_starting_a_server(payload):
    app = build_app(payload)
    assert app.layout is not None
    assert type(app).__name__ == "Dash"
