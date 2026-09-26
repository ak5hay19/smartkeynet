"""Tests for the National Backbone illustrative-scenario panel: the
real config/seed-search result, tenant-identity fidelity, exact
tier-color correctness, and no-fabrication (rendered values must equal
one real episode's values)."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from dashboard.render_migration_wave import _TIER_COLOR
from dashboard.render_national_backbone import (
    NAMES_DISCLAIMER,
    AgencyView,
    render_national_backbone_html,
    validate_identity_fidelity,
)
from dashboard.render_national_backbone_demo import (
    _AGENCIES,
    collect_real_national_backbone_data,
    search_graph_seed,
)
from env.contracts import Action, SensitivityClass
from env.request_generator import build_tenant_graph
from experiments.train import load_full_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = _REPO_ROOT / "configs" / "scenarios" / "s_narrative_national_backbone.yaml"
_S6 = _REPO_ROOT / "configs" / "scenarios" / "s6_migration.yaml"


@pytest.fixture(scope="module")
def data():
    return collect_real_national_backbone_data()


@pytest.fixture(scope="module")
def page(data):
    return render_national_backbone_html(data)


# --- config / seed search -------------------------------------------------


def test_seed_search_finds_seed_30():
    assert search_graph_seed() == 30


def test_config_seed_gives_real_story_shape_without_relabeling():
    cfg = load_full_config(_CONFIG)
    graph = build_tenant_graph(n_nodes=cfg["tenant_graph"]["n_nodes"], seed=cfg["migration_graph_seed"])
    cls = {t: int(graph.nodes[t]["sensitivity_class"]) for t in ("tenant_0", "tenant_1", "tenant_2", "tenant_3")}
    assert cls == {"tenant_0": 3, "tenant_1": 3, "tenant_2": 2, "tenant_3": 1}


def test_config_differs_from_s6_only_in_seed_and_schedule():
    a, b = load_full_config(_S6), load_full_config(_CONFIG)
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differing == {"migration_graph_seed", "migration_schedule"}
    assert b["scenario"] == "S6" and b["train_eligible"] is False
    assert b["migration_schedule"] == [{"step": 100, "tenant_index": 3, "new_sensitivity_class": 3}]


# --- real episode: no fabrication ----------------------------------------


def test_reclassification_is_real_and_unconfounded(data):
    ev = data.reclass_event
    assert (ev.tenant_id, ev.old_sensitivity_class, ev.new_sensitivity_class) == ("tenant_3", SensitivityClass.S1, SensitivityClass.S3)
    assert ev.attribution == "scripted"
    assert ev.before.floor == Action.SERVE_CLASSICAL and ev.after.floor == Action.SERVE_HYBRID
    assert ev.before.posture == ev.after.posture


def test_only_reclassified_agency_changes_class_and_only_after_the_event(data):
    for a in data.agencies:
        classes = [(o.step, o.sensitivity_class) for o in a.observations]
        if a.display_name == data.reclass_agency:
            assert all(c == SensitivityClass.S1 for s, c in classes if s < 100)
            assert classes[-1][1] == SensitivityClass.S3
        else:
            assert {c for _, c in classes} == {a.initial_class}


def test_every_floor_matches_real_floor_table(data):
    from env.masking import _PLACEHOLDER_FLOOR_TABLE

    for a in data.agencies:
        for o in a.observations:
            assert o.floor == _PLACEHOLDER_FLOOR_TABLE[(o.sensitivity_class, o.posture)]


def test_no_decision_served_below_floor(data):
    assert data.n_below_floor == 0
    for a in data.agencies:
        assert all(int(o.served_tier) >= int(o.floor) for o in a.observations)


def test_pool_pressure_absence_is_disclosed(data, page):
    assert max(p.pool_fill for p in data.trajectory) - min(p.pool_fill for p in data.trajectory) < 0.01
    assert "did NOT come under pressure" in page


# --- identity fidelity -----------------------------------------------------


def test_agency_names_map_to_expected_tenants(data):
    assert [(a.display_name, a.tenant_id) for a in data.agencies] == list(_AGENCIES)
    for a in data.agencies:
        assert a.observations and all(o.tenant == a.tenant_id for o in a.observations)


def test_validate_rejects_mislabeled_observation(data):
    a0, a1 = data.agencies[0], data.agencies[1]
    bad = dataclasses.replace(a0, observations=a0.observations + a1.observations[:1])
    with pytest.raises(ValueError):
        validate_identity_fidelity(dataclasses.replace(data, agencies=(bad,) + data.agencies[1:]))


def test_validate_rejects_reclass_agency_mismatch(data):
    with pytest.raises(ValueError):
        validate_identity_fidelity(dataclasses.replace(data, reclass_agency="Tax Authority"))


# --- rendering ---------------------------------------------------------------


def test_page_shows_disclaimer_names_and_tenant_ids(data, page):
    assert NAMES_DISCLAIMER.split(";")[0] in page and 'data-role="names-disclaimer"' in page
    for name, tenant in _AGENCIES:
        assert name in page and tenant in page


def test_tier_colors_exact_for_every_dot(data, page):
    served = re.findall(r'<circle[^>]*fill="(#\w+)"[^>]*data-tenant="(\w+)" data-step="(\d+)"[^>]*data-tier="(\w+)"', page)
    floors = re.findall(r'<rect[^>]*fill="(#\w+)"[^>]*data-kind="floor" data-tenant="(\w+)" data-step="(\d+)" data-tier="(\w+)"', page)
    n = sum(len(a.observations) for a in data.agencies)
    assert len(served) == len(floors) == n
    for color, _t, _s, tier in served:
        assert color == _TIER_COLOR[Action[f"SERVE_{tier}"]]
    for color, _t, _s, tier in floors:
        assert color == _TIER_COLOR[Action[tier]]


def test_rendered_dots_equal_real_observations(data, page):
    rendered = {(t, int(s), tier) for _c, t, s, tier in re.findall(r'<circle[^>]*fill="(#\w+)"[^>]*data-tenant="(\w+)" data-step="(\d+)"[^>]*data-tier="(\w+)"', page)}
    real = {(o.tenant, o.step, o.served_tier.name) for a in data.agencies for o in a.observations}
    assert rendered == real


def test_no_red_ring_when_nothing_below_floor(page):
    assert 'stroke="#FF5C6C"' not in page


def test_red_ring_appears_if_a_below_floor_decision_is_supplied(data):
    a = data.agencies[0]
    o = dataclasses.replace(a.observations[0], floor=Action.SERVE_HYBRID, served_tier=type(a.observations[0].served_tier).CLASSICAL)
    forged = dataclasses.replace(data, agencies=(dataclasses.replace(a, observations=(o,)),) + data.agencies[1:], n_below_floor=1)
    assert 'stroke="#FF5C6C"' in render_national_backbone_html(forged)


def test_event_marker_at_scripted_step(page):
    assert 'data-event-step="100"' in page


def test_panel_registered_and_served_on_both_shells():
    from dashboard.app import PANELS, render_index_v2_html, resolve_route

    assert any(p.title.startswith("National Backbone") for p in PANELS)
    assert resolve_route("/samples/national_backbone.html")[0] == 200
    assert "national_backbone.html" in render_index_v2_html()
    assert "national_backbone.html" in resolve_route("/classic")[2].decode("utf-8")
