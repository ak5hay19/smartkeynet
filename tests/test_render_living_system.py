"""Behavioral tests for `dashboard.render_living_system` (Hard Rule 10 by
analogy with every prior rendered panel -- the rendered graph must show
only real tenant identities/tiers actually present in the input, never
the mockup's fabricated hospital/fintech/logging/iot-telemetry names,
and the rendered node count must reflect the real `build_tenant_graph`
topology, never a fixed/hardcoded count).

Tenant graphs used here are built via the real, real
`env.request_generator.build_tenant_graph` (never a hand-authored
`nx.Graph`), mirroring `tests/test_request_generator.py`'s own
established use of that function -- this is what lets the "real node
count" tests below vary `n_nodes` and assert the render tracks it.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from dashboard.render_living_system import (
    LivingSystemSnapshot,
    RecentDecisionView,
    TenantNodeView,
    build_snapshot,
    render_living_system_html,
    write_living_system_html,
)
from env.contracts import Action, KeyType, SensitivityClass, ThreatPosture
from env.request_generator import build_tenant_graph, _HUB_NODE_ID


class _BalancedTagChecker(HTMLParser):
    _VOID_ELEMENTS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self._VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack, f"</{tag}> with no matching open tag"
        assert self.stack[-1] == tag, f"expected </{self.stack[-1]}>, got </{tag}>"
        self.stack.pop()


def _assert_well_formed(html_text: str) -> None:
    checker = _BalancedTagChecker()
    checker.feed(html_text)
    assert checker.stack == [], f"unclosed tags: {checker.stack}"


def _tenant_attrs_from_graph(n_nodes: int, seed: int) -> tuple[dict, str]:
    graph = build_tenant_graph(n_nodes=n_nodes, seed=seed)
    tenant_attrs = {
        node: dict(attrs) for node, attrs in graph.nodes(data=True) if attrs.get("kind") == "tenant"
    }
    hub_id = next(node for node, attrs in graph.nodes(data=True) if attrs.get("kind") == "hub")
    return tenant_attrs, hub_id


def _decision(step: int, tenant: str, action: Action, served_tier: KeyType, *, service="auth", sens=SensitivityClass.S1) -> RecentDecisionView:
    return RecentDecisionView(
        step=step, tenant=tenant, service=service, sensitivity_class=sens, action=action, served_tier=served_tier
    )


# ---------------------------------------------------------------------------
# build_snapshot: real fold-over-decisions correctness
# ---------------------------------------------------------------------------


def test_build_snapshot_last_served_tier_is_the_most_recent_real_decision():
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=3, seed=1)
    tenant_ids = list(tenant_attrs.keys())
    decisions = [
        _decision(0, tenant_ids[0], Action.SERVE_CLASSICAL, KeyType.CLASSICAL),
        _decision(1, tenant_ids[0], Action.SERVE_PQC, KeyType.PQC),  # supersedes the CLASSICAL decision above
        _decision(2, tenant_ids[1], Action.SERVE_HYBRID, KeyType.HYBRID),
    ]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=2, pool_fill=0.5, posture=ThreatPosture.CALM,
    )
    tiers_by_id = {t.tenant_id: t.last_served_tier for t in snapshot.tenants}
    assert tiers_by_id[tenant_ids[0]] == KeyType.PQC  # most recent, not the first
    assert tiers_by_id[tenant_ids[1]] == KeyType.HYBRID
    assert tiers_by_id[tenant_ids[2]] is None  # never touched -- honest "no traffic yet"


def test_build_snapshot_ordinal_cutoff_ignores_later_decisions():
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=2, seed=2)
    tenant_ids = list(tenant_attrs.keys())
    decisions = [
        _decision(0, tenant_ids[0], Action.SERVE_CLASSICAL, KeyType.CLASSICAL),
        _decision(0, tenant_ids[1], Action.SERVE_HYBRID, KeyType.HYBRID),  # same step, later ordinal
    ]
    # snapshot_index=0 must NOT see the second (later-ordinal) decision, even though it shares step=0
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=0, pool_fill=0.1, posture=ThreatPosture.CALM,
    )
    tiers_by_id = {t.tenant_id: t.last_served_tier for t in snapshot.tenants}
    assert tiers_by_id[tenant_ids[0]] == KeyType.CLASSICAL
    assert tiers_by_id[tenant_ids[1]] is None


def test_build_snapshot_recent_decisions_window_and_order():
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=1, seed=3)
    tenant_id = next(iter(tenant_attrs))
    decisions = [
        _decision(i, tenant_id, Action.SERVE_PQC, KeyType.PQC) for i in range(20)
    ]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=19, pool_fill=0.5, posture=ThreatPosture.CALM, recent_window=5,
    )
    assert [d.step for d in snapshot.recent_decisions] == [19, 18, 17, 16, 15]  # newest first, windowed
    assert snapshot.step == 19


def test_build_snapshot_rejects_empty_cutoff():
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=1, seed=4)
    with pytest.raises(ValueError):
        build_snapshot(
            label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=[],
            snapshot_index=0, pool_fill=0.0, posture=ThreatPosture.CALM,
        )


# ---------------------------------------------------------------------------
# Rendering: real node count / real topology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [3, 7, 10])
def test_rendered_node_count_matches_real_graph_n_nodes(n_nodes):
    """The rendered graph's tenant-node count must equal the REAL
    build_tenant_graph node count for whatever n_nodes was requested --
    never a fixed/hardcoded number."""
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=n_nodes, seed=5)
    tenant_ids = list(tenant_attrs.keys())
    decisions = [_decision(0, tenant_ids[0], Action.SERVE_PQC, KeyType.PQC)]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=0, pool_fill=0.5, posture=ThreatPosture.CALM,
    )
    out = render_living_system_html(snapshot)
    # every real tenant node has exactly one data-tenant circle + one data-tenant edge line
    rendered_tenant_ids = set(re.findall(r'data-tenant="([^"]+)"', out))
    assert rendered_tenant_ids == set(tenant_ids)
    assert len(tenant_ids) == n_nodes  # sanity: build_tenant_graph honored n_nodes
    assert out.count("<circle") == n_nodes + 1  # + 1 hub node
    assert out.count("<line") == n_nodes  # one spoke per tenant, real hub-and-spoke topology


def test_rendered_hub_id_is_the_real_hub_node():
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=4, seed=6)
    assert hub_id == _HUB_NODE_ID
    tenant_ids = list(tenant_attrs.keys())
    decisions = [_decision(0, tenant_ids[0], Action.SERVE_PQC, KeyType.PQC)]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=0, pool_fill=0.5, posture=ThreatPosture.CALM,
    )
    out = render_living_system_html(snapshot)
    assert f'>{hub_id}</text>' in out


# ---------------------------------------------------------------------------
# Rendering: tier -> color correctness (direct check across all three tiers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier,expected_color",
    [
        (KeyType.CLASSICAL, "#8B95A5"),
        (KeyType.PQC, "#E8A33D"),
        (KeyType.HYBRID, "#33D687"),
    ],
)
def test_tier_color_mapping_is_exact(tier, expected_color):
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=1, seed=7)
    tenant_id = next(iter(tenant_attrs))
    action_by_tier = {
        KeyType.CLASSICAL: Action.SERVE_CLASSICAL,
        KeyType.PQC: Action.SERVE_PQC,
        KeyType.HYBRID: Action.SERVE_HYBRID,
    }
    decisions = [_decision(0, tenant_id, action_by_tier[tier], tier)]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=0, pool_fill=0.5, posture=ThreatPosture.CALM,
    )
    out = render_living_system_html(snapshot)
    assert f'data-tenant="{tenant_id}" data-tier="{tier.name}"' in out
    # the node circle for this tenant carries exactly the real tier's mapped color
    node_pattern = re.compile(
        rf'<circle[^>]*stroke="([^"]+)"[^>]*data-tenant="{re.escape(tenant_id)}" data-tier="{tier.name}"'
    )
    match = node_pattern.search(out)
    assert match is not None
    assert match.group(1) == expected_color
    # the edge to this tenant is colored identically
    edge_pattern = re.compile(rf'<line[^>]*stroke="([^"]+)"[^>]*data-tenant="{re.escape(tenant_id)}"')
    edge_match = edge_pattern.search(out)
    assert edge_match is not None
    assert edge_match.group(1) == expected_color


def test_no_traffic_yet_tenant_renders_the_no_traffic_color_not_a_fabricated_tier():
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=2, seed=8)
    tenant_ids = list(tenant_attrs.keys())
    untouched_tenant = tenant_ids[1]
    decisions = [_decision(0, tenant_ids[0], Action.SERVE_HYBRID, KeyType.HYBRID)]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=0, pool_fill=0.5, posture=ThreatPosture.CALM,
    )
    out = render_living_system_html(snapshot)
    assert f'data-tenant="{untouched_tenant}" data-tier="NONE"' in out
    node_pattern = re.compile(
        rf'<circle[^>]*stroke="([^"]+)"[^>]*data-tenant="{re.escape(untouched_tenant)}" data-tier="NONE"'
    )
    match = node_pattern.search(out)
    assert match is not None
    assert match.group(1) == "#4C5A6B"  # the documented "no traffic yet" color, not a tier color


# ---------------------------------------------------------------------------
# No-fabrication: rendered tenant ids / tiers are exactly the real input set
# ---------------------------------------------------------------------------


def test_no_fabrication_every_rendered_tenant_and_tier_is_in_the_real_input():
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=6, seed=9)
    tenant_ids = list(tenant_attrs.keys())
    decisions = [
        _decision(0, tenant_ids[0], Action.SERVE_CLASSICAL, KeyType.CLASSICAL),
        _decision(1, tenant_ids[1], Action.SERVE_PQC, KeyType.PQC),
        _decision(2, tenant_ids[2], Action.SERVE_HYBRID, KeyType.HYBRID),
    ]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=2, pool_fill=0.42, posture=ThreatPosture.ELEVATED,
    )
    out = render_living_system_html(snapshot)

    real_tenant_ids = set(tenant_ids) | {hub_id}
    rendered_node_labels = set(re.findall(r'class="node-label(?: hub)?">([^<]+)</text>', out))
    assert rendered_node_labels <= real_tenant_ids

    real_tiers = {t.name for t in KeyType} | {"NONE"}
    rendered_tiers = set(re.findall(r'data-tier="([^"]+)"', out))
    assert rendered_tiers <= real_tiers

    # every recent-decision row's tenant/service/action/tier is a real decision field, verbatim
    for decision in decisions:
        assert f'<span class="req-tenant">{decision.tenant}</span>' in out
        assert f'<span class="req-action">{decision.action.name}</span>' in out
        assert f'<span class="req-tier">{decision.served_tier.name}</span>' in out

    # never the mockup's fabricated example tenant names
    for fake_name in ("hospital", "fintech", "logging", "iot-telemetry"):
        assert fake_name not in out


def test_real_pool_fill_and_posture_are_shown_verbatim_not_derived():
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=1, seed=10)
    tenant_id = next(iter(tenant_attrs))
    decisions = [_decision(0, tenant_id, Action.SERVE_HYBRID, KeyType.HYBRID)]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=0, pool_fill=0.6789, posture=ThreatPosture.HIGH,
    )
    out = render_living_system_html(snapshot)
    assert "68%" in out  # pool_fill * 100, rounded for display -- 0.6789 -> 67.89 -> "68%"
    assert '<span class="badge high">high</span>' in out


# ---------------------------------------------------------------------------
# HTML/SVG well-formedness + round trip
# ---------------------------------------------------------------------------


def test_rendered_html_is_well_formed():
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=5, seed=11)
    tenant_ids = list(tenant_attrs.keys())
    decisions = [_decision(i, tenant_ids[i % len(tenant_ids)], Action.SERVE_PQC, KeyType.PQC) for i in range(5)]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=4, pool_fill=0.3, posture=ThreatPosture.ELEVATED,
    )
    _assert_well_formed(render_living_system_html(snapshot))


def test_rendered_html_is_well_formed_with_no_recent_decisions_window_edge_case():
    # a single-decision episode: recent_decisions has exactly one row
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=2, seed=12)
    tenant_id = next(iter(tenant_attrs))
    decisions = [_decision(0, tenant_id, Action.REUSE, KeyType.CLASSICAL)]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=0, pool_fill=0.5, posture=ThreatPosture.CALM,
    )
    _assert_well_formed(render_living_system_html(snapshot))


def test_write_living_system_html_round_trip(tmp_path):
    tenant_attrs, hub_id = _tenant_attrs_from_graph(n_nodes=3, seed=13)
    tenant_id = next(iter(tenant_attrs))
    decisions = [_decision(0, tenant_id, Action.SERVE_HYBRID, KeyType.HYBRID)]
    snapshot = build_snapshot(
        label="test", hub_id=hub_id, tenant_attrs=tenant_attrs, all_decisions=decisions,
        snapshot_index=0, pool_fill=0.5, posture=ThreatPosture.CALM,
    )
    out_path = tmp_path / "nested" / "living_system.html"
    written = write_living_system_html(snapshot, out_path)
    assert written == out_path
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == render_living_system_html(snapshot)
