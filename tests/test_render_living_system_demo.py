"""Tests for `dashboard.render_living_system_demo` -- specifically the
S1 (steady/calm traffic) addition alongside the pre-existing S2 run.

`pick_snapshot_indices`'s `prefix` param is tested as a pure function
(fast). `collect_real_episode` against S1 is exercised with a real, but
deliberately short, episode (`max_steps` well below the demo's default
250) -- still the exact real `SmartKeyNetEnv` + `StaticThresholdPolicy`
+ graph-driven request stream, just a shorter real run, so these stay
fast without touching any protected file or faking any value (Hard Rule
10 by analogy with `tests/test_render_living_system.py`).
"""

from __future__ import annotations

import re

import pytest

from dashboard.render_living_system import (
    build_snapshot,
    render_living_system_html,
)
from dashboard.render_living_system_demo import (
    _S1_CONFIG_PATH,
    _S2_CONFIG_PATH,
    _TENANT_GRAPH_SEED,
    collect_real_episode,
    pick_snapshot_indices,
)
from env.contracts import KeyType
from env.request_generator import build_tenant_graph

_MAX_STEPS = 60  # short but real -- enough for several real decisions per tenant


# ---------------------------------------------------------------------------
# pick_snapshot_indices: prefix behavior (pure function, no episode needed)
# ---------------------------------------------------------------------------


def test_default_prefix_matches_original_s2_filenames():
    # Guards the S2 driver call site: passing no prefix must still
    # produce byte-identical keys to before this session's change, so
    # the committed S2 samples are never silently renamed/reshuffled.
    class _FakeDecision:
        def __init__(self, tenant):
            self.tenant = tenant

    class _FakeEpisode:
        tenant_attrs = {"tenant_0": {}, "tenant_1": {}}
        decisions = [_FakeDecision("tenant_0"), _FakeDecision("tenant_1")]

    picks = pick_snapshot_indices(_FakeEpisode())
    assert set(picks) == {
        "living_system_01_first_decision",
        "living_system_02_graph_fully_populated",
        "living_system_03_final_decision",
    }


def test_s1_prefix_produces_distinct_non_colliding_keys():
    class _FakeDecision:
        def __init__(self, tenant):
            self.tenant = tenant

    class _FakeEpisode:
        tenant_attrs = {"tenant_0": {}, "tenant_1": {}}
        decisions = [_FakeDecision("tenant_0"), _FakeDecision("tenant_1")]

    default_picks = set(pick_snapshot_indices(_FakeEpisode()))
    s1_picks = set(pick_snapshot_indices(_FakeEpisode(), prefix="living_system_s1"))
    assert s1_picks.isdisjoint(default_picks)
    assert all(name.startswith("living_system_s1_") for name in s1_picks)


# ---------------------------------------------------------------------------
# collect_real_episode against S1: real fidelity, no fabrication
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def s1_episode():
    return collect_real_episode(seed=0, max_steps=_MAX_STEPS, scenario_config_path=_S1_CONFIG_PATH)


def test_s1_config_path_is_the_real_default_config_with_scenario_s1():
    # configs/default.yaml is the real S1 dispatch (env/environment.py:
    # self._threat_schedule_cfg is only ever populated when scenario ==
    # "S2") -- not a hand-authored/duplicated config for this session.
    assert _S1_CONFIG_PATH.name == "default.yaml"
    assert _S1_CONFIG_PATH != _S2_CONFIG_PATH


def test_s1_episode_uses_the_real_tenant_graph_node_identities(s1_episode):
    graph = build_tenant_graph(n_nodes=len(s1_episode.tenant_attrs), seed=_TENANT_GRAPH_SEED)
    real_tenant_ids = {node for node, attrs in graph.nodes(data=True) if attrs.get("kind") == "tenant"}
    assert set(s1_episode.tenant_attrs) == real_tenant_ids
    # every decision's tenant is a real graph node -- never a fabricated identity
    for decision in s1_episode.decisions:
        assert decision.tenant in real_tenant_ids


def test_s1_snapshot_tier_colors_match_the_real_served_tiers(s1_episode):
    picks = pick_snapshot_indices(s1_episode, prefix="living_system_s1")
    for name, idx in picks.items():
        snapshot = build_snapshot(
            label=name,
            hub_id=s1_episode.hub_id,
            tenant_attrs=s1_episode.tenant_attrs,
            all_decisions=s1_episode.decisions,
            snapshot_index=idx,
            pool_fill=s1_episode.pool_fill_by_index[idx],
            posture=s1_episode.posture_by_index[idx],
        )
        out = render_living_system_html(snapshot)

        # rebuild the real "most recently served tier per tenant" by hand
        # from the same real decision list, independently of build_snapshot,
        # and check the rendered node color against it directly.
        expected_tier: dict[str, KeyType] = {}
        for decision in s1_episode.decisions[: idx + 1]:
            expected_tier[decision.tenant] = decision.served_tier

        for tenant_id, tier in expected_tier.items():
            assert f'data-tenant="{tenant_id}" data-tier="{tier.name}"' in out

        # no fabricated tenant/tier: every rendered tenant id is a real
        # snapshot tenant, every rendered tier is a real KeyType or NONE
        rendered_tenant_ids = set(re.findall(r'data-tenant="([^"]+)"', out))
        real_tenant_ids = {t.tenant_id for t in snapshot.tenants} | {snapshot.hub_id}
        assert rendered_tenant_ids <= real_tenant_ids
        rendered_tiers = set(re.findall(r'data-tier="([^"]+)"', out))
        assert rendered_tiers <= {t.name for t in KeyType} | {"NONE"}


def test_s1_rendered_html_is_well_formed(s1_episode):
    picks = pick_snapshot_indices(s1_episode, prefix="living_system_s1")
    for name, idx in picks.items():
        snapshot = build_snapshot(
            label=name,
            hub_id=s1_episode.hub_id,
            tenant_attrs=s1_episode.tenant_attrs,
            all_decisions=s1_episode.decisions,
            snapshot_index=idx,
            pool_fill=s1_episode.pool_fill_by_index[idx],
            posture=s1_episode.posture_by_index[idx],
        )
        out = render_living_system_html(snapshot)
        # cheap balanced-tag check (mirrors test_render_living_system.py's
        # _BalancedTagChecker without re-importing a private test helper)
        assert out.count("<svg") == out.count("</svg>")
        assert out.count("<html") == out.count("</html>")
        assert out.count("<body") == out.count("</body>")
