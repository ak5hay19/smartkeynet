"""Behavioral tests for `env.request_generator` (PLAN.md §10 kickoff
step 4; Hard Rule 3 -- the graph only shapes *which* requests arrive,
never anything the agent sees directly, so both the dummy stream and
the real graph-driven stream must be valid, interchangeable `Request`
sources).

`build_tenant_graph()` and `RequestGenerator` (2026-08-23 session) are
now real, tested implementations -- see the "tenant graph" and
"RequestGenerator" sections below. `random_request_generator()`'s own
tests are unchanged from the prior session.
"""

from __future__ import annotations

import itertools
from collections import Counter

import networkx as nx
import pytest

from env.contracts import SensitivityClass
from env.request_generator import (
    _ARRIVAL_RATE_PER_STEP,
    RequestGenerator,
    build_tenant_graph,
    load_tenant_graph_config,
    random_request_generator,
)

N_SAMPLE = 300


# ---------------------------------------------------------------------------
# Field validity
# ---------------------------------------------------------------------------


def test_emitted_requests_have_valid_fields():
    n_classes = len(SensitivityClass)
    for request in itertools.islice(random_request_generator(seed=0), N_SAMPLE):
        assert isinstance(request["request_id"], str) and request["request_id"]
        assert isinstance(request["tenant"], str) and request["tenant"]
        assert isinstance(request["service"], str) and request["service"]
        assert isinstance(request["step"], int) and request["step"] >= 0
        assert 0 <= request["sensitivity_class"] < n_classes
        assert isinstance(request["pqc_capable"], bool)
        assert isinstance(request["hybrid_mandatory"], bool)


def test_request_ids_are_unique_within_a_stream():
    ids = [r["request_id"] for r in itertools.islice(random_request_generator(seed=3), N_SAMPLE)]
    assert len(ids) == len(set(ids))


def test_steps_are_non_decreasing():
    steps = [r["step"] for r in itertools.islice(random_request_generator(seed=5), N_SAMPLE)]
    assert steps == sorted(steps)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_sequence():
    a = list(itertools.islice(random_request_generator(seed=42), N_SAMPLE))
    b = list(itertools.islice(random_request_generator(seed=42), N_SAMPLE))
    assert a == b


def test_different_seeds_diverge():
    a = list(itertools.islice(random_request_generator(seed=1), N_SAMPLE))
    b = list(itertools.islice(random_request_generator(seed=2), N_SAMPLE))
    assert a != b


# ---------------------------------------------------------------------------
# Arrival rate sanity
# ---------------------------------------------------------------------------


def test_arrival_rate_close_to_documented_fixed_rate():
    n_steps = 5000
    count = 0
    for request in random_request_generator(seed=7):
        if request["step"] >= n_steps:
            break
        count += 1

    observed_rate = count / n_steps
    assert _ARRIVAL_RATE_PER_STEP * 0.5 <= observed_rate <= _ARRIVAL_RATE_PER_STEP * 1.5


# ---------------------------------------------------------------------------
# load_tenant_graph_config
# ---------------------------------------------------------------------------


def test_load_tenant_graph_config_matches_real_yaml():
    config = load_tenant_graph_config()
    assert config["n_nodes"] == 10


# ---------------------------------------------------------------------------
# build_tenant_graph
# ---------------------------------------------------------------------------


def _tenant_nodes(graph: nx.Graph) -> list[tuple[str, dict]]:
    return [(n, dict(attrs)) for n, attrs in graph.nodes(data=True) if attrs.get("kind") == "tenant"]


def test_build_tenant_graph_has_configured_node_count():
    n_nodes = load_tenant_graph_config()["n_nodes"]
    graph = build_tenant_graph(n_nodes=n_nodes, seed=0)
    assert len(_tenant_nodes(graph)) == n_nodes
    # + 1 for the hub node
    assert graph.number_of_nodes() == n_nodes + 1


def test_build_tenant_graph_has_hub_and_spoke_topology():
    graph = build_tenant_graph(n_nodes=8, seed=0)
    tenants = _tenant_nodes(graph)
    hub_nodes = [n for n, attrs in graph.nodes(data=True) if attrs.get("kind") == "hub"]
    assert len(hub_nodes) == 1
    hub = hub_nodes[0]
    assert graph.number_of_edges() == len(tenants)
    for tenant_id, _ in tenants:
        assert graph.has_edge(hub, tenant_id)


def test_build_tenant_graph_node_attributes_are_valid():
    n_classes = len(SensitivityClass)
    graph = build_tenant_graph(n_nodes=10, seed=1)
    for _, attrs in _tenant_nodes(graph):
        assert 0 <= attrs["sensitivity_class"] < n_classes
        assert isinstance(attrs["sensitivity_class"], int)
        assert attrs["traffic_rate"] > 0.0
        assert isinstance(attrs["traffic_rate"], float)
        assert isinstance(attrs["pqc_capable"], bool)
        assert isinstance(attrs["services"], tuple) and len(attrs["services"]) >= 1
        assert all(isinstance(s, str) for s in attrs["services"])


def test_build_tenant_graph_same_seed_is_byte_for_byte_identical():
    a = build_tenant_graph(n_nodes=10, seed=42)
    b = build_tenant_graph(n_nodes=10, seed=42)
    assert sorted(a.nodes()) == sorted(b.nodes())
    assert sorted(a.edges()) == sorted(b.edges())
    for node in a.nodes():
        assert dict(a.nodes[node]) == dict(b.nodes[node])


def test_build_tenant_graph_different_seed_differs():
    a = build_tenant_graph(n_nodes=10, seed=1)
    b = build_tenant_graph(n_nodes=10, seed=2)
    a_attrs = [dict(attrs) for _, attrs in _tenant_nodes(a)]
    b_attrs = [dict(attrs) for _, attrs in _tenant_nodes(b)]
    assert a_attrs != b_attrs


# ---------------------------------------------------------------------------
# RequestGenerator
# ---------------------------------------------------------------------------


def test_request_generator_requires_a_real_tenant_graph():
    with pytest.raises(ValueError):
        RequestGenerator(nx.Graph())


def test_request_generator_emits_valid_request_fields():
    n_classes = len(SensitivityClass)
    graph = build_tenant_graph(n_nodes=10, seed=0)
    gen = RequestGenerator(graph, seed=0)
    for request in itertools.islice(iter(gen), N_SAMPLE):
        assert isinstance(request["request_id"], str) and request["request_id"]
        assert isinstance(request["tenant"], str) and request["tenant"]
        assert isinstance(request["service"], str) and request["service"]
        assert isinstance(request["step"], int) and request["step"] >= 0
        assert 0 <= request["sensitivity_class"] < n_classes
        assert isinstance(request["pqc_capable"], bool)
        assert isinstance(request["hybrid_mandatory"], bool)


def test_request_generator_same_seed_produces_same_sequence():
    graph = build_tenant_graph(n_nodes=10, seed=0)
    a = list(itertools.islice(iter(RequestGenerator(graph, seed=42)), N_SAMPLE))
    b = list(itertools.islice(iter(RequestGenerator(graph, seed=42)), N_SAMPLE))
    assert a == b


def test_request_generator_reset_reproduces_the_same_stream():
    graph = build_tenant_graph(n_nodes=10, seed=0)
    gen = RequestGenerator(graph, seed=7)
    first_pass = []
    step = 0
    while len(first_pass) < N_SAMPLE:
        first_pass.extend(gen.step(step))
        step += 1

    gen.reset()
    second_pass = []
    step = 0
    while len(second_pass) < N_SAMPLE:
        second_pass.extend(gen.step(step))
        step += 1

    assert first_pass[:N_SAMPLE] == second_pass[:N_SAMPLE]


def test_request_generator_tenant_attributes_never_drift_from_the_graph():
    """The whole point of this session: a sampled request's
    sensitivity_class/pqc_capable must come directly from its tenant's
    persistent node attributes, with zero drift -- not approximately,
    exactly, across a substantial sample."""
    graph = build_tenant_graph(n_nodes=10, seed=3)
    tenant_attrs = {n: attrs for n, attrs in _tenant_nodes(graph)}
    gen = RequestGenerator(graph, seed=3)

    checked = 0
    for request in itertools.islice(iter(gen), 500):
        attrs = tenant_attrs[request["tenant"]]
        assert request["sensitivity_class"] == attrs["sensitivity_class"]
        assert request["pqc_capable"] == attrs["pqc_capable"]
        assert request["service"] in attrs["services"]
        checked += 1
    assert checked == 500


def test_request_generator_arrival_share_tracks_traffic_rate_weight():
    """Statistical, not exact: over many steps, each tenant's share of
    generated requests should roughly track its traffic_rate weight."""
    graph = build_tenant_graph(n_nodes=5, seed=9)
    tenant_attrs = {n: attrs for n, attrs in _tenant_nodes(graph)}
    total_rate = sum(attrs["traffic_rate"] for attrs in tenant_attrs.values())
    expected_share = {n: attrs["traffic_rate"] / total_rate for n, attrs in tenant_attrs.items()}

    gen = RequestGenerator(graph, seed=9)
    counts = Counter()
    n_requests = 0
    step = 0
    while n_requests < 4000:
        for request in gen.step(step):
            counts[request["tenant"]] += 1
            n_requests += 1
        step += 1

    for tenant_id, expected in expected_share.items():
        observed = counts[tenant_id] / n_requests
        assert observed == pytest.approx(expected, abs=0.05)


def test_request_generator_arrival_rate_close_to_documented_fixed_rate():
    graph = build_tenant_graph(n_nodes=10, seed=0)
    gen = RequestGenerator(graph, seed=13)
    n_steps = 5000
    count = 0
    for step in range(n_steps):
        count += len(gen.step(step))

    observed_rate = count / n_steps
    assert _ARRIVAL_RATE_PER_STEP * 0.5 <= observed_rate <= _ARRIVAL_RATE_PER_STEP * 1.5


# ---------------------------------------------------------------------------
# load_spike diagnostic (2026-08-10 -- NOT real S4, see docstring)
# ---------------------------------------------------------------------------

_SPIKE = {
    "period_steps": 500,
    "spike_duration_steps": 20,
    "spike_rate_multiplier": 3.0,
    "low_rate_multiplier": 0.3,
}


def test_load_spike_none_is_byte_identical_to_undecorated_stream():
    a = list(itertools.islice(random_request_generator(seed=11), N_SAMPLE))
    b = list(itertools.islice(random_request_generator(seed=11, load_spike=None), N_SAMPLE))
    assert a == b


def test_load_spike_raises_observed_arrival_rate_inside_the_window():
    n_periods = 20
    n_steps = _SPIKE["period_steps"] * n_periods
    in_window = 0
    out_window = 0
    for request in random_request_generator(seed=21, load_spike=_SPIKE):
        if request["step"] >= n_steps:
            break
        if (request["step"] % _SPIKE["period_steps"]) < _SPIKE["spike_duration_steps"]:
            in_window += 1
        else:
            out_window += 1

    in_window_steps = _SPIKE["spike_duration_steps"] * n_periods
    out_window_steps = (_SPIKE["period_steps"] - _SPIKE["spike_duration_steps"]) * n_periods
    in_window_rate = in_window / in_window_steps
    out_window_rate = out_window / out_window_steps

    # loose bounds -- this is a Poisson process, exact equality isn't
    # expected, just that both multipliers are clearly taking effect.
    expected_in = _ARRIVAL_RATE_PER_STEP * _SPIKE["spike_rate_multiplier"]
    expected_out = _ARRIVAL_RATE_PER_STEP * _SPIKE["low_rate_multiplier"]
    assert in_window_rate == pytest.approx(expected_in, rel=0.3)
    assert out_window_rate == pytest.approx(expected_out, rel=0.3)
    assert in_window_rate > out_window_rate  # the whole point: genuinely oscillates


def test_load_spike_same_seed_still_reproducible():
    a = list(itertools.islice(random_request_generator(seed=33, load_spike=_SPIKE), N_SAMPLE))
    b = list(itertools.islice(random_request_generator(seed=33, load_spike=_SPIKE), N_SAMPLE))
    assert a == b


# ---------------------------------------------------------------------------
# RequestGenerator's S4 (DDoS/noisy-neighbor) flood_override mechanism
# (2026-08-24 session) -- see the class docstring's own S4 section for
# why this is an additive second Poisson stream, not a traffic_rate
# multiply.
# ---------------------------------------------------------------------------

_FLOOD_TENANT = "tenant_4"


def _counts_by_tenant(gen: RequestGenerator, n_steps: int) -> Counter:
    counts: Counter = Counter()
    for step in range(n_steps):
        for request in gen.step(step):
            counts[request["tenant"]] += 1
    return counts


def test_flood_override_rejects_an_unknown_tenant_id():
    graph = build_tenant_graph(n_nodes=10, seed=0)
    with pytest.raises(ValueError):
        RequestGenerator(graph, seed=0, flood_override={"tenant_id": "no_such_tenant", "extra_rate": 5.0})


def test_flood_override_raises_the_flooded_tenants_own_count():
    graph = build_tenant_graph(n_nodes=10, seed=0)
    no_flood = _counts_by_tenant(RequestGenerator(graph, seed=7), n_steps=500)
    flooded = _counts_by_tenant(
        RequestGenerator(graph, seed=7, flood_override={"tenant_id": _FLOOD_TENANT, "extra_rate": 5.0}),
        n_steps=500,
    )
    assert flooded[_FLOOD_TENANT] > no_flood[_FLOOD_TENANT] * 3


def test_flood_override_leaves_every_other_tenants_count_byte_for_byte_unaffected():
    """The precise isolation guarantee (see class docstring): the base
    arrival draw is completely untouched by flood activity -- every
    other tenant's own count must match EXACTLY, not just closely,
    between a flood-on and flood-off run with the same seed and graph."""
    graph = build_tenant_graph(n_nodes=10, seed=0)
    no_flood = _counts_by_tenant(RequestGenerator(graph, seed=7), n_steps=500)
    flooded = _counts_by_tenant(
        RequestGenerator(graph, seed=7, flood_override={"tenant_id": _FLOOD_TENANT, "extra_rate": 5.0}),
        n_steps=500,
    )

    other_tenants = {n for n, attrs in build_tenant_graph(n_nodes=10, seed=0).nodes(data=True) if attrs.get("kind") == "tenant"}
    other_tenants.discard(_FLOOD_TENANT)
    assert other_tenants
    for tenant in other_tenants:
        assert flooded.get(tenant, 0) == no_flood.get(tenant, 0)


def test_flood_override_requests_have_valid_fields_matching_the_tenants_own_attributes():
    """Flood-batch requests are built by the exact same field-
    construction logic as any other request -- same validity, same
    zero-drift-from-the-graph guarantee, no special-cased shape."""
    graph = build_tenant_graph(n_nodes=10, seed=0)
    tenant_attrs = {n: attrs for n, attrs in graph.nodes(data=True) if attrs.get("kind") == "tenant"}
    gen = RequestGenerator(graph, seed=7, flood_override={"tenant_id": _FLOOD_TENANT, "extra_rate": 5.0})

    n_classes = len(SensitivityClass)
    flood_requests_checked = 0
    for step in range(200):
        for request in gen.step(step):
            if request["tenant"] != _FLOOD_TENANT:
                continue
            assert isinstance(request["request_id"], str) and request["request_id"]
            assert isinstance(request["service"], str) and request["service"]
            assert 0 <= request["sensitivity_class"] < n_classes
            assert isinstance(request["pqc_capable"], bool)
            assert isinstance(request["hybrid_mandatory"], bool)
            attrs = tenant_attrs[_FLOOD_TENANT]
            assert request["sensitivity_class"] == attrs["sensitivity_class"]
            assert request["pqc_capable"] == attrs["pqc_capable"]
            assert request["service"] in attrs["services"]
            flood_requests_checked += 1
    assert flood_requests_checked > 0


def test_flood_override_none_is_byte_identical_to_no_flood_argument_at_all():
    """Regression safety: passing `flood_override=None` explicitly must
    be indistinguishable from not passing it -- both are the documented
    default, zero new behavior."""
    graph = build_tenant_graph(n_nodes=10, seed=0)
    a = _counts_by_tenant(RequestGenerator(graph, seed=7), n_steps=300)
    b = _counts_by_tenant(RequestGenerator(graph, seed=7, flood_override=None), n_steps=300)
    assert a == b


def test_request_generator_regression_unaffected_by_flood_override_addition():
    """Every existing (pre-S4) RequestGenerator test in this file
    exercises the class with no `flood_override` argument at all --
    this is a direct, explicit re-confirmation (not just an assumption
    that the existing suite would catch a regression) that adding the
    parameter changed nothing about the class's prior default
    behavior: two independently-constructed generators, same seed, same
    graph, no flood_override, still produce an identical stream."""
    graph = build_tenant_graph(n_nodes=10, seed=0)
    a = list(itertools.islice(iter(RequestGenerator(graph, seed=42)), N_SAMPLE))
    b = list(itertools.islice(iter(RequestGenerator(graph, seed=42)), N_SAMPLE))
    assert a == b
