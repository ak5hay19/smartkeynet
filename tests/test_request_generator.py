"""Behavioral tests for `env.request_generator` (PLAN.md §10 kickoff
step 4; Hard Rule 3 -- the graph only shapes *which* requests arrive,
never anything the agent sees directly, so the dummy stream must be a
valid drop-in `Request` source).

All three request sources are covered: `random_request_generator()`
(the graph-free stub), `build_tenant_graph()`, and the graph-driven
`RequestGenerator`. The stub is deliberately kept alongside the real
sampler -- it is the Hard Rule 3 swappability check.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from env.contracts import SensitivityClass
from env.request_generator import (
    _ARRIVAL_RATE_PER_STEP,
    _TENANT_PROFILES,
    RequestGenerator,
    build_tenant_graph,
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
# build_tenant_graph (2026-08-19: implemented -- these replaced the two
# `raises(NotImplementedError)` placeholders that stood here)
# ---------------------------------------------------------------------------


def test_build_tenant_graph_has_requested_node_count_and_contract_edge_attrs():
    graph = build_tenant_graph(n_nodes=10, seed=0)

    assert graph.number_of_nodes() == 10
    assert graph.number_of_edges() > 0
    for node, data in graph.nodes(data=True):
        assert data["tenant"] in _TENANT_PROFILES
        assert isinstance(data["service"], str)
    for _u, _v, data in graph.edges(data=True):
        # exactly the three attributes env/contracts.py's Request needs
        assert data["sensitivity_class"] in {int(c) for c in SensitivityClass}
        assert data["traffic_rate"] > 0.0
        assert isinstance(data["pqc_capable"], bool)


def test_build_tenant_graph_is_connected_and_spans_every_tenant():
    """A disconnected graph would mean tenants that never contend for
    the same pool, which is the opposite of the multi-tenant KMS this
    models."""
    graph = build_tenant_graph(n_nodes=12, seed=3)
    assert nx.is_connected(graph)
    tenants = {data["tenant"] for _n, data in graph.nodes(data=True)}
    assert tenants == set(_TENANT_PROFILES)


def test_build_tenant_graph_is_seed_reproducible_and_seed_sensitive():
    a = build_tenant_graph(n_nodes=12, seed=5)
    b = build_tenant_graph(n_nodes=12, seed=5)
    c = build_tenant_graph(n_nodes=12, seed=6)

    def signature(g):
        return sorted(
            (u, v, d["sensitivity_class"], round(d["traffic_rate"], 9), d["pqc_capable"])
            for u, v, d in g.edges(data=True)
        )

    assert signature(a) == signature(b)
    assert signature(a) != signature(c)


def test_build_tenant_graph_allocates_every_node_and_rejects_too_few():
    for n in range(len(_TENANT_PROFILES), 40):
        assert build_tenant_graph(n_nodes=n, seed=1).number_of_nodes() == n
    with pytest.raises(ValueError):
        build_tenant_graph(n_nodes=len(_TENANT_PROFILES) - 1)


def test_hospital_flows_carry_higher_sensitivity_than_telemetry_flows():
    """The graph has to make tenants genuinely different, or S4 ("one
    *low-sensitivity* tenant floods the API") has nothing to mean."""
    graph = build_tenant_graph(n_nodes=40, seed=0)
    by_tenant: dict[str, list[int]] = {}
    for _u, _v, data in graph.edges(data=True):
        by_tenant.setdefault(data["tenant"], []).append(data["sensitivity_class"])

    assert np.mean(by_tenant["hospital"]) > np.mean(by_tenant["iot-telemetry"])
    assert np.mean(by_tenant["fintech"]) > np.mean(by_tenant["logging"])


# ---------------------------------------------------------------------------
# RequestGenerator (graph-driven stream)
# ---------------------------------------------------------------------------


def test_request_generator_rejects_an_edgeless_graph():
    with pytest.raises(ValueError):
        RequestGenerator(nx.Graph())


def test_request_generator_emits_contract_shaped_requests_from_graph_edges():
    graph = build_tenant_graph(n_nodes=10, seed=0)
    generator = RequestGenerator(graph, seed=0)

    edge_classes = {d["sensitivity_class"] for _u, _v, d in graph.edges(data=True)}
    edge_tenants = {d["tenant"] for _u, _v, d in graph.edges(data=True)}

    seen = 0
    for step in range(200):
        for request in generator.step(step):
            seen += 1
            assert request["step"] == step
            assert request["tenant"] in edge_tenants
            assert request["sensitivity_class"] in edge_classes
            assert isinstance(request["pqc_capable"], bool)
            assert isinstance(request["hybrid_mandatory"], bool)
    assert seen > 100  # the stream actually produced traffic


def test_request_generator_stream_is_swappable_for_the_graph_free_stub():
    """Hard Rule 3, made literal: both request sources are
    `Iterator[Request]` at the same aggregate arrival rate, so the
    environment (and therefore every line of agent code) cannot tell
    them apart structurally."""
    graph_stream = RequestGenerator(build_tenant_graph(n_nodes=10, seed=0), seed=0).stream()
    stub_stream = random_request_generator(seed=0)

    graph_requests = [next(graph_stream) for _ in range(300)]
    stub_requests = [next(stub_stream) for _ in range(300)]

    assert set(graph_requests[0]) == set(stub_requests[0])  # identical key sets
    # step counters are monotonically non-decreasing in both
    for requests in (graph_requests, stub_requests):
        steps = [r["step"] for r in requests]
        assert steps == sorted(steps)
    # and the two sit at a comparable aggregate rate (within 25%)
    graph_span = graph_requests[-1]["step"] + 1
    stub_span = stub_requests[-1]["step"] + 1
    assert 0.75 < (graph_span / stub_span) < 1.25


def test_request_generator_reset_rewinds_the_stream():
    generator = RequestGenerator(build_tenant_graph(n_nodes=10, seed=0), seed=7)
    first = [r["sensitivity_class"] for step in range(50) for r in generator.step(step)]
    generator.reset()
    second = [r["sensitivity_class"] for step in range(50) for r in generator.step(step)]
    assert first == second


def test_tenant_rate_multiplier_floods_that_tenant_and_raises_aggregate_load():
    """S4's mechanism, tested at the generator level: a noisy neighbour
    sends *more* requests, it does not merely take a larger share of a
    fixed budget."""
    graph = build_tenant_graph(n_nodes=10, seed=0)
    baseline = RequestGenerator(graph, seed=0)
    flooded = RequestGenerator(graph, seed=0, tenant_rate_multipliers={"iot-telemetry": 12.0})

    assert flooded.arrival_rate > baseline.arrival_rate

    def tenant_share(generator, tenant):
        requests = [r for step in range(400) for r in generator.step(step)]
        return sum(r["tenant"] == tenant for r in requests) / len(requests)

    assert tenant_share(flooded, "iot-telemetry") > tenant_share(baseline, "iot-telemetry")


def test_set_tenant_rate_multipliers_takes_effect_mid_episode():
    generator = RequestGenerator(build_tenant_graph(n_nodes=10, seed=0), seed=0)
    before = generator.arrival_rate
    generator.set_tenant_rate_multipliers({"iot-telemetry": 12.0})
    assert generator.arrival_rate > before
    generator.set_tenant_rate_multipliers({})
    assert generator.arrival_rate == pytest.approx(before)


def test_graph_hybrid_mandatory_rate_matches_the_calibrated_pool_regime():
    """`_HYBRID_MANDATORY_BY_CLASS` rises with sensitivity, but its
    aggregate rate over the default graph must stay near the flat 0.2
    the graph-free stub used -- that is the demand level the 2026-08-19
    pool recalibration was frozen against. If this drifts far, the pool
    regime silently stops being the one that was validated."""
    generator = RequestGenerator(build_tenant_graph(n_nodes=10, seed=0), seed=0)
    requests = [r for step in range(3000) for r in generator.step(step)]
    rate = sum(r["hybrid_mandatory"] for r in requests) / len(requests)
    assert 0.10 < rate < 0.30


def test_hybrid_mandatory_probability_rises_with_sensitivity_class():
    generator = RequestGenerator(build_tenant_graph(n_nodes=40, seed=0), seed=0)
    requests = [r for step in range(4000) for r in generator.step(step)]

    by_class: dict[int, list[bool]] = {}
    for request in requests:
        by_class.setdefault(request["sensitivity_class"], []).append(request["hybrid_mandatory"])

    rates = {c: np.mean(v) for c, v in by_class.items() if len(v) > 50}
    ordered = [rates[c] for c in sorted(rates)]
    assert ordered == sorted(ordered)  # monotonically non-decreasing in class


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
