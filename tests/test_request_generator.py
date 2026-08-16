"""Behavioral tests for `env.request_generator` (PLAN.md §10 kickoff
step 4; Hard Rule 3 -- the graph only shapes *which* requests arrive,
never anything the agent sees directly, so both sources must be valid
drop-in `Request` streams).

Two request sources live here and both are covered: the plain
stationary Poisson `random_request_generator`, and the NetworkX tenant
graph sampled by `RequestGenerator` with tenant-level MMPP bursts
(built 2026-08-15, previously `NotImplementedError`).
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from env.contracts import SensitivityClass
from env.request_generator import (
    _ARRIVAL_RATE_PER_STEP,
    _HYBRID_MANDATORY_MIN_CLASS,
    _LEGACY_MAX_CLASS,
    _TENANT_PROFILES,
    RequestGenerator,
    TenantFlood,
    build_tenant_graph,
    measure_fano_factor,
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
# build_tenant_graph (PLAN.md §10 step 4; SMARTKEYNET_BUILD_SPEC.md §S3)
# ---------------------------------------------------------------------------


def test_graph_is_connected_and_carries_the_three_edge_attributes():
    graph = build_tenant_graph(n_nodes=50, seed=0)

    assert graph.number_of_nodes() == 50
    assert nx.is_connected(graph)
    for _, _, attrs in graph.edges(data=True):
        assert 0 <= attrs["sensitivity_class"] < len(SensitivityClass)
        assert attrs["traffic_rate"] > 0.0
        assert isinstance(attrs["pqc_capable"], bool)
        assert attrs["tenant"] in {profile.name for profile in _TENANT_PROFILES}


def test_graph_rejects_too_few_nodes_for_the_tenant_profiles():
    with pytest.raises(ValueError):
        build_tenant_graph(n_nodes=4, seed=0)


def test_graph_is_deterministic_under_seed():
    a = build_tenant_graph(n_nodes=50, seed=11)
    b = build_tenant_graph(n_nodes=50, seed=11)
    c = build_tenant_graph(n_nodes=50, seed=12)

    def signature(graph):
        return sorted(
            (u, v, attrs["sensitivity_class"], round(attrs["traffic_rate"], 9), attrs["pqc_capable"])
            for u, v, attrs in graph.edges(data=True)
        )

    assert signature(a) == signature(b)
    assert signature(a) != signature(c)


def test_legacy_edges_carry_only_low_classes():
    """SMARTKEYNET_BUILD_SPEC.md §S3's consistency assertion: a
    `pqc_capable: false` edge must never carry a class that could
    become unservable once interoperability masking exists. See
    `_LEGACY_MAX_CLASS` for the known tension around this rule."""
    for seed in range(10):
        graph = build_tenant_graph(n_nodes=50, seed=seed)
        legacy_edges = [attrs for _, _, attrs in graph.edges(data=True) if not attrs["pqc_capable"]]
        assert legacy_edges  # the scenario is actually exercised
        for attrs in legacy_edges:
            assert attrs["sensitivity_class"] <= int(_LEGACY_MAX_CLASS)


def test_class_assignment_is_tenant_conditioned_not_uniform():
    """The heterogeneity that makes budgeting non-trivial: hospital
    flows must skew high-sensitivity and telemetry flows low. If every
    tenant looked alike, a single threshold rule would be optimal and
    the project's premise would fail (spec §S3)."""
    hospital_rate_by_class = np.zeros(len(SensitivityClass))
    telemetry_rate_by_class = np.zeros(len(SensitivityClass))

    for seed in range(8):
        graph = build_tenant_graph(n_nodes=50, seed=seed)
        for _, _, attrs in graph.edges(data=True):
            if attrs["tenant"] == "hospital":
                hospital_rate_by_class[attrs["sensitivity_class"]] += attrs["traffic_rate"]
            elif attrs["tenant"] == "telemetry":
                telemetry_rate_by_class[attrs["sensitivity_class"]] += attrs["traffic_rate"]

    hospital_high_share = hospital_rate_by_class[2:].sum() / hospital_rate_by_class.sum()
    telemetry_high_share = telemetry_rate_by_class[2:].sum() / telemetry_rate_by_class.sum()

    assert hospital_high_share > 0.5
    assert telemetry_high_share < 0.05
    assert hospital_high_share > telemetry_high_share


def test_aggregate_class_mix_is_near_the_spec_target():
    """Rate-weighted class mix against SMARTKEYNET_BUILD_SPEC.md §3.2's
    `class_mix` target. The tolerance is deliberately loose -- see
    `_TENANT_PROFILES`' docstring for why the residual is accepted
    rather than optimised away."""
    target = np.array([0.35, 0.35, 0.20, 0.10])
    rate_by_class = np.zeros(len(SensitivityClass))
    for seed in range(8):
        graph = build_tenant_graph(n_nodes=50, seed=seed)
        for _, _, attrs in graph.edges(data=True):
            rate_by_class[attrs["sensitivity_class"]] += attrs["traffic_rate"]

    mix = rate_by_class / rate_by_class.sum()
    assert float(np.abs(mix - target).sum()) < 0.20

    # the hybrid-mandatory fraction is the number the scarcity
    # calibration in configs/default.yaml actually depends on
    assert mix[2] + mix[3] == pytest.approx(0.30, abs=0.05)


def test_total_arrival_rate_matches_the_poisson_source():
    """Both request sources must present the same mean load, so the
    Hard Rule 3 substitution changes burstiness and tenant structure
    without silently changing how much work arrives."""
    graph = build_tenant_graph(n_nodes=50, seed=0, total_arrival_rate=_ARRIVAL_RATE_PER_STEP)
    generator = RequestGenerator(graph, seed=0)
    counts = [len(generator.step(t)) for t in range(20_000)]
    assert float(np.mean(counts)) == pytest.approx(_ARRIVAL_RATE_PER_STEP, rel=0.15)


# ---------------------------------------------------------------------------
# RequestGenerator
# ---------------------------------------------------------------------------


def test_generator_emits_contract_valid_requests():
    graph = build_tenant_graph(n_nodes=50, seed=0)
    generator = RequestGenerator(graph, seed=0)

    seen = 0
    for step in range(500):
        for request in generator.step(step):
            seen += 1
            assert request["step"] == step
            assert isinstance(request["request_id"], str) and request["request_id"]
            assert isinstance(request["tenant"], str) and request["tenant"]
            assert isinstance(request["service"], str) and request["service"]
            assert 0 <= request["sensitivity_class"] < len(SensitivityClass)
            assert isinstance(request["pqc_capable"], bool)
            assert isinstance(request["hybrid_mandatory"], bool)
    assert seen > 0


def test_generator_rejects_a_graph_with_no_edges():
    with pytest.raises(ValueError):
        RequestGenerator(nx.Graph(), seed=0)


def test_hybrid_mandatory_is_derived_from_sensitivity_class():
    """The graph source derives the flag from the class (mirroring the
    policy table's highest-posture column) rather than flipping an
    independent coin, so its demand is internally consistent with the
    floors the env will compute."""
    graph = build_tenant_graph(n_nodes=50, seed=0)
    generator = RequestGenerator(graph, seed=0)
    for step in range(500):
        for request in generator.step(step):
            expected = request["sensitivity_class"] >= int(_HYBRID_MANDATORY_MIN_CLASS)
            assert request["hybrid_mandatory"] is expected


def test_generator_is_deterministic_and_resettable():
    graph = build_tenant_graph(n_nodes=50, seed=0)
    generator = RequestGenerator(graph, seed=5)

    first = [generator.step(t) for t in range(200)]
    generator.reset()
    second = [generator.step(t) for t in range(200)]
    assert first == second

    other = RequestGenerator(graph, seed=6)
    assert [other.step(t) for t in range(200)] != first


def test_burstiness_exceeds_the_plain_poisson_source():
    """Spec §S3 test 2. Measured at `_FANO_BIN_STEPS`-wide bins -- see
    that constant for why a per-step Fano factor cannot express
    burstiness in this environment."""
    n_steps = 40_000
    graph = build_tenant_graph(n_nodes=50, seed=0)
    generator = RequestGenerator(graph, seed=0)
    mmpp_counts = [len(generator.step(t)) for t in range(n_steps)]

    poisson_counts = [0] * n_steps
    for request in random_request_generator(seed=0):
        if request["step"] >= n_steps:
            break
        poisson_counts[request["step"]] += 1

    mmpp_fano = measure_fano_factor(mmpp_counts)
    poisson_fano = measure_fano_factor(poisson_counts)

    assert mmpp_fano > 1.5
    assert poisson_fano == pytest.approx(1.0, abs=0.2)
    assert mmpp_fano > poisson_fano


def test_burstiness_does_not_wash_out_as_the_graph_grows():
    """Regression guard for the per-tenant-vs-per-edge MMPP decision
    (`_MMPP_ON_RATE`): with independent per-edge chains the Fano factor
    *fell* from 2.35 to 1.32 as the graph grew from 10 to 55 edges,
    which would have made the ~50-node graph PLAN.md §4 asks for
    strictly less bursty than a toy one."""
    fano_by_size = {}
    for n_nodes in (10, 50):
        graph = build_tenant_graph(n_nodes=n_nodes, seed=0)
        generator = RequestGenerator(graph, seed=0)
        counts = [len(generator.step(t)) for t in range(40_000)]
        fano_by_size[n_nodes] = measure_fano_factor(counts)

    assert fano_by_size[50] > 1.5
    assert fano_by_size[50] > 0.75 * fano_by_size[10]


def test_measure_fano_factor_rejects_a_too_short_series():
    with pytest.raises(ValueError):
        measure_fano_factor([1, 2, 3], bin_steps=25)


# ---------------------------------------------------------------------------
# S4 tenant flood (PLAN.md §5 S4)
# ---------------------------------------------------------------------------


def test_tenant_flood_multiplies_only_the_target_tenant_inside_the_window():
    graph = build_tenant_graph(n_nodes=50, seed=0)
    flood = TenantFlood(tenant="telemetry", start_step=600, end_step=1200, rate_multiplier=50.0)
    generator = RequestGenerator(graph, seed=0, tenant_flood=flood)

    in_window: dict[str, int] = {}
    out_window: dict[str, int] = {}
    for step in range(2000):
        bucket = in_window if 600 <= step < 1200 else out_window
        for request in generator.step(step):
            bucket[request["tenant"]] = bucket.get(request["tenant"], 0) + 1

    in_rate = {name: count / 600 for name, count in in_window.items()}
    out_rate = {name: count / 1400 for name, count in out_window.items()}

    assert in_rate["telemetry"] > 10 * out_rate["telemetry"]
    for tenant in out_rate:
        if tenant == "telemetry":
            continue
        # untargeted tenants keep their nominal rate (loose bound: MMPP
        # bursts make these counts noisy over a single window)
        assert in_rate.get(tenant, 0.0) == pytest.approx(out_rate[tenant], rel=0.6)


def test_flood_target_carries_no_hybrid_mandatory_flows():
    """S4 asks whether a critical tenant's pool share survives a
    neighbour's load, so the flooding tenant must not itself be a
    source of hybrid-mandatory demand (that is S3's question)."""
    graph = build_tenant_graph(n_nodes=50, seed=0)
    telemetry_classes = {
        attrs["sensitivity_class"] for _, _, attrs in graph.edges(data=True) if attrs["tenant"] == "telemetry"
    }
    assert max(telemetry_classes) < int(_HYBRID_MANDATORY_MIN_CLASS)


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
