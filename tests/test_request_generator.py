"""Behavioral tests for `env.request_generator` (PLAN.md §10 kickoff
step 4; Hard Rule 3 -- the graph only shapes *which* requests arrive,
never anything the agent sees directly, so the dummy stream must be a
valid drop-in `Request` source).

`build_tenant_graph()` and `RequestGenerator` are intentionally left
`NotImplementedError` this session -- only `random_request_generator()`
is implemented -- and that's verified here too.
"""

from __future__ import annotations

import itertools

import networkx as nx
import pytest

from env.contracts import SensitivityClass
from env.request_generator import (
    _ARRIVAL_RATE_PER_STEP,
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
# build_tenant_graph / RequestGenerator untouched this session
# ---------------------------------------------------------------------------


def test_build_tenant_graph_is_still_not_implemented():
    with pytest.raises(NotImplementedError):
        build_tenant_graph()


def test_request_generator_class_is_still_not_implemented():
    graph = nx.Graph()
    with pytest.raises(NotImplementedError):
        RequestGenerator(graph)


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
