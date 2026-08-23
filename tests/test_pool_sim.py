"""Behavioral tests for `env.pool_sim` (PLAN.md §10 kickoff step 2:
"Build the pool simulator first ... with unit tests.").

Covers: refill arithmetic against trace SKR values, drain-by-draw,
exhaustion signaling, the never-negative invariant, config-driven
construction (nothing hardcoded), and the documented synthetic trace's
generation procedure (mean rate + dialed-in QBER spike for S3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pytest

from env.pool_sim import (
    PoolExhaustedError,
    PoolSim,
    SyntheticSKRQBERTrace,
    load_pool_config,
)


@dataclass
class FixedSKRQBERTrace:
    """Deterministic `SKRQBERTrace` for exact-arithmetic assertions."""

    pairs: list[tuple[float, float]]

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return iter(self.pairs)


def make_pool(capacity: float, pairs: list[tuple[float, float]], initial_fill_frac: float = 0.0) -> PoolSim:
    return PoolSim(capacity=capacity, trace=FixedSKRQBERTrace(pairs), initial_fill_frac=initial_fill_frac)


# ---------------------------------------------------------------------------
# Refill arithmetic
# ---------------------------------------------------------------------------


def test_refill_rate_matches_trace_skr():
    """step() must refill by exactly skr_kbps * 1000 bits (1 step == 1 second)."""
    pool = make_pool(capacity=1_000_000, pairs=[(10.0, 0.01), (20.0, 0.01)])

    state = pool.step()
    assert state.fill == pytest.approx(10.0 * 1000.0)
    assert state.skr == pytest.approx(10.0)
    assert state.qber == pytest.approx(0.01)

    state = pool.step()
    assert state.fill == pytest.approx((10.0 + 20.0) * 1000.0)


def test_refill_never_exceeds_capacity():
    pool = make_pool(capacity=5_000, pairs=[(100.0, 0.01)])  # 100 kbps -> 100_000 bits, way over cap
    state = pool.step()
    assert state.fill == pytest.approx(5_000)
    assert state.fill <= state.capacity


def test_reset_rewinds_trace_and_restores_initial_fill():
    pool = make_pool(capacity=1_000_000, pairs=[(10.0, 0.01)], initial_fill_frac=0.5)
    pool.step()
    assert pool.fill != pytest.approx(500_000)

    state = pool.reset()
    assert state.fill == pytest.approx(500_000)

    # trace should yield the same first value again after rewind
    state = pool.step()
    assert state.skr == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Draw / drain
# ---------------------------------------------------------------------------


def test_draw_drains_by_exact_amount():
    pool = make_pool(capacity=1_000_000, pairs=[(10.0, 0.01)], initial_fill_frac=0.5)
    before = pool.fill
    pool.draw(1_000.0)
    assert pool.fill == pytest.approx(before - 1_000.0)


def test_can_draw_true_when_sufficient():
    pool = make_pool(capacity=1_000_000, pairs=[], initial_fill_frac=0.5)
    assert pool.can_draw(500_000.0) is True
    assert pool.can_draw(500_001.0) is False


def test_draw_raises_when_exceeds_fill():
    pool = make_pool(capacity=1_000_000, pairs=[], initial_fill_frac=0.1)
    with pytest.raises(PoolExhaustedError):
        pool.draw(200_000.0)


def test_draw_rejects_negative_bits():
    pool = make_pool(capacity=1_000_000, pairs=[], initial_fill_frac=0.5)
    with pytest.raises(ValueError):
        pool.draw(-1.0)


# ---------------------------------------------------------------------------
# Exhaustion / never-negative invariant
# ---------------------------------------------------------------------------


def test_exhaustion_condition_fires_at_zero():
    pool = make_pool(capacity=1_000_000, pairs=[], initial_fill_frac=0.0)
    assert pool.can_draw(1.0) is False
    with pytest.raises(PoolExhaustedError):
        pool.draw(1.0)


def test_pool_level_never_goes_negative():
    pool = make_pool(capacity=1_000_000, pairs=[], initial_fill_frac=0.001)
    exact = pool.fill
    pool.draw(exact)  # drain to exactly zero
    assert pool.fill == pytest.approx(0.0)
    assert pool.fill >= 0.0

    with pytest.raises(PoolExhaustedError):
        pool.draw(1.0)
    assert pool.fill >= 0.0


def test_pool_level_never_negative_across_many_draws():
    pool = make_pool(capacity=10_000, pairs=[(1.0, 0.01)] * 5, initial_fill_frac=0.0)
    for _ in range(5):
        state = pool.step()
        if pool.can_draw(2_000.0):
            pool.draw(2_000.0)
        assert pool.fill >= 0.0
        assert state.fill >= 0.0


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        make_pool(capacity=0, pairs=[])


def test_rejects_out_of_range_initial_fill_frac():
    with pytest.raises(ValueError):
        make_pool(capacity=1_000, pairs=[], initial_fill_frac=1.5)


# ---------------------------------------------------------------------------
# Config-driven construction (nothing hardcoded -- pulls from configs/default.yaml)
# ---------------------------------------------------------------------------


def test_load_pool_config_matches_yaml_file():
    import yaml
    from pathlib import Path

    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    loaded = load_pool_config()
    assert loaded == raw["pool"]
    assert "capacity_bits" in loaded
    assert "initial_fill_frac" in loaded


def test_pool_sim_constructs_from_config():
    cfg = load_pool_config()
    trace = SyntheticSKRQBERTrace(n_steps=10, seed=0)
    pool = PoolSim(
        capacity=cfg["capacity_bits"],
        trace=trace,
        initial_fill_frac=cfg["initial_fill_frac"],
    )
    assert pool.capacity == pytest.approx(cfg["capacity_bits"])
    assert pool.fill == pytest.approx(cfg["capacity_bits"] * cfg["initial_fill_frac"])


# ---------------------------------------------------------------------------
# SyntheticSKRQBERTrace generation procedure
# ---------------------------------------------------------------------------


def test_synthetic_trace_mean_skr_is_approximately_centered():
    trace = SyntheticSKRQBERTrace(n_steps=2000, mean_skr_kbps=200.0, skr_noise_frac=0.1, seed=42)
    skrs = [skr for skr, _ in trace]
    mean = sum(skrs) / len(skrs)
    assert mean == pytest.approx(200.0, rel=0.05)


def test_synthetic_trace_is_deterministic_and_reiterable():
    trace = SyntheticSKRQBERTrace(n_steps=50, seed=7)
    first_pass = list(trace)
    second_pass = list(trace)
    assert first_pass == second_pass


def test_synthetic_trace_qber_spike_window_raises_qber_and_lowers_skr():
    trace = SyntheticSKRQBERTrace(
        n_steps=100,
        mean_skr_kbps=200.0,
        baseline_qber=0.02,
        spike_start=40,
        spike_duration=20,
        spike_magnitude=0.3,
        seed=1,
    )
    pairs = list(trace)

    pre_spike_qber = sum(q for _, q in pairs[:40]) / 40
    spike_qber = sum(q for _, q in pairs[40:60]) / 20
    assert spike_qber > pre_spike_qber

    pre_spike_skr = sum(s for s, _ in pairs[:40]) / 40
    spike_skr = sum(s for s, _ in pairs[40:60]) / 20
    assert spike_skr < pre_spike_skr


def test_synthetic_trace_values_stay_in_valid_ranges():
    trace = SyntheticSKRQBERTrace(
        n_steps=500,
        spike_start=100,
        spike_duration=50,
        spike_magnitude=0.9,
        seed=3,
    )
    for skr, qber in trace:
        assert skr >= 0.0
        assert 0.0 <= qber <= 0.999


def test_pool_drains_correctly_under_synthetic_trace_s3_degradation():
    """End-to-end sanity check: pool fed by a degrading synthetic trace
    (S3-style) refills less during the spike window than before it."""
    trace = SyntheticSKRQBERTrace(
        n_steps=200,
        mean_skr_kbps=200.0,
        spike_start=100,
        spike_duration=50,
        spike_magnitude=0.4,
        seed=5,
    )
    pool = PoolSim(capacity=10_000_000_000, trace=trace, initial_fill_frac=0.0)

    for _ in range(100):
        pool.step()
    fill_before_spike = pool.fill

    for _ in range(50):
        pool.step()
    fill_after_spike = pool.fill
    refill_during_spike = fill_after_spike - fill_before_spike

    pool.reset()
    for _ in range(100):
        pool.step()
    refill_pre_spike_equivalent = pool.fill

    assert refill_during_spike < refill_pre_spike_equivalent


# ---------------------------------------------------------------------------
# spike_skr_multiplier (2026-08-24, Gate W3 S3 recalibration -- see this
# module's docstring step 4a and configs/scenarios/s3_degradation.yaml)
# ---------------------------------------------------------------------------


def test_spike_skr_multiplier_none_is_byte_identical_to_prior_formula():
    """Default (unset) must reproduce the pre-existing qber-derived,
    50%-capped formula exactly -- every caller/config that predates
    this session must be completely unaffected."""
    kwargs = dict(n_steps=250, seed=0, spike_start=50, spike_duration=150, spike_magnitude=0.6)
    trace_default = SyntheticSKRQBERTrace(**kwargs)
    trace_explicit_none = SyntheticSKRQBERTrace(**kwargs, spike_skr_multiplier=None)
    assert list(trace_default) == list(trace_explicit_none)


def test_spike_skr_multiplier_breaks_the_50_percent_ceiling():
    """The pre-existing formula can never reduce in-window SKR by more
    than 50% (verified: magnitudes 0.6/0.9/0.99/5.0 all saturate
    identically). `spike_skr_multiplier` must be able to go far
    beyond that ceiling when set."""
    old_formula = SyntheticSKRQBERTrace(
        n_steps=250, seed=0, spike_start=50, spike_duration=150, spike_magnitude=5.0
    )
    old_in_window = [s for s, _ in old_formula][50:200]
    old_mean = sum(old_in_window) / len(old_in_window)

    new_formula = SyntheticSKRQBERTrace(
        n_steps=250,
        seed=0,
        spike_start=50,
        spike_duration=150,
        spike_magnitude=0.6,
        spike_skr_multiplier=0.0,
    )
    new_in_window = [s for s, _ in new_formula][50:200]

    # old formula's ceiling: at most ~50% reduction from the ~200 kbps mean
    assert old_mean == pytest.approx(99.7448, rel=0.01)
    # new mechanism: a literal 0.0 multiplier drives in-window SKR to exactly zero
    assert all(s == 0.0 for s in new_in_window)


def test_spike_skr_multiplier_is_decoupled_from_qber_value():
    """The multiplier must apply regardless of what qber itself lands
    on that step -- it directly scales skr, it does not derive from
    qber (unlike the pre-existing formula)."""
    trace = SyntheticSKRQBERTrace(
        n_steps=100,
        seed=2,
        spike_start=20,
        spike_duration=30,
        spike_magnitude=0.01,  # deliberately tiny -- qber barely moves in-window
        spike_skr_multiplier=0.001,  # but skr should still collapse ~1000x
    )
    pairs = list(trace)
    pre_spike_mean_skr = sum(s for s, _ in pairs[:20]) / 20
    in_window_mean_skr = sum(s for s, _ in pairs[20:50]) / 30
    assert in_window_mean_skr < pre_spike_mean_skr / 100  # collapsed despite tiny qber shift
