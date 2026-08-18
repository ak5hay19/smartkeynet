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
    slice_skr_kbps,
)


_EVAL_EPISODE_STEPS = 2_000
"""configs/default.yaml's `training.eval_max_steps` / the harness default.
Spelled here so the exhaustibility check below stays honest if either moves."""


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
# Slice-SKR derivation (2026-08-19 pool recalibration -- see
# configs/default.yaml's `pool:` block for the full note)
# ---------------------------------------------------------------------------


# Measured on the configured 50-node graph over 2000-step episodes, seeds 0-2,
# with an effectively unlimited pool -- so these are *demands*, not outcomes.
# See configs/default.yaml's `pool:` block for the full derivation.
_FLOOR_MANDATED_DEMAND_BITS_PER_STEP = 8.64   # S2, HIGH posture
_MAXIMAL_DEMAND_BITS_PER_STEP = 20.98         # always-hybrid


def test_slice_skr_converts_configured_bits_per_step_into_the_trace_unit():
    """`slice_skr_kbps` is a unit conversion, nothing more -- the config
    states bits per decision epoch, `PoolSim.step()` speaks kbps."""
    assert slice_skr_kbps({"refill_bits_per_step": 15.0}) == pytest.approx(0.015)
    assert slice_skr_kbps({"refill_bits_per_step": 200.0}) == pytest.approx(0.2)


def test_slice_skr_still_accepts_the_interim_two_factor_spelling():
    """tests/test_environment.py's extreme-scarcity fixtures ask for a
    deliberately huge refill in the two-factor form."""
    assert slice_skr_kbps({"link_skr_kbps": 200.0, "kms_requests_per_decision_epoch": 1.0}) == pytest.approx(200.0)
    with pytest.raises(ValueError):
        slice_skr_kbps({"link_skr_kbps": 200.0, "kms_requests_per_decision_epoch": 0.0})


def test_slice_skr_rejects_a_non_positive_refill():
    with pytest.raises(ValueError):
        slice_skr_kbps({"refill_bits_per_step": 0.0})


def test_slice_skr_falls_back_for_pre_recalibration_pool_blocks():
    """A hand-built `pool:` dict with neither spelling still resolves,
    to the documented default."""
    assert slice_skr_kbps({"capacity_bits": 1.0}) == pytest.approx(0.015)


def test_configured_refill_sits_inside_the_measured_demand_bracket():
    """The premise check (PLAN2 §3.2; Hard Rule 7's "investigate the
    environment design first").

    Refill must sit strictly between floor-MANDATED hybrid demand and
    MAXIMAL hybrid demand, or the budgeting problem stops existing in
    one direction or the other:

      * at or below mandated demand -> floors become unservable and
        every policy drowns in deferrals regardless of skill;
      * at or above maximal demand  -> nothing can ever exhaust the
        pool, `pool_fill` pins at 1.0, every threshold in the grid
        collapses onto always-hybrid, and budgeting skill is
        irrelevant. That was this repo's state as received.
    """
    cfg = load_pool_config()
    bits_per_step = slice_skr_kbps(cfg) * 1000.0 * PoolSim._SECONDS_PER_STEP
    assert _FLOOR_MANDATED_DEMAND_BITS_PER_STEP < bits_per_step < _MAXIMAL_DEMAND_BITS_PER_STEP, (
        f"refill of {bits_per_step:.2f} bits/step is outside the measured demand bracket "
        f"({_FLOOR_MANDATED_DEMAND_BITS_PER_STEP}, {_MAXIMAL_DEMAND_BITS_PER_STEP}) -- "
        "the QKD budgeting problem no longer exists at this sizing"
    )


def test_configured_pool_capacity_is_exhaustible_within_an_episode():
    """Capacity must be small enough that the deficit between maximal
    demand and refill can genuinely drain the initial buffer inside one
    evaluation episode, or pool exhaustion is unobservable in every
    result the project reports."""
    cfg = load_pool_config()
    bits_per_step = slice_skr_kbps(cfg) * 1000.0 * PoolSim._SECONDS_PER_STEP
    deficit_per_step = _MAXIMAL_DEMAND_BITS_PER_STEP - bits_per_step
    assert deficit_per_step > 0
    steps_to_drain = (cfg["capacity_bits"] * cfg["initial_fill_frac"]) / deficit_per_step
    assert steps_to_drain < _EVAL_EPISODE_STEPS
