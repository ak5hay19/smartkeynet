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
    QberDriftSchedule,
    SyntheticSKRQBERTrace,
    load_pool_config,
    load_qkd_config,
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


# ---------------------------------------------------------------------------
# Reconciliation gate + S3 drift (SMARTKEYNET_BUILD_SPEC.md §S1)
# ---------------------------------------------------------------------------


def test_qber_gate_is_monotone_and_vanishes_at_abort():
    """Spec §S1 test 5: SKR is non-increasing in QBER, and zero at
    `qber >= qber_abort`."""
    trace = SyntheticSKRQBERTrace(n_steps=1, baseline_qber=0.02, qber_abort=0.11)

    qbers = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.109, 0.11, 0.2]
    gates = [trace.reconciliation_gate(q) for q in qbers]

    assert all(later <= earlier + 1e-12 for earlier, later in zip(gates, gates[1:]))
    assert gates[0] == pytest.approx(1.0)  # at/below baseline: no loss
    assert trace.reconciliation_gate(0.02) == pytest.approx(1.0)
    assert trace.reconciliation_gate(0.11) == pytest.approx(0.0)
    assert trace.reconciliation_gate(0.5) == pytest.approx(0.0)


def test_gate_does_not_touch_skr_at_or_below_baseline_qber():
    """The gate is expressed relative to the link's baseline operating
    point precisely so that a scenario with no drift and no spike is
    unaffected by it -- otherwise turning the gate on would silently
    re-scale the S1 baseline."""
    trace = SyntheticSKRQBERTrace(n_steps=1, baseline_qber=0.02)
    for qber in (0.0, 0.005, 0.019, 0.02):
        assert trace.reconciliation_gate(qber) == pytest.approx(1.0)


def test_drift_schedule_ramps_then_partially_recovers():
    drift = QberDriftSchedule.for_episode(n_steps=900, peak_qber=0.099, residual_frac=0.25)
    baseline = 0.02
    peak_excess = 0.099 - baseline

    assert drift.excess_at(0, baseline) == pytest.approx(0.0)
    assert drift.excess_at(299, baseline) == pytest.approx(0.0)
    # ramps across [300, 450), holds across [450, 600), recovers to 900
    assert drift.excess_at(375, baseline) == pytest.approx(peak_excess * 0.5, rel=0.05)
    assert drift.excess_at(500, baseline) == pytest.approx(peak_excess)
    assert drift.excess_at(599, baseline) == pytest.approx(peak_excess)
    assert drift.peak_hold_window() == (450, 600)
    # partial recovery: ends at residual_frac of the peak, not at zero
    assert drift.excess_at(899, baseline) == pytest.approx(peak_excess * 0.25, rel=0.05)
    assert drift.excess_at(5000, baseline) == pytest.approx(peak_excess * 0.25)


def test_drift_excess_is_never_negative():
    """A drift schedule must not be able to *lower* QBER below
    baseline, which would raise SKR and make S3 a boon."""
    drift = QberDriftSchedule.for_episode(n_steps=600, peak_qber=0.099)
    for step in range(0, 1200, 7):
        assert drift.excess_at(step, 0.02) >= 0.0


def test_s3_drift_collapses_refill():
    """Spec §S1 test 6: mean refill in the middle third of an S3
    episode is under 30% of the first third. This is the test that
    decides whether scenario S3 tests anything at all."""
    n_steps = 3000
    drift = QberDriftSchedule.for_episode(n_steps=n_steps, peak_qber=0.9 * 0.11)
    trace = SyntheticSKRQBERTrace(n_steps=n_steps, drift=drift, seed=0)

    skrs = [skr for skr, _ in trace]
    third = n_steps // 3
    first_third_mean = sum(skrs[:third]) / third
    middle_third_mean = sum(skrs[third : 2 * third]) / third

    assert middle_third_mean < 0.30 * first_third_mean


def test_scarcity_ratio_in_target_band():
    """Spec §S1 test 11 -- the test that protects the thesis.

    `rho = keys demanded per step / keys refilled per step`. If
    `rho << 0.8` the pool never binds, every policy looks identical,
    and the DQN ties the tuned threshold baseline. This repo measured
    `rho = 0.0013` on 2026-08-15 before recalibration; see
    `configs/default.yaml`'s scarcity calibration block for the full
    arithmetic and `SESSION_LOG.md` for the measurement.

    Demand is measured against a **sensible** policy, not the
    always-hybrid villain. Sizing the link so always-hybrid struggles
    is what produced the misleading rho = 1.14 in the first pass at
    this calibration: that policy rekeys roughly 500x more often than
    necessary, so a link that strains it is still twenty times
    over-provisioned for anything anyone would deploy. See
    `configs/default.yaml`'s calibration block for the full reasoning
    and the measurement.

    Both ratios are asserted, because the environment needs both to be
    true: enough key material for a deliberate policy, nowhere near
    enough for a profligate one.
    """
    pool_config = load_pool_config()
    qkd_config = load_qkd_config()

    bits_per_step = qkd_config["mean_skr_kbps"] * 1000.0
    keys_refilled_per_step = bits_per_step / pool_config["bits_per_hybrid_draw"]

    # Both measured directly on 2026-08-15; see the config's calibration
    # block. Sensible demand ~= sessions / key_lifetime, because a
    # policy that reuses its keys draws one per session per lifetime.
    sensible_demand_per_step = 0.043
    always_hybrid_demand_per_step = 0.98

    rho_sensible = sensible_demand_per_step / keys_refilled_per_step
    rho_villain = always_hybrid_demand_per_step / keys_refilled_per_step
    print(f"\nrho (tuned threshold, S1) = {rho_sensible:.3f}")
    print(f"rho (always-hybrid, S1)   = {rho_villain:.3f}")

    assert 0.2 <= rho_sensible <= 1.3, (
        f"rho_sensible={rho_sensible:.4f} is outside the band the pool must bind in. "
        "Too low and no policy can misbudget; too high and even a careful policy "
        "starves. See SMARTKEYNET_BUILD_SPEC.md §S1 test 11 and §11.2."
    )
    assert rho_villain > 2.0, (
        f"rho_villain={rho_villain:.3f}: always-hybrid must comfortably exhaust the "
        "pool, or Demo Beat 2 has no villain."
    )


def test_s3_scarcity_ratio_exceeds_the_s1_band():
    """Spec §S1 test 11, S3 half: degradation must actually bite, i.e.
    `rho_S3 > 1.3`. Measured against the *irreducible* hybrid-mandatory
    demand rather than always-hybrid demand -- S3 has to hurt even a
    perfectly frugal agent, otherwise it is testing nothing."""
    n_steps = 3000
    pool_config = load_pool_config()
    qkd_config = load_qkd_config()

    drift = QberDriftSchedule.for_episode(
        n_steps=n_steps, peak_qber=qkd_config["s3_peak_qber_frac"] * qkd_config["qber_abort"]
    )
    trace = SyntheticSKRQBERTrace(
        n_steps=n_steps,
        mean_skr_kbps=qkd_config["mean_skr_kbps"],
        baseline_qber=qkd_config["baseline_qber"],
        qber_abort=qkd_config["qber_abort"],
        gate_kappa=qkd_config["gate_kappa"],
        drift=drift,
        seed=0,
    )

    hold_start, hold_end = drift.peak_hold_window()
    worst_skrs = [skr for skr, _ in trace][hold_start:hold_end]
    mean_skr = sum(worst_skrs) / len(worst_skrs)
    keys_refilled_per_step = mean_skr * 1000.0 / pool_config["bits_per_hybrid_draw"]

    # hybrid-mandatory fraction of arrivals (env/request_generator.py's
    # graph source realises 0.30, matching spec §11.2)
    keys_demanded_per_step = 0.988 * 0.30

    rho_s3 = keys_demanded_per_step / keys_refilled_per_step
    print(f"\nrho (hybrid-mandatory only, S3 peak-hold window) = {rho_s3:.3f}")

    assert rho_s3 > 1.3, (
        f"rho_S3={rho_s3:.3f} does not exceed 1.3 -- S3 degradation is not biting, "
        "so the scenario tests nothing about scarcity."
    )


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
