"""Behavioral tests for `env.pool_sim` (PLAN.md §10 kickoff step 2:
"Build the pool simulator first ... with unit tests.").

Covers: refill arithmetic against trace SKR values, drain-by-draw,
exhaustion signaling, the never-negative invariant, config-driven
construction (nothing hardcoded), and the documented synthetic trace's
generation procedure (mean rate + dialed-in QBER spike for S3).
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


KEY_BITS = 256
"""ETSI GS QKD 014 key size. The pool counts whole keys of this size, so
every bit quantity in this file is a multiple of it -- see
`test_refill_conserves_bits` for why that matters."""


def make_pool(
    capacity: float,
    pairs: list[tuple[float, float]],
    initial_fill_frac: float = 0.0,
    max_key_age_steps: int | None = None,
) -> PoolSim:
    return PoolSim(
        capacity=capacity,
        trace=FixedSKRQBERTrace(pairs),
        initial_fill_frac=initial_fill_frac,
        max_key_age_steps=max_key_age_steps,
    )


# ---------------------------------------------------------------------------
# Refill arithmetic
# ---------------------------------------------------------------------------


def test_refill_rate_matches_trace_skr():
    """step() refills by skr_kbps * 1000 bits (1 step == 1 second), rounded
    down to whole keys with the remainder banked in the carry."""
    pool = make_pool(capacity=1_000_000, pairs=[(10.0, 0.01), (20.0, 0.01)])

    state = pool.step()
    assert state.fill == pytest.approx(KEY_BITS * int(10.0 * 1000.0 / KEY_BITS))
    assert state.skr == pytest.approx(10.0)
    assert state.qber == pytest.approx(0.01)

    state = pool.step()
    # Two steps supplied 30_000 bits; the pool holds whole keys, so it is
    # within one key of that and never above it.
    assert 30_000 - KEY_BITS < state.fill <= 30_000


def test_refill_conserves_bits():
    """SMARTKEYNET_BUILD_SPEC.md §S1 test 1, the one that catches the
    classic dropped-remainder bug.

    Each step's distilled bits rarely divide evenly into 256-bit keys.
    Truncating independently every step would silently discard up to one
    key per step -- at this project's calibrated 0.859 keys/step that is
    most of the link's output, and it would show up only as an
    unexplained shortfall in the scarcity ratio. The fractional carry is
    what makes the long-run rate exact.
    """
    steps = 10_000
    skr_kbps = 7.3  # deliberately not a multiple of a key per step
    pool = make_pool(capacity=10**9, pairs=[(skr_kbps, 0.01)] * steps)

    for _ in range(steps):
        pool.step()

    supplied_bits = skr_kbps * 1000.0 * steps
    banked_bits = pool.level * KEY_BITS + pool._fractional_carry * KEY_BITS
    assert banked_bits == pytest.approx(supplied_bits, abs=KEY_BITS)


def test_refill_never_exceeds_capacity():
    pool = make_pool(capacity=5_120, pairs=[(100.0, 0.01)])  # 20 keys cap; 100 kbps is way over
    state = pool.step()
    assert state.fill == pytest.approx(5_120)
    assert state.fill <= state.capacity
    assert pool.level == pool.capacity_keys


def test_overflow_is_reported():
    """Overflow is a result, not a nuisance (§S1): it is the quantum
    material the link produced and the pool was too full to hold."""
    pool = make_pool(capacity=5_120, pairs=[(100.0, 0.01)])  # 20-key cap
    state = pool.step()
    # 100 kbps == 100_000 bits == 390 keys distilled into a 20-key pool.
    assert state.overflow_keys == 390 - 20
    assert pool.overflow_keys_total == 370


def test_no_overflow_when_pool_has_room():
    pool = make_pool(capacity=10**9, pairs=[(1.0, 0.01)])
    state = pool.step()
    assert state.overflow_keys == 0
    assert pool.overflow_keys_total == 0


def test_reset_rewinds_trace_and_restores_initial_fill():
    pool = make_pool(capacity=1_000_000, pairs=[(10.0, 0.01)], initial_fill_frac=0.5)
    starting_keys = pool.level
    pool.step()
    assert pool.level != starting_keys

    state = pool.reset()
    assert pool.level == starting_keys
    assert state.fill == pytest.approx(starting_keys * KEY_BITS)

    # trace should yield the same first value again after rewind
    state = pool.step()
    assert state.skr == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Draw / drain
# ---------------------------------------------------------------------------


def test_draw_drains_by_exact_amount():
    pool = make_pool(capacity=1_000_000, pairs=[(10.0, 0.01)], initial_fill_frac=0.5)
    before = pool.level
    pool.draw(4 * KEY_BITS)
    assert pool.level == before - 4
    assert pool.fill == pytest.approx((before - 4) * KEY_BITS)


def test_draw_of_a_partial_key_still_costs_a_whole_key():
    """A fraction of a key cannot establish a session, so a sub-key draw
    consumes a whole one. This is the behaviour the old float-bits pool
    could not express."""
    pool = make_pool(capacity=1_000_000, pairs=[], initial_fill_frac=0.5)
    before = pool.level
    pool.draw(1.0)
    assert pool.level == before - 1


def test_can_draw_true_when_sufficient():
    pool = make_pool(capacity=1_000_000, pairs=[], initial_fill_frac=0.5)
    held_bits = pool.level * KEY_BITS
    assert pool.can_draw(held_bits) is True
    assert pool.can_draw(held_bits + 1.0) is False


def test_draw_refuses_when_insufficient():
    """§S1 test 3: at level 0 and at exactly one key short, the draw is
    refused and the level is unchanged."""
    pool = make_pool(capacity=2_560, pairs=[], initial_fill_frac=0.0)
    assert pool.draw_keys(1).ok is False
    assert pool.level == 0

    pool = make_pool(capacity=2_560, pairs=[], initial_fill_frac=0.5)  # 5 keys
    assert pool.draw_keys(6).ok is False
    assert pool.level == 5


def test_draw_is_fifo_over_batches():
    """§S1 test 4: a draw spanning two batches reports the older batch
    first with the correct split."""
    pool = make_pool(capacity=100 * KEY_BITS, pairs=[], initial_fill_frac=0.0)
    pool.refill(skr_kbps=2 * KEY_BITS / 1000.0, step_seconds=1.0, now=1)  # 2 keys
    pool.refill(skr_kbps=3 * KEY_BITS / 1000.0, step_seconds=1.0, now=2)  # 3 keys
    first, second = pool.batches()

    result = pool.draw_keys(4)
    assert result.ok is True
    assert result.lineage == ((first.batch_id, 2), (second.batch_id, 2))
    assert pool.level == 1


def test_age_out_discards_oldest_first():
    """§S1 test 8: real key stores do not hold material forever."""
    pool = make_pool(capacity=100 * KEY_BITS, pairs=[], initial_fill_frac=0.0, max_key_age_steps=5)
    pool.refill(skr_kbps=2 * KEY_BITS / 1000.0, step_seconds=1.0, now=1)
    pool.refill(skr_kbps=3 * KEY_BITS / 1000.0, step_seconds=1.0, now=4)
    assert pool.level == 5

    # now=6 is 5 steps past the first batch (refilled at 1) but not the second.
    result = pool.refill(skr_kbps=0.0, step_seconds=1.0, now=6)
    assert result.expired_keys == 2
    assert pool.level == 3
    assert pool.expired_keys_total == 2


def test_age_out_is_off_by_default():
    """Ageing is config-gated so it cannot silently change the calibrated
    scarcity ratio of runs that never asked for it."""
    pool = make_pool(capacity=100 * KEY_BITS, pairs=[], initial_fill_frac=0.0)
    pool.refill(skr_kbps=2 * KEY_BITS / 1000.0, step_seconds=1.0, now=1)
    pool.refill(skr_kbps=0.0, step_seconds=1.0, now=10_000)
    assert pool.level == 2


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
    from pathlib import Path

    import yaml

    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, encoding="utf-8") as f:
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

    assert all(later <= earlier + 1e-12 for earlier, later in zip(gates, gates[1:], strict=False))
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


def _binding_diagnostics(policy, scenario: str, seeds=(0, 1, 2), steps: int = 900) -> dict:
    """Run `policy` and report how hard the pool bound: the fraction of
    steps it sat empty and full, regret events, the peak and final deferral
    queue length, and wasted overflow.

    Imported lazily so `tests/test_pool_sim.py` stays runnable as a unit
    test file when the env or the baselines are mid-edit.
    """
    import numpy as np
    import yaml

    from agents.baselines import AlwaysHybridPolicy, AlwaysPQCPolicy  # noqa: F401
    from env.environment import SmartKeyNetEnv

    config_path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    empties, fulls, regrets, queue_peaks, queue_ends, overflows = [], [], [], [], [], []
    for seed in seeds:
        env = SmartKeyNetEnv(
            {
                **base,
                "scenario": scenario,
                "max_steps": steps,
                "scenario_steps": steps + 200,
                "seed": seed,
            }
        )
        state, info = env.reset(seed=seed)
        levels, queue_lengths, regret_events = [], [], 0
        for _ in range(steps):
            levels.append(env._pool_sim.level)
            queue_lengths.append(len(env._deferral_queue))
            state, _reward, _terminated, truncated, info = env.step(
                policy.act(state, info["action_mask"])
            )
            regret_events += len(info["regret_events"])
            if truncated:
                break
        levels_array = np.array(levels)
        empties.append(float(np.mean(levels_array == 0)))
        fulls.append(float(np.mean(levels_array == env._pool_sim.capacity_keys)))
        regrets.append(regret_events)
        queue_peaks.append(max(queue_lengths))
        queue_ends.append(queue_lengths[-1])
        overflows.append(env.pool_overflow_keys)

    return {
        "empty_fraction": float(np.mean(empties)),
        "full_fraction": float(np.mean(fulls)),
        "regret_events": float(np.mean(regrets)),
        "queue_peak": float(np.mean(queue_peaks)),
        "queue_end": float(np.mean(queue_ends)),
        "overflow_keys": float(np.mean(overflows)),
    }


def test_scarcity_ratio_in_target_band():
    """Spec §S1 test 11 -- the test that protects the thesis.

    §S1 test 11 exists to guarantee one property: **the pool binds, but does
    not collapse.** If it never binds, every policy looks identical and the
    DQN ties the tuned threshold in week 3. If it binds so hard that demand
    permanently exceeds supply, every policy drowns equally and the
    differences between them are noise on top of a huge constant starvation
    cost -- which is just as fatal and much harder to notice.

    This test asserted a *hardcoded* demand figure (0.043 and 0.98
    keys/step, "measured directly on 2026-08-15") divided by the live refill
    rate, until 2026-08-19. That is not a measurement: the numerator was
    frozen while the denominator tracked the config, so the ratio drifted
    away from the environment it claimed to describe and the test passed
    right through the environment being in permanent deficit -- pool empty
    85% of steps, deferral queue growing monotonically to 303 and never
    draining, and the `starve` reward term at 99.5% of total reward
    magnitude. It now measures the environment it is testing.

    The behavioural signature asserted below is the one SMARTKEYNET_BUILD_SPEC
    §S7 predicts for a correctly-sized link, and each half is load-bearing:
      - always-hybrid must drain the pool (Demo Beat 2 needs a villain);
      - always-PQC must waste the link instead (high overflow, ~no regret),
        which is what makes overflow "a free extra axis of evidence";
      - neither may accumulate an unbounded queue on the benign scenario.
    """
    from agents.baselines import AlwaysHybridPolicy, AlwaysPQCPolicy

    villain = _binding_diagnostics(AlwaysHybridPolicy(), "S1")
    hoarder = _binding_diagnostics(AlwaysPQCPolicy(), "S1")

    print("\nS1 always-hybrid:", {k: round(v, 3) for k, v in villain.items()})
    print("S1 always-PQC   :", {k: round(v, 3) for k, v in hoarder.items()})

    # 1. The villain must actually exhaust the pool.
    assert villain["empty_fraction"] >= 0.20, (
        f"always-hybrid leaves the pool empty only {villain['empty_fraction']:.1%} of steps -- "
        "the pool does not bind, so no policy can misbudget and Gate W3 is unwinnable. "
        "See SMARTKEYNET_BUILD_SPEC.md §7.1 fix A."
    )
    assert villain["regret_events"] > 0, "always-hybrid caused no regret: no scarcity at all"

    # 2. But the queue must stay bounded -- scarcity, not permanent deficit.
    assert villain["queue_end"] <= 0.5 * villain["queue_peak"] + 20, (
        f"deferral queue ends at {villain['queue_end']:.0f} against a peak of "
        f"{villain['queue_peak']:.0f}: the backlog never drains, so the environment is in "
        "permanent deficit rather than intermittent scarcity. Raise supply (§7.1 fix A)."
    )

    # 3. The hoarder must waste the link rather than starve it. This is the
    #    contrast that makes the two metrics independent evidence.
    assert hoarder["overflow_keys"] > 0, (
        "always-PQC wasted no key material -- overflow cannot discriminate policies, "
        "and §S1's 'free extra axis of evidence' is unavailable"
    )
    assert hoarder["regret_events"] < villain["regret_events"], (
        "always-PQC caused at least as much regret as always-hybrid, which inverts the "
        "expected ordering (§S7 tests 2-3)"
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


# ---------------------------------------------------------------------------
# Property-based invariants (SMARTKEYNET_BUILD_SPEC.md §S1 tests 9-10)
#
# These are specified as property-based rather than example-based because the
# invariants must hold for *any* interleaving of refills and draws, and the
# interesting failures live at boundaries (a draw that exactly empties the
# pool, a refill that exactly fills it) that hand-written cases miss.
# ---------------------------------------------------------------------------

REFILL_OR_DRAW = st.one_of(
    st.tuples(st.just("refill"), st.floats(min_value=0.0, max_value=5.0)),
    st.tuples(st.just("draw"), st.integers(min_value=0, max_value=8)),
)


@given(operations=st.lists(REFILL_OR_DRAW, min_size=1, max_size=200))
@settings(max_examples=200, deadline=None)
def test_level_invariant(operations):
    """§S1 test 9: for any random sequence of refills and draws,
    `0 <= level <= capacity` AND `level == sum(batch.keys_remaining)`.

    The second half is the one that matters: it says the lineage deque and
    the integer counter can never disagree. If they drift apart, the
    attribution ledger starts describing key material that was never
    actually spent, and §S2's "bits attributed <= bits spent" invariant
    becomes unfalsifiable rather than true.
    """
    pool = make_pool(capacity=20 * KEY_BITS, pairs=[], initial_fill_frac=0.25)

    for step_index, (kind, amount) in enumerate(operations, start=1):
        if kind == "refill":
            pool.refill(skr_kbps=amount, step_seconds=1.0, now=step_index)
        else:
            pool.draw_keys(amount)

        assert 0 <= pool.level <= pool.capacity_keys
        assert pool.level == sum(batch.keys_remaining for batch in pool.batches())
        # No empty batch may linger: a drained batch is popped, so every
        # batch in the deque carries real material.
        assert all(batch.keys_remaining > 0 for batch in pool.batches())


@given(
    query_keys=st.integers(min_value=-3, max_value=30),
    prior_refills=st.lists(st.floats(min_value=0.0, max_value=3.0), max_size=20),
)
@settings(max_examples=200, deadline=None)
def test_peek_is_pure(query_keys, prior_refills):
    """§S1 test 10: `peek_can_cover` never changes level or lineage.

    Compared by deep snapshot, not by eyeballing the method: masking calls
    this on every single step, so a peek with a side effect would corrupt
    the pool once per decision and look like a mysterious drift.
    """
    pool = make_pool(capacity=20 * KEY_BITS, pairs=[], initial_fill_frac=0.5)
    for step_index, skr in enumerate(prior_refills, start=1):
        pool.refill(skr_kbps=skr, step_seconds=1.0, now=step_index)

    before = (pool.level, pool.batches(), pool.overflow_keys_total, pool._fractional_carry)
    pool.peek_can_cover(query_keys)
    after = (pool.level, pool.batches(), pool.overflow_keys_total, pool._fractional_carry)
    assert before == after


@given(skr_values=st.lists(st.floats(min_value=0.0, max_value=20.0), min_size=1, max_size=300))
@settings(max_examples=100, deadline=None)
def test_refill_conserves_bits_property(skr_values):
    """Bit conservation as a property, over arbitrary SKR sequences.

    The example-based `test_refill_conserves_bits` uses one awkward rate;
    this asserts the carry works for any sequence, including ones that
    alternate between zero and large rates (where a naive carry
    implementation loses or double-counts the remainder).
    """
    pool = make_pool(capacity=10**9, pairs=[], initial_fill_frac=0.0)
    for step_index, skr in enumerate(skr_values, start=1):
        pool.refill(skr_kbps=skr, step_seconds=1.0, now=step_index)

    supplied_bits = sum(skr_values) * 1000.0
    banked_bits = pool.level * KEY_BITS + pool._fractional_carry * KEY_BITS
    assert banked_bits == pytest.approx(supplied_bits, abs=KEY_BITS)
    assert 0.0 <= pool._fractional_carry < 1.0


@given(draw_sizes=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=40))
@settings(max_examples=150, deadline=None)
def test_draw_lineage_accounts_for_exactly_the_keys_taken(draw_sizes):
    """Every draw's lineage must sum to exactly the keys it removed.

    This is the pool-side half of §S2 test 7 (attribution conservation):
    if lineage over- or under-reports, attribution inherits the error and
    the regret ledger silently stops adding up.
    """
    pool = make_pool(capacity=500 * KEY_BITS, pairs=[], initial_fill_frac=0.0)
    for step_index in range(1, 60):
        pool.refill(skr_kbps=2 * KEY_BITS / 1000.0, step_seconds=1.0, now=step_index)

    for keys in draw_sizes:
        level_before = pool.level
        result = pool.draw_keys(keys)
        if not result.ok:
            assert pool.level == level_before  # a refused draw changes nothing
            continue
        assert sum(taken for _, taken in result.lineage) == keys
        assert pool.level == level_before - keys
        # FIFO: batch ids in a lineage are strictly increasing (oldest first)
        batch_ids = [batch_id for batch_id, _ in result.lineage]
        assert batch_ids == sorted(batch_ids)


# ---------------------------------------------------------------------------
# SKR process: log-space OU (spec §S1) and trace mode with the 70/30 split
# ---------------------------------------------------------------------------


def test_ou_process_preserves_the_configured_mean():
    """`mean_skr_kbps` must actually be the process mean.

    A stationary log-space OU process has `E[exp(x)] = exp(mu + sigma^2/(4*theta))`,
    so taking `mu = log(mean_skr_kbps)` literally -- as the spec's formula does --
    overshoots. That would be a quiet disaster here specifically: the entire
    scarcity calibration (§11.2) is a ratio against this supply figure, so a
    process whose mean disagrees with its own config key would invalidate it.
    """
    trace = SyntheticSKRQBERTrace(n_steps=100_000, mean_skr_kbps=0.10, seed=0)
    values = np.array([skr for skr, _qber in trace])
    # Within 5%: the small shortfall is the reconciliation gate, which bites
    # whenever QBER noise pushes above baseline, not the OU correction.
    assert values.mean() == pytest.approx(0.10, rel=0.05)
    assert (values > 0.0).all()  # log space cannot go negative, with no clip


def test_ou_process_is_autocorrelated_unlike_iid():
    """Mean reversion is the point: an i.i.d. sequence has nothing for a
    forecaster to learn, so the pool head could never beat persistence on it.
    This process was i.i.d. Gaussian until 2026-08-19."""
    trace = SyntheticSKRQBERTrace(n_steps=50_000, mean_skr_kbps=0.10, seed=1)
    values = np.array([skr for skr, _qber in trace])
    lag_one = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    assert lag_one > 0.5, f"lag-1 autocorrelation {lag_one:.3f} is too low to forecast"


def test_ou_process_is_deterministic_under_seed():
    first = list(SyntheticSKRQBERTrace(n_steps=500, seed=7))
    second = list(SyntheticSKRQBERTrace(n_steps=500, seed=7))
    assert first == second


def _write_trace_csv(path, n_rows: int = 1000) -> None:
    import csv as _csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("# SYNTHETIC test trace\n")
        writer = _csv.writer(handle)
        writer.writerow(["step", "skr_kbps", "qber"])
        for step in range(n_rows):
            writer.writerow([step, f"{0.10 + step * 1e-6:.8f}", "0.02"])


def test_trace_split_is_seventy_thirty_and_disjoint(tmp_path):
    """§S1: "first 70% for training scenarios, last 30% reserved for
    evaluation. Reusing the same trace segment for train and eval is a silent
    leak that a reviewer will find."

    The SKR values here increase monotonically, so the two segments are
    trivially distinguishable and an overlap would be visible rather than
    merely improbable.
    """
    from env.pool_sim import TraceSKRQBERSource

    csv_path = tmp_path / "skr_qber.csv"
    _write_trace_csv(csv_path, n_rows=1000)

    train = TraceSKRQBERSource(trace_path=csv_path, n_steps=10, split="train", seed=0)
    evaluation = TraceSKRQBERSource(trace_path=csv_path, n_steps=10, split="eval", seed=0)

    assert train.segment_length == 700
    assert evaluation.segment_length == 300
    assert max(skr for skr, _ in train._segment) < min(skr for skr, _ in evaluation._segment)


def test_trace_cycles_within_its_own_split(tmp_path):
    """A long episode must wrap inside its split, never spill into the other."""
    from env.pool_sim import TraceSKRQBERSource

    csv_path = tmp_path / "skr_qber.csv"
    _write_trace_csv(csv_path, n_rows=1000)

    train = TraceSKRQBERSource(trace_path=csv_path, n_steps=5000, split="train", seed=3)
    highest_train_skr = max(skr for skr, _ in train._segment)
    for skr, _qber in train:
        assert skr <= highest_train_skr


def test_trace_rejects_an_unknown_split(tmp_path):
    from env.pool_sim import TraceSKRQBERSource

    csv_path = tmp_path / "skr_qber.csv"
    _write_trace_csv(csv_path, n_rows=100)
    with pytest.raises(ValueError, match="split"):
        TraceSKRQBERSource(trace_path=csv_path, n_steps=10, split="both")


def test_trace_missing_file_says_how_to_generate_one(tmp_path):
    from env.pool_sim import TraceSKRQBERSource

    with pytest.raises(FileNotFoundError, match="get_data"):
        TraceSKRQBERSource(trace_path=tmp_path / "absent.csv", n_steps=10)
