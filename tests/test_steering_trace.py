"""Behavioral tests for `attack.steering_trace` (PLAN.md §5 S5 -- the
headline contribution, never cut).
"""

from __future__ import annotations

import numpy as np
import pytest

from attack.steering_trace import (
    SteeringTraceConfig,
    default_honest_trace,
    dose_response_sweep,
    generate_steering_trace,
    suppression_ratio,
)


def test_config_rejects_a_dose_outside_the_unit_interval():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            SteeringTraceConfig(dose=bad, duration_steps=100)


def test_config_rejects_a_non_positive_duration():
    with pytest.raises(ValueError):
        SteeringTraceConfig(dose=0.5, duration_steps=0)


def test_dose_zero_is_a_genuine_no_op():
    """The control arm of the sweep has to be exactly the honest trace,
    or every comparison against it is measuring the generator's noise."""
    honest = default_honest_trace(500)
    attacked = generate_steering_trace(
        SteeringTraceConfig(dose=0.0, duration_steps=500, seed=0), honest_trace=honest
    )
    assert attacked == pytest.approx(honest)


def test_suppression_increases_monotonically_with_dose():
    honest = default_honest_trace(500)
    suppressions = []
    for dose in (0.0, 0.25, 0.5, 0.75, 1.0):
        attacked = generate_steering_trace(
            SteeringTraceConfig(dose=dose, duration_steps=500, seed=0), honest_trace=honest
        )
        suppressions.append(suppression_ratio(honest, attacked))
    assert suppressions == sorted(suppressions)
    assert suppressions[0] == pytest.approx(0.0)
    assert suppressions[-1] > 0.9


def test_attack_only_ever_suppresses_never_amplifies():
    """The threat model is a *quiet* adversary. An amplifying trace would
    make both agents serve stronger keys, which is costly but not a
    security failure -- and it is not what Hard Rule 2 is written
    against."""
    honest = default_honest_trace(500)
    for dose in (0.25, 0.5, 0.75, 1.0):
        attacked = generate_steering_trace(
            SteeringTraceConfig(dose=dose, duration_steps=500, seed=1), honest_trace=honest
        )
        assert np.mean(attacked) < np.mean(honest)


def test_trace_never_goes_below_the_benign_level():
    """An adversary gains nothing from a negative threat level, and no
    real sensor could report one."""
    honest = default_honest_trace(300)
    for dose in (0.0, 0.5, 1.0):
        attacked = generate_steering_trace(
            SteeringTraceConfig(dose=dose, duration_steps=300, seed=2), honest_trace=honest
        )
        assert min(attacked) >= 0.0


def test_trace_is_seed_reproducible_and_the_right_length():
    config = SteeringTraceConfig(dose=0.6, duration_steps=250, seed=11)
    a = generate_steering_trace(config)
    b = generate_steering_trace(config)
    assert a == b
    assert len(a) == 250


def test_a_short_honest_trace_is_held_at_its_last_value():
    attacked = generate_steering_trace(
        SteeringTraceConfig(dose=0.0, duration_steps=100, seed=0), honest_trace=[2.0, 2.0]
    )
    assert len(attacked) == 100
    assert attacked[-1] == pytest.approx(2.0)


def test_default_honest_trace_starts_calm_and_ramps_up():
    """An attack on an already-calm signal has nothing to suppress,
    which would make the dose axis meaningless."""
    trace = default_honest_trace(500)
    assert trace[0] == pytest.approx(0.0)
    assert trace[-1] > 3.0
    assert trace == sorted(trace)  # monotone ramp, never falls back


def test_dose_response_sweep_holds_the_honest_trace_fixed_across_doses():
    """The only thing that may vary across the sweep is the adversary's
    strength."""
    sweep = dose_response_sweep([0.0, 0.5, 1.0], SteeringTraceConfig(dose=0.0, duration_steps=200, seed=3))
    assert set(sweep) == {0.0, 0.5, 1.0}
    assert all(len(trace) == 200 for trace in sweep.values())
    assert sweep[0.0] == pytest.approx(default_honest_trace(200))


def test_suppression_ratio_is_zero_for_an_all_calm_honest_trace():
    assert suppression_ratio([0.0] * 10, [0.0] * 10) == 0.0
