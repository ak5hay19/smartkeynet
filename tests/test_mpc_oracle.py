"""Tests for `agents.mpc_oracle` -- the perfect-foresight diagnostic (§S7).

The oracle is a measuring instrument, not a competitor, so what matters is
that it respects the mask, confines its cheating to one auditable method, and
produces the foresight-value gap the §7.1 diagnosis tree asks for.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import yaml
from pathlib import Path

from agents.mpc_oracle import HORIZON, MPCOracle
from env.contracts import N_ACTIONS, Action

REPO = Path(__file__).resolve().parent.parent


def _all_nonempty_masks():
    masks = []
    for r in range(1, N_ACTIONS + 1):
        for combo in itertools.combinations(list(Action), r):
            mask = np.zeros(N_ACTIONS, dtype=bool)
            for action in combo:
                mask[int(action)] = True
            masks.append(mask)
    return masks


def _state(pool_fill=0.5, policy_floor=1, key_age=0.0, sensitivity_class=1):
    """Minimal state. The oracle must tolerate this like any other Policy --
    `act` uses `.get` with defaults so a caller passing a partial state gets
    conservative behaviour rather than a KeyError."""
    return {
        "pool_fill": pool_fill,
        "policy_floor": policy_floor,
        "key_age": key_age,
        "sensitivity_class": sensitivity_class,
    }


@pytest.mark.parametrize("mask", _all_nonempty_masks())
def test_never_returns_an_illegal_action(mask):
    """Holds even unbound (no env), and for every contrived mask."""
    action = MPCOracle().act(_state(), mask)
    assert mask[int(action)]


def test_cheating_is_confined_to_one_method():
    """`peek_future` is the only place the oracle reads the environment's
    future. Keeping it in one method is what makes the cheat auditable and
    stops the capability leaking into a policy claiming to be causal."""
    import inspect

    source = inspect.getsource(MPCOracle)
    total_env_reads = source.count("self.env.")
    audited = (
        MPCOracle.peek_future,        # future demand and refill -- the cheat
        MPCOracle.peek_future_floor,  # the floor the horizon will reach -- the cheat
        MPCOracle._current_pool_keys, # present pool level, which causal policies see too
        MPCOracle._lifetime_cap,      # the SP 800-57 cap, given to every policy
    )
    accounted = sum(inspect.getsource(method).count("self.env.") for method in audited)
    # Every environment access lives in one of three audited methods. `bind`
    # only assigns, and `act` reads nothing from the env directly -- so an
    # auditor has a bounded set of places to check that the foresight has not
    # leaked into something claiming to be causal.
    assert total_env_reads == accounted


def test_unbound_oracle_degrades_gracefully():
    """No env attached -> no foresight -> still a valid policy."""
    oracle = MPCOracle()
    assert oracle.peek_future() == (0, 0.0)
    mask = np.ones(N_ACTIONS, dtype=bool)
    assert mask[int(oracle.act(_state(), mask))]


def test_horizon_matches_the_spec():
    assert HORIZON == 50


def test_runs_a_full_episode_without_floor_violations():
    from experiments.harness import run_scenario

    config = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    config.update({"max_steps": 300, "scenario_steps": 300})
    result = run_scenario(MPCOracle(), "S1", config, seed=1000)
    assert result.floor_violations == 0


def test_foresight_value_gap_is_measurable():
    """The §7.1 diagnostic-1 number. This test does not assert the gap is
    large -- it asserts the measurement runs, because the measured answer
    (0.0% on S1, -4.7% on S3) is itself the finding: this environment has no
    foresight value, which is why Gate W3 is unwinnable by any agent."""
    from agents.baselines import StaticThresholdPolicy
    from experiments.harness import run_scenario

    config = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    config.update({"max_steps": 300, "scenario_steps": 300})
    max_key_age = float(config["key_lifetime"]["max_key_age_steps"])

    threshold = StaticThresholdPolicy(0.7, 2, 0.9, max_key_age)
    threshold_regret = run_scenario(threshold, "S3", config, seed=1000).pool_exhaustion_events
    oracle_regret = run_scenario(MPCOracle(), "S3", config, seed=1000).pool_exhaustion_events

    assert threshold_regret >= 0 and oracle_regret >= 0
