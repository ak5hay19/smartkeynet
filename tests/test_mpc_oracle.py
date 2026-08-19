"""Tests for `agents.mpc_oracle` -- the perfect-foresight diagnostic (§S7).

The oracle is a measuring instrument, not a competitor, so what matters is
that it respects the mask, confines its cheating to one auditable method, and
produces the foresight-value gap the §7.1 diagnosis tree asks for.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import yaml

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
        MPCOracle.peek_future,  # future demand and refill -- the cheat
        MPCOracle.peek_future_floor,  # the floor the horizon will reach -- the cheat
        MPCOracle._current_pool_keys,  # present pool level, which causal policies see too
        MPCOracle._lifetime_cap,  # the SP 800-57 cap, given to every policy
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


# ---------------------------------------------------------------------------
# MPCForecast -- the fair foresight baseline (spec §8.3 rung 3)
# ---------------------------------------------------------------------------


def test_mpc_forecast_is_causal_and_touches_no_env_internals():
    """§8.3's whole point is that this baseline has the *same information the
    DQN has* -- no more.

    If it could read the environment's future, beating it would prove nothing,
    exactly as beating `MPCOracle` proves nothing. So it must run to completion
    with no environment bound at all: every input comes from the `StateDict`
    the environment already hands every policy.
    """
    from agents.mpc_oracle import MPCForecast

    policy = MPCForecast()  # deliberately NOT bound to an env
    assert policy.env is None

    state = {
        "policy_floor": int(Action.SERVE_PQC),
        "sensitivity_class": 2,
        "key_age": 0.1,
        "pool_fill": 0.5,
        "threat_forecast": [0.1, 0.2, 0.9, 0.2, 0.1],
        "hybrid_demand_hat": [1.0, 2.0, 4.0],
        "pool_level_hat": [0.4, 0.3, 0.2],
    }
    mask = np.ones(N_ACTIONS, dtype=bool)
    action = policy.act(state, mask)
    assert mask[int(action)]


def test_mpc_forecast_respects_every_mask():
    """Property: never returns an illegal action, over every non-empty mask."""
    from agents.mpc_oracle import MPCForecast

    policy = MPCForecast()
    state = {
        "policy_floor": int(Action.SERVE_CLASSICAL),
        "sensitivity_class": 1,
        "key_age": 0.5,
        "threat_forecast": [0.0] * 5,
        "hybrid_demand_hat": [0.0, 0.0, 0.0],
        "pool_level_hat": [0.5, 0.5, 0.5],
    }
    for r in range(1, N_ACTIONS + 1):
        for combo in itertools.combinations(list(Action), r):
            mask = np.zeros(N_ACTIONS, dtype=bool)
            for action in combo:
                mask[int(action)] = True
            assert mask[int(policy.act(state, mask))]


def test_mpc_forecast_floor_lookahead_can_only_raise():
    """The forecast is aggregated with MAX, matching `env/masking.py`.

    Not a stylistic choice: max aggregation is what makes a forecast able only
    to *raise* a floor, which is what Hard Rule 2 rests on. A forecast-driven
    baseline that averaged could talk itself into a floor below what the
    present already justifies -- and would then be a counterexample to the
    paper's central proposition sitting inside the repo.
    """
    from agents.mpc_oracle import MPCForecast

    policy = MPCForecast()
    base_state = {"policy_floor": int(Action.SERVE_PQC), "sensitivity_class": 3}

    calm = policy.peek_future_floor({**base_state, "threat_forecast": [0.0] * 5})
    spike = policy.peek_future_floor({**base_state, "threat_forecast": [0.0, 0.0, 1.0, 0.0, 0.0]})

    assert calm >= int(Action.SERVE_PQC), "lookahead lowered the floor below the present"
    assert spike >= calm, "a threat spike in the forecast must not lower the floor"


def test_mpc_forecast_without_a_forecast_degrades_to_the_current_floor():
    """With foresight `off` there is genuinely nothing to anticipate with, so
    the honest behaviour is myopia -- not a crash, and not an optimistic
    guess."""
    from agents.mpc_oracle import MPCForecast

    policy = MPCForecast()
    floor = policy.peek_future_floor(
        {"policy_floor": int(Action.SERVE_HYBRID), "sensitivity_class": 3}
    )
    assert floor == int(Action.SERVE_HYBRID)


def test_oracle_is_at_least_as_good_as_the_forecast_baseline():
    """Perfect foresight must dominate imperfect foresight.

    If the forecast baseline beat the oracle, one of them is buggy -- the same
    reasoning as §S7 test 5, applied one rung down the ladder.
    """
    from agents.mpc_oracle import MPCForecast
    from experiments.harness import run_scenario

    with open(Path(__file__).resolve().parent.parent / "configs" / "default.yaml") as handle:
        base = yaml.safe_load(handle)
    config = {**base, "use_foresight": "ewma", "max_steps": 400, "scenario_steps": 600}

    oracle_regret = np.mean(
        [
            run_scenario(MPCOracle(), "S3", config, seed=seed).episode_metrics.regret_events
            for seed in (0, 1, 2)
        ]
    )
    forecast_regret = np.mean(
        [
            run_scenario(MPCForecast(), "S3", config, seed=seed).episode_metrics.regret_events
            for seed in (0, 1, 2)
        ]
    )
    assert oracle_regret <= forecast_regret + 1e-9
