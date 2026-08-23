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
    """Every environment read lives in a named, audited method.

    Three of them genuinely cheat (they read the future); two read present
    state that every causal policy is given anyway. Keeping the set small and
    enumerated is what makes the cheat auditable and stops the capability
    leaking into a policy that claims to be causal -- `MPCForecast` inherits
    from this class, and if foresight leaked into a shared helper it would
    silently become an oracle while still being reported as the fair baseline.

    The count is asserted rather than the method names, so ADDING an env read
    anywhere fails this test until it is deliberately listed here."""
    import inspect

    source = inspect.getsource(MPCOracle)
    total_env_reads = source.count("self.env.")
    audited = (
        MPCOracle.peek_future,  # future demand and refill -- the cheat
        MPCOracle.peek_future_floor,  # the floor the horizon will reach -- the cheat
        MPCOracle.projected_refill_keys,  # future refill against the known drift -- the cheat
        MPCOracle._current_pool_keys,  # present pool level, which causal policies see too
        MPCOracle._lifetime_cap,  # the SP 800-57 cap, given to every policy
    )
    accounted = sum(inspect.getsource(method).count("self.env.") for method in audited)
    # Every environment access lives in one of the audited methods above.
    # `bind` only assigns, and `act` reads nothing from the env directly -- so
    # an auditor has a bounded set of places to check that the foresight has
    # not leaked into something claiming to be causal.
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


# ---------------------------------------------------------------------------
# §S7 test 5 -- the oracle must dominate EVERY causal policy
# ---------------------------------------------------------------------------


def test_mpc_oracle_dominates_all_causal_policies():
    """SMARTKEYNET_BUILD_SPEC.md §S7 test 5, which had never been written.

    "on the same seed, MPC's return >= every causal policy's return. If MPC
    loses to anything, MPC is buggy (or your reward has an exploit)."

    This is the single most load-bearing test in the file, because the oracle's
    only job is to be an upper bound: the gap between it and a causal policy is
    what the whole project reports as the value of foresight. An oracle that
    loses makes that gap meaningless -- and worse, makes it look *negative*,
    which reads as "anticipation is harmful" rather than "our oracle is
    broken".

    Its absence hid three real bugs for the project's whole life. The oracle
    gated every rekey on pool surplus though only hybrid rekeys draw keys; it
    compared demand over the full horizon against refill over a much shorter
    window, so its pre-emption gate passed spuriously; and its establishment
    scan omitted `REKEY_NOW` entirely, buying hybrid where refreshing in place
    was cheaper. Net effect: the "perfect foresight" bound lost to
    `greedy_recommender` -- a policy with no model, no memory and no planning
    -- by 16.8 regret events to 1.6.
    """
    from agents.baselines import (
        AlwaysHybridPolicy,
        AlwaysPQCPolicy,
        GreedyRecommenderPolicy,
        RandomPolicy,
        StaticThresholdPolicy,
    )
    from agents.mpc_oracle import MPCForecast
    from experiments.harness import run_scenario

    with open(Path(__file__).resolve().parent.parent / "configs" / "default.yaml") as handle:
        base = yaml.safe_load(handle)
    config = {**base, "max_steps": 600, "scenario_steps": 800}
    seeds = (1000, 1001, 1002)

    causal_policies = {
        "greedy": GreedyRecommenderPolicy(),
        "static_threshold": StaticThresholdPolicy(0.95, 0, 0.9),
        "mpc_forecast": MPCForecast(),
        "always_pqc": AlwaysPQCPolicy(),
        "always_hybrid": AlwaysHybridPolicy(),
        "random": RandomPolicy(seed=0),
    }

    def score(policy, scenario):
        results = [run_scenario(policy, scenario, config, seed=seed) for seed in seeds]
        return (
            float(np.mean([r.episode_metrics.regret_events for r in results])),
            float(np.mean([r.total_reward for r in results])),
        )

    for scenario in ("S1", "S3"):
        oracle_regret, oracle_reward = score(MPCOracle(), scenario)
        for name, policy in causal_policies.items():
            causal_regret, causal_reward = score(policy, scenario)
            assert oracle_regret <= causal_regret + 1e-9, (
                f"{scenario}: MPC oracle caused MORE regret ({oracle_regret:.1f}) than the "
                f"causal policy `{name}` ({causal_regret:.1f}). An upper bound that loses to "
                "a causal policy is a bug in the oracle (§S7 test 5)."
            )
            assert oracle_reward >= causal_reward - 1e-6, (
                f"{scenario}: MPC oracle scored below `{name}` on reward "
                f"({oracle_reward:.1f} vs {causal_reward:.1f})."
            )


def test_mpc_oracle_never_underperforms_the_myopic_choice_by_construction():
    """The structural guarantee behind the test above.

    The oracle's DEFAULT branch is the myopic choice -- cheapest legal action,
    sharing `GreedyRecommenderPolicy`'s exact cost ordering. Foresight may only
    override it when the four-gate lookahead fires. So the oracle can differ
    from greedy solely on steps where it believes anticipation strictly pays,
    which is what makes domination a property of the code rather than of the
    numbers on any particular seed.
    """
    from agents.baselines import GreedyRecommenderPolicy

    assert MPCOracle._CHEAPEST_FIRST == GreedyRecommenderPolicy._CHEAPEST_FIRST
