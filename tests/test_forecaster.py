"""Behavioral tests for the dual-head LSTM forecaster (PLAN.md
Addition A).

The properties under test are the ones Addition A names: correct output
shapes per head, interchangeability with the EWMA fallback, unchanged
state length under each flag, and -- the load-bearing one -- **no
gradient flow from the DQN's loss into the forecaster**.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.dqn import _FORECAST_STATE_DIM, _OFF_STATE_DIM, flatten_state
from env.contracts import ForecastObservation, ForecastProvider, ThreatPosture
from env.forecast_provider import MovingAverageForecaster
from forecaster.model import (
    HORIZONS,
    N_FEATURES,
    N_POOL_OUTPUTS,
    THREAT_HORIZON_STEPS,
    WINDOW,
    LSTMForecastProvider,
    SmartKeyForecaster,
    observation_to_features,
)


def make_observation(qber: float = 0.02, skr: float = 0.025) -> ForecastObservation:
    return ForecastObservation(
        qber=qber,
        skr=skr,
        pool_fill=0.5,
        arrivals_per_class=[1, 0, 2, 0],
        hybrid_serves=1,
        threat_features=[qber, 0.1, 0.0],
    )


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_feature_vector_matches_declared_width():
    assert len(observation_to_features(make_observation())) == N_FEATURES


def test_feature_order_is_stable_against_class_count_changes():
    """`arrivals_per_class` is padded/truncated to 4, so a config with a
    different class count cannot silently shift every later feature."""
    short = make_observation()
    short["arrivals_per_class"] = [1, 2]
    long = make_observation()
    long["arrivals_per_class"] = [1, 2, 3, 4, 5, 6]

    for observation in (short, long):
        assert len(observation_to_features(observation)) == N_FEATURES
    # hybrid_serves stays at its fixed index either way -- it sits just
    # before the reserved threat-feature block
    from forecaster.model import N_THREAT_FEATURES_IN_MODEL

    hybrid_index = N_FEATURES - N_THREAT_FEATURES_IN_MODEL - 1
    assert observation_to_features(short)[hybrid_index] == 1.0
    assert observation_to_features(long)[hybrid_index] == 1.0


def test_threat_features_are_padded_to_a_fixed_width():
    """Both sources share one input layer: RT-IoT2022 supplies 9 threat
    features, the synthetic fallback 3, and the model sees the reserved
    width either way."""
    from forecaster.model import N_THREAT_FEATURES_IN_MODEL

    synthetic = make_observation()
    synthetic["threat_features"] = [0.02, 0.1, 0.5]  # 3, the fallback
    real = make_observation()
    real["threat_features"] = [0.1] * 9  # 9, RT-IoT2022

    for observation in (synthetic, real):
        assert len(observation_to_features(observation)) == N_FEATURES

    # the short vector is zero-padded, not silently shifted
    padded = observation_to_features(synthetic)[-N_THREAT_FEATURES_IN_MODEL:]
    assert padded[:3] == [0.02, 0.1, 0.5]
    assert padded[3:] == [0.0] * (N_THREAT_FEATURES_IN_MODEL - 3)


def test_model_output_shapes_per_head():
    model = SmartKeyForecaster()
    windows = torch.zeros(7, WINDOW, N_FEATURES)
    threat_logits, pool_outputs = model(windows)

    assert threat_logits.shape == (7, THREAT_HORIZON_STEPS, len(ThreatPosture))
    assert pool_outputs.shape == (7, N_POOL_OUTPUTS)


def test_provider_returns_contract_shaped_forecasts():
    provider = LSTMForecastProvider(SmartKeyForecaster())
    provider.update(make_observation())

    threat = provider.get_threat_forecast()
    pool = provider.get_pool_forecast()

    assert len(threat.posture_probs) == len(ThreatPosture)
    assert len(threat.horizon_scores) == THREAT_HORIZON_STEPS
    assert 0.0 <= threat.threat_score <= 1.0
    assert sum(threat.posture_probs) == pytest.approx(1.0)

    assert len(pool.pool_level_hat) == len(HORIZONS)
    assert len(pool.skr_mean_hat) == len(HORIZONS)
    assert len(pool.hybrid_demand_hat) == len(HORIZONS)


# ---------------------------------------------------------------------------
# Interchangeability with the EWMA fallback
# ---------------------------------------------------------------------------


def test_lstm_provider_satisfies_the_forecast_provider_interface():
    provider = LSTMForecastProvider(SmartKeyForecaster())
    assert isinstance(provider, ForecastProvider)


def test_both_providers_are_well_formed_before_any_update():
    """A fresh instance must return sensible output rather than
    crashing -- the environment builds a state before the first
    observation exists."""
    for provider in (MovingAverageForecaster(), LSTMForecastProvider(SmartKeyForecaster())):
        threat = provider.get_threat_forecast()
        pool = provider.get_pool_forecast()
        assert len(threat.posture_probs) == len(ThreatPosture)
        assert len(pool.pool_level_hat) == len(HORIZONS)


def test_state_length_is_identical_under_ewma_and_lstm():
    """The E-A ablation has to compare forecast *quality*, not input
    dimensionality -- if the two modes produced different state widths
    they would be different MDPs and the comparison would be void."""
    observation = make_observation()

    lengths = set()
    for provider in (MovingAverageForecaster(), LSTMForecastProvider(SmartKeyForecaster())):
        provider.update(observation)
        threat = provider.get_threat_forecast()
        pool = provider.get_pool_forecast()
        state = {
            "threat_score": threat.threat_score,
            "threat_forecast": threat.horizon_scores,
            "posture_probs": list(threat.posture_probs) + [0.0],
            "qber": 0.2,
            "skr": 1.0,
            "pool_fill": 0.5,
            "arrival_rate": 1.0,
            "load": 0.1,
            "avg_latency": 0.01,
            "key_age": 0.02,
            "key_type_onehot": [0.0, 0.0, 1.0, 0.0],
            "request_class_onehot": [0.0, 0.0, 1.0, 0.0],
            "floor_onehot": [0.0, 1.0, 0.0, 0.0],
            "pqc_capable": 1.0,
            "queue_len_norm": 0.0,
            "queue_head_wait_norm": 0.0,
            "steps_since_rekey_norm": 0.02,
            "sensitivity_class": 2,
            "policy_floor": 1,
            "pool_level_hat": pool.pool_level_hat,
            "skr_mean_hat": pool.skr_mean_hat,
            "skr_trend": 0.0,
            "hybrid_demand_hat": pool.hybrid_demand_hat,
            "regret_event_recent": False,
        }
        lengths.add(flatten_state(state, has_forecast=True).shape[0])

    assert lengths == {_FORECAST_STATE_DIM}
    assert _FORECAST_STATE_DIM != _OFF_STATE_DIM


# ---------------------------------------------------------------------------
# Frozen during DQN training (Addition A's named unit test)
# ---------------------------------------------------------------------------


def test_provider_parameters_are_frozen():
    model = SmartKeyForecaster()
    assert any(p.requires_grad for p in model.parameters())  # trainable before wrapping

    provider = LSTMForecastProvider(model)
    assert not any(p.requires_grad for p in provider._model.parameters())
    assert not provider._model.training


def test_forecast_outputs_carry_no_gradient():
    """ "No gradient flow from DQN loss into forecaster" -- the outputs
    are plain floats, so the gradient path does not merely go unused,
    it does not exist."""
    provider = LSTMForecastProvider(SmartKeyForecaster())
    provider.update(make_observation())

    pool = provider.get_pool_forecast()
    threat = provider.get_threat_forecast()
    for value in (*pool.pool_level_hat, *pool.skr_mean_hat, *threat.horizon_scores):
        assert isinstance(value, float)
        assert not isinstance(value, torch.Tensor)


def test_padding_lets_the_provider_run_before_the_window_fills():
    provider = LSTMForecastProvider(SmartKeyForecaster())
    for _ in range(3):  # far fewer than WINDOW
        provider.update(make_observation())
    assert len(provider.get_pool_forecast().pool_level_hat) == len(HORIZONS)


def test_checkpoint_round_trip_preserves_outputs():
    model = SmartKeyForecaster()
    path = Path("checkpoints/_test_forecaster.pt")
    model.save(path)
    try:
        reloaded = SmartKeyForecaster.load(path)
        windows = torch.randn(2, WINDOW, N_FEATURES)
        with torch.no_grad():
            a_threat, a_pool = model(windows)
            b_threat, b_pool = reloaded(windows)
        assert torch.allclose(a_threat, b_threat)
        assert torch.allclose(a_pool, b_pool)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Hard Rule 2: the pool head must never reach the policy table
# ---------------------------------------------------------------------------


def test_masking_never_imports_the_forecaster():
    """A floor that depended on a learned pool regression would be a
    soft-security design -- exactly what this project argues against."""
    source = Path(__file__).resolve().parent.parent / "env" / "masking.py"
    tree = ast.parse(source.read_text())

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert not any(name.startswith("forecaster") for name in imported)
    assert not any("forecast" in name for name in imported)


def test_pool_forecast_is_absent_from_the_policy_table_signature():
    """`PolicyTable.floor` takes a class, a posture and a capability
    flag -- and nothing forecast-derived."""
    import inspect

    from env.masking import PolicyTable

    parameters = set(inspect.signature(PolicyTable.floor).parameters)
    assert parameters == {"self", "sensitivity_class", "threat_posture", "pqc_capable"}
