"""
forecaster/model.py

Dual-head LSTM forecaster: threat head + pool head (PLAN.md Addition A).
Owned by Person A (split.md §1).

Shared LSTM encoder over a W=64-timestep window, with two heads:
  - threat head: class distribution over threat posture, next k=5
    steps -> feeds `env/masking.py`'s PolicyTable (floors).
  - pool head: `pool_level_hat` / `skr_mean_hat` / `hybrid_demand_hat`
    at H in {10, 25, 50} -> feeds the DQN state only, never the policy
    table (Hard Rule 2).

Frozen during DQN training -- no end-to-end gradients (PLAN.md
Addition A).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from env.contracts import (
    ForecastObservation,
    ForecastProvider,
    PoolForecast,
    ThreatForecast,
)

WINDOW = 64
HORIZONS = (10, 25, 50)
K_THREAT_STEPS = 5


class SmartKeyForecaster(nn.Module):
    """Shared LSTM encoder with a threat head and a pool head."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`window`: shape `(batch, WINDOW, input_dim)`.

        Returns `(threat_logits, pool_head_outputs)` -- see
        `forecaster/train.py` for the exact per-head shapes and losses.
        """
        raise NotImplementedError


class LSTMForecastProvider(ForecastProvider):
    """`ForecastProvider` implementation wrapping a trained
    `SmartKeyForecaster` checkpoint.

    Selected via `configs/default.yaml`'s `use_foresight: lstm`.
    """

    def __init__(self, checkpoint_path: str) -> None:
        raise NotImplementedError

    def update(self, observation: ForecastObservation) -> None:
        raise NotImplementedError

    def get_threat_forecast(self) -> ThreatForecast:
        raise NotImplementedError

    def get_pool_forecast(self) -> PoolForecast:
        raise NotImplementedError
