"""
env/forecast_provider.py

EWMA fallback forecaster (PLAN.md Addition A). Owned by Person A
(split.md §1).

`MovingAverageForecaster` implements the `ForecastProvider` interface
frozen in `env/contracts.py` so `env/environment.py` can run in month 1
before `forecaster.model.LSTMForecastProvider` exists. Selected via
`configs/default.yaml`'s `use_foresight: ewma`.
"""

from __future__ import annotations

from env.contracts import (
    ForecastObservation,
    ForecastProvider,
    PoolForecast,
    ThreatForecast,
)


class MovingAverageForecaster(ForecastProvider):
    """EWMA-of-SKR-and-arrivals fallback forecaster (Addition A).

    Deliberately simple: no learned parameters. Exists so the env, the
    DQN, and the E-A ablation's `off` vs `ewma` comparison all have a
    working, non-LSTM forecast source from week 1 (PLAN.md Addition A:
    "This means the env is buildable in month 1 before the LSTM
    exists.").
    """

    def __init__(self, alpha: float = 0.3) -> None:
        raise NotImplementedError

    def update(self, observation: ForecastObservation) -> None:
        raise NotImplementedError

    def get_threat_forecast(self) -> ThreatForecast:
        raise NotImplementedError

    def get_pool_forecast(self) -> PoolForecast:
        raise NotImplementedError
