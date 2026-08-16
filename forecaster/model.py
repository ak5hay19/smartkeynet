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

---------------------------------------------------------------------
Why the two heads must not be wired symmetrically (Hard Rule 2)
---------------------------------------------------------------------
The heads share an encoder but their outputs go to completely different
places, and that asymmetry is load-bearing:

  * The **threat head** feeds the policy table, which sets floors. It
    is inside the security path, so `LSTMForecastProvider` routes it
    through the same posture-probability contract the EWMA fallback
    uses, and the ratchet in `env/masking.py` still applies -- a
    forecast can still only ever *raise* a floor.
  * The **pool head** feeds the DQN's state vector and nothing else. A
    forecast of "the pool will be fine in 50 steps" must never be able
    to relax a floor; that would make the floor a function of a learned
    regression, which is exactly the soft-security design this project
    argues against.

`tests/test_forecaster_model.py` asserts the pool head never reaches
`env/masking.py`.

---------------------------------------------------------------------
Interchangeability with the EWMA fallback
---------------------------------------------------------------------
`LSTMForecastProvider` implements the same `ForecastProvider` interface
as `env.forecast_provider.MovingAverageForecaster`, including its
"works before `update()` has ever been called" behaviour. The
environment picks between them purely by config flag
(`use_foresight: ewma | lstm`), and the flattened state vector is the
same length either way -- which is what makes the E-A ablation a
comparison of forecast *quality* rather than of input dimensionality.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import torch
import torch.nn as nn

from env.contracts import (
    ForecastObservation,
    ForecastProvider,
    PoolForecast,
    ThreatForecast,
    ThreatPosture,
)

WINDOW = 64
"""Input window W, per PLAN.md Addition A ("input window W=64 timesteps")."""

HORIZONS: tuple[int, int, int] = (10, 25, 50)
"""Pool-head horizons H, per `env/contracts.py`'s `PoolForecast`."""

THREAT_HORIZON_STEPS = 5
"""Threat-head horizon k, per Addition A ("next k=5 steps")."""

N_FEATURES = 8
"""Per-timestep input features, in a fixed order that must match
`observation_to_features` exactly:

    0  qber
    1  skr
    2  pool_fill
    3-6  arrivals_per_class[0..3]
    7  hybrid_serves

`threat_features` is deliberately NOT passed through raw: its width is
dataset-dependent (today a 3-vector placeholder, eventually
RT-IoT2022-derived), and baking a variable width into a checkpoint's
input layer would make every saved model incompatible with the next
change to that vector. The threat head instead learns posture from the
*observable dynamics* -- QBER, arrivals, pool behaviour -- which is
also the more defensible modelling claim.
"""

N_POOL_OUTPUTS = 3 * len(HORIZONS)
"""pool_level_hat(3) + skr_mean_hat(3) + hybrid_demand_hat(3)."""


def observation_to_features(observation: ForecastObservation) -> list[float]:
    """Flatten one `ForecastObservation` into the model's fixed input
    order.

    Single source of truth for that order: `dataset.py` and
    `LSTMForecastProvider` both call this rather than re-deriving it,
    so a training set and a live rollout cannot silently disagree about
    which column is which.
    """
    arrivals = list(observation["arrivals_per_class"])
    # pad/truncate defensively so a config with a different class count
    # cannot silently shift every downstream feature index
    arrivals = (arrivals + [0, 0, 0, 0])[:4]

    return [
        float(observation["qber"]),
        float(observation["skr"]),
        float(observation["pool_fill"]),
        *[float(count) for count in arrivals],
        float(observation["hybrid_serves"]),
    ]


class SmartKeyForecaster(nn.Module):
    """Shared LSTM encoder with a threat head and a pool head.

    The threat head emits logits over `ThreatPosture` for each of the
    next `THREAT_HORIZON_STEPS` steps; the pool head emits
    `N_POOL_OUTPUTS` regression values.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = 64,
        n_postures: int = len(ThreatPosture),
        threat_horizon: int = THREAT_HORIZON_STEPS,
        n_pool_outputs: int = N_POOL_OUTPUTS,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_postures = n_postures
        self.threat_horizon = threat_horizon

        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        self.threat_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_postures * threat_horizon),
        )
        self.pool_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_pool_outputs),
        )

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`windows` is (batch, WINDOW, n_features).

        Returns `(threat_logits, pool_outputs)` with `threat_logits`
        shaped (batch, threat_horizon, n_postures) and `pool_outputs`
        shaped (batch, n_pool_outputs).
        """
        encoded, _ = self.encoder(windows)
        final_hidden = encoded[:, -1, :]  # last timestep's representation

        threat_logits = self.threat_head(final_hidden)
        threat_logits = threat_logits.view(-1, self.threat_horizon, self.n_postures)
        pool_outputs = self.pool_head(final_hidden)
        return threat_logits, pool_outputs

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden_size": self.hidden_size,
                "n_postures": self.n_postures,
                "threat_horizon": self.threat_horizon,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SmartKeyForecaster":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            hidden_size=checkpoint["hidden_size"],
            n_postures=checkpoint["n_postures"],
            threat_horizon=checkpoint["threat_horizon"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class LSTMForecastProvider(ForecastProvider):
    """`ForecastProvider` backed by a trained `SmartKeyForecaster`.

    Selected by `configs/default.yaml`'s `use_foresight: lstm`.

    **Frozen by construction.** The model is put in `eval()` mode, every
    parameter has `requires_grad` cleared, and every forward pass runs
    under `torch.no_grad()`. PLAN.md Addition A requires the forecaster
    to be "frozen during DQN training (no end-to-end gradients --
    simpler, stabler, and keeps the ablation clean)"; doing it here
    rather than trusting the caller means the gradient path does not
    exist at all. `tests/test_forecaster_model.py` asserts it.

    Before `WINDOW` observations have accumulated, the window is
    left-padded with its earliest observation, so the provider returns
    well-formed output from step one exactly as the EWMA fallback does
    -- no warm-up branch is needed in the environment.
    """

    def __init__(self, model: SmartKeyForecaster) -> None:
        self._model = model
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

        self._window: deque[list[float]] = deque(maxlen=WINDOW)
        self._last_threat: ThreatForecast | None = None
        self._last_pool: PoolForecast | None = None

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "LSTMForecastProvider":
        return cls(SmartKeyForecaster.load(path))

    def _padded_window(self) -> torch.Tensor:
        if not self._window:
            rows = [[0.0] * N_FEATURES for _ in range(WINDOW)]
        else:
            rows = list(self._window)
            if len(rows) < WINDOW:
                rows = [rows[0]] * (WINDOW - len(rows)) + rows
        return torch.tensor([rows], dtype=torch.float32)

    def update(self, observation: ForecastObservation) -> None:
        self._window.append(observation_to_features(observation))

        with torch.no_grad():
            threat_logits, pool_outputs = self._model(self._padded_window())

        posture_probs_per_step = torch.softmax(threat_logits[0], dim=-1)
        posture_probs = posture_probs_per_step[0].tolist()

        # `threat_score` is the probability-weighted posture index,
        # normalised to [0, 1] so it lands on the same scale as the EWMA
        # fallback's score. The DQN's state vector has to *mean* the
        # same thing under both providers, or the E-A ablation would be
        # comparing two different MDPs rather than two forecasters.
        posture_indices = torch.arange(posture_probs_per_step.shape[-1], dtype=torch.float32)
        max_index = max(1.0, float(posture_indices[-1]))
        horizon_scores = (
            (posture_probs_per_step * posture_indices).sum(dim=-1) / max_index
        ).tolist()

        self._last_threat = ThreatForecast(
            threat_score=float(horizon_scores[0]),
            posture_probs=posture_probs,
            horizon_scores=[float(score) for score in horizon_scores],
        )

        values = pool_outputs[0].tolist()
        n = len(HORIZONS)
        self._last_pool = PoolForecast(
            pool_level_hat=[float(v) for v in values[0:n]],
            skr_mean_hat=[float(v) for v in values[n : 2 * n]],
            hybrid_demand_hat=[float(v) for v in values[2 * n : 3 * n]],
        )

    def get_threat_forecast(self) -> ThreatForecast:
        if self._last_threat is None:
            # Fresh instance, before update() -- mirror the EWMA
            # fallback's CALM-biased default rather than crashing.
            uniform = [0.0] * len(ThreatPosture)
            uniform[0] = 1.0
            return ThreatForecast(
                threat_score=0.0,
                posture_probs=uniform,
                horizon_scores=[0.0] * THREAT_HORIZON_STEPS,
            )
        return self._last_threat

    def get_pool_forecast(self) -> PoolForecast:
        if self._last_pool is None:
            zeros = [0.0] * len(HORIZONS)
            return PoolForecast(
                pool_level_hat=list(zeros),
                skr_mean_hat=list(zeros),
                hybrid_demand_hat=list(zeros),
            )
        return self._last_pool
