"""
forecaster/model.py

Dual-head LSTM forecaster (PLAN.md Addition A; PLAN2 §4A, §5.1's
architecture diagram). Owned by Person A (split.md §1).

    ForecastObservation (per env step)
        threat_features  (16, RT-IoT2022-derived, standardized)
        pool_signals     (4: qber, skr, pool_fill, hybrid_serves)
                    |
                    v
         +----------------------+
         |  shared LSTM trunk   |   window of the last W env steps
         +----------+-----------+
                    |
          +---------+---------+
          |                   |
     THREAT HEAD          POOL HEAD
     threat logit         pool_level_hat  (H = 10/25/50)
     + 5 horizon          skr_mean_hat    (H = 10/25/50)
       logits (k=5)       hybrid_demand_hat (H = 10/25/50)
          |                   |
          v                   v
   env/masking.py        DQN state only
   (RAISE ONLY)          (never the policy table)

Hard Rules this module is load-bearing for
------------------------------------------
* **Hard Rule 2** -- the threat head's output reaches `PolicyTable` and
  may only ever *raise* a floor. That is enforced structurally in
  `env/masking.py` (the ratchet is one-way and there is no
  `ratchet_down`), so nothing this model can predict, however wrong or
  however adversarially induced, can lower a floor. The pool head
  reaches the DQN's state vector and *nothing else* -- routing it into
  the floor computation would break the rule, and
  `env/contracts.py`'s `PoolForecast` docstring says so explicitly.
* **Frozen during DQN training.** `LSTMForecastProvider` puts the model
  in `eval()` mode, never builds a graph (`torch.no_grad()` on every
  forward), and holds no optimizer. No gradient from the agent's loss
  can reach these weights, so the agent cannot learn to manipulate its
  own threat signal into a lower floor.
* **Hard Rule 1** -- nothing here touches the reward.

Divergence from the dashboard mockup, stated rather than papered over
--------------------------------------------------------------------
`mock.html`'s Threat Input panel depicts an autoencoder + XGBoost
classifier + fusion stage. What is built here is the **LSTM dual-head
specified in PLAN2 §4A**, because the plan is authoritative over the
mockup (PLAN2's header rule 1: the HTML is a UI/UX mockup, and none of
it is a real result). The three depicted stages do have honest
counterparts -- a reconstruction/anomaly stage (the benign-referenced
standardization in `forecaster/dataset.py`, which is what makes an
attack window read as a positive deviation), a classifier stage (this
threat head), and a fusion stage (the posture mapping below) -- but
they are not an autoencoder and not XGBoost, and the dashboard labels
them for what they actually are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from env.contracts import (
    ForecastObservation,
    ForecastProvider,
    PoolForecast,
    ThreatForecast,
)
from forecaster.dataset import (
    N_FEATURES,
    N_POOL_SIGNALS,
    POOL_HORIZONS,
    FeatureStandardizer,
    pool_signal_vector,
)

THREAT_HORIZON_STEPS = 5
"""Addition A: "next k=5-step threat signal"; matches
`ThreatForecast.horizon_scores`."""

DEFAULT_WINDOW_STEPS = 16
"""How many env steps of history the trunk sees. Long enough for a scan
ramp or a flood onset to be a *shape* rather than a point, short enough
that the provider's per-step cost stays negligible against the
environment's own step cost."""

_POOL_OUTPUTS = 3 * len(POOL_HORIZONS)  # pool_level, skr_mean, hybrid_demand x 3 horizons


@dataclass
class DualHeadConfig:
    """Architecture + training hyperparameters (values live in
    `configs/default.yaml`'s `forecaster:` block)."""

    hidden_size: int = 64
    num_layers: int = 1
    window_steps: int = DEFAULT_WINDOW_STEPS
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 8
    threat_loss_weight: float = 1.0
    pool_loss_weight: float = 1.0
    seed: int = 0


class DualHeadLSTM(nn.Module):
    """Shared LSTM trunk, threat head + pool head.

    Genuinely shared: one recurrent encoder over the concatenated
    `[threat_features, pool_signals]` sequence, which is what makes this
    a dual-*head* model rather than two models in a trenchcoat. It is
    trainable that way because `forecaster/dataset.py`'s rollout dataset
    carries a threat label *and* a pool target on every single step, by
    construction -- the rollout injects real RT-IoT2022 windows into the
    environment, so both supervision signals come off the same sequence.
    """

    def __init__(self, config: DualHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else DualHeadConfig()

        self.input_size = N_FEATURES + N_POOL_SIGNALS
        self.trunk = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
        )
        # Threat head: 1 "now" logit + k future logits. Logits, not
        # probabilities, so training can use BCEWithLogits (numerically
        # stabler) and inference squashes once, in one place.
        self.threat_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1 + THREAT_HORIZON_STEPS),
        )
        self.pool_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, _POOL_OUTPUTS),
        )

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`window`: (batch, window_steps, N_FEATURES + N_POOL_SIGNALS).

        Returns `(threat_logits, pool_outputs)` with shapes
        `(batch, 1 + THREAT_HORIZON_STEPS)` and `(batch, 9)`.
        """
        _sequence, (hidden, _cell) = self.trunk(window)
        summary = hidden[-1]  # last layer's final hidden state
        return self.threat_head(summary), self.pool_head(summary)


# ---------------------------------------------------------------------------
# Posture mapping -- shared with the EWMA fallback on purpose
# ---------------------------------------------------------------------------

_POSTURE_ANCHORS = (0.0, 0.5, 1.0)
_POSTURE_TEMPERATURE = 0.15


def posture_probs_from_score(threat_score: float) -> list[float]:
    """Map a threat score in (0, 1) onto CALM/ELEVATED/HIGH probabilities.

    Deliberately the *same* fixed-temperature RBF-softmax
    `env/forecast_provider.py`'s `MovingAverageForecaster` uses, and
    deliberately not a learned 3-way classifier.

    Two reasons. First, RT-IoT2022 labels flows as benign or as one of
    eleven attack types; it does not label an *operational posture*, so
    a learned 3-class head would need posture labels invented here --
    which is precisely the kind of thing Hard Rule 4 exists to stop.
    Second, the two providers have to be interchangeable behind
    `ForecastProvider` (Addition A's "provider interchangeability" unit
    test) and feed the same `PolicyTable`; if `ewma` and `lstm` disagreed
    about what a score of 0.6 *means*, the E-A ablation would be
    comparing two different floor policies rather than two forecasters.

    So the learned part is the score (what the threat head predicts from
    real traffic), and the score-to-posture mapping is a fixed,
    documented, shared function.
    """
    anchors = np.asarray(_POSTURE_ANCHORS, dtype=float)
    logits = -((float(threat_score) - anchors) ** 2) / _POSTURE_TEMPERATURE
    exp_logits = np.exp(logits - logits.max())
    return (exp_logits / exp_logits.sum()).tolist()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LSTMForecastProvider(ForecastProvider):
    """The trained, frozen dual-head forecaster behind
    `env/contracts.py`'s `ForecastProvider` interface.

    Selected by `configs/default.yaml`'s `use_foresight: lstm`.
    Interchangeable with `MovingAverageForecaster` by construction: same
    interface, same posture mapping, same standardized-feature contract.

    Frozen means frozen: the module is in `eval()` mode, every forward
    pass is inside `torch.no_grad()`, and this class owns no optimizer
    and exposes no training entry point. Nothing the DQN does can move
    these weights.
    """

    def __init__(
        self,
        model: DualHeadLSTM,
        standardizer: FeatureStandardizer | None = None,
        window_steps: int | None = None,
        pool_target_scale: Sequence[float] | None = None,
    ) -> None:
        self._model = model
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

        self._standardizer = standardizer
        self._window_steps = int(window_steps or model.config.window_steps)
        self._pool_target_scale = (
            np.asarray(pool_target_scale, dtype=np.float32)
            if pool_target_scale is not None
            else np.ones(_POOL_OUTPUTS, dtype=np.float32)
        )

        input_size = model.input_size
        self._window = np.zeros((self._window_steps, input_size), dtype=np.float32)
        self._observations_seen = 0
        self._threat_score = 0.0
        self._horizon_scores = [0.0] * THREAT_HORIZON_STEPS
        self._pool_outputs = np.zeros(_POOL_OUTPUTS, dtype=np.float32)

    # -- ForecastProvider -------------------------------------------------

    def update(self, observation: ForecastObservation) -> None:
        threat_features = np.asarray(observation["threat_features"], dtype=np.float32)
        if threat_features.size != N_FEATURES:
            # The scenario-driven threat source emits a single
            # standardized scalar rather than a real feature window (see
            # env/environment.py's `_threat_features`). Broadcasting it
            # across the feature axis keeps the provider usable in that
            # mode instead of crashing -- the model was trained on real
            # windows, so this is a degraded but well-defined input, and
            # `configs/default.yaml`'s `threat_input.source` documents
            # which mode is which.
            fill = float(threat_features.mean()) if threat_features.size else 0.0
            threat_features = np.full(N_FEATURES, fill, dtype=np.float32)

        step_vector = np.concatenate(
            [threat_features, np.asarray(pool_signal_vector(observation), dtype=np.float32)]
        )
        self._window = np.roll(self._window, shift=-1, axis=0)
        self._window[-1] = step_vector
        self._observations_seen += 1

        with torch.no_grad():
            batch = torch.from_numpy(self._window).unsqueeze(0)
            threat_logits, pool_outputs = self._model(batch)
            scores = torch.sigmoid(threat_logits).squeeze(0).numpy()

        self._threat_score = float(scores[0])
        self._horizon_scores = [float(v) for v in scores[1:]]
        self._pool_outputs = pool_outputs.squeeze(0).numpy() * self._pool_target_scale

    def get_threat_forecast(self) -> ThreatForecast:
        return ThreatForecast(
            threat_score=self._threat_score,
            posture_probs=posture_probs_from_score(self._threat_score),
            horizon_scores=list(self._horizon_scores),
        )

    def get_pool_forecast(self) -> PoolForecast:
        n = len(POOL_HORIZONS)
        values = self._pool_outputs
        return PoolForecast(
            pool_level_hat=[float(v) for v in values[0:n]],
            skr_mean_hat=[float(v) for v in values[n : 2 * n]],
            hybrid_demand_hat=[float(v) for v in values[2 * n : 3 * n]],
        )

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "config": self._model.config.__dict__,
                "standardizer": self._standardizer.to_dict() if self._standardizer else None,
                "window_steps": self._window_steps,
                "pool_target_scale": self._pool_target_scale.tolist(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LSTMForecastProvider":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = DualHeadConfig(**checkpoint["config"])
        model = DualHeadLSTM(config)
        model.load_state_dict(checkpoint["model_state"])
        standardizer = (
            FeatureStandardizer.from_dict(checkpoint["standardizer"])
            if checkpoint.get("standardizer")
            else None
        )
        return cls(
            model=model,
            standardizer=standardizer,
            window_steps=checkpoint.get("window_steps"),
            pool_target_scale=checkpoint.get("pool_target_scale"),
        )


def load_forecaster_config(path: str | Path | None = None) -> DualHeadConfig:
    """Read the `forecaster:` block out of `configs/default.yaml`."""
    import yaml

    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    return DualHeadConfig(**config.get("forecaster", {}))
