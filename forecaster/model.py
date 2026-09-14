"""
forecaster/model.py

LSTM threat forecaster (PLAN.md Addition A; PLAN2.md §8A Addition E,
Phases 2-4). Owned by Person A (split.md §1).

`SmartKeyForecaster`: an LSTM encoder over a W=64-timestep window of
RT-IoT2022-derived flow features (`forecaster/rt_iot_features.py`,
`FEATURE_DIM=39` columns), with a single threat head -- a binary
benign-vs-attack logit -- trained in `forecaster/train_threat_lstm.py`
on the real windowed RT-IoT2022 dataset
(`data/processed/rt_iot2022/rt_iot2022_windows.npz`, Addition E Phase
1). `threat_score = sigmoid(logit)` is used exactly like
`env/forecast_provider.py::MovingAverageForecaster`'s squashed EWMA
signal: a scalar in (0, 1) that is then mapped to a `ThreatPosture`
class distribution via the same RBF-softmax anchor scheme, so posture
semantics stay identical between the two `ForecastProvider`
implementations.

**Scope decision, reported explicitly rather than silently narrowed
(Addition E Phases 2-4 session instruction):** PLAN.md's original
Addition A design and this file's own prior stub docstring described a
*dual*-head forecaster (threat head + pool head: `pool_level_hat`/
`skr_mean_hat`/`hybrid_demand_hat`). This session trains and returns
ONLY the threat head. RT-IoT2022 is a network-flow intrusion dataset --
it carries no QKD pool telemetry (`qber`/`skr`/`pool_fill`) at all, so
there is no real data in scope to supervise a pool head from. A real
pool head needs logged `SmartKeyNetEnv` rollouts
(`forecaster/dataset.py::RolloutWindowDataset`'s intended data source),
which have not been collected -- that remains a separate, not-yet-
started future session, and `forecaster/dataset.py`/`forecaster/train.py`
are left untouched (still real `NotImplementedError` stubs) rather than
built around with a fabricated pool head. `LSTMForecastProvider.
get_pool_forecast()` below reuses `MovingAverageForecaster`'s own EWMA
pool-head computation verbatim (composition, not duplication) so the
interface stays fully implemented and honest about which half is
learned.

Frozen during DQN training -- no end-to-end gradients (PLAN.md
Addition A). This file only defines the model + the live
`ForecastProvider` wrapper; training lives in
`forecaster/train_threat_lstm.py` (kept separate from
`forecaster/train.py`, which remains the future rollout-based
dual-head trainer -- PLAN2.md Addition D's "one implementation, two
callers" precedent for a clean split between this session's real,
RT-IoT2022-direct pipeline and the still-unbuilt rollout-based one).
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from env.contracts import (
    ForecastObservation,
    ForecastProvider,
    PoolForecast,
    ThreatForecast,
)
from env.forecast_provider import MovingAverageForecaster

WINDOW = 64
HORIZONS = (10, 25, 50)
K_THREAT_STEPS = 5


class SmartKeyForecaster(nn.Module):
    """LSTM encoder + single threat head (binary attack logit).

    See module docstring for why this is threat-only, not the
    originally-envisioned dual-head design.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.threat_head = nn.Linear(hidden_dim, 1)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """`window`: shape `(batch, WINDOW, input_dim)`.

        Returns `threat_logits`, shape `(batch,)` -- one raw
        (pre-sigmoid) benign-vs-attack logit per window, read off the
        top LSTM layer's final hidden state.
        """
        _, (h_n, _) = self.lstm(window)
        last_hidden = h_n[-1]  # (batch, hidden_dim) -- top layer's final step
        return self.threat_head(last_hidden).squeeze(-1)


class LSTMForecastProvider(ForecastProvider):
    """`ForecastProvider` implementation wrapping a trained
    `SmartKeyForecaster` checkpoint (real, threat-head-only -- see
    module docstring).

    Selected via `configs/default.yaml`'s `use_foresight: lstm`, or by
    direct construction through `env/environment.py`'s
    `forecast_provider_factory` injection point (design decision 17)
    for a live sanity-check episode -- see SESSION_LOG.md's Addition E
    Phases 2-4 entry: this session does NOT edit
    `env/environment.py::_build_forecaster`'s `"lstm"` stub branch
    (that file is explicitly out of this session's scope), so
    `use_foresight: lstm` is not yet wired as a config choice; the
    factory injection point already covers the required "sanity-check
    the wire-in" step without touching that file.

    **Live-feed caveat, load-bearing for interpreting any live sanity
    check (Addition F, PLAN2.md §8B, is what actually fixes this):**
    `env/environment.py::_threat_features_placeholder` still returns
    the 2-dimensional `[qber, load]` placeholder (documented there as
    "not a real threat signal"), not a real `FEATURE_DIM=39`
    RT-IoT2022-shaped flow-feature vector -- that requires a live/pcap
    feature source, which is Addition F's explicit job, sequenced
    after this one. `_prepare_step_features` below zero-pads whatever
    width it is given up to `feature_dim` so the model never crashes on
    the current placeholder width, but until Addition F exists, any
    live episode run against the real environment exercises this
    class's *plumbing* (window buffering, forward pass, output shape/
    range) only -- not genuine live threat detection, since the input
    it is fed bears no relation to the RT-IoT2022 columns it was
    trained on.
    """

    # Kept numerically identical to `MovingAverageForecaster`'s own
    # anchors/temperature (not imported as private attributes) so
    # discrete posture semantics stay consistent across both
    # `ForecastProvider` implementations.
    _POSTURE_ANCHORS: tuple[float, float, float] = (0.0, 0.5, 1.0)
    _POSTURE_TEMPERATURE: float = 0.15
    _THREAT_HORIZON_STEPS: int = K_THREAT_STEPS

    def __init__(self, checkpoint_path: str, decision_threshold: float | None = None) -> None:
        # weights_only=False: this checkpoint dict carries numpy arrays
        # (the persisted scaler) alongside the state_dict, and is always
        # a file this training pipeline itself produced (never an
        # untrusted third-party file), same trust model as
        # `agents/dqn.py::load`.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self._feature_dim: int = int(checkpoint["feature_dim"])
        self._window: int = int(checkpoint["window"])
        self._model = SmartKeyForecaster(
            input_dim=self._feature_dim,
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint.get("num_layers", 1)),
        )
        self._model.load_state_dict(checkpoint["model"])
        self._model.eval()

        self._scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
        self._scaler_std = np.asarray(checkpoint["scaler_std"], dtype=np.float64)
        self._decision_threshold: float = (
            float(decision_threshold) if decision_threshold is not None else float(checkpoint.get("decision_threshold", 0.5))
        )

        self._buffer: deque[np.ndarray] = deque(maxlen=self._window)
        # Pool head reuse -- see module docstring: no real pool-head
        # training data is in scope this session, so pool forecasting
        # is delegated verbatim to the same EWMA computation
        # `MovingAverageForecaster` already implements and tests.
        self._pool_ewma = MovingAverageForecaster()

    def _prepare_step_features(self, threat_features: Sequence[float]) -> np.ndarray:
        """Pads/truncates one step's raw feature vector to
        `feature_dim` and applies the persisted training-time scaler
        to the leading continuous columns (see class docstring's
        live-feed caveat for why this is a real, honest zero-pad, not
        a semantic reconstruction, when fed the current 2-dim env
        placeholder)."""
        raw = np.zeros(self._feature_dim, dtype=np.float64)
        n = min(len(threat_features), self._feature_dim)
        if n > 0:
            raw[:n] = np.asarray(list(threat_features)[:n], dtype=np.float64)

        n_continuous = self._scaler_mean.shape[0]
        scaled = raw.copy()
        scaled[:n_continuous] = (raw[:n_continuous] - self._scaler_mean) / self._scaler_std
        return scaled

    def update(self, observation: ForecastObservation) -> None:
        self._pool_ewma.update(observation)
        self._buffer.append(self._prepare_step_features(observation["threat_features"]))

    def get_threat_forecast(self) -> ThreatForecast:
        history = list(self._buffer)
        pad = self._window - len(history)
        if pad > 0:
            # Warm-start fix (2026-09-14, S5-flat-curve follow-up): the
            # model was only ever trained on full `window`-length real
            # windows, so a mostly-zero window at cold start is wildly
            # out-of-distribution and spikes `threat_score` regardless of
            # true class (measured: ~0.96 at the very first decision of a
            # real S2 episode, before this fix) -- the same class of bug
            # `RTIoT2022ReplayProvider._warm_start_buffer()` fixed for the
            # replay provider. That fix can't be reused verbatim: it
            # warm-starts at __init__ time from a fully pre-loaded replay
            # array, so it can borrow real rows that PRECEDE the episode.
            # This provider has no such array -- it only ever sees rows
            # one at a time via `update()`, so at cold start there is no
            # "real preceding row" to reach for, by construction, only
            # ever the earliest real row THIS episode has actually
            # delivered so far (`history[0]`). Replicating that row to
            # fill the missing slots is the direct analog of the replay
            # provider's own documented fallback (replicate the earliest
            # available real row rather than zero-pad) -- a real-
            # magnitude, in-distribution-ish constant window instead of
            # an all-zero one, not equivalent to real history, same as
            # that fallback's own honest framing. `history` is only ever
            # empty before this episode's first `update()` call, which
            # `env/environment.py`'s tick loop always runs before the
            # first `get_threat_forecast()` call -- the plain-zero pad
            # below is an unreachable-in-practice last resort, not the
            # real fix.
            filler = history[0] if history else np.zeros(self._feature_dim, dtype=np.float64)
            history = [filler] * pad + history
        window_arr = np.stack(history, axis=0).astype(np.float32)

        with torch.no_grad():
            logit = self._model(torch.from_numpy(window_arr).unsqueeze(0))
            threat_score = float(torch.sigmoid(logit).item())

        anchors = np.array(self._POSTURE_ANCHORS)
        logits = -((threat_score - anchors) ** 2) / self._POSTURE_TEMPERATURE
        exp_logits = np.exp(logits - logits.max())
        posture_probs = (exp_logits / exp_logits.sum()).tolist()

        # Flat-hold, same rationale `MovingAverageForecaster` documents:
        # no multi-step-ahead trend supervision exists for this
        # binary-classification target, so all k=5 horizon steps repeat
        # the current estimate rather than extrapolating one.
        horizon_scores = [threat_score] * self._THREAT_HORIZON_STEPS

        return ThreatForecast(threat_score=threat_score, posture_probs=posture_probs, horizon_scores=horizon_scores)

    def get_pool_forecast(self) -> PoolForecast:
        return self._pool_ewma.get_pool_forecast()
