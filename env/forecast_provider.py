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

import numpy as np

from env.contracts import (
    ForecastObservation,
    ForecastProvider,
    PoolForecast,
    ThreatForecast,
)


def _squash_non_negative(raw_signal: float) -> float:
    """Map a non-negative raw threat signal onto `[0, 1)` with 0 -> 0.

        squash(x) = 2 * sigmoid(x) - 1

    WHY NOT A PLAIN SIGMOID (fixed 2026-08-15). The threat features
    this forecaster is fed are all non-negative quantities (QBER, a
    normalised load, a scenario threat boost), so a plain
    `sigmoid(mean(features))` can never return less than 0.5. With
    posture anchors at {0.0, 0.5, 1.0}, that put the nearest anchor at
    ELEVATED or above *always*: `ThreatPosture.CALM` was unreachable,
    and a benign S1 episode sat at ELEVATED from its very first step.

    That is not a cosmetic problem. It silently raised every floor in
    the baseline scenario, it left S2 demonstrating ELEVATED -> HIGH
    rather than the intended CALM -> ELEVATED -> HIGH progression, and
    it meant the "posture" input to the policy table carried almost no
    information -- one of its three values was dead.

    `2 * sigmoid(x) - 1` is the same monotone squashing with the
    half-line `[0, inf)` mapped onto `[0, 1)` instead of `[0.5, 1)`. It
    is 0 at 0, so a quiet link with no load reads as genuinely calm,
    and it is strictly increasing, so the Hard Rule 2 property that a
    larger raw signal can only produce a higher posture is preserved
    exactly.
    """
    return float(2.0 / (1.0 + np.exp(-raw_signal)) - 1.0)


class MovingAverageForecaster(ForecastProvider):
    """EWMA-of-SKR-and-arrivals fallback forecaster (Addition A).

    Deliberately simple: no learned parameters. Exists so the env, the
    DQN, and the E-A ablation's `off` vs `ewma` comparison all have a
    working, non-LSTM forecast source from week 1 (PLAN.md Addition A:
    "This means the env is buildable in month 1 before the LSTM
    exists.").

    Threat head: `threat_features` (an arbitrary-length raw feature
    vector meant for the real LSTM threat head) is collapsed to a
    single scalar via its mean, squashed through a sigmoid into (0, 1),
    then EWMA-smoothed -- this scalar is `threat_score`.
    `posture_probs` is a fixed-temperature RBF-softmax over three
    anchor points at 0.0/0.5/1.0 (CALM/ELEVATED/HIGH) in that squashed
    space, which is what guarantees it always sums to 1 (softmax
    normalization) regardless of `threat_score`'s value. This is a
    placeholder heuristic, not a calibrated classifier -- real posture
    classification is the LSTM threat head's job.

    Flat-hold horizon design (Hard Rule 2 note: this only ever affects
    the *pool* head's output and `threat_forecast.horizon_scores`,
    never `env/masking.py`'s floor computation directly -- only
    `ThreatForecast` is meant to feed the policy table, and only in
    the raise direction, per Hard Rule 2; `PoolForecast` must never
    reach it at all): `get_pool_forecast()`'s three horizons
    (H in {10, 25, 50}) and `get_threat_forecast()`'s five-step
    `horizon_scores` all repeat the *current* EWMA-smoothed estimate
    at every horizon step -- there is no learned trend/extrapolation
    model here (that's exactly what
    `forecaster.model.LSTMForecastProvider`'s pool head adds later).
    This is an intentional simplification, not a bug: it's what "no
    learned parameters" means in practice for a multi-step-ahead
    forecast. In particular, `PoolForecast.hybrid_demand_hat` is
    documented in `contracts.py` as an *expected count of
    hybrid-mandatory arrivals over the horizon* (a quantity that would
    normally grow with H); this fallback flat-holds the current
    per-step hybrid-serve EWMA instead of scaling it by H, which
    under-estimates demand at longer horizons by design -- an
    explicitly accepted fallback-only simplification, not something a
    trend-aware forecaster (like the LSTM) should replicate.

    Fresh-instance behavior: before `update()` is ever called, all
    EWMA state defaults to 0.0 (a CALM-biased threat score, and empty
    pool/SKR/demand estimates), so `get_threat_forecast()` /
    `get_pool_forecast()` return well-formed, sensible output rather
    than crashing or returning garbage.
    """

    _POSTURE_ANCHORS: tuple[float, float, float] = (
        0.0,
        0.5,
        1.0,
    )  # CALM, ELEVATED, HIGH in squashed-threat-score space
    _POSTURE_TEMPERATURE: float = (
        0.15  # fixed RBF-softmax spread; smaller -> sharper posture assignment
    )
    _THREAT_HORIZON_STEPS: int = 5  # Addition A: "next k=5-step threat signal"
    _POOL_HORIZONS: tuple[int, int, int] = (10, 25, 50)  # H in {10, 25, 50} per contracts.py

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha

        self._threat_score: float = 0.0
        self._pool_fill: float = 0.0
        self._skr: float = 0.0
        self._hybrid_serve_rate: float = 0.0

    def _ewma(self, previous: float, new_value: float) -> float:
        return self._alpha * new_value + (1.0 - self._alpha) * previous

    def update(self, observation: ForecastObservation) -> None:
        threat_features = observation["threat_features"]
        # CONVENTION: the LAST element of `threat_features` is a scalar threat
        # summary already in [0, 1]; everything before it is raw per-flow
        # detail for the LSTM head.
        #
        # This used to average the whole vector. That was fine while the
        # vector was `[qber, load, boost]` and every entry rose with threat,
        # and it broke the moment real RT-IoT2022 features arrived: those are
        # standardised, so they average to ~0 regardless of posture, and they
        # outnumber the summary 8 to 1. Measured on an S2 episode, the mean
        # read 0.124 before the attack and -0.119 during reconnaissance --
        # i.e. an escalation made the signal look *calmer*. Reading the
        # summary directly is both correct and simpler.
        summary = float(threat_features[-1]) if len(threat_features) > 0 else 0.0
        self._threat_score = self._ewma(self._threat_score, min(1.0, max(0.0, summary)))
        self._pool_fill = self._ewma(self._pool_fill, float(observation["pool_fill"]))
        self._skr = self._ewma(self._skr, float(observation["skr"]))
        self._hybrid_serve_rate = self._ewma(
            self._hybrid_serve_rate, float(observation["hybrid_serves"])
        )

    def get_threat_forecast(self) -> ThreatForecast:
        anchors = np.array(self._POSTURE_ANCHORS)
        logits = -((self._threat_score - anchors) ** 2) / self._POSTURE_TEMPERATURE
        exp_logits = np.exp(logits - logits.max())  # numerically stable softmax
        posture_probs = (exp_logits / exp_logits.sum()).tolist()

        horizon_scores = [self._threat_score] * self._THREAT_HORIZON_STEPS

        return ThreatForecast(
            threat_score=self._threat_score,
            posture_probs=posture_probs,
            horizon_scores=horizon_scores,
        )

    def get_pool_forecast(self) -> PoolForecast:
        n_horizons = len(self._POOL_HORIZONS)
        return PoolForecast(
            pool_level_hat=[self._pool_fill] * n_horizons,
            skr_mean_hat=[self._skr] * n_horizons,
            hybrid_demand_hat=[self._hybrid_serve_rate] * n_horizons,
        )
