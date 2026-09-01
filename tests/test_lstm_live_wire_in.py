"""Step-5 sanity-check for PLAN2.md §8A Addition E, Phases 2-4: does the
real, trained `forecaster.model.LSTMForecastProvider` actually run live
against `env.environment.SmartKeyNetEnv` without crashing, and does the
posture signal it produces vary sensibly?

**Scope, matching this session's instruction precisely**: this is a
PLUMBING sanity check, not a live threat-detection validation, and not
the full experimental campaign (Gate W3 / S3 comparison / S5 dose-
response / dashboard panels / Table V) -- that is a later, separate,
deliberate session (PLAN2.md §8A Phase 4). It also does NOT edit
`env/environment.py`: `LSTMForecastProvider` is injected purely through
the pre-existing `forecast_provider_factory` constructor parameter
(design decision 17, already used by `attack/attacking_provider.py` for
exactly this kind of external-provider injection), so `use_foresight:
lstm` remains unwired in `_build_forecaster()`'s existing stub branch --
flagged as a small, explicitly out-of-scope follow-up rather than done
here.

**Read `forecaster/model.py::LSTMForecastProvider`'s class docstring
("live-feed caveat") before trusting these results as more than
plumbing**: `env/environment.py::_threat_features_placeholder` still
returns the 2-dimensional `[qber, load]` placeholder, not a real
`FEATURE_DIM=39` RT-IoT2022-shaped feature vector -- that requires
Addition F's pcap replay, sequenced after this one. So "posture varies
sensibly" here demonstrates the wiring (window buffering -> forward
pass -> `ThreatForecast`) works end-to-end and reacts to its input, not
that it is making genuine live threat detections yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.baselines import StaticThresholdPolicy
from env.environment import SmartKeyNetEnv
from experiments.train import load_full_config
from forecaster.model import LSTMForecastProvider

REAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "lstm_threat_forecaster.pt"
requires_real_checkpoint = pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="lstm_threat_forecaster.pt not present")


@requires_real_checkpoint
def test_lstm_forecast_provider_drives_a_handful_of_live_episodes_without_crashing():
    config = load_full_config()
    config = {**config, "max_steps": 60}
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)

    all_threat_scores: list[float] = []
    all_policy_floors: list[int] = []

    for seed in (0, 1, 2):
        env = SmartKeyNetEnv(
            config,
            forecast_provider_factory=lambda _seed: LSTMForecastProvider(str(REAL_CHECKPOINT)),
        )
        state, info = env.reset(seed=seed)

        truncated = False
        while not truncated:
            mask = info["action_mask"]
            action = policy.act(state, mask)
            state, reward, terminated, truncated, info = env.step(action)

            all_threat_scores.append(state["threat_score"])
            all_policy_floors.append(state["policy_floor"])

    # env doesn't crash across 3 full episodes -- the core "wire it in"
    # requirement.
    assert len(all_threat_scores) > 0

    # threat_score must stay in its documented range throughout
    assert all(0.0 <= s <= 1.0 for s in all_threat_scores)

    # "posture signal varies sensibly (not stuck at one value)" -- the
    # LSTM's threat_score reacts to its (zero-padded, plumbing-only)
    # input rather than being a frozen constant across an entire episode.
    assert len(set(round(s, 6) for s in all_threat_scores)) > 1
    # the downstream floor the LSTM-driven posture produces is a
    # well-formed Action value throughout, never crashing masking/floor
    # computation.
    assert all(0 <= f <= 4 for f in all_policy_floors)
