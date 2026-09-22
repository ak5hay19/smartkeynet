"""Live-episode sanity check for `PCAPExtractProvider`, mirroring
`tests/test_rt_iot_replay_live_episodes.py`'s pattern exactly -- real
tshark-extracted pcap flow features driving the trained LSTM through a
real `SmartKeyNetEnv` episode, injected via the same
`forecast_provider_factory` seam. Does NOT edit `env/environment.py`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from agents.baselines import StaticThresholdPolicy
from env.environment import SmartKeyNetEnv
from experiments.train import load_full_config
from forecaster.pcap_extract_provider import PCAPExtractProvider

REAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "lstm_threat_forecaster.pt"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pcap_flow_features_fixture.pcap"
REAL_SAMPLE = Path(__file__).resolve().parents[1] / "scratchpad" / "sample.pcap"

requires_tshark = pytest.mark.skipif(shutil.which("tshark") is None, reason="tshark not found on PATH")
requires_real_checkpoint = pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="lstm_threat_forecaster.pt not present")
requires_real_sample = pytest.mark.skipif(
    not REAL_SAMPLE.exists(), reason="scratchpad/sample.pcap not present (gitignored, session-local capture)"
)


@requires_tshark
@requires_real_checkpoint
def test_provider_extracts_real_features_from_the_fixture_pcap():
    provider = PCAPExtractProvider(str(REAL_CHECKPOINT), FIXTURE)
    assert provider.extraction.n_flows == 10
    assert provider.extraction.features.shape[0] == 10


@requires_tshark
@requires_real_checkpoint
def test_provider_drives_a_live_episode_with_sane_finite_threat_scores():
    """Real, end-to-end: the fixture pcap's real extracted flows drive
    the trained LSTM through a full `SmartKeyNetEnv` episode. Not a
    class-conditioned-variance test (the fixture pcap is a handful of
    synthetic, mostly-benign-shaped flows, not a real
    benign-vs-attack-labeled span like `RTIoT2022ReplayProvider`'s
    held-out data) -- this checks the wiring/shape/range end-to-end,
    matching this session's Step 3 instruction."""
    config = {**load_full_config(), "max_steps": 30}
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)

    env = SmartKeyNetEnv(
        config,
        forecast_provider_factory=lambda _seed: PCAPExtractProvider(str(REAL_CHECKPOINT), FIXTURE),
    )
    state, info = env.reset(seed=0)
    scores = []
    truncated = False
    while not truncated:
        mask = info["action_mask"]
        action = policy.act(state, mask)
        state, reward, terminated, truncated, info = env.step(action)
        scores.append(state["threat_score"])

    assert len(scores) > 0
    assert all(np.isfinite(s) for s in scores)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # inherited cold-start warm-start fix (RTIoT2022ReplayProvider) must
    # still hold for a pcap-sourced provider: no false-positive spike at
    # the very first decision relative to the rest of the episode.
    assert scores[0] - min(scores) < 0.3


@requires_tshark
@requires_real_checkpoint
def test_provider_raises_a_clear_value_error_on_a_shape_mismatch_not_a_silent_pad():
    """Regression guard: `PCAPExtractProvider` must inherit
    `RTIoT2022ReplayProvider`'s real shape guard (refuses to silently
    truncate/zero-pad a wrong-width feature array) -- exercised here by
    constructing against a checkpoint whose declared `feature_dim`
    cannot match a hand-built wrong-width array, mirroring how that
    guard is proven in `tests/test_rt_iot_replay_live_episodes.py`."""
    import torch

    from forecaster.pcap_flow_features import PCAPExtractionResult

    checkpoint = torch.load(str(REAL_CHECKPOINT), map_location="cpu", weights_only=False)
    real_feature_dim = int(checkpoint["feature_dim"])

    provider = PCAPExtractProvider(str(REAL_CHECKPOINT), FIXTURE)
    wrong_width = provider.extraction.features[:, : real_feature_dim - 1]
    assert wrong_width.shape[1] != real_feature_dim

    from forecaster.rt_iot_replay import RTIoT2022ReplayProvider

    with pytest.raises(ValueError):
        RTIoT2022ReplayProvider(str(REAL_CHECKPOINT), wrong_width)


@requires_tshark
@requires_real_checkpoint
@requires_real_sample
def test_provider_drives_a_live_episode_from_the_real_captured_sample():
    """Step 3's actual real-file validation: the real ~10s/1,716-packet
    own-traffic capture (own browsing traffic, not RT-IoT2022 IoT
    traffic -- a real distribution-shift caveat, not asserted away
    here) driving a real 60-step episode end-to-end."""
    config = {**load_full_config(), "max_steps": 60}
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)

    env = SmartKeyNetEnv(
        config,
        forecast_provider_factory=lambda _seed: PCAPExtractProvider(str(REAL_CHECKPOINT), REAL_SAMPLE),
    )
    state, info = env.reset(seed=0)
    scores = []
    truncated = False
    while not truncated:
        mask = info["action_mask"]
        action = policy.act(state, mask)
        state, reward, terminated, truncated, info = env.step(action)
        scores.append(state["threat_score"])

    assert len(scores) == 60
    assert all(np.isfinite(s) and 0.0 <= s <= 1.0 for s in scores)
