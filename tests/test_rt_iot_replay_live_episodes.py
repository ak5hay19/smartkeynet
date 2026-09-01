"""Live-episode sanity check for `RTIoT2022ReplayProvider`, mirroring
`tests/test_lstm_live_wire_in.py`'s pattern exactly but with real,
held-out RT-IoT2022 replay data driving the LSTM instead of the
synthetic `[qber, load]` placeholder. Does NOT edit `env/environment.py`
-- injected via the same `forecast_provider_factory` seam.

This is the fast, always-run smoke check; the full multi-episode
Step-5 campaign (`forecaster/replay_campaign.py`) is run manually and
its real numbers are reported in SESSION_LOG.md/PROGRESS.md, not
re-asserted here as a flaky statistical test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.baselines import StaticThresholdPolicy
from env.environment import SmartKeyNetEnv
from experiments.train import load_full_config
from forecaster.rt_iot_replay import RTIoT2022ReplayProvider, held_out_replay_rows

REAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "lstm_threat_forecaster.pt"
REAL_NPZ = Path(__file__).resolve().parents[1] / "data" / "processed" / "rt_iot2022" / "rt_iot2022_windows.npz"
REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "rt_iot2022" / "RT_IOT2022.csv"
requires_real_data = pytest.mark.skipif(
    not (REAL_CHECKPOINT.exists() and REAL_NPZ.exists() and REAL_CSV.exists()),
    reason="lstm_threat_forecaster.pt / rt_iot2022 windows.npz / RT_IOT2022.csv not present",
)


@requires_real_data
def test_replay_provider_drives_live_episodes_with_genuine_class_conditioned_variance():
    holdout = held_out_replay_rows()
    config = {**load_full_config(), "max_steps": 60}
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)

    def run(features, seed):
        env = SmartKeyNetEnv(
            config,
            forecast_provider_factory=lambda _seed: RTIoT2022ReplayProvider(str(REAL_CHECKPOINT), features),
        )
        state, info = env.reset(seed=seed)
        scores, floors = [], []
        truncated = False
        while not truncated:
            mask = info["action_mask"]
            action = policy.act(state, mask)
            state, reward, terminated, truncated, info = env.step(action)
            scores.append(state["threat_score"])
            floors.append(state["policy_floor"])
        return scores, floors

    benign_scores, benign_floors = run(holdout.benign_features, seed=0)
    attack_scores, attack_floors = run(holdout.attack_features, seed=0)

    assert len(benign_scores) > 0 and len(attack_scores) > 0
    assert all(0.0 <= s <= 1.0 for s in benign_scores + attack_scores)
    assert all(0 <= f <= 4 for f in benign_floors + attack_floors)

    # the central real-data-driven property: genuinely replaying a real
    # attack-labeled held-out span produces a materially higher mean
    # threat_score, live through the full env, than a real benign-labeled
    # span -- not just in the isolated provider-level test.
    import statistics

    assert statistics.mean(attack_scores) > statistics.mean(benign_scores)


@requires_real_data
def test_cold_start_zero_padding_causes_a_real_transient_spike_regardless_of_replayed_class():
    """A real finding from this session's Step-5 campaign (see
    SESSION_LOG.md), locked in as a regression test rather than left as
    a one-off observation: at the very first decision of an episode,
    `LSTMForecastProvider.get_threat_forecast()`'s buffer holds only 1
    real replayed row (63 zero-padded, per its pre-existing,
    Phase-4a-built cold-start behavior -- untouched by this session).
    That mostly-zero window reads as spuriously high-threat to the
    model REGARDLESS of whether the one real row is benign or attack
    (the model was only ever trained on FULL 64-row real windows, never
    a near-all-zero one), and -- because Hard Rule 2's floor ratchet
    only ever raises -- this transient artifact can lock an elevated
    floor in for a genuinely benign episode's entire remaining
    duration. This is NOT fixed here (deliberately out of this
    session's scope -- see SESSION_LOG.md); this test exists so a
    future session addressing it can see exactly what changes."""
    holdout = held_out_replay_rows()
    config = {**load_full_config(), "max_steps": 60}
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)

    env = SmartKeyNetEnv(
        config,
        forecast_provider_factory=lambda _seed: RTIoT2022ReplayProvider(str(REAL_CHECKPOINT), holdout.benign_features),
    )
    state, info = env.reset(seed=0)
    first_decision_score = state["threat_score"]

    truncated = False
    later_scores = []
    i = 0
    while not truncated:
        mask = info["action_mask"]
        action = policy.act(state, mask)
        state, reward, terminated, truncated, info = env.step(action)
        i += 1
        if i >= 5:  # buffer has by now filled with several real benign rows
            later_scores.append(state["threat_score"])

    # real, measured this session: step-0 spuriously reads elevated
    # (mean 0.92 across 15 seeds) even though the replayed data is
    # genuinely benign; once the buffer fills with real rows the score
    # correctly drops (post-warmup mean 0.08, max 0.31).
    assert first_decision_score > 0.5
    assert max(later_scores) < 0.5
