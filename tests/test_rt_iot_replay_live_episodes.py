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

import statistics
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
def test_cold_start_buffer_warm_start_fixes_the_false_positive_spike():
    """Regression test for the cold-start artifact found in the
    original replay-wiring session (see SESSION_LOG.md/PROGRESS.md) and
    fixed in the "cold-start buffer fix" session:
    `RTIoT2022ReplayProvider` used to leave its inherited LSTM window
    buffer empty at construction, so the very first decision of an
    episode saw only 1 real replayed row + 63 zero-padded rows --
    wildly out-of-distribution for a model trained only on full 64-row
    real windows, which responded by spiking `threat_score` to ~0.92
    REGARDLESS of whether the replayed data was genuinely benign or
    attack. Because Hard Rule 2's floor ratchet only ever raises, that
    transient false positive could lock an elevated floor in for a
    genuinely benign episode's entire remaining duration.

    Fix: `RTIoT2022ReplayProvider._warm_start_buffer()` now pre-fills
    the buffer with real context before the first `update()` call --
    real preceding held-out rows when available, or the earliest
    available real row replicated as a documented approximation when
    the episode starts at the true beginning of the held-out span
    (this benign-class case, `start_offset=0`, always hits the
    fallback -- see that method's docstring). This test would fail
    again if the buffer were ever left empty/zero-padded at cold start.

    Real, measured before/after this session (15-seed mean, matching
    `forecaster/replay_campaign.py`'s own seed/start_offset scheme):
    benign step-0 threat_score mean was 0.9221 (false positive) before
    the fix, 0.0808 after -- landing right next to the true post-warmup
    benign mean (~0.08, unchanged by this fix). Attack step-0 stayed
    correctly high throughout: 0.9723 before, 0.9750 after -- the fix
    does not suppress genuine early detection."""
    holdout = held_out_replay_rows()
    config = {**load_full_config(), "max_steps": 60}
    policy = StaticThresholdPolicy(pool_fill_threshold=0.5)

    def first_decision_score(features):
        env = SmartKeyNetEnv(
            config,
            forecast_provider_factory=lambda _seed, feats=features: RTIoT2022ReplayProvider(
                str(REAL_CHECKPOINT), feats
            ),
        )
        state, info = env.reset(seed=0)
        return state["threat_score"]

    benign_score = first_decision_score(holdout.benign_features)
    attack_score = first_decision_score(holdout.attack_features)

    # the false-positive artifact this fix removes: benign step-0 must
    # now read low, close to the true post-warmup benign signal, not
    # the false ~0.92 the empty/zero-padded buffer used to produce.
    assert benign_score < 0.5
    # genuine early detection must be preserved: attack step-0 must
    # still read correctly high.
    assert attack_score > 0.5


@requires_real_data
def test_cold_start_no_longer_spikes_relative_to_post_warmup_benign_scores():
    """Companion check to the regression test above, using the same
    shape as the original bug-documenting test: step-0 and post-warmup
    (decision >=5) benign scores must now be of a similar, low
    magnitude -- there must be no large transient spike at episode
    start. Before the fix, step-0 (~0.92) was far above every
    post-warmup score (max 0.31); after the fix they sit in the same
    range."""
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

    assert first_decision_score < 0.5
    assert max(later_scores) < 0.5
    # no large cold-start spike relative to the post-warmup regime
    assert first_decision_score - statistics.mean(later_scores) < 0.3
