"""v2 posture-variance/floor-monotonicity validation (2026-09-23 QA
session). PROGRESS.md/SESSION_LOG.md's matching entry reports the
full real numbers from a larger, ad-hoc 100-decision run; this is the
committed regression test guarding the same real property against
future regressions, trimmed to run fast in the normal suite.

**The question this answers**: the upload demo's results table showed
a case where a HIGHER threat_score (0.1376, S0) produced a LOWER
floor (SERVE_CLASSICAL) than a lower threat_score (0.0689, S3,
SERVE_HYBRID). This is expected -- both scores fall in the same CALM
posture bucket (posture is CALM/ELEVATED/HIGH, not a continuous
input to the floor table; sensitivity class does the differentiating
work within a bucket) -- but that needs a real, broad check, not one
anecdote.

**Real data only, no fabrication**, three real sources, mirroring the
project's existing conventions rather than a new pattern:
  - `tests/fixtures/pcap_flow_features_fixture.pcap` (the same
    committed fixture `tests/test_dashboard_upload.py`/
    `tests/test_pcap_extract_provider.py` already use) via the
    unmodified `PCAPExtractProvider` -- real calm-end traffic.
  - Real held-out RT-IoT2022 rows via the unmodified
    `RTIoT2022ReplayProvider`/`held_out_replay_rows()` (the same
    held-out val split `tests/test_rt_iot_replay*.py` already
    validates) -- a real benign->attack transition span (captures a
    genuine, if transient, CALM->ELEVATED->HIGH escalation) plus a
    real steady-state attack span (confirms sustained HIGH).

**Posture-threshold finding this session confirmed** (see
`env/forecast_provider.py`/`forecaster/model.py`: `posture =
nearest of anchors (0.0, 0.5, 1.0)` to `threat_score`, independent of
the RBF-softmax temperature): CALM is `threat_score < 0.25`, ELEVATED
is `0.25 <= threat_score < 0.75`, HIGH is `threat_score >= 0.75`.
`0.0689` and `0.1376` are both comfortably inside CALM (0.1124 and
0.1811 below the 0.25 boundary respectively) -- neither is borderline.

Scope: read-only validation, reuses existing tested components
unmodified -- no change to `env/`, `agents/`, the reward function,
the action space, the floor table, or any checkpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from agents.dqn import DQNAgent, flatten_state, load_dqn_config
from dashboard.explain import explain_decision_from_env
from env.contracts import Action, ThreatPosture
from env.environment import SmartKeyNetEnv
from experiments.train import GreedyDQNPolicy, load_full_config
from forecaster.pcap_extract_provider import PCAPExtractProvider
from forecaster.rt_iot_replay import RTIoT2022ReplayProvider, held_out_replay_rows

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LSTM_CHECKPOINT = _REPO_ROOT / "checkpoints" / "lstm_threat_forecaster.pt"
_DQN_CHECKPOINT = _REPO_ROOT / "checkpoints" / "dqn_s1.pt"
_NPZ = _REPO_ROOT / "data" / "processed" / "rt_iot2022" / "rt_iot2022_windows.npz"
_CSV = _REPO_ROOT / "data" / "raw" / "rt_iot2022" / "RT_IOT2022.csv"
_PCAP_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "pcap_flow_features_fixture.pcap"

requires_real_data = pytest.mark.skipif(
    not (
        _LSTM_CHECKPOINT.exists()
        and _DQN_CHECKPOINT.exists()
        and _NPZ.exists()
        and _CSV.exists()
        and _PCAP_FIXTURE.exists()
    ),
    reason="real checkpoints / rt_iot2022 windows.npz / RT_IOT2022.csv / pcap fixture not present",
)

_TIER_ORDER = {Action.SERVE_CLASSICAL: 0, Action.SERVE_PQC: 1, Action.SERVE_HYBRID: 2}


def _run_episode(config, provider_factory, max_steps, seed):
    """Real live episode: real checkpoint-loaded masked DQN, real
    `SmartKeyNetEnv`, real `dashboard.explain.explain_decision_from_env`
    trace per decision -- mirrors `dashboard/upload_demo.py`'s own
    reload pattern exactly, reused rather than reimplemented."""
    cfg = {**config, "max_steps": max_steps}
    has_forecast = cfg.get("use_foresight", "off") != "off"
    probe_env = SmartKeyNetEnv({**cfg, "seed": 0}, forecast_provider_factory=provider_factory)
    state, _info = probe_env.reset(seed=0)
    state_dim = flatten_state(state, has_forecast).shape[0]
    agent = DQNAgent(state_dim=state_dim, has_forecast=has_forecast, config=load_dqn_config())
    agent.load(str(_DQN_CHECKPOINT))
    policy = GreedyDQNPolicy(agent)

    env = SmartKeyNetEnv(cfg, forecast_provider_factory=provider_factory)
    state, info = env.reset(seed=seed)
    rows = []
    truncated = False
    while not truncated and len(rows) < max_steps:
        mask = info["action_mask"]
        chosen = policy.act(state, mask)
        trace = explain_decision_from_env(env, state, chosen)
        rows.append(
            {
                "sensitivity_class": trace.sensitivity_class,
                "threat_score": float(trace.threat_score),
                "posture": trace.resolved_posture,
                "floor": trace.floor,
                "chosen_action": trace.chosen_action,
            }
        )
        state, _reward, _terminated, truncated, info = env.step(chosen)
    return rows


@requires_real_data
def test_real_decisions_span_calm_elevated_and_high_with_monotonic_floors_and_action_variety():
    config = load_full_config()

    # Real source 1: committed pcap fixture (calm end -- ordinary,
    # non-adversarial captured traffic).
    def pcap_factory(_seed):
        return PCAPExtractProvider(str(_LSTM_CHECKPOINT), _PCAP_FIXTURE)

    pcap_rows = _run_episode(config, pcap_factory, max_steps=10, seed=0)

    # Real source 2: held-out RT-IoT2022 benign->attack transition span
    # -- real chronological data straddling the benign/attack boundary
    # in the held-out replay stream, genuinely exercises a real
    # CALM->ELEVATED->HIGH escalation (not fabricated/forced).
    held_out = held_out_replay_rows(npz_path=_NPZ, csv_path=_CSV)
    combined = np.concatenate([held_out.benign_features, held_out.attack_features], axis=0)
    transition_start = len(held_out.benign_features) - 40

    def transition_factory(_seed):
        return RTIoT2022ReplayProvider(str(_LSTM_CHECKPOINT), combined, start_offset=transition_start)

    transition_rows = _run_episode(config, transition_factory, max_steps=50, seed=11)

    # Real source 3: held-out RT-IoT2022 steady-state attack span --
    # confirms sustained HIGH posture outside the transition window.
    def attack_factory(_seed):
        return RTIoT2022ReplayProvider(str(_LSTM_CHECKPOINT), held_out.attack_features, start_offset=500)

    attack_rows = _run_episode(config, attack_factory, max_steps=20, seed=21)

    all_rows = pcap_rows + transition_rows + attack_rows
    assert 50 <= len(all_rows) <= 100

    # -- Posture variety: real data genuinely reaches all three buckets. --
    posture_counts = Counter(r["posture"].name for r in all_rows)
    assert posture_counts[ThreatPosture.CALM.name] > 0
    assert posture_counts[ThreatPosture.ELEVATED.name] > 0
    assert posture_counts[ThreatPosture.HIGH.name] > 0

    # -- Floor monotonicity: holding posture constant, a higher
    # sensitivity_class must never produce a strictly lower floor. --
    by_posture = defaultdict(list)
    for r in all_rows:
        by_posture[r["posture"]].append(r)

    for posture, rows in by_posture.items():
        per_class_floor_values = defaultdict(set)
        for r in rows:
            per_class_floor_values[r["sensitivity_class"]].add(_TIER_ORDER[r["floor"]])
        classes_present = sorted(per_class_floor_values.keys(), key=lambda c: c.value)
        for i in range(len(classes_present)):
            for j in range(i + 1, len(classes_present)):
                c_lo, c_hi = classes_present[i], classes_present[j]
                max_floor_lo = max(per_class_floor_values[c_lo])
                min_floor_hi = min(per_class_floor_values[c_hi])
                assert min_floor_hi >= max_floor_lo, (
                    f"floor monotonicity violated at posture={posture.name}: "
                    f"{c_lo.name} floor(s)={per_class_floor_values[c_lo]} vs. "
                    f"{c_hi.name} floor(s)={per_class_floor_values[c_hi]}"
                )

    # -- Action variety: the DQN's chosen action does not saturate to
    # a single action across this real, varied sample. --
    action_counts = Counter(r["chosen_action"].name for r in all_rows)
    assert len(action_counts) >= 2


def test_posture_boundaries_are_the_midpoints_between_the_rbf_softmax_anchors():
    """Direct, checkpoint-free confirmation of the exact posture cutoffs
    reported in PROGRESS.md/SESSION_LOG.md: `posture = nearest of
    anchors (0.0, 0.5, 1.0)` to `threat_score` is mathematically
    equivalent to `argmax` over `-((threat_score - anchor) ** 2) /
    TEMPERATURE` for any TEMPERATURE > 0 (a positive, anchor-independent
    denominator does not change which anchor is closest) -- so the
    CALM/ELEVATED boundary is exactly the 0.0/0.5 midpoint (0.25) and
    the ELEVATED/HIGH boundary is exactly the 0.5/1.0 midpoint (0.75),
    regardless of the RBF-softmax temperature actually configured."""
    from forecaster.model import LSTMForecastProvider

    anchors = np.array(LSTMForecastProvider._POSTURE_ANCHORS)
    temperature = LSTMForecastProvider._POSTURE_TEMPERATURE
    assert tuple(anchors.tolist()) == (0.0, 0.5, 1.0)

    def posture_index(threat_score: float) -> int:
        logits = -((threat_score - anchors) ** 2) / temperature
        return int(np.argmax(logits))

    # Just under/over each analytic boundary.
    assert posture_index(0.2499) == ThreatPosture.CALM.value
    assert posture_index(0.2501) == ThreatPosture.ELEVATED.value
    assert posture_index(0.7499) == ThreatPosture.ELEVATED.value
    assert posture_index(0.7501) == ThreatPosture.HIGH.value

    # The two real upload-demo threat_scores this session investigated:
    # both comfortably inside CALM, well clear of the 0.25 boundary.
    assert posture_index(0.0689) == ThreatPosture.CALM.value
    assert posture_index(0.1376) == ThreatPosture.CALM.value
    assert 0.25 - 0.1376 > 0.1  # not borderline -- >0.1 of real margin left
