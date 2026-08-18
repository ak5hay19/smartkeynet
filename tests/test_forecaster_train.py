"""Behavioral tests for `forecaster.train` -- offline, frozen dual-head
training (PLAN.md Addition A; Hard Rule 8).

Kept to a small, fast configuration: this asserts the training path is
correct and that its Hard Rule guards hold, not that a 6-second run
reaches a good model. The real training run's numbers are recorded in
SESSION_LOG.md and docs/report.md.
"""

from __future__ import annotations

import pytest

from forecaster.model import LSTMForecastProvider
from forecaster.train import ForecasterTrainingRecord, train_forecaster
from forecaster.dataset import resolve_dataset_path


def _dataset_available() -> bool:
    try:
        resolve_dataset_path()
        return True
    except FileNotFoundError:
        return False


requires_dataset = pytest.mark.skipif(
    not _dataset_available(),
    reason="RT-IoT2022 is gitignored; skipped where the operator has not placed it",
)

_SMOKE = {
    "hidden_size": 8,
    "window_steps": 4,
    "epochs": 1,
    "batch_size": 32,
    "rollout_scenarios": ["S1"],
    "rollout_seeds": [0],
    "rollout_max_steps": 120,
}


def test_training_refuses_the_held_out_scenario():
    """Hard Rule 8 -- S6 is held-out evaluation only, and the forecaster
    must not see it during training either. Enforced in code rather than
    left to discipline: it is a one-character mistake to make and an
    invisible one to detect afterwards."""
    with pytest.raises(ValueError, match="held-out"):
        train_forecaster(overrides={**_SMOKE, "rollout_scenarios": ["S1", "S6"]})


@requires_dataset
def test_smoke_run_produces_a_frozen_provider_and_a_record(tmp_path):
    provider, record = train_forecaster(
        overrides={**_SMOKE, "checkpoint_path": str(tmp_path / "smoke.pt")}
    )

    assert isinstance(provider, LSTMForecastProvider)
    assert isinstance(record, ForecasterTrainingRecord)
    assert record.n_train > 0 and record.n_val > 0
    assert len(record.epochs) == 1
    assert 0.0 <= record.val_threat_accuracy[-1] <= 1.0
    assert 0.0 <= record.val_threat_balanced_accuracy[-1] <= 1.0

    # The saved artefact is what env/environment.py loads, so it has to
    # come back frozen.
    reloaded = LSTMForecastProvider.load(tmp_path / "smoke.pt")
    assert not reloaded._model.training
    assert all(not p.requires_grad for p in reloaded._model.parameters())


@requires_dataset
def test_majority_class_rate_is_recorded_alongside_accuracy(tmp_path):
    """Raw accuracy alone is uninformative on this label mixture and
    would flatter the model. The base rate has to travel with it."""
    _provider, record = train_forecaster(
        overrides={**_SMOKE, "checkpoint_path": str(tmp_path / "smoke.pt")}
    )
    assert 0.5 <= record.majority_class_rate <= 1.0


@requires_dataset
def test_rollout_windows_carry_both_supervision_signals(tmp_path):
    """The claim that makes a *shared trunk* trainable: every rollout
    step has a threat label and a pool target, from the same sequence."""
    from experiments.train import load_full_config
    from forecaster.dataset import build_rollout_dataset

    rollouts = build_rollout_dataset(
        config=load_full_config(),
        scenarios=["S1"],
        seeds=[0],
        window_steps=4,
        max_steps=120,
    )
    assert len(rollouts) > 0
    assert rollouts.threat_windows.shape[0] == rollouts.threat_labels.shape[0]
    assert rollouts.pool_signals.shape[0] == rollouts.pool_targets.shape[0]
    assert rollouts.threat_windows.shape[0] == rollouts.pool_targets.shape[0]
    # label vector is [now, t+1 .. t+k] -- the horizon head forecasts
    assert rollouts.threat_labels.shape[1] == 6
    assert set(rollouts.threat_labels.flatten().tolist()) <= {0.0, 1.0}
