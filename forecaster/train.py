"""
forecaster/train.py

Offline training for the dual-head LSTM (PLAN.md Addition A). Owned by
Person A (split.md §1).

    python -m forecaster.train

Offline and frozen are the load-bearing words. This module is the only
place the forecaster's weights ever move. `env/environment.py`
constructs the trained provider through
`forecaster.model.LSTMForecastProvider.load`, which puts the module in
`eval()` mode with `requires_grad=False` and runs every forward pass
under `torch.no_grad()` -- so during DQN training there is no path,
even accidentally, from the agent's loss back into the threat head.
That matters for more than tidiness: an agent that could shape its own
threat signal could shape its own floors, which is the exact failure
mode (Hard Rule 2, and the steering attack in PLAN2 §7.5) this
architecture exists to rule out.

Data provenance (PLAN2 §6):
  * threat head  -- RT-IoT2022, the one real-network slot.
  * pool head    -- the environment's own pool trajectory during
                    baseline rollouts.
  * neither head -- `rl_experiment_*` or `context_dataset_*`, ever.

Hard Rule 8: training scenarios only. S6 is refused here as it is in
`experiments/train.py`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from forecaster.dataset import (
    RolloutDataset,
    build_rollout_dataset,
    load_rt_iot2022,
)
from forecaster.model import (
    THREAT_HORIZON_STEPS,
    DualHeadConfig,
    DualHeadLSTM,
    LSTMForecastProvider,
    load_forecaster_config,
)

_HELD_OUT_SCENARIOS = frozenset({"S6"})


@dataclass
class ForecasterTrainingRecord:
    """Per-epoch curves plus the held-out evaluation the run ended on."""

    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    threat_losses: list[float] = field(default_factory=list)
    pool_losses: list[float] = field(default_factory=list)
    val_threat_accuracy: list[float] = field(default_factory=list)
    val_threat_balanced_accuracy: list[float] = field(default_factory=list)
    val_pool_mae: list[float] = field(default_factory=list)
    checkpoint_path: str | None = None
    n_train: int = 0
    n_val: int = 0
    majority_class_rate: float = 0.0
    """Fraction of validation labels belonging to the larger class.

    Reported next to raw accuracy because raw accuracy alone is
    uninformative here and would flatter the model: the rollout label
    mixture is dominated by benign windows (the scenarios the forecaster
    trains on sit at or near a calm threat level for most of their
    length), so a model that predicted "benign" unconditionally would
    already score close to this number. Balanced accuracy -- the mean of
    the two per-class recalls -- is the metric that actually says
    whether the threat head detects attacks."""


def _stack_inputs(dataset: RolloutDataset) -> torch.Tensor:
    """`[threat_features | pool_signals]` per step -- the shared trunk's input."""
    return torch.from_numpy(
        np.concatenate([dataset.threat_windows, dataset.pool_signals], axis=2).astype(np.float32)
    )


def _split(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random train/validation split.

    Random rather than a trailing slice: windows are collected
    scenario-by-scenario and policy-by-policy, so a trailing slice would
    hand the validation set exactly one policy on exactly one scenario
    and measure something other than generalization.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    return order[n_val:], order[:n_val]


def train_forecaster(
    full_config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[LSTMForecastProvider, ForecasterTrainingRecord]:
    """Collect baseline rollouts, train both heads jointly, save the
    frozen provider.

    Joint training on one shared trunk is the point of "dual-head":
    every rollout step carries a threat label *and* a pool target, so
    both losses are computed on the same forward pass and the trunk
    learns a representation that serves both. The two losses are summed
    with the config's `threat_loss_weight`/`pool_loss_weight`.
    """
    from experiments.train import load_full_config

    full_config = full_config if full_config is not None else load_full_config()
    forecaster_cfg = {**full_config.get("forecaster", {}), **(overrides or {})}

    scenarios = list(forecaster_cfg.get("rollout_scenarios", ["S1", "S2", "S3"]))
    held_out = _HELD_OUT_SCENARIOS.intersection(scenarios)
    if held_out:
        raise ValueError(
            f"{sorted(held_out)} is held-out evaluation only (Hard Rule 8) -- "
            "the forecaster must not see it during training either"
        )

    config = DualHeadConfig(
        **{
            key: forecaster_cfg[key]
            for key in DualHeadConfig.__dataclass_fields__
            if key in forecaster_cfg
        }
    )
    torch.manual_seed(config.seed)

    dataset = load_rt_iot2022(
        path=(full_config.get("threat_input") or {}).get("dataset_path"),
        max_rows=(full_config.get("threat_input") or {}).get("max_rows"),
    )
    rollouts = build_rollout_dataset(
        config=full_config,
        scenarios=scenarios,
        seeds=list(forecaster_cfg.get("rollout_seeds", [0, 1])),
        window_steps=config.window_steps,
        max_steps=int(forecaster_cfg.get("rollout_max_steps", 600)),
        dataset=dataset,
    )

    inputs = _stack_inputs(rollouts)
    threat_targets = torch.from_numpy(rollouts.threat_labels)
    pool_targets_raw = rollouts.pool_targets

    # Pool targets span wildly different magnitudes (pool_fill in [0,1],
    # skr ~1e-2, hybrid_serves counts), so an unscaled MSE would be
    # almost entirely the largest-magnitude column. Scale each column to
    # unit RMS for training and undo it at inference (the provider
    # multiplies by `pool_target_scale`), so the model is not implicitly
    # told that one horizon matters more than another.
    pool_scale = np.sqrt((pool_targets_raw**2).mean(axis=0))
    pool_scale = np.where(pool_scale < 1e-8, 1.0, pool_scale).astype(np.float32)
    pool_targets = torch.from_numpy((pool_targets_raw / pool_scale).astype(np.float32))

    train_idx, val_idx = _split(len(rollouts), val_fraction=0.2, seed=config.seed)

    model = DualHeadLSTM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    threat_criterion = nn.BCEWithLogitsLoss()
    pool_criterion = nn.MSELoss()

    val_labels = rollouts.threat_labels[val_idx]
    positive_rate = float(val_labels.mean())
    record = ForecasterTrainingRecord(
        n_train=len(train_idx),
        n_val=len(val_idx),
        majority_class_rate=max(positive_rate, 1.0 - positive_rate),
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = np.random.default_rng(config.seed + epoch).permutation(train_idx)
        epoch_total, epoch_threat, epoch_pool, n_batches = 0.0, 0.0, 0.0, 0

        for start in range(0, len(permutation), config.batch_size):
            batch_idx = permutation[start : start + config.batch_size]
            if len(batch_idx) < 2:
                continue
            batch = torch.as_tensor(batch_idx, dtype=torch.long)

            threat_logits, pool_out = model(inputs[batch])
            threat_loss = threat_criterion(threat_logits, threat_targets[batch])
            pool_loss = pool_criterion(pool_out, pool_targets[batch])
            loss = (
                config.threat_loss_weight * threat_loss + config.pool_loss_weight * pool_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += float(loss.item())
            epoch_threat += float(threat_loss.item())
            epoch_pool += float(pool_loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val = torch.as_tensor(val_idx, dtype=torch.long)
            val_threat_logits, val_pool_out = model(inputs[val])
            predicted = (torch.sigmoid(val_threat_logits) > 0.5).float()
            truth = threat_targets[val]
            accuracy = float((predicted == truth).float().mean().item())

            attack = truth > 0.5
            benign = ~attack
            recall_attack = (
                float((predicted[attack] > 0.5).float().mean().item()) if attack.any() else 0.0
            )
            recall_benign = (
                float((predicted[benign] < 0.5).float().mean().item()) if benign.any() else 0.0
            )
            balanced_accuracy = 0.5 * (recall_attack + recall_benign)
            pool_mae = float((val_pool_out - pool_targets[val]).abs().mean().item())

        record.epochs.append(epoch)
        record.train_losses.append(epoch_total / max(1, n_batches))
        record.threat_losses.append(epoch_threat / max(1, n_batches))
        record.pool_losses.append(epoch_pool / max(1, n_batches))
        record.val_threat_accuracy.append(accuracy)
        record.val_threat_balanced_accuracy.append(balanced_accuracy)
        record.val_pool_mae.append(pool_mae)

    provider = LSTMForecastProvider(
        model=model,
        standardizer=dataset.standardizer,
        window_steps=config.window_steps,
        pool_target_scale=pool_scale.tolist(),
    )

    checkpoint_path = forecaster_cfg.get(
        "checkpoint_path", "checkpoints/forecaster_dual_head.pt"
    )
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    provider.save(path)
    record.checkpoint_path = str(path)

    return provider, record


def main() -> None:
    provider, record = train_forecaster()
    print(f"train windows: {record.n_train}  validation windows: {record.n_val}")
    print(f"majority-class rate on validation: {record.majority_class_rate:.4f} (the number to beat)")
    for i, epoch in enumerate(record.epochs):
        print(
            f"  epoch {epoch:2d}  loss={record.train_losses[i]:.4f} "
            f"(threat={record.threat_losses[i]:.4f} pool={record.pool_losses[i]:.4f})  "
            f"val_acc={record.val_threat_accuracy[i]:.4f} "
            f"val_balanced_acc={record.val_threat_balanced_accuracy[i]:.4f}  "
            f"val_pool_mae={record.val_pool_mae[i]:.4f}"
        )
    print(f"saved frozen provider to {record.checkpoint_path}")
    print(
        f"threat head: accuracy {record.val_threat_accuracy[-1]:.4f} vs "
        f"majority-class {record.majority_class_rate:.4f}; "
        f"balanced accuracy {record.val_threat_balanced_accuracy[-1]:.4f} "
        "(mean of per-class recall -- the metric that says whether attacks are detected)"
    )


if __name__ == "__main__":
    main()
