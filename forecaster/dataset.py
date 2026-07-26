"""
forecaster/dataset.py

Sliding-window supervised datasets for `forecaster/train.py` (PLAN.md
Addition A). Owned by Person A (split.md §1).

Built from logged environment rollouts: inputs are QBER, SKR,
per-class arrival counts, hybrid serves, pool level, and threat
features (RT-IoT2022-derived); targets are the same signals shifted by
each pool-head horizon H in {10, 25, 50} or the threat head's k=5
steps.

Never build this dataset from `rl_experiment_*` / `synthetic_rl_*`
Q-OPSEC logs (PLAN.md "Datasets & Provenance": those are for baseline
reproduction + feedback calibration only).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from forecaster.model import HORIZONS, K_THREAT_STEPS, WINDOW


class RolloutWindowDataset(Dataset):
    """Sliding windows of length `WINDOW` over one or more logged
    rollouts, with threat-head and pool-head targets computed by
    shifting the log."""

    def __init__(self, rollout_log_paths: list[Path]) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns a dict with keys: `window`, `threat_target`,
        `pool_level_target`, `skr_mean_target`, `hybrid_demand_target`."""
        raise NotImplementedError


def build_datasets(
    rollout_log_paths: list[Path], train_frac: float = 0.8
) -> tuple[RolloutWindowDataset, RolloutWindowDataset]:
    """Split logged rollouts into train/val `RolloutWindowDataset`s."""
    raise NotImplementedError
