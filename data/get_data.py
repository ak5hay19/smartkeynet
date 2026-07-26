"""
data/get_data.py

Dataset download/prep stub (PLAN.md "Datasets & Provenance"; §10
kickoff step 1, Person A). Owned by Person A (split.md §1).

Fetches the three external-data slots this project actually needs:
RT-IoT2022 (threat forecaster), Q-OPSEC (`confidentiality_train/valid`,
`synthetic_context_dataset`), and a CV-QKD SKR/QBER trace. Never
fetches or loads `context_dataset_basic.csv` / `context_dataset_advanced.csv`
(verified degenerate) or `rl_experiment_*` / `synthetic_rl_*` logs as
training data (those are baseline-reproduction inputs only).

Read `data/README.md`'s licensing section before adding a real
Q-OPSEC download step -- their repo has no LICENSE file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"

SUPPORTED_DATASETS = (
    "rt_iot2022",
    "qopsec-confidentiality",
    "qopsec-synthetic-context",
    "qkd-trace",
)


def _download_rt_iot2022(out_dir: Path) -> None:
    """Download RT-IoT2022 (real IoT intrusion flows) into `out_dir`."""
    raise NotImplementedError


def _download_qopsec_confidentiality(out_dir: Path) -> None:
    """Fetch Q-OPSEC's `confidentiality_train`/`valid` (320/80 rows,
    4-class) into `out_dir`. See licensing note in `data/README.md`
    before redistributing anything derived from this."""
    raise NotImplementedError


def _download_qopsec_synthetic_context(out_dir: Path) -> None:
    """Fetch Q-OPSEC's `synthetic_context_dataset` (939 rows, 6 balanced
    classes) into `out_dir`. Policy-table calibration only."""
    raise NotImplementedError


def _download_qkd_trace(out_dir: Path) -> None:
    """Fetch (or point to) a published CV-QKD experimental SKR/QBER
    trace into `out_dir`. If no citable trace is available, generate a
    documented synthetic SKR process instead (PLAN.md "Datasets &
    Provenance": synthetic fallback is acceptable if the generation
    procedure is stated and rate ranges are cited)."""
    raise NotImplementedError


_DOWNLOADERS = {
    "rt_iot2022": _download_rt_iot2022,
    "qopsec-confidentiality": _download_qopsec_confidentiality,
    "qopsec-synthetic-context": _download_qopsec_synthetic_context,
    "qkd-trace": _download_qkd_trace,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _DOWNLOADERS[args.dataset](args.out_dir)


if __name__ == "__main__":
    main()
