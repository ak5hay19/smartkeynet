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
import hashlib
import csv
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"

SUPPORTED_DATASETS = (
    "rt_iot2022",
    "qopsec-confidentiality",
    "qopsec-synthetic-context",
    "qkd-trace",
)


@dataclass(frozen=True)
class DatasetSlot:
    """One external-data slot: where it comes from and what uses it."""

    name: str
    filename: str
    source: str
    used_by: str
    licence_note: str = ""


_SLOTS: dict[str, DatasetSlot] = {
    "rt_iot2022": DatasetSlot(
        name="RT-IoT2022",
        filename="rt_iot2022/RT_IOT2022.csv",
        source="UCI Machine Learning Repository -- 'RT-IoT2022' (accept the terms, download the CSV)",
        used_by=(
            "the forecaster's threat features. NOT currently wired: "
            "env/environment.py feeds a documented placeholder instead, and "
            "forecaster/model.py deliberately learns posture from observable "
            "dynamics rather than raw threat features (see its N_FEATURES note)."
        ),
    ),
    "qopsec-confidentiality": DatasetSlot(
        name="Q-OPSEC confidentiality_train/valid",
        filename="qopsec/confidentiality_train.csv",
        source="the Q-OPSEC repository (Noetzold)",
        used_by="a sensitivity classifier. NOT currently wired -- classes come from the tenant graph.",
        licence_note=(
            "The Q-OPSEC repo has NO LICENSE FILE, so all rights are reserved by "
            "default. Do not redistribute their CSVs inside this repo. Cite them, "
            "and see data/README.md."
        ),
    ),
    "qopsec-synthetic-context": DatasetSlot(
        name="Q-OPSEC synthetic_context_dataset",
        filename="qopsec/synthetic_context_dataset.csv",
        source="the Q-OPSEC repository (Noetzold)",
        used_by=(
            "sanity-checking the (risk, confidentiality) -> tier mapping. NOT "
            "currently wired -- env/masking.py's table is a documented placeholder."
        ),
        licence_note="Same no-LICENSE caveat as above.",
    ),
    "qkd-trace": DatasetSlot(
        name="CV-QKD SKR/QBER trace",
        filename="cvqkd/skr_qber.csv",
        source="a published CV-QKD field-trial or testbed time series",
        used_by=(
            "env/pool_sim.py's pool refill. NOT currently wired: no citable trace "
            "was sourced, so SyntheticSKRQBERTrace is used instead. PLAN.md "
            "explicitly permits this if the generation procedure is stated, which "
            "it is, in that class's docstring."
        ),
    ),
}


def _report_slot(key: str, out_dir: Path) -> None:
    """Report a slot's status and, if it is absent, how to obtain it.

    THESE DOWNLOADS ARE DELIBERATELY NOT AUTOMATED, and that is a
    decision rather than an omission:

      * **RT-IoT2022** is behind a terms-acceptance step. Scripting
        around a click-through is discourteous to the host and fragile.
      * **The two Q-OPSEC files** come from a repository with **no
        LICENSE file**, so all rights are reserved by default.
        Auto-downloading and vendoring them into this repo would be a
        redistribution we have no permission for.
      * **The CV-QKD trace** has no single canonical source to fetch.

    More importantly: **nothing in this project requires any of them.**
    Every result in docs/report.md was produced from the documented
    synthetic processes and the NetworkX tenant graph. This script
    therefore tells you what each slot is for, whether it is present,
    and how to obtain it -- rather than pretending an automated
    pipeline exists.
    """
    slot = _SLOTS[key]
    target = out_dir / slot.filename
    status = "PRESENT" if target.exists() else "absent"

    print(f"[{status}] {slot.name}")
    print(f"  expected at : {target}")
    print(f"  source      : {slot.source}")
    print(f"  used by     : {slot.used_by}")
    if slot.licence_note:
        print(f"  LICENCE     : {slot.licence_note}")
    if status == "absent":
        print("  -> Place the file at the path above by hand. Nothing in this")
        print("     project needs it; every reported result runs without it.")
    print()


def check_datasets(out_dir: Path = RAW_DIR) -> dict[str, bool]:
    """Presence of every slot, as a dict -- importable by tests."""
    return {key: (out_dir / slot.filename).exists() for key, slot in _SLOTS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=None)
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys = [args.dataset] if args.dataset else list(_SLOTS)

    print("SmartKeyNet external-data slots\n")
    for key in keys:
        _report_slot(key, args.out_dir)
    print("CV-QKD SKR/QBER trace (SYNTHETIC fallback):")
    trace_path = write_synthetic_cvqkd_trace()
    print(f"  wrote    {trace_path}  sha256 {sha256_of(trace_path)[:16]}...")
    print("           NOT measured data -- see the header comment in the file.\n")

    samples = write_committed_samples()
    print("Committed 100-row samples (spec §S0 item 4):")
    for sample in samples:
        print(f"  wrote    {sample}  sha256 {sha256_of(sample)[:16]}...")
    if not samples:
        print("  (none -- no source files present to sample)")
    print()

    print(
        "None of the external downloads are required. Every number in docs/report.md\n"
        "was produced from the documented synthetic processes -- see that report's\n"
        "Limitations section."
    )




# ---------------------------------------------------------------------------
# CV-QKD SKR/QBER trace + committed 100-row samples (spec §S0 item 4)
# ---------------------------------------------------------------------------

CVQKD_TRACE_PATH = Path(__file__).resolve().parent / "cvqkd" / "skr_qber.csv"
SAMPLE_DIR = Path(__file__).resolve().parent / "sample"
SAMPLE_ROWS = 100
"""§S0 item 4 asks for "100-row samples committed" so the repo's data-handling
paths can be exercised, and reviewed, without anyone downloading anything."""


def write_synthetic_cvqkd_trace(
    path: Path = CVQKD_TRACE_PATH,
    n_rows: int = 20_000,
    mean_skr_kbps: float = 0.10,
    seed: int = 0,
) -> Path:
    """Write a synthetic SKR/QBER CSV in the shape `TraceSKRQBERSource` reads.

    No published CV-QKD trace was sourced for this capstone; PLAN.md's
    "Datasets & Provenance" section explicitly permits a documented synthetic
    fallback, and every result in this repo comes from
    `env.pool_sim.SyntheticSKRQBERTrace`. This writes that same process to
    disk so the trace-loading path -- including the 70/30 train/eval split
    §S1 insists on -- is exercised, and so swapping in a real trace is a
    config change rather than a code change.

    Reuses `SyntheticSKRQBERTrace` rather than reimplementing the maths, so
    the CSV and the in-memory source cannot drift apart. The file carries a
    header comment marking it synthetic: it must never be described as
    measured data.
    """
    from env.pool_sim import SyntheticSKRQBERTrace

    path.parent.mkdir(parents=True, exist_ok=True)
    trace = SyntheticSKRQBERTrace(n_steps=n_rows, mean_skr_kbps=mean_skr_kbps, seed=seed)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(
            "# SYNTHETIC CV-QKD SKR/QBER trace, generated by data/get_data.py.\n"
            "# NOT measured data. Produced by env.pool_sim.SyntheticSKRQBERTrace\n"
            f"# (log-space OU process, mean_skr_kbps={mean_skr_kbps}, seed={seed}).\n"
            "# PLAN.md 'Datasets & Provenance' permits a documented synthetic\n"
            "# fallback; this is it. Do not cite it as a published testbed trace.\n"
        )
        writer = csv.writer(handle)
        writer.writerow(["step", "skr_kbps", "qber"])
        for step, (skr_kbps, qber) in enumerate(trace):
            writer.writerow([step, f"{skr_kbps:.6f}", f"{qber:.6f}"])
    return path


def write_committed_samples(sample_dir: Path = SAMPLE_DIR, n_rows: int = SAMPLE_ROWS) -> list[Path]:
    """Write small committed samples of every dataset slot that is present.

    For the synthetic CV-QKD trace this is always possible. For RT-IoT2022 it
    only happens if the real file has been downloaded -- a sample of a dataset
    nobody has is not something this script can invent, and inventing one
    would be worse than having none.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    trace_sample = sample_dir / "skr_qber_sample.csv"
    if CVQKD_TRACE_PATH.exists():
        with open(CVQKD_TRACE_PATH, "r", encoding="utf-8") as source:
            lines = [line for line in source if not line.startswith("#")]
        with open(trace_sample, "w", encoding="utf-8") as destination:
            destination.write("# 100-row sample of the SYNTHETIC CV-QKD trace (spec §S0).\n")
            destination.writelines(lines[: n_rows + 1])  # +1 for the header row
        written.append(trace_sample)

    for real_csv in sorted((Path(__file__).resolve().parent / "raw" / "rt_iot2022").glob("*.csv")):
        sample_path = sample_dir / f"{real_csv.stem}_sample.csv"
        with open(real_csv, "r", encoding="utf-8") as source:
            lines = [next(source) for _ in range(n_rows + 1)]
        with open(sample_path, "w", encoding="utf-8") as destination:
            destination.write(f"# 100-row sample of {real_csv.name} (RT-IoT2022, UCI).\n")
            destination.writelines(lines)
        written.append(sample_path)

    return written


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
