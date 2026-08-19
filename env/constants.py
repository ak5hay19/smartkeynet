"""
env/constants.py

Loads `configs/constants.yaml` and makes it the *source of truth* for every
cited security constant (spec §3.1's `constants.py`; Hard Rule 4).

---------------------------------------------------------------------
Why this module exists
---------------------------------------------------------------------
`configs/constants.yaml` and its HR4 source-lint were added on 2026-08-19,
and for one session the file was **loaded by nothing**. The values that
actually drove the simulator lived in `configs/default.yaml` and, for the
cost model, as hardcoded dicts in `env/environment.py`. So the lint checked
a file that was not the source of truth, the numbers were duplicated in two
places, and Hard Rule 4 was being enforced on a decoration -- a citation
attached to a copy of a constant is not a citation on the constant.

This module closes that. The cost model now reads its per-tier latency and
energy from `constants.yaml` (they had no config source at all before), and
`assert_consistent_with_default_config` fails loudly if the values that must
appear in both files ever drift apart. Deleting the duplicates outright
would be cleaner still, but `configs/default.yaml` is read by roughly every
module and every recorded run, so the duplication is guarded rather than
removed -- and the guard is what makes it safe.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONSTANTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "constants.yaml"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class ConstantsError(Exception):
    """Raised when a cited constant is missing, or when the two config files
    disagree about a value that appears in both."""


@lru_cache(maxsize=1)
def load_constants(path: str | Path | None = None) -> dict[str, Any]:
    """Parse `configs/constants.yaml`.

    Cached because this is read during environment construction and the file
    is immutable within a run. Pass an explicit `path` in tests to bypass the
    cache with a different file.
    """
    resolved = Path(path) if path is not None else _CONSTANTS_PATH
    if not resolved.exists():
        raise ConstantsError(f"cited-constants file missing: {resolved}")
    with open(resolved, encoding="utf-8") as handle:
        constants: dict[str, Any] = yaml.safe_load(handle)
    if not constants:
        raise ConstantsError(f"cited-constants file is empty: {resolved}")
    return constants


def key_bits() -> int:
    """ETSI GS QKD 014 key-delivery size, in bits."""
    return int(load_constants()["etsi_qkd_014"]["key_bits"])


def max_key_age_steps() -> int:
    """The SP 800-57-derived key-lifetime cap `L`, in simulator steps."""
    return int(load_constants()["key_lifetime"]["max_key_age_steps"])


def qber_abort() -> float:
    """Reconciliation abort threshold. Calibrated, not a device figure --
    see the `source` string in `configs/constants.yaml`."""
    return float(load_constants()["cvqkd_link"]["qber_abort"])


def gate_kappa() -> float:
    """Exponent of the reconciliation gate."""
    return float(load_constants()["cvqkd_link"]["gate_kappa"])


def handshake_latency_ms() -> dict[str, float]:
    """Per-tier handshake latency in ms, keyed by
    `{"reuse", "classical", "pqc", "hybrid"}`.

    An ordinal model, NOT measured on the evaluation host -- the block is
    flagged `measured: false` in the YAML and a test stops the report
    claiming otherwise. What the reward needs is the ordering
    reuse < classical < PQC < hybrid; the magnitudes are simulator units
    against the spec's 100 ms reference.
    """
    block = load_constants()["handshake_latency_ms"]
    return {tier: float(block[tier]) for tier in ("reuse", "classical", "pqc", "hybrid")}


def handshake_energy_mj() -> dict[str, float]:
    """Per-tier energy in mJ. Ordinal model, as above."""
    block = load_constants()["handshake_energy_mj"]
    return {tier: float(block[tier]) for tier in ("reuse", "classical", "pqc", "hybrid")}


def assert_consistent_with_default_config(
    constants_path: str | Path | None = None,
    default_config_path: str | Path | None = None,
) -> None:
    """Fail if `constants.yaml` and `default.yaml` disagree.

    Four constants appear in both files, because `default.yaml` is the
    runtime config every module already reads while `constants.yaml` is the
    citation-bearing record HR4's lint walks. Duplication that nothing checks
    is how a cited value silently stops describing the value in use, so this
    is called at environment construction.
    """
    constants = load_constants(constants_path)
    resolved_default = (
        Path(default_config_path) if default_config_path is not None else _DEFAULT_CONFIG_PATH
    )
    with open(resolved_default, encoding="utf-8") as handle:
        runtime_config: dict[str, Any] = yaml.safe_load(handle)

    mismatches: list[str] = []

    def compare(name: str, cited: float, runtime: float) -> None:
        if float(cited) != float(runtime):
            mismatches.append(f"{name}: constants.yaml={cited} but default.yaml={runtime}")

    compare(
        "key_bits / pool.bits_per_hybrid_draw",
        constants["etsi_qkd_014"]["key_bits"],
        runtime_config["pool"]["bits_per_hybrid_draw"],
    )
    compare(
        "key_lifetime.max_key_age_steps",
        constants["key_lifetime"]["max_key_age_steps"],
        runtime_config["key_lifetime"]["max_key_age_steps"],
    )
    compare(
        "cvqkd_link.qber_abort",
        constants["cvqkd_link"]["qber_abort"],
        runtime_config["qkd"]["qber_abort"],
    )
    compare(
        "cvqkd_link.gate_kappa",
        constants["cvqkd_link"]["gate_kappa"],
        runtime_config["qkd"]["gate_kappa"],
    )

    if mismatches:
        raise ConstantsError(
            "cited constants disagree with the runtime config (Hard Rule 4) -- "
            + "; ".join(mismatches)
        )
