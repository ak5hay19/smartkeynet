"""
experiments/ea_ablation.py

Experiment E-A, the foresight ablation (PLAN.md §5, Addition A).

Runs the identical agent with `use_foresight = off / ewma / lstm` and
reports the deltas in regret events, pool exhaustion and latency, per
Addition A: "identical agent trained/evaluated with `use_foresight =
off / ewma / lstm` on S3 and S6."

Addition A also fixes the success criterion, and it is worth quoting
because this experiment did not meet it:

    "**Success criterion:** LSTM foresight measurably reduces regret
     events on S3 vs `off`. If EWMA ties LSTM, report honestly (still
     beats `off`; the claim becomes 'foresight matters,' not 'LSTMs
     matter')."

---------------------------------------------------------------------
Read this before interpreting the output
---------------------------------------------------------------------
**This is a null result.** Measured on the final code: S3 regret events
`off 317.2`, `ewma 317.8`, `lstm 722.0`. Foresight buys nothing here,
and the LSTM is actively worse.

A previous revision of this file reported that EWMA foresight cut S3
regret by 23%, and that claim has been **withdrawn**. It was measured
before two DQN training bugs were fixed (missing observation
normalisation, and an absorbing-state training loop that left the agent
trapped in a starved regime for ~95% of training). Once the agent's
inputs were correctly scaled and its training no longer collapsed, the
apparent benefit vanished -- because it had been an artifact of a
broken agent reacting to whichever features dominated its unnormalised
input, not a property of foresight.

The lesson is worth carrying: **an ablation is only as trustworthy as
the agent underneath it.** Anything measured before a training fix has
to be re-measured after it, and comparing results-file timestamps
against source timestamps is the cheap way to catch what needs redoing.

Why the LSTM is worse rather than merely useless: its threat head does
learn (balanced accuracy 0.334 -> 0.852 after fixing a ratcheted-label
bug and severe class imbalance), but class weighting recovers rare
escalations by trading precision for recall. Every false positive
raises a floor, and under a scarce pool a raised floor converts
directly into deferrals. A more *sensitive* threat forecaster is not
automatically a better one.

Two environment bugs were also fixed along the way, both of which had
produced flat nulls of their own (see SESSION_LOG.md):

  1. **Threat windows were rectangular** -- the signal jumped to full
     intensity in one step with no precursor, and absolute time is
     excluded from the state, so nothing observable predicted an
     escalation. There was no forecasting problem to solve.
  2. **The threat head was trained on the *ratcheted* posture** under
     unweighted cross-entropy. The ratchet is monotone and sticky, so
     the label was "same as now" in 99.9% of samples, and raw accuracy
     (0.838) hid the collapse completely -- that is exactly what
     answering "calm" to everything scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.harness import run_scenario
from experiments.train import GreedyDQNPolicy, load_full_config, train

FORESIGHT_MODES: tuple[str, ...] = ("off", "ewma", "lstm")
ABLATION_SCENARIOS: tuple[str, ...] = ("S3", "S6")
"""Addition A names S3 and S6. S6 is eval-only (Hard Rule 8), so the
agent for it is trained on S3 and evaluated on S6 -- never trained on
the migration timeline."""

_EVAL_SEED_OFFSET = 2000


def run_ablation(
    n_train_seeds: int = 3,
    n_eval_seeds: int = 5,
    total_steps: int = 20_000,
    eval_max_steps: int = 2_000,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config if config is not None else load_full_config()
    config = {**config, "scenario_steps": eval_max_steps}

    train_seeds = list(range(n_train_seeds))
    eval_seeds = [_EVAL_SEED_OFFSET + i for i in range(n_eval_seeds)]

    report: dict[str, Any] = {
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "total_steps": total_steps,
        "modes": {},
    }

    print(f"{'=' * 78}\nE-A FORESIGHT ABLATION\n{'=' * 78}")
    print(f"  modes  : {list(FORESIGHT_MODES)}")
    print(f"  eval on: {list(ABLATION_SCENARIOS)}  (S6 is held-out: trained on S3)\n")

    for mode in FORESIGHT_MODES:
        mode_config = {**config, "use_foresight": mode}
        print(f"  --- use_foresight = {mode} ---", flush=True)

        per_scenario: dict[str, dict[str, float]] = {}
        agents = []
        for seed in train_seeds:
            # Always trained on S3: it is the scenario the ablation is
            # about (scarcity under degradation) and it is trainable,
            # whereas S6 is held out (Hard Rule 8).
            agent, _record = train(
                full_config=mode_config,
                training_overrides={
                    "seed": seed,
                    "total_steps": total_steps,
                    "eval_every": max(1, total_steps),
                    "eval_max_steps": eval_max_steps,
                    "checkpoint_path": f"checkpoints/ea_{mode}_seed{seed}.pt",
                },
                scenario="S3",
            )
            agents.append(agent)

        for scenario in ABLATION_SCENARIOS:
            rewards, regrets, latencies, violations = [], [], [], []
            eval_config = {**mode_config, "max_steps": eval_max_steps}
            if scenario == "S6":
                # S6's migration schedule ratchets per tenant cohort, so
                # it needs the graph source to have cohorts to target.
                eval_config = {**eval_config, "request_source": "graph"}

            for agent in agents:
                for seed in eval_seeds:
                    result = run_scenario(GreedyDQNPolicy(agent), scenario, eval_config, seed=seed)
                    rewards.append(result.total_reward)
                    regrets.append(result.pool_exhaustion_events)
                    latencies.append(result.p99_latency)
                    violations.append(result.floor_violations)

            per_scenario[scenario] = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_regret_events": float(np.mean(regrets)),
                "mean_p99_latency": float(np.mean(latencies)),
                "total_floor_violations": int(np.sum(violations)),
            }
            summary = per_scenario[scenario]
            print(
                f"    {scenario}: reward {summary['mean_reward']:12.1f} "
                f"+/- {summary['std_reward']:9.1f}   regret {summary['mean_regret_events']:7.2f}   "
                f"viol {summary['total_floor_violations']}"
            )

        report["modes"][mode] = per_scenario
        print()

    # --- the criterion Addition A actually set ---
    print(f"{'=' * 78}\nVERDICT\n{'=' * 78}")
    off_regret = report["modes"]["off"]["S3"]["mean_regret_events"]
    ewma_regret = report["modes"]["ewma"]["S3"]["mean_regret_events"]
    lstm_regret = report["modes"]["lstm"]["S3"]["mean_regret_events"]

    report["s3_regret_by_mode"] = {
        "off": off_regret,
        "ewma": ewma_regret,
        "lstm": lstm_regret,
    }
    report["lstm_beats_off_on_s3_regret"] = bool(lstm_regret < off_regret)
    report["lstm_beats_ewma_on_s3_regret"] = bool(lstm_regret < ewma_regret)

    print(
        f"  S3 regret events -- off {off_regret:.2f}  ewma {ewma_regret:.2f}  lstm {lstm_regret:.2f}"
    )
    print(
        "  Addition A success criterion (LSTM measurably reduces S3 regret vs off): "
        f"{'MET' if report['lstm_beats_off_on_s3_regret'] else 'NOT MET'}"
    )
    best_mode = min(report["s3_regret_by_mode"], key=report["s3_regret_by_mode"].get)
    report["best_mode_on_s3_regret"] = best_mode
    ewma_beats_off = ewma_regret < off_regret
    report["ewma_beats_off_on_s3_regret"] = bool(ewma_beats_off)

    print(f"  best mode on S3 regret: {best_mode}")
    if ewma_beats_off and not report["lstm_beats_off_on_s3_regret"]:
        reduction = 100.0 * (off_regret - ewma_regret) / max(1e-9, off_regret)
        print(
            f"\n  FORESIGHT MATTERS, LSTMs DO NOT. EWMA foresight cuts S3 regret events by\n"
            f"  {reduction:.0f}% against no foresight ({off_regret:.1f} -> {ewma_regret:.1f}), so anticipation is\n"
            "  worth something in this environment. The LSTM is worse than both, and the\n"
            "  reason is instructive rather than a training failure: its threat head is\n"
            "  class-weighted to recover rare escalations (balanced accuracy 0.334 -> 0.852),\n"
            "  which necessarily trades precision for recall. Every false positive raises a\n"
            "  floor, and under a pool this scarce a raised floor converts directly into\n"
            "  deferrals. A more *sensitive* threat forecaster is not automatically a better\n"
            "  one: on shared scarce infrastructure, over-triggering costs availability.\n"
            "  Addition A anticipated exactly this reporting shape -- 'the claim becomes\n"
            "  foresight matters, not LSTMs matter'."
        )
    elif not report["lstm_beats_off_on_s3_regret"]:
        print(
            "\n  Null result, reported per Addition A's instruction to report honestly\n"
            "  rather than tune until the number comes out the desired way."
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the E-A foresight ablation.")
    parser.add_argument("--train-seeds", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--eval-max-steps", type=int, default=2_000)
    parser.add_argument("--out", type=str, default="results/ea_ablation.json")
    args = parser.parse_args()

    report = run_ablation(
        n_train_seeds=args.train_seeds,
        n_eval_seeds=args.eval_seeds,
        total_steps=args.steps,
        eval_max_steps=args.eval_max_steps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {out_path}")


if __name__ == "__main__":
    main()
