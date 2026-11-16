# S5 attacked-floor check on the current dose-response sweep

2026-11-16, branch `camera-ready-baselines`. Read-only: no training, no
checkpoint written, nothing in `env/`, `agents/`, `attack/`,
`experiments/harness.py`, the reward, or the floor table was touched.
Raw data: [`s5_attacked_floor_check_results.json`](./s5_attacked_floor_check_results.json)
(per episode, all 110 episodes).

**Question.** The paper says: *"at every strength the masked agent's
decisions cleared the floor it was shown."* That was verified on the
original 2026-08-26 sweep. Is it also true on the CURRENT sweep behind
Fig. 4 (V(π) = 0.0000 through α = 0.1, up to 0.0072 by α = 0.5, a step to
0.2984 at α ≥ 0.6)?

**Answer: yes.** Across all 55 masked-agent episodes (11 alphas × 5 seeds,
13,750 decisions) there are **0 decisions below the floor derived from the
attacked posture**, and 0 choices outside the action mask. The harness
exposes this count already, so nothing was modified.

## Command

```bash
python -m experiments.s5_attacked_floor_check     # ~1 minute, writes docs/s5_attacked_floor_check_results.json
```

Needs the gitignored `checkpoints/dqn_s2.pt`,
`checkpoints/soft_reward_baseline_s2.pt`,
`checkpoints/lstm_threat_forecaster.pt`. Exit code 0 means Step 1
reproduced; 2 means the published V(π) did not.

## Setup (same as the Phase 4b / 2026-09-14 sweep)

Scenario S2; masked DQN `checkpoints/dqn_s2.pt` with
`configs/scenarios/s2_hndl.yaml`; soft-reward baseline
`checkpoints/soft_reward_baseline_s2.pt` with
`configs/soft_reward_baseline_s2.yaml`; the real trained
`LSTMForecastProvider` (cold-start fix in) as both the attacked base
provider and the true-posture shadow provider; α = 0.0, 0.1, …, 1.0; eval
seeds 900-904; greedy frozen policies. The driver calls the harness's own
`run_scenario_under_attack`, once per (alpha, seed, policy) — the function
`evaluate_attack_multi_seed` loops over. The setup was re-derived from the
gitignored `scratchpad/phase4b_s5_dose_response.py`.

## The two quantities (both come from the harness)

- **Below the ATTACKED floor** = `scenario_result.floor_violations`:
  decisions whose *delivered* tier was below the floor the agent was shown
  (`state["policy_floor"]`, computed from the attacked posture). This is
  the quantity in the paper's sentence.
- **V(π), below the TRUE floor** = `below_floor_true_count / decisions`
  (equation 4, what Fig. 4 plots): decisions whose delivered tier was below
  the floor the *true*, unshaped posture would have required.

Cross-checks added by the script, without touching the harness: a
transparent wrapper records each decision's shown floor and whether the
chosen action was legal under its mask; both harness counts are then
**recomputed independently** from the harness's `delivered_tier_log` /
`true_floor_log` and must agree exactly (the script aborts on any
disagreement; none occurred).

## Step 1 — the published V(π) reproduces exactly

Every one of the 22 (agent, alpha) cells matches the published sweep
(PROGRESS.md 2026-09-14) **and** the committed
`dashboard/samples/results_data.json` to 4 decimals (the `Published`
column below). `step1_all_match = true`.

## Step 2 — masked DQN (11 alphas × 5 seeds × 250 decisions)

| α | V(π) vs TRUE floor (mean ± std over seeds) | published | below TRUE floor (count / decisions) | **below ATTACKED floor** | choices outside the mask | decisions where the shown floor < the true floor |
|---|---|---|---|---|---|---|
| 0.0 | 0.0000 ± 0.0000 | 0.0000 | 0 / 1250 | **0** | 0 | 0 |
| 0.1 | 0.0000 ± 0.0000 | 0.0000 | 0 / 1250 | **0** | 0 | 0 |
| 0.2 | 0.0008 ± 0.0016 | 0.0008 | 1 / 1250 | **0** | 0 | 1 |
| 0.3 | 0.0024 ± 0.0020 | 0.0024 | 3 / 1250 | **0** | 0 | 3 |
| 0.4 | 0.0048 ± 0.0030 | 0.0048 | 6 / 1250 | **0** | 0 | 6 |
| 0.5 | 0.0072 ± 0.0016 | 0.0072 | 9 / 1250 | **0** | 0 | 9 |
| 0.6 | 0.2984 ± 0.0070 | 0.2984 | 373 / 1250 | **0** | 0 | 492 |
| 0.7 | 0.2984 ± 0.0070 | 0.2984 | 373 / 1250 | **0** | 0 | 492 |
| 0.8 | 0.2984 ± 0.0070 | 0.2984 | 373 / 1250 | **0** | 0 | 492 |
| 0.9 | 0.2984 ± 0.0070 | 0.2984 | 373 / 1250 | **0** | 0 | 492 |
| 1.0 | 0.2984 ± 0.0070 | 0.2984 | 373 / 1250 | **0** | 0 | 492 |

Totals: 13,750 decisions, **0 below the attacked floor, 0 episodes with
any violation, 0 illegal choices.** (At α = 0.6, per-seed counts below the
true floor are 74/77/74/76/72 out of 250 each.)

## Step 3 — soft-reward baseline, for contrast

| α | V(π) vs TRUE floor (mean ± std) | published | below TRUE floor (count / decisions) | **below ATTACKED floor** | choices outside its (relaxed) mask | shown floor < true floor |
|---|---|---|---|---|---|---|
| 0.0 | 0.3912 ± 0.0723 | 0.3912 | 489 / 1250 | **489** | 0 | 0 |
| 0.1 | 0.3912 ± 0.0723 | 0.3912 | 489 / 1250 | **489** | 0 | 0 |
| 0.2 | 0.3912 ± 0.0723 | 0.3912 | 489 / 1250 | **489** | 0 | 1 |
| 0.3 | 0.3912 ± 0.0723 | 0.3912 | 489 / 1250 | **488** | 0 | 3 |
| 0.4 | 0.3912 ± 0.0723 | 0.3912 | 489 / 1250 | **487** | 0 | 6 |
| 0.5 | 0.3912 ± 0.0723 | 0.3912 | 489 / 1250 | **486** | 0 | 9 |
| 0.6 | 0.4272 ± 0.0874 | 0.4272 | 534 / 1250 | **353** | 0 | 492 |
| 0.7 | 0.4272 ± 0.0874 | 0.4272 | 534 / 1250 | **353** | 0 | 492 |
| 0.8 | 0.4272 ± 0.0874 | 0.4272 | 534 / 1250 | **353** | 0 | 492 |
| 0.9 | 0.4272 ± 0.0874 | 0.4272 | 534 / 1250 | **353** | 0 | 492 |
| 1.0 | 0.4272 ± 0.0874 | 0.4272 | 534 / 1250 | **353** | 0 | 492 |

Totals: 4,693 of 13,750 decisions below the attacked floor, in **all 55**
episodes. Its mask has no floor rule (`security_masking: false`), so
"outside the mask" is 0 for it too and says nothing about security; the
violations are against the floor it was shown but is not forced to obey.

## What this means for the paper

1. **The sentence is true for the current sweep**, as measured here:
   "at every strength the masked agent's decisions cleared the floor it
   was shown" — 0 of 13,750 decisions fell below the attacked floor, at
   every α and every seed, and the soft-reward baseline does not (4,693).
2. **It is a statement about the floor the agent was shown, not the true
   floor.** The attack does work against the masked agent from α = 0.2:
   the shown floor is lower than the true floor on 1, 3, 6, 9, then 492 of
   1,250 decisions, and the masked agent follows the lowered floor
   faithfully, so V(π) vs the TRUE posture is 0.0008 … 0.0072, then 0.2984
   at α ≥ 0.6. At the plateau 373 of those 492 decisions were delivered
   below the true floor; the other 119 were delivered at or above it
   (the agent happened to serve a tier at least as strong as the true
   floor anyway). Fig. 4 and this sentence are therefore consistent, but the
   text must not let a reader infer V(π) = 0 or "the attack fails". At
   low α (≤ 0.5), every decision with a lowered shown floor was also
   delivered below the true floor (e.g. 9 of 9 at α = 0.5).
3. **The zero is by construction, not a lucky outcome.** The masked
   policy can only choose inside a mask that already forbids delivering
   below the shown floor (Hard Rule 2). This check confirms empirically,
   via a recount independent of the harness's own counter, that the mask
   held on every decision of the current sweep; it does not add evidence
   that the floor the agent was shown is the right one. That is exactly
   the gap Fig. 4 measures.

## Limits of this check

- One masked checkpoint (`dqn_s2.pt`, training seed 0), 5 eval seeds
  (900-904), the same as the published sweep; no new seeds were run.
- The check says nothing about decisions on other scenarios; it covers the
  S2/S5 sweep only.
- The sweep still feeds the LSTM the zero-padded 2-dim `[qber, load]`
  placeholder (README "Known limitations"); the 0.2984 plateau is a
  property of that setup.
