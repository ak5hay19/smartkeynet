# Non-learned baselines, re-measured under the current configuration

Camera-ready baselines session, 2026-11-16, branch `camera-ready-baselines`.
No training was done and no checkpoint was written. Raw numbers:
[`baselines_remeasured_results.json`](./baselines_remeasured_results.json).

**Why:** the paper quotes always-PQC and random only from "an earlier
configuration". Every non-learned baseline is re-measured below on S1 and
S3 under the same setup Gate W3 used (Addition E Phase 4b): real trained
LSTM forecaster via `RTIoT2022ReplayProvider` over the real held-out
RT-IoT2022 rows, recalibrated floor table, eval seeds 900-907, 250
decisions per episode.

## Command

```bash
python -m experiments.remeasure_baselines        # ~2 minutes; writes docs/baselines_remeasured_results.json
python -m pytest tests/test_reuse_first.py       # legality tests for the new policy
```

Needs the gitignored `checkpoints/lstm_threat_forecaster.pt` and
`data/raw/rt_iot2022/RT_IOT2022.csv`; the supplementary DQN profile also
reads (never writes) `checkpoints/dqn_s1_seed{1,4,7}.pt` and
`checkpoints/s3_masked_seed{1,4,7}.pt` and is skipped if they are absent
(`--skip-dqn-profile`). The forecaster-wiring helper from the gitignored
`scratchpad/phase4b_common.py` is restated, logic unchanged, inside
`experiments/remeasure_baselines.py`.

## Step 1 - the reference reproduces exactly

`StaticThresholdPolicy(0.9)`, same configs / forecaster wiring / seeds as
the Gate W3 run, measured by this script:

| Scenario | Measured | Paper / Gate W3 log | abs diff |
|---|---|---|---|
| S1 | -64,648.4475 | -64,648.45 (logged -64,648.447499999944) | 0.0 |
| S3 | -38,566.8700 | -38,566.87 (logged -38,566.87000000001) | 0.0 |

The script hard-fails (exit 2) before measuring anything else if this does
not match to 2 decimals. As a second cross-check, re-evaluating the three
trained masked-DQN checkpoints per scenario reproduces the published
pooled figures exactly: **-11,524.57 (S1)** and **-10,784.76 (S3)**.

## Results

`× masked DQN` = the policy's `total_reward` mean divided by the
published masked-DQN pooled `total_reward` (both are negative; **> 1
means costlier than the DQN, < 1 means cheaper**). Std for the baselines
is across the 8 eval seeds (population std, as the harness reports it).
The DQN row's std is across its 3 *training* seeds, as published; its
per-eval-seed std is ~1,096 (S1) and 790-1,148 (S3, by training seed).

"Fresh-key issuances / episode" is every non-`REUSE` decision, i.e.
the harness's `rekeys_per_100_requests` × 250 / 100 (cross-checked in the
script against directly counted actions; they agree). "Ticks waiting"
is `deferred_critical_steps`: ticks hybrid-mandatory requests spent in
the deferral queue.

### S1 (steady state)

| Policy | total_reward (mean ± std) | × masked DQN | below-floor rate | regret events | forced-rekey ratio | fresh-key issuances / episode | ticks waiting / episode |
|---|---|---|---|---|---|---|---|
| Masked DQN (published, 3 training seeds pooled) | -11,524.57 (std 0.13 across training seeds) | 1.00 | 0.0000 | 0.00 | 0.099 | 201.6 | 0.0 |
| Static threshold (tuned, t=0.9) | -64,648.45 ± 209.80 | 5.61 | 0.0000 | 0.000 | 0.080 | 250.0 | 0.0 |
| Always-PQC | -16,939.03 ± 1,551.70 | 1.47 | 0.0000 | 0.000 | 0.080 | 250.0 | 0.0 |
| Always-hybrid | -64,776.61 ± 42.54 | 5.62 | 0.0000 | 0.000 | 0.080 | 250.0 | 0.0 |
| Random (legal set, RNG seed 0) | -29,562.86 ± 1,515.82 | 2.57 | 0.0000 | 0.000 | 0.098 | 204.8 | 0.0 |
| **Reuse-first** (new) | **-5,062.76 ± 171.15** | **0.44** | 0.0000 | 0.000 | 0.504 | 39.9 | 0.0 |

### S3 (QKD degradation)

| Policy | total_reward (mean ± std) | × masked DQN | below-floor rate | regret events | forced-rekey ratio | fresh-key issuances / episode | ticks waiting / episode |
|---|---|---|---|---|---|---|---|
| Masked DQN (published, 3 training seeds pooled) | -10,784.76 (std 1,369.43 across training seeds) | 1.00 | 0.0000 | 0.00 | 0.116 | 180.6 | 0.0 |
| Static threshold (tuned, t=0.9) | -38,566.87 ± 1,636.12 | 3.58 | 0.0000 | 0.000 | 0.080 | 250.0 | 0.0 |
| Always-PQC | -16,939.03 ± 1,551.70 | 1.57 | 0.0000 | 0.000 | 0.080 | 250.0 | 0.0 |
| Always-hybrid | -59,346.00 ± 4,096.23 | 5.50 | 0.0000 | 23.375 | 0.082 | 245.2 | 779.4 |
| Random (legal set, RNG seed 0) | -29,562.86 ± 1,515.82 | 2.74 | 0.0000 | 0.000 | 0.098 | 204.8 | 0.0 |
| **Reuse-first** (new) | **-5,062.76 ± 171.15** | **0.47** | 0.0000 | 0.000 | 0.504 | 39.9 | 0.0 |

### Action mix per episode (mean over the 8 eval seeds; DQN = mean over its 3 checkpoints)

| Policy | Scenario | CLASSICAL | PQC | HYBRID | REUSE | REKEY_NOW |
|---|---|---|---|---|---|---|
| Masked DQN | S1 | 127.6 | 27.2 | 1.0 | 48.4 | 45.7 |
| Masked DQN | S3 | 67.1 | 50.5 | 21.4 | 69.4 | 41.6 |
| Static threshold (t=0.9) | S1 / S3 | 0.0 / 0.0 | 0.5 / 102.2 | 249.5 / 147.8 | 0.0 / 0.0 | 0.0 / 0.0 |
| Always-PQC | both | 0.0 | 186.6 | 63.4 | 0.0 | 0.0 |
| Always-hybrid | S1 / S3 | 0.0 / 32.6 | 0.0 / 14.2 | 250.0 / 198.4 | 0.0 / 4.8 | 0.0 |
| Random | both | 27.9 | 42.1 | 67.8 | 45.2 | 67.0 |
| Reuse-first | both | 10.2 | 10.4 | 19.2 | 210.1 | 0.0 |

## What these numbers say

1. **Reuse-first beats the masked DQN on total reward, on both
   scenarios, by about 2×** (-5,062.76 vs -11,524.57 on S1; vs -10,784.76
   on S3), with the same 0.0000 below-floor rate and zero regret events.
   Its *worst* eval-seed reward (-5,258.7; best -4,731.6) is better than
   the DQN's *best* training-seed mean on either scenario (-8,978.51, S3
   seed 7). This is **not a tie**. The paper's remark that a reuse-first heuristic "could close
   much of the gap" understates it: under this reward it does not close
   the gap to the DQN, it overshoots it.
2. **What it does and why.** It takes `REUSE` 210 times in 250 decisions;
   the DQN takes it 48 (S1) and 69 (S3). `REUSE` is the cheapest action in
   the environment's cost model (latency unit 0.2 vs 1.0-1.5 for a fresh
   tier), so a policy that reuses whenever the mask allows is cheap by
   construction. The DQN does not exploit this fully: it issues ~202
   (S1) / ~181 (S3) fresh keys per episode versus 40 for reuse-first. On
   S3 the DQN's own training seeds are consistent with this reading: the
   seed that reuses most (117.9 `REUSE`/episode, seed 7) has the best
   reward (-8,978.51), the one that reuses least (30.1, seed 4) the
   worst (-12,292.95). I did **not** investigate *why* the DQN under-reuses
   (training horizon, exploration, reward scale, state features); that is
   open.
3. **Security is not what separates them.** Every policy, including
   reuse-first, has a 0.0000 below-floor rate. That is by construction:
   the policy only ever chooses inside the action mask, `compute_mask`
   forbids a `REUSE` whose delivered tier is below the current floor or
   whose key is past the age cap, and the new tests prove reuse-first never
   leaves the mask (exhaustively over all 31 non-empty masks, and on real
   S1 and S3 episodes with 0 illegal choices and 0 floor violations).
   The comparison is purely about reward (latency/energy/freshness/pool
   cost).
4. **The other baselines (well-tuned or not) are far behind.** The tuned
   static threshold is 5.61× (S1) and 3.58× (S3) the DQN's cost, as in
   Gate W3; the grid's best threshold is the top of the grid (0.9), so
   the "tuned" threshold is grid-edge, as it was in the Gate W3 run. On S1
   `pool_fill` stays above 0.9 on ~249.5 of 250 decisions, so the policy
   is effectively always-hybrid (249.5 hybrid serves/episode; its reward
   is within 0.2% of always-hybrid's). Always-PQC is the strongest
   non-reusing baseline (1.47× / 1.57×); random is 2.57× / 2.74×.
5. **Overhead (rekey / "handshake") question.** Fresh-key issuances per
   episode: always-PQC / always-hybrid / threshold 250 (they never reuse),
   random 204.8, masked DQN 201.6 (S1) / 180.6 (S3), reuse-first 39.9.
   Reuse-first's forced-rekey *ratio* is high (0.504) because it keeps
   keys until the age cap forces a rekey, so about half of its few rekeys
   are forced; the ratio is a share of rekeys, not a rate (the always-X
   policies show 0.080 while issuing 250). Only these counts, and the
   environment's latency units, are measured: **no wall-clock handshake or
   KMS latency is measured** anywhere here, and the harness exposes no
   per-decision handshake cost beyond the `_LATENCY_UNITS` table (REUSE
   0.2, CLASSICAL 1.0, PQC 1.2, HYBRID 1.5).

## Consequences to decide on (for the author, not for this document)

- PLAN.md Hard Rule 7: "If a tuned threshold ties the DQN, report it
  honestly and investigate environment design first." The tuned
  threshold does not tie the DQN (it is 3.6-5.6× worse), but a
  one-line heuristic does better than the DQN under the current reward,
  which is a stronger version of the same concern and should be
  disclosed in the camera-ready, not left implicit.
- The Gate W3 claim (DQN beats a tuned static threshold) still holds as
  stated. What no longer holds is any implication that the DQN is
  near-best among simple policies.
- Caveat on reading reuse-first as "better": the reward only prices a
  reused key through the 0.1-weighted freshness term plus the hard
  key-age cap, so "cheaper under this reward" is not the same as
  "better risk posture".

## Caveats on this measurement

- **S1 and S3 are identical for every policy that never exhausts the
  pool**: always-PQC, random, and reuse-first have bit-identical
  rewards on both scenarios (verified: always-PQC's per-step reward trace
  is identical on S1 and S3 on seed 900, while S3's pool fill does fall
  to 0.60 vs 1.00 on S1). S3's degradation only changes the outcome for
  policies that draw heavily on the pool - here, always-hybrid (23.4
  regret events, 779 waiting ticks) and, in the published results, the
  soft-reward baseline. So **S3 does not stress reuse-first at all**; this
  says nothing about how it would do under harsher scarcity.
- **Random depends on its RNG seed** (one stateful `RandomPolicy`
  instance runs the 8 episodes in sequence, as `evaluate_multi_seed`
  requires). Primary row: seed 0. Sensitivity: seed 1 -30,686.53 ± 1,906.44,
  seed 2 -30,399.91 ± 2,091.30 (S1 and S3 identical). All stay within ~4% of
  the primary row.
- The previously published always-PQC / random S3 figures (-16,906.98 and
  -31,904.80, "earlier configuration") differ from these (-16,939.03 and
  -29,562.86). The earlier measurement
  (`dashboard/samples/baselines_s3_data.json`) was run through
  `evaluate_multi_seed` without the real-replay `forecast_provider_factory`,
  which is the likely cause; I did not isolate it. The static-threshold
  figure is unchanged to the cent (-38,566.87) between the two setups;
  I did not investigate why.
- `total_reward` is one scalar; this document does not re-measure the
  paper's other Table V columns beyond those listed above (p99 latency
  saturates at 1.5000 for most policies and is omitted, see README).
- Single configuration (S1 default, S3 as committed), 8 eval seeds, one
  trained DQN family (3 training seeds). Nothing was retrained or tuned
  for the DQN here.
