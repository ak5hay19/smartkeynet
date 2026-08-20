# SmartKeyNet — Progress Tracker

> **Update convention:** updating this file's checkboxes and the "Next task"
> line is part of the same end-of-session step as updating `SESSION_LOG.md`.
> This file gets *updated*, not rewritten, at the end of every session.
> It exists so a fresh Claude Code session (or a new person) can read
> `PLAN.md` + `SESSION_LOG.md` + this file and know what's done and what
> the single next task is, without reconstructing status from log prose.

---

## Next task

**2026-08-19/20 full build session. Gate W3 was attempted for real and
FAILED; the S5 steering result HOLDS.** See SESSION_LOG.md 2026-08-19/20
for the complete account, and `docs/report.md` §5 and §7 for the writeup.

### The single next task

**Decide what to do about Gate W3's negative result.** The tuned
threshold beats the masked DQN on both S1 and S3, by a wide margin,
checkpoint-averaged across 5 training seeds:

| scenario | masked DQN (total_reward) | tuned threshold |
|---|---|---|
| S1 | **-3820.8 +/- 1623.9** | **-955.5 +/- 7.7** |
| S3 | **-97475.6 +/- 204475.5** | **-945.8 +/- 8.6** |

PLAN2 §3.2's disqualification rule is triggered. Two concrete,
pre-committed options:

**(a) Test the heavy-tailed-reward hypothesis.** The proximate cause is
visible: the agent rekeys ~6x too often
(`rekeys_per_100_requests` 66.8 vs 10.6 for baselines), which at ~2.5
reward units per rekey accounts for almost the whole S1 gap. With a
genuinely scarce pool the reward is now heavy-tailed -- a single step can
score -442 while a typical step scores -0.4 -- which is a hard
regression target for an unclipped DQN. Reward clipping or normalization
is standard practice (the DQN Nature paper clips to [-1,1]), is an
*agent-side* change that touches neither the environment nor the
reported objective, and does not go near Hard Rule 1. Pre-commit the
decision rule before running: if clipped training does not clear the
tuned threshold on S1 within the same 25,000-step budget across 5 seeds,
option (b).

**(b) Report the negative result and reframe.** This is already written
up honestly in `docs/report.md` §5.1 and §6.3, and the project's stated
contribution (PLAN2 §2.3, the steering result) does not depend on it.
"The masking is what resists steering, and a tuned threshold behind the
same mask would be equally immune" is a defensible and arguably more
interesting thesis than a marginal RL win.

**Do NOT** add a security term, weaken masking, or move an environment
parameter toward the agent. Every environment change this session was
made *before* any training run and in the direction that makes the
agent's case harder; that property is worth more than the gate.

### Still open, unchanged

The **checkpoint-to-checkpoint oscillation** (items 1-6 below, six
sessions deep) is unresolved and was deliberately not chased this
session. It is present at full amplitude in the Gate W3 numbers:
within-run `total_reward` stdev is **1447 +/- 1411** on S1, comparable to
the mean itself. Every DQN number in the repo is now produced through
`experiments/campaign.py`, which is checkpoint-averaged, eval-seed-
averaged and multi-seed by construction, with `SeedSpread` making the
spread a required field so a mean cannot be reported without it.

### Newly opened this session

1. **With a realistic noisy detector, the one-way ratchet saturates.**
   A single momentary threat peak -- 1 decision in 2,000, score 0.269 --
   trips the ratchet and changes the floor regime for the remaining 795
   decisions. That is Hard Rule 2 working exactly as specified, not a
   bug, but it is an operational property that deserves a paragraph in
   the report's limitations and possibly a discussion of episode-scoped
   vs. deployment-scoped ratchet lifetime.
2. **The E-A ablation and the closing results table are long runs.**
   Both are implemented and smoke-tested; `results/*.json` needs
   regenerating whenever the environment changes, and the dashboard
   renders an explicit "not yet run" until they exist.

### Prior thread (unchanged, for context)

The checkpoint-oscillation investigation, five sessions deep before this
one. Reproduced verbatim; nothing below was re-run or revised this
session.

1. First load-spike session: `forced_rekey_ratio` dropped from flat
   S1's `1.000` to `0.256`/`0.872` across two training seeds —
   confirmed direction, spread unquantified.
2. 10-seed follow-up sized the spread (`0.190`-`1.000`, mean `0.735`,
   stdev `0.275`, leaning bimodal) and found why it was worth
   distrusting: `agents/dqn.py` never seeded its own weight
   init/exploration/replay sampling at all — `training.seed` only ever
   reached the environment. Flagged, not fixed.
3. **This session fixed that**: `DQNAgent.__init__` gained a `seed`
   parameter (seeds `random`+`torch` before `QNetwork` construction),
   `experiments/train.py` now threads `training.seed` to both the env
   and the agent, and a new test (`tests/test_dqn.py`) proves same-seed
   agents produce identical action sequences and different seeds
   diverge. Re-ran the identical 10-seed load-spike sweep with the fix
   in place:

   | | sorted `forced_rekey_ratio` | mean | stdev |
   |---|---|---|---|
   | Before fix (unseeded DQN) | `0.190, 0.417, 0.418, 0.703, 0.872, 0.895, 0.914, 0.971, 0.971, 1.000` | 0.735 | 0.275 |
   | **After fix (seeded DQN)** | `0.102, 0.148, 0.463, 0.553, 0.730, 1.000, 1.000, 1.000, 1.000, 1.000` | 0.700 | 0.345 |

   **The fix made the spread wider, not tighter** — exactly half the
   seeds (5/10) now land at the *exact* never-proactive ceiling
   `1.000` (vs. 1/10 before), while the other half ranges `0.102`-
   `0.730`, including the best result seen across either sweep
   (`0.102` — 90% proactive). With RNG-conflation ruled out as the
   explanation (both sweeps used identical config; only the seeding
   fix changed), this looks like a genuine, structural
   learn/don't-learn split: roughly half of random weight-init +
   exploration trajectories discover any proactive rekeying at all
   within a 25,000-step budget, and half settle into "wait until
   forced" and stay there. `reward.w_fr`/`reward.c_rekey_base` remain
   uninvolved in any of this — no `reward.*` change made or requested
   across any of the three sessions.

4. **2026-08-17 budget probe** ran a strict, pre-committed decision
   rule to test the training-budget hypothesis before building
   anything: 3 of the 5
   ceiling-stuck seeds (`1`, `4`, `7`) trained to `50,000` and `75,000`
   steps (up from `25,000`), load-spike enabled, same config. Verdict:
   **DID NOT BUDGE** — no seed cleared `forced_rekey_ratio <= 0.5` at
   75k, and none held a monotonically non-increasing trajectory. This
   isn't just "training was too short": seeds `1` and `4` reached
   genuinely good intermediate values at 50k (`0.1020` and `0.6585`,
   `seed=1`'s matching the best result seen in any sweep) and then
   **regressed back to the exact `1.000` ceiling by 75k**. Seed `7`
   never moved off the ceiling at any budget tested. A real, substantial
   improvement appearing and then being lost mid-training is a
   different, more concerning phenomenon than either of the two
   hypotheses framed on 2026-08-10 (marginal budget vs. static bimodal
   landscape) — it looks more like training instability / policy
   forgetting within a single continuous run than a simple
   convergence-speed question. See SESSION_LOG.md 2026-08-17 for all
   six data points.

5. **2026-08-18 dense diagnostic** re-ran seeds `1`, `4`, `7` to
   75,000 steps with `eval_every=1000` (~75 snapshots instead of 3) to
   localize the 2026-08-17 regression's onset. Found none — max
   single-window swings of `0.90`-`0.92` occur throughout the entire
   run for every seed, not just around the buffer-capacity crossing at
   step `50,000`. **Buffer-capacity hypothesis: ruled out** — loss is
   unremarkable around step 50,000 (matches its neighbors, no spike),
   and swing amplitude doesn't shrink or shift near that step. There
   *is* a real, separate, noisy long-run trend — the fraction of
   snapshots sitting exactly at the `1.000` ceiling roughly doubles
   each third of a run (seed 1: `8% -> 48% -> 64%`) — but it's a
   smooth drift across the whole run, not a step-change tied to
   `12,500` (epsilon-decay-complete) or `50,000` (buffer-capacity)
   specifically, and the swings persist at full amplitude even in the
   run's final third.

6. **2026-08-18's second same-day diagnostic** directly tested both
   candidate explanations item 5 left open, using a measurement-only
   reimplementation of `train()`'s loop (`DQNAgent`/`SmartKeyNetEnv`/
   `GreedyDQNPolicy`/`run_scenario` imported and called exactly as
   `train()` does, not a new training path) run for seeds `1`, `4`, `7`
   to `75,000` steps, `eval_every=750` (deliberately not a multiple of
   `target_update_every=1000`), 8 fixed eval seeds (`900`-`907`) per
   snapshot instead of one. **Hypothesis (a) — single-episode eval
   noise — RULED OUT**: the spread across a checkpoint's own 8 eval
   draws is small (mean `0.05`-`0.07`), but the *mean* of those 8 draws
   still swings 4-6x larger (mean `0.21`-`0.30`, still `>0.5` on
   19-27% of adjacent 750-step snapshots) from one checkpoint to the
   next — averaging away eval-episode randomness does not tame the
   swing. **Hypothesis (b) — eval-cadence/target-sync aliasing —
   evidence against, not a clean rule-out** (between-session
   comparison, not a controlled ablation): the non-aligned
   `eval_every=750` cadence swings with the same character and
   magnitude as 2026-08-18's earlier aligned `eval_every=1000` data
   (max swing `~0.89`-`0.91` both), if anything slightly *less*
   frequently — the opposite of what aliasing inflation would predict.
   The 2026-08-10 ceiling-fraction long-run drift (item 5) was
   confirmed to survive 8-seed averaging too. **Net effect: both
   leading explanations for the swings are now disfavored, and the
   standing "report as a distribution across eval seeds"
   recommendation from item 5 is revised** — it doesn't fix the
   problem, since the distribution's own mean is nearly as unstable
   checkpoint-to-checkpoint as a single draw. See SESSION_LOG.md's
   second 2026-08-18 entry for the full comparison tables.

**Concretely next (pick one — a real decision, not more solo running):**
(a) a direct Q-value-margin inspection at a known swing (e.g. seed 1's
step `71250->72000` transition, `forced_rekey_ratio` mean dropping from
`~0.90` to `0.149`) — check whether the greedy action at the eval
episode's early, trajectory-determining decision points sits at a
near-tie in Q-value between two actions, such that one `learn()` call's
weight update is enough to flip the argmax and cascade into a very
different downstream key-age trajectory for the rest of a short
250-step episode (this would explain instability surviving 8-seed
averaging without the underlying Q-values needing to move much) — not
run yet, no repo code exists for this, would need new (flagged-first)
instrumentation in a copy of the eval loop, not `agents/dqn.py` itself;
or (b) treat "the greedy policy genuinely oscillates checkpoint-to-
checkpoint, mechanism unknown" as a standing, now twice-confirmed
property of this training setup and move on to real S4 regardless,
using **checkpoint-averaged** (e.g. mean of several `eval_every`
windows near the end of training), not single-checkpoint, comparisons
for any future DQN-vs-baseline number. Real S4 still needs the tenant
graph (`build_tenant_graph`/`RequestGenerator`, still
`NotImplementedError`) for genuine per-tenant flooding regardless of
this thread — the load-spike diagnostic remains a cruder, tenant-blind
stand-in, not real S4 itself. Do not mistake `configs/default.yaml`'s
`load_spike:` block or `random_request_generator`'s `load_spike` kwarg
for finished S4 work — both are documented as a diagnostic stub
throughout.

**Superseded by this session:** the concluding paragraph above says real
S4 "still needs the tenant graph ... still `NotImplementedError`". That
graph now exists (`build_tenant_graph`/`RequestGenerator`), real S1-S6
scenario dispatch is wired, and the `load_spike` diagnostic stub is
retained only so that session's runs stay reproducible -- no scenario in
the current grid uses it. The recommendation in option (b) -- "treat the
oscillation as a standing property and use checkpoint-averaged
comparisons" -- was the path taken, and `experiments/campaign.py` makes
it structural.

---

## Milestone checklist

Pulled from PLAN.md §10 (kickoff order) and §7 / split.md §2 (weekly gates).

- [x] Repo scaffolded (folder skeleton, `env/contracts.py` frozen, CI + PR template)
- [x] Gate W1 — `pytest` green on stubs; contracts frozen and committed
- [x] Spine wired end-to-end into `environment.py` — pool sim + deferral queue
      + masking + forecaster (EWMA) + request stream (random) + full reward
      formula, `reset()`/`step()` running a complete S1 episode with regret
      logging
- [x] Gate W2 — env `step()` runs end-to-end with a random valid agent across
      a full S1 episode; regret events logged
- [x] Real NetworkX tenant graph + graph-driven `RequestGenerator`
      (`build_tenant_graph`, `RequestGenerator.reset/step`) — PLAN.md §10 step 4
      *(2026-08-19: 50-node graph, four differentiated tenant profiles, documented
      six-step generator, Hard Rule 3 swappability asserted by test.)*
- [x] Masked DQN agent (`agents/dqn.py`)
- [x] Four tuned baselines — always-PQC, always-hybrid, static-threshold
      (grid-searched), random (`agents/baselines.py`) + comparison harness
      (`experiments/harness.py`) — Hard Rule 7
- [x] 🚩 Gate W3 (make-or-break) — attempted for real 2026-08-19. **FAILED.**
      The tuned threshold beats the masked DQN on both S1 and S3, checkpoint-averaged
      across 5 training seeds (S1: `-3820.8 +/- 1623.9` vs `-955.5 +/- 7.7`;
      S3: `-97475.6 +/- 204475.5` vs `-945.8 +/- 8.6`). PLAN2 §3.2's disqualification
      rule is triggered. Reported as measured — no reward term added, no masking
      weakened, no environment parameter moved toward the agent afterwards.
      **The agent did learn proactive rekeying** (`forced_rekey_ratio` 0.107-0.211 vs
      0.93-1.00 for every baseline — the behaviour four prior sessions chased and
      never got), but rekeys ~6x too often (`rekeys_per_100_requests` 66.8 vs 10.6),
      which accounts for almost the whole S1 gap. See "Next task" for the two
      pre-committed options.
- [x] Soft-reward baseline agent reproducing Noetzold (`agents/soft_reward_baseline.py`)
- [x] Scenario dispatch S2-S4 wired into `environment.py`
      *(2026-08-19: plus S5's external-trace hook and S6's schedule. Each scenario's
      distinguishing behaviour verified and pinned by test before any result was
      reported.)*
- [x] Real LSTM dual-head forecaster (Addition A) — `forecaster/model.py`,
      `forecaster/dataset.py`, `forecaster/train.py`, `LSTMForecastProvider`
      in `env/forecast_provider.py`
      *(2026-08-19: shared trunk, threat head balanced accuracy 0.9312 vs a 0.6817
      majority-class rate; pool head val MAE 0.189. Frozen during DQN training,
      asserted by test.)*
- [x] E-A foresight ablation (off / ewma / lstm on S3 + S6) — `experiments/ablation.py`
      *(implemented and smoke-tested; all arms run on `threat_input.source:
      rt_iot2022` because evaluating the LSTM under the scalar `scenario` source
      measures a distribution mismatch, not the value of foresight)*
- [x] 🎯 Steering attack — adversarial threat-trace generator (`attack/steering_trace.py`)
      + attack run producing the split-screen result — Gate W5, headline contribution, never cut
      *(2026-08-19 **RESULT HOLDS**: soft-reward arm 14.0% → 27.8% below the
      sensitivity-class floor as the attack strengthens; masked arm identically
      0.0% at every dose. See SESSION_LOG.md and docs/report.md §5.2, including
      three qualifications reported rather than smoothed.)*
- [x] S6 migration wave (scripted schedule, held-out eval only)
      *(2026-08-19: three cohort phases; a schedule entry that would LOWER a floor
      is rejected at construction; Hard Rule 8 enforced in code at both training
      entry points.)*
- [x] Live dashboard (`dashboard/app.py`) — all 7 panels of PLAN2 §7
      *(2026-08-19: Dash app + self-contained static export, every panel from real
      runs; missing artefacts render "not yet run", never a placeholder.)*
- [x] AWS-KMS-style API facade (`api/main.py`)
      *(2026-08-19: primitive-honesty matrix published on `/Health` — ML-KEM is
      simulated and every response says so.)*
- [x] Report (`docs/report.md`) — written, with real numbers and §7 recording
      the four environment corrections made this session

---

## Per-file status

Status values: **not started** (stub, `raise NotImplementedError`, 11-line
import-smoke test only) · **stub (partial)** (some functions real, some
still `NotImplementedError`) · **implemented+tested** (real logic, real
behavioral tests, part of the green `pytest` run).

### env/

| File | Status | Notes |
|---|---|---|
| `env/contracts.py` | implemented+tested | Frozen interface contract — `Action`, `StateDict`, `ForecastProvider` ABC, `Request`, event-log TypedDicts. 5 real tests (`test_contracts.py`). |
| `env/pool_sim.py` | implemented+tested | `PoolSim` (refill/drain/exhaustion) + `SyntheticSKRQBERTrace`. 19 tests (`test_pool_sim.py`). |
| `env/deferral_queue.py` | implemented+tested | `DeferralQueue.enqueue/tick/pop_servable`, priority+FIFO, cumulative-headroom draw. 8 tests (`test_deferral_queue.py`). |
| `env/masking.py` | implemented+tested | `PolicyTable` (placeholder floor table, sticky ratchet) + `compute_mask`. 14 tests (`test_masking.py`). Floor table not yet calibrated against Q-OPSEC data. |
| `env/forecast_provider.py` | implemented+tested | `MovingAverageForecaster` (EWMA fallback), 12 tests. **2026-08-19: threat squash recalibrated** — it was `sigmoid(mean)` over non-negative placeholder features, so `threat_score` could never fall below 0.5 (the ELEVATED anchor) and, with the one-way ratchet, every episode pinned at ELEVATED from its second tick (measured 249/250 decisions on benign S1) leaving the floor table's CALM row unreachable. Now calibrated for standardized features: benign → ~0.05 (CALM), +3σ → ~0.90 (HIGH). `LSTMForecastProvider` is re-exported here lazily from `forecaster.model` (PEP 562 `__getattr__`, so `off`/`ewma` runs never import torch). |
| `env/request_generator.py` | implemented+tested | `random_request_generator()` implemented+tested (11 tests, `test_request_generator.py`), incl. 3 new 2026-08-10 tests for its optional `load_spike` kwarg — a periodic, config-driven arrival-rate diagnostic (**explicitly not real S4** — see that session's SESSION_LOG.md entry and `configs/default.yaml`'s `load_spike:` block). `build_tenant_graph()` and `RequestGenerator` (graph-driven stream) still `raise NotImplementedError` — real S4 needs these. |
| `env/decision_trace.py` | implemented+tested | PLAN2 §7.3's six-step trace, Hard Rule 10. One source of truth, imported by both `api/` and `dashboard/`, reimplemented by neither. No model, no heuristic, no narration. 81 tests, including a parametrized check that `illegal_reason()` agrees with `compute_mask()` on every (floor × key_age × pool × key_type) combination — which caught a real disagreement on first run. |
| `env/environment.py` | implemented+tested | `SmartKeyNetEnv.reset/step/action_mask` fully wired (pool + deferral + masking + forecast + reward + session-key state). 17 behavioral tests incl. the split.md Gate W2 tests (`test_environment.py`). Only S1 scenario dispatch exists; S2-S6 config is read but not acted on. 2026-08-10: wired `config["load_spike"]` through to `random_request_generator` (design decision 9) — orthogonal to scenario dispatch, not a substitute for it. |

### agents/

| File | Status | Notes |
|---|---|---|
| `agents/dqn.py` | implemented+tested | `flatten_state(state, has_forecast)` (genuinely variable-length: 13 dims under `off`, 28 under `ewma`/`lstm`; `has_forecast` is now an **explicit required parameter** — the earlier `state["threat_score"] != 0.0` inference trick was removed 2026-08-08 as fragile-by-accident, see that session's log entry). `QNetwork` (2-hidden-layer MLP), `DQNConfig`/`load_dqn_config` (reads `configs/default.yaml`'s `dqn:` block), `DQNAgent(state_dim, has_forecast, config, seed=None)` (internal circular-buffer replay, `act`/`observe`/`learn`/`save`/`load`, `has_forecast` fixed once at construction and threaded through every internal `flatten_state` call) — masking applied structurally at both action-selection *and* bootstrap-target time (Hard Rule 2), no security term anywhere (Hard Rule 1). **2026-08-10: `seed` parameter added** — reseeds `random`+`torch`'s global RNGs before `QNetwork` construction when given, so weight init/exploration/replay sampling are genuinely reproducible (`seed=None` default is unchanged prior behavior); fixes a gap the same day's earlier sessions found and flagged (see `experiments/train.py`'s row and SESSION_LOG.md). 26 tests (`test_dqn.py`, up from 23), incl. a regression test for a foresight-mode state with `threat_score == 0.0` still flattening to 28 dims, an integration test training against the real `SmartKeyNetEnv` on S1 for 3000 steps with loss trending down, and 3 new seed tests (same-seed reproducibility, different-seed divergence, `seed=None` leaves ambient state alone). Not yet run to convergence — that's `experiments/train.py`. |
| `agents/baselines.py` | implemented+tested | `AlwaysPQCPolicy`, `AlwaysHybridPolicy`, `StaticThresholdPolicy` (incl. `grid_search`), `RandomPolicy` — all real, sharing a `_lowest_legal_action` fallback. 261 tests (`test_baselines.py`), incl. an adversarial parametrized sweep over all 31 non-empty action masks per policy (never returns an illegal action, however contrived the mask). |
| `agents/soft_reward_baseline.py` | implemented+tested | `soft_reward()` + tabular `SoftRewardAgent` (no mask — the critique target). 11 tests. Q-rows initialize to the **myopic value**, not zeros: every real soft-reward value is negative, so zero-init made unvisited actions dominate learned ones and the greedy policy returned whatever it had never tried — a dose-response curve flat at 44.0% across every dose, i.e. the agent reporting its initialization. |

### forecaster/

| File | Status | Notes |
|---|---|---|
| `forecaster/model.py` | implemented+tested | `DualHeadLSTM` (shared trunk + threat/pool heads), `LSTMForecastProvider`, shared posture mapping. 10 tests, including one that drives 20 updates and asserts not a single weight moved. |
| `forecaster/dataset.py` | implemented+tested | `extract_flow_features` (ONE implementation, two callers — Hard Rule 11), `FeatureStandardizer`, `RTIoT2022Dataset` with a degeneracy guard, `ThreatTraceSampler`, `build_rollout_dataset`. 19 tests. Standardization is **benign-referenced**: against the whole (90%-attack) capture the threat signal inverts — Cohen's d −0.98 the wrong way vs +4.43 the right way. Parsed captures are cached (a campaign resets thousands of times; re-parsing 123k rows each time dominated everything else). |
| `forecaster/train.py` | implemented+tested | Joint offline training of both heads on one shared trunk. Real run: balanced accuracy **0.9312** vs a **0.6817** majority-class rate, pool MAE 0.189. Refuses S6 (Hard Rule 8). 4 tests. |

### metrics/

| File | Status | Notes |
|---|---|---|
| `metrics/regret.py` | implemented+tested | `compute_episode_metrics()` + `attribute_regret()`. 11 tests (`test_regret.py`). |

### experiments/

| File | Status | Notes |
|---|---|---|
| `experiments/harness.py` | implemented+tested | `run_scenario` (one policy x scenario x seed episode → `ScenarioResult`, truncated via `max_steps`, default 250) + `run_grid` (every combination). Recomputes per-decision latency/hybrid-draw resolution from public `StateDict` fields (mirrors `env.environment`'s private cost tables/`REKEY_NOW` resolution, since `step()` doesn't surface them directly). `ScenarioResult` gained `total_reward: float` 2026-08-10 (raw summed episode reward — a sharper policy discriminator than `p99_latency`, see that session's log entry). 8 tests (`test_harness.py`), incl. the S1 x four-baselines zero-floor-violations check, a `run_grid` combination-count check, and a `total_reward` check against a manually-summed reference. Only S1 exercised this session (S2-S6 dispatch not wired in `environment.py` yet). |
| `experiments/campaign.py` | implemented+tested | The **only** supported way to produce a DQN-vs-baseline number here: checkpoint-averaged, eval-seed-averaged, multi-seed. `SeedSpread` makes the spread a required field so a mean cannot be reported without it; `verdict()` treats TIE as a first-class outcome (Hard Rule 7). |
| `experiments/ablation.py` | implemented+tested | E-A, `off`/`ewma`/`lstm` on S3 then held-out S6. All arms on `threat_input.source: rt_iot2022` — evaluating the LSTM under the scalar `scenario` source measures a distribution mismatch, not foresight (observed: −341,702 vs `off`'s −6,506 on S6 before the fix). |
| `experiments/results_table.py` | implemented+tested | PLAN2 §7.7's closing table, S1-S4 + S6, agents trained on S1 only. Prints "0 — structural" only where the count is genuinely 0. |
| `experiments/train.py` | implemented+tested | `train()` (one continuous S1 episode, `total_steps` from `configs/default.yaml`'s new `training:` block, periodic greedy-mode eval snapshots via the harness, final `DQNAgent.save` checkpoint), `GreedyDQNPolicy` (wraps a trained agent's `q_network` directly for deterministic epsilon=0 evaluation without touching `agents/dqn.py` or the agent's training epsilon-decay counter), `evaluate_against_baseline()` (trained agent vs. grid-searched `StaticThresholdPolicy`, same fixed eval seed). **2026-08-10: `train()` now passes `training_cfg["seed"]` to `DQNAgent(..., seed=...)` too**, not just `env.reset(seed=...)` — see `agents/dqn.py`'s row. 6 tests (`test_train.py`), incl. a smoke run (100 steps) and a determinism check contrasting `GreedyDQNPolicy` against `DQNAgent.act()`'s genuine epsilon=1 stochasticity. **Six real 25,000-step campaigns executed 2026-08-10 across four sessions** (~40-46s/run): an epsilon-schedule fix (`epsilon_decay_steps` 50k→12.5k) let training genuinely converge but the converged flat-S1 policy still tied the tuned threshold on `p99_latency` and never rekeyed proactively (`forced_rekey_ratio=1.000`); a load-spike diagnostic (see `env/request_generator.py`'s row) re-ran under it and got `0.256`/`0.872` across two seeds; a 10-seed sweep sized that spread properly (`0.190`-`1.000`, mean `0.735`, stdev `0.275`) and found `agents/dqn.py`'s randomness was never seeded — training seed only reached the environment; a same-day fix session seeded it and re-ran the same 10-seed sweep — the spread got *wider*, not tighter (`0.102`-`1.000`, mean `0.700`, stdev `0.345`), with exactly half the seeds landing at the exact never-proactive ceiling. **2026-08-17: a budget probe (6 more real campaigns, 50k/75k steps, 3 of the 5 stuck seeds) found the stuck seeds don't respond to more training budget** — 2 of 3 reached good intermediate `forced_rekey_ratio` values at 50k steps (`0.102`, `0.659`) and then regressed back to the exact `1.000` ceiling by 75k, read at the time as a mid-training instability. **2026-08-18: a denser re-probe (3 more real 75,000-step campaigns, `eval_every=1000` — ~75 snapshots each instead of 3) overturned that framing** — there is no localized regression: `forced_rekey_ratio` swings 0.5+ between adjacent 1,000-step snapshots roughly 1 in 3 of the time, continuously across the entire run, for every seed including one previously read as "flat, never found a good policy." Buffer-capacity crossing at step 50,000 (`agents/dqn.py`'s `_REPLAY_BUFFER_CAPACITY`) is ruled out as the cause — no loss anomaly or swing-amplitude change near that step. A real but separate long-run drift toward the ceiling exists (noisy, not step-changed at any specific step) layered on top of the noise. **A second 2026-08-18 diagnostic (3 more real 75,000-step campaigns, `eval_every=750`, 8 fixed eval seeds/snapshot instead of 1) tested the two remaining hypotheses that session left open**: single-episode eval noise (RULED OUT — 8-seed averaging only shrinks the checkpoint-to-checkpoint swing to ~4-6x smaller than the swing itself, nowhere near enough to explain it) and eval-cadence/target-sync aliasing (evidence against, not a clean rule-out since it's a between-session comparison — a non-aligned cadence swings just as hard as the aligned one did). The ceiling-fraction drift was confirmed to survive 8-seed averaging. The actual mechanism behind the swings is still unidentified. See SESSION_LOG.md's two 2026-08-18 entries for the full data, and PROGRESS.md's "Next task" for the current candidate follow-ups. |

### attack/

| File | Status | Notes |
|---|---|---|
| `attack/run_attack.py` | implemented+tested | The S5 dose-response experiment. Both arms scored on ONE shared trajectory at key *establishments*, each under its own mask — `mask & tier_only` for the masked arm, `tier_only` alone for the unmasked one. That construction is the cleanest statement of the thesis in the codebase, and it took three tries: scoring a masked soft-reward agent measured the mask (0.0% at every dose for both arms), and scoring every decision let a REUSE deliver whichever tier the *driving* policy had established (0.0% at full suppression — the opposite of the truth). 5 tests. |
| `attack/steering_trace.py` | implemented+tested | Suppression-only adversarial trace; `dose=0` is a genuine no-op (the sweep's control arm). 11 tests. Plus new `attack/run_attack.py` — the dose-response experiment, 5 tests. |

### dashboard/

| File | Status | Notes |
|---|---|---|
| `dashboard/data.py` | implemented+tested | Assembles all seven panels from real environment replays and `results/` artefacts. `mock.html` is layout truth and nothing else — nothing is copied from it, and a missing or corrupt artefact yields `available: false` with the command to run. |
| `dashboard/app.py` | implemented+tested | Dash app (`build_app()` still returns without starting a server) + self-contained static HTML export. All seven PLAN2 §7 panels. New `dashboard/data.py` assembles them from real runs; a missing or corrupt artefact renders "not yet run", never a placeholder. 18 tests. |

### api/

| File | Status | Notes |
|---|---|---|
| `api/main.py` | implemented+tested | `GenerateDataKey` / `ExplainDecision` / `Decisions` / `PoolStatus` / `ThreatStatus` / `Scenario` / `Health`. The primitive-honesty matrix is published on `/Health`: classical is REAL, ML-KEM-768 is SIMULATED (liboqs not installed) and says so on every response, hybrid is PARTIAL. A caller-supplied tenant cannot steer a decision (Hard Rule 3, asserted by test). 15 tests. |

### data/

| File | Status | Notes |
|---|---|---|
| `data/get_data.py` | not started (unblocking) | All four `_download_*` helpers still `raise NotImplementedError`. **No longer on any critical path:** RT-IoT2022 is the one real-network slot (PLAN2 §6's golden rule) and is operator-placed and gitignored; `forecaster/dataset.py` locates it at either accepted path and raises a message saying what to do if absent. The other three slots this file was written for are `rl_experiment_*` / `context_dataset_*`, which the README bans as training data anyway. |
| `data/README.md` | present (doc) | Download instructions written. |
| `data/sample/` | empty | Only a `README.md` placeholder — no committed sample CSVs yet for CI/others to run against. |
| `data/raw/RT_IOT2022.csv` | ingested | 123,117 rows, 12 classes (89.8% attack). Read by `forecaster/dataset.py`; both `data/raw/` and `data/raw/rt_iot2022/` are accepted paths. Gitignored, never committed. |

### docs/

| File | Status | Notes |
|---|---|---|
| `docs/report.md` | written | Full draft with real numbers throughout. §5.1 reports Gate W3's negative result, §5.2 the steering result, §7 the four environment corrections made this session. |

### configs/

| File | Status | Notes |
|---|---|---|
| `configs/default.yaml` | implemented | Gained `scenarios:` (S1-S6 dispatch), `threat_input:`, `forecaster:`, a real `migration_schedule:`, `training.eval_seeds`/`campaign_seeds`, and a fully re-derived `pool:` block carrying the measured demand bracket its sizing was chosen inside. `load_spike:` is retained only so the 2026-08-10 diagnostic runs stay reproducible; no scenario in the grid uses it. Previously: | `pool`, `key_lifetime`, `reward`, `use_foresight`, `tenant_graph.n_nodes`, `load_spike`, `dqn`, `baselines`, `steering_attack`, `training` keys all present. `migration_schedule: []` empty (S6 not yet authored). `scenario: S1` — S2-S6 read but not dispatched by `environment.py`. `load_spike.enabled: false` by default (2026-08-10 diagnostic stub, NOT real S4 — see `env/request_generator.py`'s row and SESSION_LOG.md). |

---

## Last verified

- **Date:** 2026-08-20
- **Branch:** `dev` (cut from a fresh `main` baseline — the working copy as
  received had no `.git` directory; see SESSION_LOG.md §0)
- **`pytest` pass count:** **629 passed**, 0 failed, 3 deselected (`-m "not slow"`;
  the three deselected are the end-to-end steering-attack runs). Was 400 at the
  start of the session.
- **Artefacts:** `results/steering_dose_response.json` regenerated;
  `results/foresight_ablation.json` and `results/closing_table.json` are produced
  by long runs and must be regenerated whenever the environment changes.
