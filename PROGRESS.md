# SmartKeyNet — Progress Tracker

> **Update convention:** updating this file's checkboxes and the "Next task"
> line is part of the same end-of-session step as updating `SESSION_LOG.md`.
> This file gets *updated*, not rewritten, at the end of every session.
> It exists so a fresh Claude Code session (or a new person) can read
> `PLAN.md` + `SESSION_LOG.md` + this file and know what's done and what
> the single next task is, without reconstructing status from log prose.

---

## Next task

**Three independent threads are open. Pick whichever this session serves.**

**Thread 1 — DQN training-instability (PAUSED 2026-08-19, not resolved, not
abandoned).** Five sessions deep with a genuine fork left unpicked, not more
solo running. The fork, unchanged since 2026-08-18:
(a) a direct Q-value-margin inspection at a known swing (e.g. seed 1's
step `71250->72000` transition, `forced_rekey_ratio` mean dropping from
`~0.90` to `0.149`) — check whether the greedy action at the eval
episode's early, trajectory-determining decision points sits at a
near-tie in Q-value between two actions; or (b) treat "the greedy policy
genuinely oscillates checkpoint-to-checkpoint, mechanism unknown" as a
standing, now twice-confirmed property of this training setup and move on
to real S4 regardless, using **checkpoint-averaged** (not single-checkpoint)
comparisons for any future DQN-vs-baseline number. See item 6 below for the
full 2026-08-18 result this fork comes from. Do not touch `agents/dqn.py`
or `experiments/train.py` for anything short of a deliberate, sign-off'd
decision on this fork.

**Thread 2 — Dashboard v2 (started 2026-08-19).** `dashboard/explain.py`
(the Explain Decision panel's backend, PLAN2.md §7.3) is now
implemented+tested — see its `dashboard/` per-file row below. The
concrete next Dashboard v2 step is the Threat Input panel (PLAN2.md
§7.1), but that's blocked on Person A's feature-extraction work (the
shared RT-IoT2022/pcap feature-extraction function PLAN2.md §6 and
Hard Rule 11 require) — dataset ingestion hasn't started (see `data/`
below). Until that exists, do not stub a placeholder extraction path for
Threat Input (Hard Rule 11: one shared extraction path, no parallel
pipeline). Other dashboard-adjacent, unblocked options: Panel 2 (Living
System) could start against `dashboard/explain.py`'s real trace output
plus `StateDict`/event-log fields, no new dependency; or start on
`agents/soft_reward_baseline.py` (needed before the steering attack,
PLAN2.md §7.5/§9 S5, can be built).

**Thread 3 — Real scenario dispatch, S4/S6 blocked on a tenant-identity
decision (opened 2026-08-19).** S1/S2/S3 scenario dispatch is now real (see
`env/environment.py`'s per-file row). S4 (DDoS/noisy-neighbor: "one
low-sensitivity tenant floods the API") and S6 (migration wave: "a tenant
cohort's floor changes") both fundamentally need a notion of **which
tenant a request belongs to that persists and can be targeted/grouped** —
`env/request_generator.py`'s `random_request_generator()` currently
assigns `tenant`/`service` strings per request with no graph, no
per-tenant identity the environment can reason about, and no way to say
"flood tenant X" or "raise cohort Y's floor." This is exactly
`build_tenant_graph()`/`RequestGenerator` (PLAN.md §10 step 4, still
`NotImplementedError` — see `env/request_generator.py`'s row) — a real
decision, not a small addition: **either** build the real NetworkX tenant
graph + graph-driven `RequestGenerator` first (the "do it properly" path,
larger scope, unblocks S4 *and* S6 *and* real Stretch-B-style per-tenant
allocation), **or** find a smaller, explicitly-flagged-as-a-shortcut way to
give the *existing* random stream a stable per-tenant identity concept just
big enough for S4/S6 (e.g. a small fixed pool of named tenants the random
generator draws from, still no graph/edges/traffic-rate modeling) — cheaper
but needs to be a deliberate, sign-off'd call given Hard Rule 3's
"deleting the tenant graph must not change one line of agent code" test,
since neither S4 nor S6 may leak tenant identity into agent-visible state
(only into the request stream / masking-adjacent floor lookups, same as
S2/S3's threat/pool inputs this session).

---

**2026-08-18's diagnostic recap (Thread 1, unchanged from before the pause):**

**A second 2026-08-18 diagnostic tested both candidate explanations for the
checkpoint-to-checkpoint swings and disfavored both — the mechanism is
still unknown, and even 8-eval-seed averaging doesn't tame it.** See item 6
below for the full result; the standing recommendation to report
`forced_rekey_ratio` "as a distribution across eval seeds" (item 5) is now
known to be insufficient on its own — the distribution's *mean* swings
checkpoint-to-checkpoint almost as hard as a single draw did.

**The 2026-08-18 dense re-probe overturned the 2026-08-17 "mid-training
regression" framing itself.** There is no localized regression event —
`forced_rekey_ratio` oscillates continuously (swings of 0.5+ between
adjacent 1,000-step snapshots, about 1 in 3 of the time) across the
*entire* 1,000-75,000 step range, for every seed tested, including the
one previously read as "flat, never found a good policy." The earlier
3-point sample (25k/50k/75k) was undersampling this noise, not
observing a real found-then-lost event. See SESSION_LOG.md 2026-08-18
for the full data. Five sessions on this thread now (three same-day
2026-08-10, one 2026-08-17, one 2026-08-18); recap in order:

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
- [ ] Real NetworkX tenant graph + graph-driven `RequestGenerator`
      (`build_tenant_graph`, `RequestGenerator.reset/step`) — PLAN.md §10 step 4
- [x] Masked DQN agent (`agents/dqn.py`)
- [x] Four tuned baselines — always-PQC, always-hybrid, static-threshold
      (grid-searched), random (`agents/baselines.py`) + comparison harness
      (`experiments/harness.py`) — Hard Rule 7
- [ ] 🚩 Gate W3 (make-or-break) — DQN beats the tuned threshold baseline on S1 and S3
      *(Still not attemptable for real — S3 doesn't exist as a scenario yet. 2026-08-10's
      load-spike diagnostics (NOT real S3/S4 — see Next task and `configs/default.yaml`'s
      `load_spike:` block) showed `forced_rekey_ratio` dropping well below flat-S1's
      never-proactive `1.000` once arrival load genuinely varies, directionally confirming
      the reward mechanism isn't broken — but even after fixing `agents/dqn.py`'s previously
      unseeded randomness and re-running the 10-seed sweep with genuinely controlled seeds,
      the spread got wider, not tighter (`0.102`-`1.000`, half the seeds at the exact
      never-proactive ceiling) — a real learn/don't-learn split by training run, not a
      measurement artifact. **2026-08-17: a budget probe on 3 stuck seeds at 50k/75k steps
      found this isn't a training-budget question either** — two of the three reached good
      intermediate values at 50k then regressed back to the exact ceiling by 75k, a
      mid-training instability, not slow convergence. **2026-08-18: a denser (every-1000-step)
      re-probe overturned even that framing** — there's no localized regression at all;
      `forced_rekey_ratio` swings 0.5+ between adjacent 1,000-step snapshots roughly 1 in 3
      of the time, continuously across an entire 75,000-step run, for every seed tested
      (including one previously read as "flat"). **A second same-day diagnostic then tested
      whether this is single-episode eval noise or eval-cadence/target-sync aliasing — both
      disfavored**: averaging 8 fixed eval seeds per checkpoint only shrinks the swing by
      ~4-6x less than the checkpoint-to-checkpoint swing itself (still >0.5 on ~1-in-4
      adjacent snapshots), and a non-aligned eval cadence swings just as hard as an aligned
      one. The mechanism behind the swings remains unknown — future Gate W3 attempts need
      **checkpoint-averaged**, not single-checkpoint or eval-seed-distribution, comparisons,
      on top of multi-seed training reporting. Evidence toward attempting the gate once S3
      exists, not the gate itself.)*
- [ ] Soft-reward baseline agent reproducing Noetzold (`agents/soft_reward_baseline.py`)
- [ ] Scenario dispatch S2-S4 wired into `environment.py` — **partially done
      2026-08-19: S2 (HNDL) + S3 (QKD degradation) are real, tested dispatch now;
      S4 (DDoS/noisy-neighbor) is still blocked** on a tenant-identity concept
      `env/request_generator.py` doesn't have yet (see "Next task"). The
      2026-08-10 `load_spike` diagnostic remains a request-rate-only stand-in,
      unrelated to this real dispatch.
- [ ] Real LSTM dual-head forecaster (Addition A) — `forecaster/model.py`,
      `forecaster/dataset.py`, `forecaster/train.py`, `LSTMForecastProvider`
      in `env/forecast_provider.py`
- [ ] E-A foresight ablation (off / ewma / lstm on S3 + S6)
- [ ] Steering attack — adversarial threat-trace generator (`attack/steering_trace.py`)
      + attack run producing the split-screen result — Gate W5, headline contribution, never cut
- [ ] S6 migration wave (scripted schedule, held-out eval only)
- [ ] Live dashboard (`dashboard/app.py`) — 4-beat demo
- [ ] AWS-KMS-style API facade (`api/main.py`)
- [ ] Report (`docs/report.md`) — currently a section-header skeleton with TODOs only

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
| `env/pool_sim.py` | implemented+tested | `PoolSim` (refill/drain/exhaustion) + `SyntheticSKRQBERTrace`. 19 tests (`test_pool_sim.py`). **2026-08-19: no code changed here** — `SyntheticSKRQBERTrace`'s existing `spike_start`/`spike_duration`/`spike_magnitude` params (already present, already documented as "the dial-in hook for the S3 'QKD degradation' scenario") are now actually exercised by real S3 dispatch in `environment.py`, not just by `test_pool_sim.py`'s own standalone degradation test. |
| `env/deferral_queue.py` | implemented+tested | `DeferralQueue.enqueue/tick/pop_servable`, priority+FIFO, cumulative-headroom draw. 8 tests (`test_deferral_queue.py`). |
| `env/masking.py` | implemented+tested | `PolicyTable` (placeholder floor table, sticky ratchet) + `compute_mask`. 14 tests (`test_masking.py`). Floor table not yet calibrated against Q-OPSEC data. |
| `env/forecast_provider.py` | stub (partial) | `MovingAverageForecaster` (EWMA fallback) implemented+tested (9 tests, `test_forecast_provider.py`). `LSTMForecastProvider` does not exist yet (Addition A) — `use_foresight: lstm` currently raises `NotImplementedError` in `environment.py`. |
| `env/request_generator.py` | stub (partial) | `random_request_generator()` implemented+tested (11 tests, `test_request_generator.py`), incl. 3 new 2026-08-10 tests for its optional `load_spike` kwarg — a periodic, config-driven arrival-rate diagnostic (**explicitly not real S4** — see that session's SESSION_LOG.md entry and `configs/default.yaml`'s `load_spike:` block). `build_tenant_graph()` and `RequestGenerator` (graph-driven stream) still `raise NotImplementedError` — real S4 needs these. |
| `env/environment.py` | implemented+tested | `SmartKeyNetEnv.reset/step/action_mask` fully wired (pool + deferral + masking + forecast + reward + session-key state). 25 behavioral tests (up from 17) incl. the split.md Gate W2 tests (`test_environment.py`). 2026-08-10: wired `config["load_spike"]` through to `random_request_generator` (design decision 9) — orthogonal to scenario dispatch, not a substitute for it. **2026-08-19 (design decision 10): real S2 (HNDL) + S3 (QKD degradation) scenario dispatch.** `config["scenario"]` now genuinely gates behavior for S2/S3 (S1, and the still-undispatched S4/S5/S6, are unaffected — confirmed by an explicit regression test). S2: `config["threat_schedule"]` (required only under `scenario: S2`) makes `_threat_features_placeholder()` return a scripted elevated signal from a configured step onward, flowing through the *existing, unmodified* `MovingAverageForecaster` → `PolicyTable.ratchet_up`/`floor` → `compute_mask` chain — `env/masking.py`'s floor table itself was never touched (Hard Rule 2), verified directly by a test cross-checking the env's real per-decision floor against a fresh `PolicyTable().floor()` call across a spread of sensitivity classes. S3: `config["qkd_degradation"]` (required only under `scenario: S3`) threads straight into `SyntheticSKRQBERTrace`'s pre-existing spike params at `reset()` — no new pool-sim code. **Observability finding, documented in `configs/scenarios/s3_degradation.yaml`'s own comments**: under `pool:`'s realistic default-scale numbers (256-bit draws vs. ~200,000-bit/tick mean SKR refill), S3's degradation is real but invisible in practice — refill dwarfs any plausible draw pattern by 2-3 orders of magnitude; the regression test demonstrates the real, measurable effect (higher regret-event count, lower minimum pool fill vs. S1) using the same scarcity-forcing small-pool override `test_environment.py`'s existing Hard Rule 9 gate test already established, not a fabricated number. New standalone scenario configs: `configs/scenarios/s2_hndl.yaml`, `configs/scenarios/s3_degradation.yaml` (both directly loadable via `experiments/train.py`'s `load_full_config(path)`, verified). S4 (DDoS/noisy-neighbor) and S6 (migration wave) deliberately NOT dispatched — both need a tenant-identity concept `env/request_generator.py` doesn't have yet; see "Next task". |

### agents/

| File | Status | Notes |
|---|---|---|
| `agents/dqn.py` | implemented+tested | `flatten_state(state, has_forecast)` (genuinely variable-length: 13 dims under `off`, 28 under `ewma`/`lstm`; `has_forecast` is now an **explicit required parameter** — the earlier `state["threat_score"] != 0.0` inference trick was removed 2026-08-08 as fragile-by-accident, see that session's log entry). `QNetwork` (2-hidden-layer MLP), `DQNConfig`/`load_dqn_config` (reads `configs/default.yaml`'s `dqn:` block), `DQNAgent(state_dim, has_forecast, config, seed=None)` (internal circular-buffer replay, `act`/`observe`/`learn`/`save`/`load`, `has_forecast` fixed once at construction and threaded through every internal `flatten_state` call) — masking applied structurally at both action-selection *and* bootstrap-target time (Hard Rule 2), no security term anywhere (Hard Rule 1). **2026-08-10: `seed` parameter added** — reseeds `random`+`torch`'s global RNGs before `QNetwork` construction when given, so weight init/exploration/replay sampling are genuinely reproducible (`seed=None` default is unchanged prior behavior); fixes a gap the same day's earlier sessions found and flagged (see `experiments/train.py`'s row and SESSION_LOG.md). 26 tests (`test_dqn.py`, up from 23), incl. a regression test for a foresight-mode state with `threat_score == 0.0` still flattening to 28 dims, an integration test training against the real `SmartKeyNetEnv` on S1 for 3000 steps with loss trending down, and 3 new seed tests (same-seed reproducibility, different-seed divergence, `seed=None` leaves ambient state alone). Not yet run to convergence — that's `experiments/train.py`. |
| `agents/baselines.py` | implemented+tested | `AlwaysPQCPolicy`, `AlwaysHybridPolicy`, `StaticThresholdPolicy` (incl. `grid_search`), `RandomPolicy` — all real, sharing a `_lowest_legal_action` fallback. 261 tests (`test_baselines.py`), incl. an adversarial parametrized sweep over all 31 non-empty action masks per policy (never returns an illegal action, however contrived the mask). |
| `agents/soft_reward_baseline.py` | not started | Stub, `test_soft_reward_baseline.py` is 1 import-smoke test. |

### forecaster/

| File | Status | Notes |
|---|---|---|
| `forecaster/model.py` | not started | Stub, `test_forecaster_model.py` is 1 import-smoke test. |
| `forecaster/dataset.py` | not started | Stub, `test_forecaster_dataset.py` is 1 import-smoke test. |
| `forecaster/train.py` | not started | Stub, `test_forecaster_train.py` is 1 import-smoke test. |

### metrics/

| File | Status | Notes |
|---|---|---|
| `metrics/regret.py` | implemented+tested | `compute_episode_metrics()` + `attribute_regret()`. 11 tests (`test_regret.py`). |

### experiments/

| File | Status | Notes |
|---|---|---|
| `experiments/harness.py` | implemented+tested | `run_scenario` (one policy x scenario x seed episode → `ScenarioResult`, truncated via `max_steps`, default 250) + `run_grid` (every combination). Recomputes per-decision latency/hybrid-draw resolution from public `StateDict` fields (mirrors `env.environment`'s private cost tables/`REKEY_NOW` resolution, since `step()` doesn't surface them directly). `ScenarioResult` gained `total_reward: float` 2026-08-10 (raw summed episode reward — a sharper policy discriminator than `p99_latency`, see that session's log entry). 8 tests (`test_harness.py`), incl. the S1 x four-baselines zero-floor-violations check, a `run_grid` combination-count check, and a `total_reward` check against a manually-summed reference. Only S1 exercised this session (S2-S6 dispatch not wired in `environment.py` yet). |
| `experiments/train.py` | implemented+tested | `train()` (one continuous S1 episode, `total_steps` from `configs/default.yaml`'s new `training:` block, periodic greedy-mode eval snapshots via the harness, final `DQNAgent.save` checkpoint), `GreedyDQNPolicy` (wraps a trained agent's `q_network` directly for deterministic epsilon=0 evaluation without touching `agents/dqn.py` or the agent's training epsilon-decay counter), `evaluate_against_baseline()` (trained agent vs. grid-searched `StaticThresholdPolicy`, same fixed eval seed). **2026-08-10: `train()` now passes `training_cfg["seed"]` to `DQNAgent(..., seed=...)` too**, not just `env.reset(seed=...)` — see `agents/dqn.py`'s row. 6 tests (`test_train.py`), incl. a smoke run (100 steps) and a determinism check contrasting `GreedyDQNPolicy` against `DQNAgent.act()`'s genuine epsilon=1 stochasticity. **Six real 25,000-step campaigns executed 2026-08-10 across four sessions** (~40-46s/run): an epsilon-schedule fix (`epsilon_decay_steps` 50k→12.5k) let training genuinely converge but the converged flat-S1 policy still tied the tuned threshold on `p99_latency` and never rekeyed proactively (`forced_rekey_ratio=1.000`); a load-spike diagnostic (see `env/request_generator.py`'s row) re-ran under it and got `0.256`/`0.872` across two seeds; a 10-seed sweep sized that spread properly (`0.190`-`1.000`, mean `0.735`, stdev `0.275`) and found `agents/dqn.py`'s randomness was never seeded — training seed only reached the environment; a same-day fix session seeded it and re-ran the same 10-seed sweep — the spread got *wider*, not tighter (`0.102`-`1.000`, mean `0.700`, stdev `0.345`), with exactly half the seeds landing at the exact never-proactive ceiling. **2026-08-17: a budget probe (6 more real campaigns, 50k/75k steps, 3 of the 5 stuck seeds) found the stuck seeds don't respond to more training budget** — 2 of 3 reached good intermediate `forced_rekey_ratio` values at 50k steps (`0.102`, `0.659`) and then regressed back to the exact `1.000` ceiling by 75k, read at the time as a mid-training instability. **2026-08-18: a denser re-probe (3 more real 75,000-step campaigns, `eval_every=1000` — ~75 snapshots each instead of 3) overturned that framing** — there is no localized regression: `forced_rekey_ratio` swings 0.5+ between adjacent 1,000-step snapshots roughly 1 in 3 of the time, continuously across the entire run, for every seed including one previously read as "flat, never found a good policy." Buffer-capacity crossing at step 50,000 (`agents/dqn.py`'s `_REPLAY_BUFFER_CAPACITY`) is ruled out as the cause — no loss anomaly or swing-amplitude change near that step. A real but separate long-run drift toward the ceiling exists (noisy, not step-changed at any specific step) layered on top of the noise. **A second 2026-08-18 diagnostic (3 more real 75,000-step campaigns, `eval_every=750`, 8 fixed eval seeds/snapshot instead of 1) tested the two remaining hypotheses that session left open**: single-episode eval noise (RULED OUT — 8-seed averaging only shrinks the checkpoint-to-checkpoint swing to ~4-6x smaller than the swing itself, nowhere near enough to explain it) and eval-cadence/target-sync aliasing (evidence against, not a clean rule-out since it's a between-session comparison — a non-aligned cadence swings just as hard as the aligned one did). The ceiling-fraction drift was confirmed to survive 8-seed averaging. The actual mechanism behind the swings is still unidentified. See SESSION_LOG.md's two 2026-08-18 entries for the full data, and PROGRESS.md's "Next task" for the current candidate follow-ups. |

### attack/

| File | Status | Notes |
|---|---|---|
| `attack/steering_trace.py` | not started | Stub, `test_steering_trace.py` is 1 import-smoke test. |

### dashboard/

| File | Status | Notes |
|---|---|---|
| `dashboard/app.py` | not started | Stub, `test_dashboard_app.py` is 1 import-smoke test. |
| `dashboard/explain.py` | implemented+tested | **2026-08-19 (new file):** Explain Decision panel backend (PLAN2.md §7.3, Addition D; Hard Rule 10) -- Person D's first real code this project. `explain_decision(...)` (pure function, all six inputs explicit) + `explain_decision_from_env(env, state, chosen_action)` (convenience wrapper pulling those inputs off a live `SmartKeyNetEnv`, mirroring `experiments/harness.py`'s established precedent for reaching into a few private env attributes the public Gym API doesn't yet surface). Returns a `DecisionTrace` dataclass covering all six PLAN2.md §7.3 steps: threat score + source, posture probs + resolved posture, floor lookup (+ the full real floor table, imported from `env.masking`, never re-typed), the action mask (calls `env.masking.compute_mask()` directly -- zero possible drift between this module's legal/reason fields and the masking layer's real behavior, by construction, not by convention), cost comparison (reads `env.environment`'s real `_LATENCY_UNITS`/`_ENERGY_UNITS`/`_KEY_TYPE_TO_SERVE_ACTION`, never re-derived), and a deterministically templated final sentence. Policy-agnostic by design (no dependency on `agents/dqn.py`) -- verified in a scratchpad sanity script against `StaticThresholdPolicy` on real S1 steps, which also caught and fixed one real bug: the final-sentence template originally said "a learned preference from the policy" for the cost-tradeoff case, which is wrong for a non-learning baseline; reworded to "the policy's own preference among legal options." 26 tests (`test_explain.py`), incl. every (sensitivity_class, posture) floor-table cell checked against a fresh `PolicyTable`, 6 parametrized mask edge cases (pool empty, key age at/over cap, cold start, all-legal, HYBRID-floor-with-empty-pool) each checked against a real `compute_mask()` call, REKEY_NOW cost-resolution cases (existing-tier and cold-start-adopts-floor), and an end-to-end test stepping a real `SmartKeyNetEnv` and cross-checking every trace against a fresh `compute_mask()` call built from the same env state. Explicitly out of scope this session (per PLAN2.md's Hard Rule 11 and Hard Rule 10's scoping): pcap ingestion / Threat Input panel (§7.1) and any dashboard HTML/frontend -- this is a Python module returning structured data only. |

### api/

| File | Status | Notes |
|---|---|---|
| `api/main.py` | not started | Stub, `test_api_main.py` is 1 import-smoke test. |

### data/

| File | Status | Notes |
|---|---|---|
| `data/get_data.py` | not started | All four `_download_*` helpers `raise NotImplementedError`; `main()` exists as a dispatcher shell. |
| `data/README.md` | present (doc) | Download instructions written. |
| `data/sample/` | empty | Only a `README.md` placeholder — no committed sample CSVs yet for CI/others to run against. |
| `data/raw/rt_iot2022/RT_IOT2022.csv` | placed, not ingested | Real dataset file present locally (gitignored, not committed); no code reads it yet. |

### docs/

| File | Status | Notes |
|---|---|---|
| `docs/report.md` | skeleton only | Section headers (Abstract, Intro, Related Work, Methodology 3.1-3.3, ...) with `_TODO_` markers and owner tags; no written content yet. |

### configs/

| File | Status | Notes |
|---|---|---|
| `configs/default.yaml` | partial | `pool`, `key_lifetime`, `reward`, `use_foresight`, `tenant_graph.n_nodes`, `load_spike`, `dqn`, `baselines`, `steering_attack`, `training` keys all present. `migration_schedule: []` empty (S6 not yet authored). `scenario: S1` — deliberately does not carry `threat_schedule`/`qkd_degradation` (those are S2/S3-only, required only when `scenario` selects them — see `env/environment.py`'s row). `load_spike.enabled: false` by default (2026-08-10 diagnostic stub, NOT real S4 — see `env/request_generator.py`'s row and SESSION_LOG.md). |
| `configs/scenarios/s2_hndl.yaml` | implemented+tested | **2026-08-19 (new file)**: standalone, directly-loadable S2 config (same shape as `default.yaml`, `scenario: S2` + a `threat_schedule` block). Verified to load and run end-to-end (`tests/test_environment.py::test_scenario_config_files_load_and_construct_a_working_env`). |
| `configs/scenarios/s3_degradation.yaml` | implemented+tested | **2026-08-19 (new file)**: standalone, directly-loadable S3 config, `scenario: S3` + a `qkd_degradation` block. Same verification as above. Its own comments document the pool-scale observability finding (see `env/environment.py`'s row). |

---

## Last verified

- **Date:** 2026-08-19
- **Commit:** `f38fff0` ("log: [solo] S2+S3 scenario dispatch wired, S4/S6 deferred pending tenant-identity decision -- 2026-08-19") — the commit this session started from; see SESSION_LOG.md for this session's own commit
- **`pytest` pass count:** 434 passed, 0 failed (unchanged — this was a pure git-history/ref reconciliation session, no source file content touched)
- **Branch:** `main` and `dev21` reconciled this session via `git merge dev21 --ff-only` + `git push origin main` — both now point at the identical commit `f38fff0fc0afd8fba120c7b09087e862568eae2d`, verified via `git rev-parse main dev21 origin/main`. Neither branch was deleted. **`main` is once again the correct single source of truth for future session prompts, per instruction** — see SESSION_LOG.md's 2026-08-19 repo-hygiene entry for the full investigation, including an unresolved, unrelated finding (`origin/capstone-build-2026-08-20`, a disconnected history with what reads as much more complete project state) flagged there for the user's attention, not acted on.
