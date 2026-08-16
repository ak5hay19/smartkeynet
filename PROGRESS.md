# SmartKeyNet — Progress Tracker

> **Update convention:** updating this file's checkboxes and the "Next task"
> line is part of the same end-of-session step as updating `SESSION_LOG.md`.
> This file gets *updated*, not rewritten, at the end of every session.
> It exists so a fresh Claude Code session (or a new person) can read
> `PLAN.md` + `SESSION_LOG.md` + this file and know what's done and what
> the single next task is, without reconstructing status from log prose.

---

## Next task

**One open design decision, and it is yours to make.** Everything buildable is
built (578 tests, no stubs). The remaining question is not an engineering task:

> **Should hybrid serving carry a genuine upside in the reward?**

Today it does not — hybrid costs more latency, more energy, and pays the scarcity
price, with nothing gained. So the optimal policy is "spend only where a floor
demands it", which is close to static, which is why Gate W3 fails to a tuned
threshold across four sessions of honest attempts. Giving hybrid an upside would
make RL earn its place — but any such benefit is a *security* benefit, and Hard
Rule 1 forbids security in the reward. Resolving that tension is a genuine
research decision that invalidates every results table.

Do NOT relitigate Gate W3 by reshaping the environment until the DQN wins. It has
been given a fixed baseline, the full spec upgrade ladder, two recalibrations, a
forecastable threat signal and a working forecaster. See `docs/report.md` §5.

Secondary: **initialise git** — this working copy still has no `.git`.

<details>
<summary>Superseded — the previous "attempt Gate W3" task (now done, not passed)</summary>

**Retrain the DQN on the calibrated environment and attempt Gate W3 for
real** — multi-seed (5+), on S1 and S3, against the grid-searched
`StaticThresholdPolicy`. This is now attemptable for the first time.

Why it changed: on 2026-08-15 the environment was found to have **no
scarcity at all**. The pool sat at 100% full for 1999 of 2000 steps of an
S1 episode, 520 hybrid serves cost nothing, and zero regret events were
ever logged — refill ran at 781 keys/step against a structural demand
ceiling of 1 key/step, i.e. **ρ = 0.0013** against the build spec's
required `[0.8, 1.3]`. The spec names this failure mode exactly: *"the
pool never binds, no policy can differ from any other, and your DQN will
tie the threshold baseline in week 3."*

That is precisely the result recorded below, and it means the whole
2026-08-10 thread was chasing a symptom. **The bimodal
`forced_rekey_ratio` investigation is closed, not resolved** — with the
pool free, rekey timing was the only signal the agent had left to learn,
and it is a weak and noisy one. Whether the split persists on the
calibrated environment is an open question, and the honest way to answer
it is to re-measure rather than to keep reasoning from void data.

**All six 25,000-step campaigns and both 10-seed sweeps recorded below
are superseded.** They ran against uncalibrated physics and a reward in
which starving was cheaper than spending (the QKD price was charged per
bit rather than per 256-bit key, making it 256× oversized). Do not carry
any of those numbers forward.

Post-calibration state: ρ = 1.150 on S1, 7.42 on S3's peak-hold window;
all four baselines separate; floor violations 0 everywhere. See
SESSION_LOG.md 2026-08-15 for the full table and for three open issues
worth deciding on before or alongside retraining (`hybrid_mandatory` is
only half-enforced, `pqc_capable` interoperability masking is missing,
and `ThreatPosture.CALM` is unreachable under the EWMA forecaster).

<details>
<summary>Superseded — the 2026-08-10 forced_rekey_ratio investigation (kept for provenance)</summary>

Three same-day 2026-08-10 sessions in a
row on this thread; see SESSION_LOG.md for full detail on each.
Recap in order:

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

**Concretely next (pick one, needs a decision, not more solo
running):** (a) spend a session isolating *why* the split is so sharp
— e.g. same-seed runs at 50,000/75,000 steps to test whether it's a
training-budget question (more steps converts `1.000` seeds toward the
low end) or a genuinely bimodal loss landscape (more steps on a
`1.000` seed doesn't move it) — before trusting a single seed's result
in any future comparison; or (b) treat "the mechanism reliably works
for roughly half of runs, and very well for some of those" as enough
evidence and proceed to real S4 regardless, treating the split as a
lower-priority parallel thread and always reporting multi-seed spread
(not single-seed point estimates) in any future DQN-vs-baseline
comparison until it's better understood. Either way, real S4 still
needs the tenant graph (`build_tenant_graph`/`RequestGenerator`, still
`NotImplementedError`) for genuine per-tenant flooding — the load-spike
diagnostic remains a cruder, tenant-blind stand-in, not real S4 itself.
Do not mistake `configs/default.yaml`'s `load_spike:` block or
`random_request_generator`'s `load_spike` kwarg for finished S4 work —
both are documented as a diagnostic stub throughout.

</details>
</details>

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
- [x] **Scarcity calibration** — the pool actually binds. Current values:
      ρ = 0.44 for a sensible policy (binds, with headroom to misuse) and
      ρ = 10.0 for always-hybrid (drains immediately); the gap between them is
      the budgeting problem. Guarded by `test_scarcity_ratio_in_target_band`.
      *(Calibrated twice on 2026-08-15: ρ was 0.0013 originally — the pool never
      bound at all — and an intermediate ρ = 1.14 was measured against the
      always-hybrid villain, which rekeys ~500x more than necessary and so left
      the link over-provisioned twenty-fold for any realistic policy.)*
- [x] Real NetworkX tenant graph + graph-driven `RequestGenerator`
      (`build_tenant_graph`, `RequestGenerator.reset/step`) — PLAN.md §10 step 4
- [x] Masked DQN agent (`agents/dqn.py`)
- [x] Four tuned baselines — always-PQC, always-hybrid, static-threshold
      (grid-searched), random (`agents/baselines.py`) + comparison harness
      (`experiments/harness.py`) — Hard Rule 7
- [ ] 🚩 Gate W3 (make-or-break) — **CLOSED: ATTEMPTED, NOT PASSED.** Final numbers
      S1 −581 (threshold) vs −1,046,440 (DQN); S3 −132,835 vs −1,675,848. Pursued
      across five sessions, during which two genuine DQN training bugs were found
      and fixed (missing observation normalisation; an absorbing-state training
      loop). The agent still loses by ~1,800× with the machinery repaired, which
      makes this a structural finding rather than an implementation failure.
      The grid-searched three-parameter threshold beats the DQN on both scenarios:
      S1 −767 vs −1,959, S3 −514,587 vs −606,507 (5 train seeds × 5 eval seeds,
      disjoint; floor violations 0 everywhere). Spec §7.1 Fix B was applied in full
      first — γ derived to 0.995 per §11.3, Double DQN, 3-step returns, Huber loss,
      gradient clipping — and the baseline itself was fixed (it had been missing the
      entire `rho`/REUSE half of the spec's rule, making it a strawman the DQN beat
      for the wrong reason). Reported rather than engineered around, per §7.1 Fix C.
      Full numbers in `results/gate_w3.json`. DQN seed variance is large and
      undiagnosed on the new calibration.
- [x] Soft-reward baseline agent reproducing Noetzold (`agents/soft_reward_baseline.py`)
      — tabular Q-learning whose reward contains the threat term; computes its own
      reward so nothing under `env/` ever emits a security term
- [x] Scenario dispatch S1-S4 wired into `environment.py` via `env/scenarios.py`
      (S3 QBER drift ramp, S4 per-tenant flood, S2 threat-elevation windows; S5/S6
      resolve to eval-only specs carrying no perturbations yet — Hard Rule 8 enforced
      by `require_trainable()`)
- [x] Real LSTM dual-head forecaster (Addition A) — `forecaster/model.py`,
      `forecaster/dataset.py`, `forecaster/train.py`, `LSTMForecastProvider`.
      Frozen by construction (params `requires_grad=False`, `no_grad` forward).
      Threat head **balanced accuracy 0.334 (chance) -> 0.852** after fixing two
      bugs: it was labelled with the *ratcheted* posture ("same as now" in 99.9%
      of samples) and trained with unweighted CE against an 89/11/0.3 imbalance,
      which raw accuracy (0.838 = majority rate) hid completely.
- [x] E-A foresight ablation — **NULL RESULT.** S3 regret events: off 317.2,
      ewma 317.8, lstm 722.0. Addition A's success criterion NOT met. An earlier
      "ewma cuts regret 23%" claim was **withdrawn**: it did not reproduce once
      the DQN's observation normalisation and absorbing-state training bugs were
      fixed, i.e. it had been an artifact of a broken agent.
      `results/ea_ablation.json`.
- [x] **Steering attack** — `attack/steering_trace.py` + `experiments/steering_attack.py`.
      Headline contribution, and it lands: the critiqued reward's optimal tier walks
      down to *classical* as the reported threat is suppressed, while the masked floor
      only moves up. 0 floor violations and 0 ratchet reversals across every agent,
      dose and seed. Evidence is analytic, so it holds for any agent maximising that
      reward — independent of seed or training budget.
- [x] S6 migration wave — scripted `FloorChange` schedule, validated at build time to
      only ratchet up, held-out eval only (Hard Rule 8 enforced at the training entry
      point by `require_trainable`)
- [x] Live dashboard (`dashboard/app.py`) — Plotly Dash, 4-beat demo. Replays
      captured episodes (scrubbable, and identical for every viewer) rather than
      stepping a shared env inside a callback. Degrades gracefully when a results
      file is absent, so the demo always comes up.
- [x] AWS-KMS-style API facade (`api/main.py`) — FastAPI. `GenerateDataKey`,
      `Encrypt`, `DescribeKeyPolicy`, `PoolStatus`. Real HKDF-SHA256 + AES-256-GCM
      (round-trip tested); ML-KEM is a clearly-named placeholder. A deferred
      request returns **503, never a weaker key** (Hard Rule 9 at the API boundary).
- [x] Report (`docs/report.md`) — written from `results/*.json`. Leads with the
      three floor holes and the steering attack; reports Gate W3's failure and the
      E-A null result plainly, with a Limitations section.

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
| `env/pool_sim.py` | implemented+tested | `PoolSim` (refill/drain/exhaustion) + `SyntheticSKRQBERTrace` + `QberDriftSchedule` (S3's ramp/hold/partial-recovery) + `load_qkd_config`. 26 tests (`test_pool_sim.py`). **2026-08-15: the SKR gate was replaced** — the old `1 - min(qber, 0.5)` spike-only gate removed only ~10% of SKR at S3's peak, far too weak for a collapse; the new reconciliation gate (relative to baseline QBER, zero at `qber_abort`) removes ~96%, and is exactly 1.0 at/below baseline so S1 is untouched by it. SKR/QBER parameters now live in `configs/default.yaml`'s `qkd:` block instead of being invisible dataclass defaults. |
| `env/deferral_queue.py` | implemented+tested | `DeferralQueue.enqueue/tick/pop_servable`, priority+FIFO, cumulative-headroom draw. 8 tests (`test_deferral_queue.py`). |
| `env/masking.py` | implemented+tested | `PolicyTable` (placeholder floor table, sticky ratchet) + `compute_mask`. 14 tests (`test_masking.py`). Floor table not yet calibrated against Q-OPSEC data. |
| `env/forecast_provider.py` | stub (partial) | `MovingAverageForecaster` (EWMA fallback) implemented+tested (9 tests, `test_forecast_provider.py`). `LSTMForecastProvider` does not exist yet (Addition A) — `use_foresight: lstm` currently raises `NotImplementedError` in `environment.py`. |
| `env/request_generator.py` | implemented+tested | Both sources real. `random_request_generator()` (plain stationary Poisson, unchanged) plus its `load_spike` diagnostic kwarg. **2026-08-15: `build_tenant_graph()` and `RequestGenerator` implemented** — ~50-node NetworkX graph, `TenantProfile`-driven tenant-conditioned class weights, stratified class allocation (i.i.d. draws were too lumpy — they dropped a whole class at `n_nodes=10`), legacy-endpoint invariant, `TenantFlood` (S4), `as_stream()` adapter, `measure_fano_factor()`. MMPP burst chains are **per-tenant, not per-edge** (per-edge chains washed out as the graph grew: binned Fano fell 2.35 → 1.32 from 10 to 55 edges; per-tenant holds 3.6 vs the Poisson source's 0.95). 25 tests. |
| `env/environment.py` | implemented+tested | `SmartKeyNetEnv.reset/step/action_mask` fully wired (pool + deferral + masking + forecast + reward + session-key state). 21 behavioral tests incl. the split.md Gate W2 tests (`test_environment.py`). **2026-08-15: scenario dispatch (design decision 10)** — `config["scenario"]` resolves through `env/scenarios.py` and is applied through three exogenous channels, with no per-scenario branch in `step()`; `config["request_source"]` selects `random` or `graph`. **Reward units fix (design decision 11)** — the QKD scarcity price was charged per *bit* rather than per 256-bit key, making it 256× oversized and making starvation cheaper than spending; `_assert_reward_weights_are_sane()` now enforces the spec's `r_starve >= 5 * w_qkd` at construction. |
| `env/scenarios.py` | implemented+tested | **New 2026-08-15.** `build_scenario()` maps S1-S6 to a frozen `ScenarioSpec` over three exogenous channels — `QberDriftSchedule` (S3), `TenantFlood` (S4), `ThreatWindow`s (S2). Unknown names raise rather than falling back to S1. `require_trainable()` machine-enforces Hard Rule 8 on S5/S6. `ThreatWindow` validates non-negative intensity, so a scenario can only ever raise floors (Hard Rule 2). 28 tests (`test_scenarios.py`). |

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
| `experiments/train.py` | implemented+tested | `train()` (one continuous S1 episode, `total_steps` from `configs/default.yaml`'s new `training:` block, periodic greedy-mode eval snapshots via the harness, final `DQNAgent.save` checkpoint), `GreedyDQNPolicy` (wraps a trained agent's `q_network` directly for deterministic epsilon=0 evaluation without touching `agents/dqn.py` or the agent's training epsilon-decay counter), `evaluate_against_baseline()` (trained agent vs. grid-searched `StaticThresholdPolicy`, same fixed eval seed). **2026-08-10: `train()` now passes `training_cfg["seed"]` to `DQNAgent(..., seed=...)` too**, not just `env.reset(seed=...)` — see `agents/dqn.py`'s row. 6 tests (`test_train.py`), incl. a smoke run (100 steps) and a determinism check contrasting `GreedyDQNPolicy` against `DQNAgent.act()`'s genuine epsilon=1 stochasticity. **Six real 25,000-step campaigns executed 2026-08-10 across four sessions** (~40-46s/run): an epsilon-schedule fix (`epsilon_decay_steps` 50k→12.5k) let training genuinely converge but the converged flat-S1 policy still tied the tuned threshold on `p99_latency` and never rekeyed proactively (`forced_rekey_ratio=1.000`); a load-spike diagnostic (see `env/request_generator.py`'s row) re-ran under it and got `0.256`/`0.872` across two seeds; a 10-seed sweep sized that spread properly (`0.190`-`1.000`, mean `0.735`, stdev `0.275`) and found `agents/dqn.py`'s randomness was never seeded — training seed only reached the environment; **this session fixed the seeding gap and re-ran the same 10-seed sweep** — the spread got *wider*, not tighter (`0.102`-`1.000`, mean `0.700`, stdev `0.345`), with exactly half the seeds landing at the exact never-proactive ceiling — see SESSION_LOG.md for full per-seed numbers and reasoning, and PROGRESS.md's "Next task" for what this implies. |

### attack/

| File | Status | Notes |
|---|---|---|
| `attack/steering_trace.py` | not started | Stub, `test_steering_trace.py` is 1 import-smoke test. |

### dashboard/

| File | Status | Notes |
|---|---|---|
| `dashboard/app.py` | not started | Stub, `test_dashboard_app.py` is 1 import-smoke test. |

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
| `configs/default.yaml` | partial | `pool`, `qkd`, `key_lifetime`, `reward`, `use_foresight`, `tenant_graph.n_nodes`, `load_spike`, `dqn`, `baselines`, `steering_attack`, `training` keys all present. **2026-08-15: recalibrated** — `pool.capacity_bits` 1_000_000 → 25_600 (3906 → 100 ETSI keys), new `qkd:` block with `mean_skr_kbps` 200.0 → 0.22, and the full worked scarcity arithmetic written into the file as a comment block so the calibration is visible where the numbers live. `migration_schedule: []` empty (S6 not yet authored). S1-S4 now dispatched by `environment.py`; S5/S6 resolve to eval-only specs with no perturbations yet. `load_spike.enabled: false` by default (2026-08-10 diagnostic stub, superseded by real S4 — see `env/request_generator.py`'s row). |

---

## Last verified

- **Date:** 2026-08-15
- **Commit:** not applicable — this working copy is **not a git repository** (it
  was unpacked from a zip; there is no `.git`, so no commit history exists locally
  and nothing from this session is version-controlled). Worth fixing before the
  next session.
- **`pytest` pass count:** 578 passed, 1 skipped, 0 failed (up from 452 — new coverage for the
  three closed floor holes, `pqc_capable` interoperability masking, the reachable-CALM
  forecaster fix, n-step/Double-DQN, the three-parameter threshold baseline, the
  steering attack and the soft-reward victim, and the S6 migration schedule)
- **Results on disk:** `results/gate_w3.json`, `results/steering_attack.json`,
  `results/ea_ablation.json`
- **Environment:** `.venv/` created this session (the repo shipped without one and
  no dependencies were installed); run tests with `.venv/bin/python -m pytest`
