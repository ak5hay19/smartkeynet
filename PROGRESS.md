# SmartKeyNet — Progress Tracker

> **Update convention:** updating this file's checkboxes and the "Next task"
> line is part of the same end-of-session step as updating `SESSION_LOG.md`.
> This file gets *updated*, not rewritten, at the end of every session.
> It exists so a fresh Claude Code session (or a new person) can read
> `PLAN.md` + `SESSION_LOG.md` + this file and know what's done and what
> the single next task is, without reconstructing status from log prose.

---

## Next task

**Still not resolved — the seeding fix landed, but it made the finding
more interesting, not less.** Three same-day 2026-08-10 sessions in a
row on this thread now; see SESSION_LOG.md for full detail on each.
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
      measurement artifact. Whenever Gate W3 is attempted for real, it needs multi-seed
      reporting, not a single run, given this. Evidence toward attempting the gate once S3
      exists, not the gate itself.)*
- [ ] Soft-reward baseline agent reproducing Noetzold (`agents/soft_reward_baseline.py`)
- [ ] Scenario dispatch S2-S4 wired into `environment.py` (`config["scenario"]`
      is currently read but not acted on — the 2026-08-10 `load_spike` diagnostic is a
      request-rate-only stand-in layered on top of S1, not scenario dispatch)
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
| `env/pool_sim.py` | implemented+tested | `PoolSim` (refill/drain/exhaustion) + `SyntheticSKRQBERTrace`. 19 tests (`test_pool_sim.py`). |
| `env/deferral_queue.py` | implemented+tested | `DeferralQueue.enqueue/tick/pop_servable`, priority+FIFO, cumulative-headroom draw. 8 tests (`test_deferral_queue.py`). |
| `env/masking.py` | implemented+tested | `PolicyTable` (placeholder floor table, sticky ratchet) + `compute_mask`. 14 tests (`test_masking.py`). Floor table not yet calibrated against Q-OPSEC data. |
| `env/forecast_provider.py` | stub (partial) | `MovingAverageForecaster` (EWMA fallback) implemented+tested (9 tests, `test_forecast_provider.py`). `LSTMForecastProvider` does not exist yet (Addition A) — `use_foresight: lstm` currently raises `NotImplementedError` in `environment.py`. |
| `env/request_generator.py` | stub (partial) | `random_request_generator()` implemented+tested (11 tests, `test_request_generator.py`), incl. 3 new 2026-08-10 tests for its optional `load_spike` kwarg — a periodic, config-driven arrival-rate diagnostic (**explicitly not real S4** — see that session's SESSION_LOG.md entry and `configs/default.yaml`'s `load_spike:` block). `build_tenant_graph()` and `RequestGenerator` (graph-driven stream) still `raise NotImplementedError` — real S4 needs these. |
| `env/environment.py` | implemented+tested | `SmartKeyNetEnv.reset/step/action_mask` fully wired (pool + deferral + masking + forecast + reward + session-key state). 17 behavioral tests incl. the split.md Gate W2 tests (`test_environment.py`). Only S1 scenario dispatch exists; S2-S6 config is read but not acted on. 2026-08-10: wired `config["load_spike"]` through to `random_request_generator` (design decision 9) — orthogonal to scenario dispatch, not a substitute for it. |

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
| `configs/default.yaml` | partial | `pool`, `key_lifetime`, `reward`, `use_foresight`, `tenant_graph.n_nodes`, `load_spike`, `dqn`, `baselines`, `steering_attack`, `training` keys all present. `migration_schedule: []` empty (S6 not yet authored). `scenario: S1` — S2-S6 read but not dispatched by `environment.py`. `load_spike.enabled: false` by default (2026-08-10 diagnostic stub, NOT real S4 — see `env/request_generator.py`'s row and SESSION_LOG.md). |

---

## Last verified

- **Date:** 2026-08-10
- **Commit:** `58bebcf` ("log: [solo] 10-seed load-spike sweep, found unseeded DQN randomness — 2026-08-10") — the commit this session started from; see SESSION_LOG.md for this session's own commit
- **`pytest` pass count:** 400 passed, 0 failed (up from 397 — 3 new seed tests in `test_dqn.py`)
