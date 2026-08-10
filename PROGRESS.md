# SmartKeyNet — Progress Tracker

> **Update convention:** updating this file's checkboxes and the "Next task"
> line is part of the same end-of-session step as updating `SESSION_LOG.md`.
> This file gets *updated*, not rewritten, at the end of every session.
> It exists so a fresh Claude Code session (or a new person) can read
> `PLAN.md` + `SESSION_LOG.md` + this file and know what's done and what
> the single next task is, without reconstructing status from log prose.

---

## Next task

**Not resolved — this is now a structural thread, not "move on to S4."**
2026-08-10's first load-spike session found `forced_rekey_ratio`
dropping from flat S1's `1.000` to `0.256`/`0.872` across two training
seeds and read that as "confirmed direction, spread worth a quick
look." A same-day follow-up ran the same load-spike config across 10
seeds (0-9) to size that spread for real, and found both a wider
spread than expected and a concrete root cause for it — see
SESSION_LOG.md for the full numbers and reasoning. Short version:

- Sorted `forced_rekey_ratio` across seeds 0-9: `0.190, 0.417, 0.418,
  0.703, 0.872, 0.895, 0.914, 0.971, 0.971, 1.000` (mean `0.735`,
  stdev `0.275`). 3/10 seeds land strongly proactive (`<0.5`), 7/10
  land weakly-to-never proactive (`>=0.70`, one at the exact
  never-proactive ceiling `1.000`) — leans bimodal, not a smooth
  spread around one typical value.
- **Root cause found**: `agents/dqn.py` never seeds anything —
  `QNetwork` weight init uses torch's default (unseeded) global RNG,
  and exploration/replay sampling use Python's global
  `random.random()`/`random.choice()`/`random.sample()`. No
  `torch.manual_seed`/`random.seed` call exists anywhere in the file.
  `training_cfg["seed"]` (`experiments/train.py`) only reaches
  `env.reset(seed=...)` — it seeds the environment's request stream,
  never the DQN's own init/exploration/replay-sampling randomness.
  Proof: today's seed=0 run gave `forced_rekey_ratio=1.000`; the prior
  session's seed=0 run (same nominal config) gave `0.256` — same
  "seed" value, unrelated outcomes, because neither run's DQN
  randomness was ever tied to that seed. Every "seed sweep" run so far
  (including today's 10-point one) is actually sampling over (env
  seed) x (uncontrolled ambient process RNG state) jointly, not a
  clean single-axis sweep.
- **Not fixed** — this is a behavior change to `agents/dqn.py`/
  `experiments/train.py` (add explicit `torch.manual_seed(seed)` +
  `random.seed(seed)` seeding tied to the training seed), flagged for
  sign-off rather than applied, per the standing rule on unrequested
  redesigns.

The mechanism direction still holds — the best seed reached 81%
proactive rekeying, so proactive rekeying clearly *can* emerge once
load varies — but given the unseeded-RNG root cause and the lopsided
cluster shape, this reads as a real reproducibility gap in the
training pipeline, not ordinary seed-to-seed training variance. `reward.w_fr`/`reward.c_rekey_base` are still not implicated by any of
this evidence — no `reward.*` change made or requested.

**Concretely next (pick one, needs a decision, not more solo
running):** (a) seed `agents/dqn.py`'s own randomness properly and
re-run the sweep to get a clean read before trusting these numbers
further, or (b) treat "the mechanism can produce proactive rekeying"
as sufficient confirmation and move on to building the real S4
scenario regardless. Either way, real S4 still needs the tenant graph
(`build_tenant_graph`/`RequestGenerator`, still `NotImplementedError`)
for genuine per-tenant flooding — today's load-spike diagnostic remains
a cruder, tenant-blind stand-in, not real S4 itself. Do not mistake
`configs/default.yaml`'s `load_spike:` block or
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
      the reward mechanism isn't broken — but a same-day 10-seed sweep found that result's
      seed-to-seed spread is wide and likely explained by `agents/dqn.py`'s own randomness
      never being seeded at all, a reproducibility gap worth resolving before trusting this
      evidence too far. Evidence toward attempting the gate once S3 exists, not the gate itself.)*
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
| `agents/dqn.py` | implemented+tested | `flatten_state(state, has_forecast)` (genuinely variable-length: 13 dims under `off`, 28 under `ewma`/`lstm`; `has_forecast` is now an **explicit required parameter** — the earlier `state["threat_score"] != 0.0` inference trick was removed 2026-08-08 as fragile-by-accident, see that session's log entry). `QNetwork` (2-hidden-layer MLP), `DQNConfig`/`load_dqn_config` (reads `configs/default.yaml`'s `dqn:` block), `DQNAgent(state_dim, has_forecast, config)` (internal circular-buffer replay, `act`/`observe`/`learn`/`save`/`load`, `has_forecast` fixed once at construction and threaded through every internal `flatten_state` call) — masking applied structurally at both action-selection *and* bootstrap-target time (Hard Rule 2), no security term anywhere (Hard Rule 1). 23 tests (`test_dqn.py`), incl. a regression test for a foresight-mode state with `threat_score == 0.0` still flattening to 28 dims, and an integration test training against the real `SmartKeyNetEnv` on S1 for 3000 steps with loss trending down. Not yet run to convergence — that's `experiments/train.py`. |
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
| `experiments/train.py` | implemented+tested | `train()` (one continuous S1 episode, `total_steps` from `configs/default.yaml`'s new `training:` block, periodic greedy-mode eval snapshots via the harness, final `DQNAgent.save` checkpoint), `GreedyDQNPolicy` (wraps a trained agent's `q_network` directly for deterministic epsilon=0 evaluation without touching `agents/dqn.py` or the agent's training epsilon-decay counter), `evaluate_against_baseline()` (trained agent vs. grid-searched `StaticThresholdPolicy`, same fixed eval seed). 6 tests (`test_train.py`), incl. a smoke run (100 steps) and a determinism check contrasting `GreedyDQNPolicy` against `DQNAgent.act()`'s genuine epsilon=1 stochasticity. **Five real 25,000-step runs executed 2026-08-10 across two sessions** (~40-45s each): an epsilon-schedule fix (`epsilon_decay_steps` 50k→12.5k) let training genuinely converge but the converged flat-S1 policy still tied the tuned threshold on `p99_latency` and never rekeyed proactively (`forced_rekey_ratio=1.000`); a same-day follow-up gave `environment.py`'s request stream an optional load-spike diagnostic (see `env/request_generator.py`'s row) and re-ran under it — `forced_rekey_ratio` dropped to `0.256`/`0.872` across two training seeds, directionally confirming the reward mechanism wasn't the problem, S1's stationarity was; a second same-day follow-up widened that to a 10-seed sweep (seeds 0-9, same load-spike config) and found both a wider spread (`0.190`-`1.000`, mean `0.735`, leaning bimodal) and its likely cause: `agents/dqn.py`'s own randomness (weight init, exploration, replay sampling) is never seeded by `training_cfg["seed"]` or anything else, so it varies uncontrolled across process runs independent of the configured seed — see SESSION_LOG.md for the full numbers and the proof (identical nominal seed=0 gave `1.000` today vs. `0.256` last session). |

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
- **Commit:** `821ce2c` ("log: [solo] load-spike diagnostic, re-run S1 comparison under it — 2026-08-10") — the commit this session started from; see SESSION_LOG.md for this session's own commit
- **`pytest` pass count:** 397 passed, 0 failed (unchanged — this session ran an external, non-repo sweep script against the existing `experiments/train.py`/`experiments/harness.py`, no source files changed)
