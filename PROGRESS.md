# SmartKeyNet — Progress Tracker

> **Update convention:** updating this file's checkboxes and the "Next task"
> line is part of the same end-of-session step as updating `SESSION_LOG.md`.
> This file gets *updated*, not rewritten, at the end of every session.
> It exists so a fresh Claude Code session (or a new person) can read
> `PLAN.md` + `SESSION_LOG.md` + this file and know what's done and what
> the single next task is, without reconstructing status from log prose.

---

## Next task

The epsilon/total_steps mismatch hypothesis (below) turned out **not**
to be the bottleneck — it's been superseded by a more specific finding,
so pick this thread up instead of re-testing step budgets:

Fixing `dqn.epsilon_decay_steps` (50,000 → 12,500, 2026-08-10) let training
genuinely converge (reward plateaus near -3 to -5 from step ~15,000
onward — more `total_steps` under the same weights won't move this
further). But the converged policy's `forced_rekey_ratio` went *to
1.000* (never proactively rekeys), and `p99_latency` still ties the
tuned threshold exactly. The likely explanation, worked out with actual
numbers: under the current `reward:` weights, `w_fr=0.1`'s max freshness
bonus (+0.1) is dwarfed by `c_rekey_base=1.0`'s minimum rekey cost (≥1.0
before considering hybrid pool-bit cost) — REUSE dominates any proactive
rekey in expected reward on benign S1, where there's no scarcity pressure
to make early rekeying pay off. So "wait until forced" may be the
genuinely reward-optimal S1 policy today, not a training shortfall.
Separately, `p99_latency` is a coarse metric here (4-value discrete
latency set, ~250 samples/episode, floor-driven HYBRID moments are
environment- not policy-determined) and may not be the right number to
judge "did the DQN learn well" by.

Two candidate next steps, **neither is a config-only change — flag to
the user before touching `reward.*`** (per `configs/default.yaml`'s own
"floor-adjacent, team ping" convention for that block):
1. Wire S2-S4 scenario dispatch into `environment.py` and re-run this
   comparison on S3 (QKD degradation) — real scarcity pressure there
   might make proactive rekeying actually reward-optimal, testing
   whether S1's specific triviality (not the agent) is why this
   didn't show up.
2. Discuss whether `reward.w_fr`/`reward.c_rekey_base` need
   recalibration, and/or whether `experiments/harness.py`'s
   `ScenarioResult` should track raw episode reward (it currently
   doesn't) as a less coarse comparison metric than `p99_latency`.

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
      *(Two S1-only checkpoint runs 2026-08-10: fixing an epsilon/total_steps mismatch let
      training genuinely converge, but the converged policy still ties — doesn't beat — the
      tuned threshold on `p99_latency`, and now never rekeys proactively at all. Likely a
      reward-weighting/metric-choice question, not a training-budget one — see Next task.
      S3 still doesn't exist as a scenario, so the gate can't be attempted for real yet either.)*
- [ ] Soft-reward baseline agent reproducing Noetzold (`agents/soft_reward_baseline.py`)
- [ ] Scenario dispatch S2-S4 wired into `environment.py` (`config["scenario"]`
      is currently read but not acted on)
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
| `env/request_generator.py` | stub (partial) | `random_request_generator()` implemented+tested (8 tests, `test_request_generator.py`). `build_tenant_graph()` and `RequestGenerator` (graph-driven stream) still `raise NotImplementedError`. |
| `env/environment.py` | implemented+tested | `SmartKeyNetEnv.reset/step/action_mask` fully wired (pool + deferral + masking + forecast + reward + session-key state). 14 behavioral tests incl. the split.md Gate W2 tests (`test_environment.py`). Only S1 scenario dispatch exists; S2-S6 config is read but not acted on. |

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
| `experiments/harness.py` | implemented+tested | `run_scenario` (one policy x scenario x seed episode → `ScenarioResult`, truncated via `max_steps`, default 250) + `run_grid` (every combination). Recomputes per-decision latency/hybrid-draw resolution from public `StateDict` fields (mirrors `env.environment`'s private cost tables/`REKEY_NOW` resolution, since `step()` doesn't surface them directly). 7 tests (`test_harness.py`), incl. the S1 x four-baselines zero-floor-violations check and a `run_grid` combination-count check. Only S1 exercised this session (S2-S6 dispatch not wired in `environment.py` yet). |
| `experiments/train.py` | implemented+tested | `train()` (one continuous S1 episode, `total_steps` from `configs/default.yaml`'s new `training:` block, periodic greedy-mode eval snapshots via the harness, final `DQNAgent.save` checkpoint), `GreedyDQNPolicy` (wraps a trained agent's `q_network` directly for deterministic epsilon=0 evaluation without touching `agents/dqn.py` or the agent's training epsilon-decay counter), `evaluate_against_baseline()` (trained agent vs. grid-searched `StaticThresholdPolicy`, same fixed eval seed). 6 tests (`test_train.py`), incl. a smoke run (100 steps) and a determinism check contrasting `GreedyDQNPolicy` against `DQNAgent.act()`'s genuine epsilon=1 stochasticity. **Two real 25,000-step runs executed 2026-08-10** (~45s / ~41s): the first (`dqn.epsilon_decay_steps=50_000`, double `training.total_steps`) tied the tuned threshold on `p99_latency` with real-but-incomplete learning; a same-day config fix (`epsilon_decay_steps` → `12_500`) let training genuinely converge (reward plateaus near -3 to -5) but still ties on `p99_latency`, and the converged policy's `forced_rekey_ratio` moved to `1.000` (never proactive) — see SESSION_LOG.md for the full numbers and the reward-weighting-based explanation. |

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
| `configs/default.yaml` | partial | `pool`, `key_lifetime`, `reward`, `use_foresight`, `tenant_graph.n_nodes`, `dqn`, `baselines`, `steering_attack`, `training` keys all present. `migration_schedule: []` empty (S6 not yet authored). `scenario: S1` — S2-S6 read but not dispatched by `environment.py`. |

---

## Last verified

- **Date:** 2026-08-10
- **Commit:** `3207eca` ("log: [solo] experiments/train.py + tests — 2026-08-10") — the commit this session started from; see SESSION_LOG.md for this session's own commit
- **`pytest` pass count:** 390 passed, 0 failed
