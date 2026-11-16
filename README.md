# SmartKeyNet: RL for Hybrid Cryptography

SmartKeyNet is a reinforcement-learning decision layer for a multi-tenant
cloud key-management system (KMS) operating in the "hybrid" cryptography
era, where classical (X25519/AES), post-quantum (ML-KEM-768), and
quantum-backed (QKD-derived) keys must all be served from the same
system, and quantum key material is a genuinely scarce, refillable
resource. A masked DQN agent picks, per incoming request, one of five
actions — serve classical, serve PQC, serve hybrid (draws from the QKD
pool), reuse the existing session key, or force a rekey — under a
reward built purely from latency, energy, key-freshness, and
pool-scarcity cost.

The project's core contribution is architectural, not just algorithmic:
**security is enforced as a hard, structural constraint (action
masking), never as a term in the reward.** A per-(sensitivity-class,
threat-posture) floor table computes the minimum legal key tier for
every request; the agent's action space is masked down to only the
options that satisfy that floor before it ever sees a Q-value. This is
demonstrated two ways in the repo: (1) a from-scratch masked DQN is
compared against a reward-shaped ("soft-reward") baseline that
reproduces a security *term* in the reward instead of masking it, and
(2) an adversarial steering attack tries to poison the threat forecast
itself to talk the floor down — the masking layer's own guarantee held
in the original 2026-08-26 sweep (zero attacked-floor violations at
every seed/alpha), even though the attack does measurably suppress
detection at high attack strength (see
[Known limitations](#known-limitations--caveats)).

The threat signal is a real, trained LSTM forecaster
(`forecaster/`) fed by real network-flow features (RT-IoT2022). On top of
the original simulator-based study, the dashboard can also run the
same forecaster -> floor -> masked-DQN pipeline on **real packet
captures**: upload a `.pcap` (`/upload`) or capture live traffic from a
local interface (`/capture`).

Six of the paper's seven planned scenarios (S1 steady-state through S6
scripted compliance migration) are real, dispatched, and independently
tested; a masked-DQN-vs-tuned-baseline "Gate W3" comparison and an
11-point steering-attack dose-response sweep are both complete and
have been re-run under the real forecaster, with honestly reported
results (including nuanced ones — see
[Known limitations](#known-limitations--caveats) below).

## Quick start / installation

```bash
# from the repo root (already cloned)
pip install -r requirements.txt
pytest
```

Python 3.12 is what this repo is developed and tested against
(`requirements.txt` pins minimum versions only, not exact ones — see
its own header comment: "pin loosely for a capstone"). No `.env` or
external service is needed to run the test suite or the dashboard.

**Optional, only for the pcap features:** `/upload` and `/capture`
shell out to **`tshark`** (Wireshark); live capture additionally needs
a capture driver (**Npcap** on Windows). Neither is in
`requirements.txt`. If `tshark` is not on `PATH`, the tshark-dependent
tests skip and `/capture` shows an explicit "unavailable" page.

**Known gotchas:**
- `checkpoints/*.pt` (trained weights) are gitignored and **not in the
  repo** — nothing is pre-supplied. Tests and demos that need a
  specific checkpoint are skip-guarded or fail with a missing-file
  error on a fresh clone; the exact dependency of every panel/demo is
  in the table under
  [How to reproduce / regenerate results](#how-to-reproduce--regenerate-results).
- `data/raw/` is gitignored too (the RT-IoT2022 CSV is not
  redistributed here). The real-forecaster replay used by the
  S1/S3/S6 panels reads `data/raw/rt_iot2022/RT_IOT2022.csv`; fetch it
  with `python data/get_data.py --dataset rt_iot2022` (UCI ML
  Repository id 942, via `ucimlrepo`; needs network). The derived
  windowed training data
  (`data/processed/rt_iot2022/rt_iot2022_windows.npz`) **is** committed.
- Serving the already-rendered panels needs neither checkpoints nor
  `data/raw/` — the committed `dashboard/samples/*.html`/`*.json` are
  served byte-for-byte. Only **regenerating** a panel, or using
  `/upload` / `/capture`, needs them.

## How to verify it works

```bash
pytest
```

This is the real command (`pytest.ini` pins `testpaths = tests`,
`pythonpath = .`) — no extra flags needed. Last full run (on the
`final-demo` merge into `main`, 2026-09-27):

```
872 passed, 1 xfailed
```

That count was measured on the development machine, where the
checkpoints, the `data/raw/` CSV, and `tshark` were all present. Tests
that need any of those carry `requires_*` skip markers (e.g.
`requires_real_checkpoint`, `requires_tshark`), so on a bare clone they
**skip rather than run** and the "passed" number will be lower. The
passed/skipped split on a clean clone has not been measured.

The green suite is the project's proof its machinery is correct, not
just its demos: it includes exact-match "honesty guard" tests that
assert a rendered value is character-for-character identical to the
real computed field it claims to show (e.g.
`tests/test_render_explain.py`, `tests/test_render_comparison_table.py`,
`tests/test_render_budgeting_brain.py`'s
`test_conserving_agent_shows_zero_exhaustion_markers` /
`test_exhausting_baseline_shows_real_exhaustion_markers_and_banner`,
`tests/test_render_migration_wave.py`'s
`attribute_floor_change` unit tests,
`tests/test_render_posture_validation.py`'s committed-data-matches-log
checks) — these exist specifically to catch a dashboard panel that looks
right but was quietly seeded with a nicer-than-real number.

## How to see the results (the dashboard)

```bash
python -m dashboard.app
```

This prints `http://127.0.0.1:8000/` — open it in a browser for the
tabbed dashboard shell (`/` aliases the tabbed `/v2` shell; the original
card-grid index is kept unchanged at `/classic`). The server is
standard-library only (`http.server`, `dashboard/app.py`).

It serves two different kinds of thing, and the difference matters:

### 1. Eleven pre-rendered, static panels

The `PANELS` registry has **11 panels** (Living System counts as four
cards). Each is a **fully static, self-contained HTML** file (inline
CSS/SVG, no JS framework, no build step) committed under
`dashboard/samples/` — you can also just open them with `file://`.
Serving them computes nothing and runs no episode.

| Panel (registry title) | File(s) in `dashboard/samples/` | What it shows |
|---|---|---|
| Living System — S1 (steady/calm) and S2 (elevated HNDL), `StaticThresholdPolicy` | `living_system_0{1,2,3}_*.html` (S2), `living_system_s1_0{1,2,3}_*.html` (S1) | Real snapshots of a tenant service graph, nodes/edges colored by the tier each tenant was most recently served. These two cards honestly show a 100%-hybrid saturation caused by `StaticThresholdPolicy` itself (it serves hybrid whenever `pool_fill > 0.5`). |
| Living System — S1 and S2, trained DQN | `living_system_s1_dqn_0{1,2,3}_*.html`, `living_system_s2_dqn_0{1,2,3}_*.html` | Same real graph/episodes driven by the trained masked-DQN checkpoints: genuine tier variety (S1: 68 classical / 83 hybrid / 99 PQC; S2: 6 / 163 / 81, over 250 decisions each). |
| Explain Decision | `01_first_decision.html`, `02_floor_driven_only_hybrid_clears.html`, `03_real_cost_tradeoff.html` | The floor lookup, the action mask (and *why* each action is legal/illegal), the cost comparison, and the final decision, for three real S2 decisions. |
| Dose-Response | `dose_response_chart.html` | The steering-attack result: V(π) (below-floor-service rate under *true* posture) vs. attack strength α, masked agent vs. soft-reward baseline, real 11-point sweep. A two-plateau **step**, not a smooth rise — see caveats. |
| S3 Comparison Table | `s3_comparison_table.html` | Masked DQN vs. soft-reward baseline on S3 (QKD degradation): p99 latency, total reward, forced-rekey ratio, regret/exhaustion, below-floor rate. |
| Budgeting Brain | `budgeting_brain.html` | Real S3 pool-trajectory comparison, masked DQN vs. `AlwaysHybridPolicy` — same seed, same conditions, real exhaustion-event markers. |
| Migration Wave | `migration_wave.html` | S6's three scripted floor-ratchet events against a real held-out episode from an agent **never trained on S6**, with an honesty gate that only credits the schedule for a floor change it can verify from the observed data. |
| Posture Validation | `posture_validation.html` | A real 100-decision check that a higher `threat_score` yielding a *lower* floor (seen in the upload demo) is expected, not a bug: posture buckets CALM 66 / ELEVATED 4 / HIGH 30, **zero** floor-monotonicity violations, plus the honest finding that ELEVATED is real but only transiently reachable. |
| National Backbone — Illustrative Scenario | `national_backbone.html` | One shared key pool serving four *labeled* government agencies (Defense, Health, Tax, Passport Office), the Passport Office reclassified S1→S3 mid-run via S6's existing mechanism. Agency names are display labels over the real `tenant_0..3`; 0 of 250 decisions served below floor. **Stated limit, on the panel itself: `pool_fill` stayed 1.00 for the whole run, so no shared-pool pressure is demonstrated here** (the S3 comparison / Budgeting Brain panels cover scarcity). |

Each panel has a matching `*_data.json` where applicable
(`results_data.json`, `budgeting_data.json`, `migration_data.json`,
`posture_validation_data.json`, `national_backbone_data.json`,
`baselines_s3_data.json`) — the raw numbers the panel was rendered from.

### 2. Two live-computed routes (real pcap in, real decisions out)

| Route | What it does | Needs |
|---|---|---|
| `/upload` | Upload a `.pcap` (25 MiB limit). At request time: real `tshark`-backed flow extraction -> the trained LSTM forecaster -> posture -> floor -> a real `SmartKeyNetEnv` episode driven by the trained masked DQN. Invalid/empty/oversized uploads get an honest error page, never a fabricated result. | `tshark`; `checkpoints/dqn_s1.pt`, `checkpoints/lstm_threat_forecaster.pt` |
| `/capture` | An alternate *trigger* for the identical pipeline: captures real traffic from a chosen local interface for a fixed 20-second window, then hands it to the same code `/upload` uses and renders the same results page. | `tshark` + a capture driver (Npcap on Windows) and permission to capture; same two checkpoints |

Each is a single bounded run-to-completion request ending in a static
result page — **not** a continuously streaming live tick of
`SmartKeyNetEnv`. The results page carries three standing caveats:

- **Honesty note (domain mismatch):** the forecaster was trained on
  RT-IoT2022 IoT-testbed flows; arbitrary real traffic is a different
  domain, so the threat scores are real model outputs but are not a
  validated detector for that traffic.
- **Demo tenant note (provenance):** each decision row's *tenant,
  service, and sensitivity class* come from an **independent synthetic
  policy stream** (`random_request_generator`, deliberately kept
  structurally separate from the threat signal per Hard Rule 2) — **not**
  from the captured/uploaded traffic. Only the **threat score, posture,
  floor, legal actions, and chosen action** are live-computed from the
  real pcap.
- **Replay note** (shown only when it applies): if the episode's
  decision count exceeds the number of real extracted flows, the flows
  are replayed by modulo wrap — real replay, not new traffic.

## How to reproduce / regenerate results

All commands are run from the repo root, after `pip install -r
requirements.txt`. Remember: `checkpoints/*.pt` and `data/raw/` are
**gitignored**; nothing below works until you've produced them.

### What each panel/demo needs

| Panel / demo (driver module) | Checkpoints needed | Also needs |
|---|---|---|
| Explain Decision — `dashboard.render_explain_demo` | `lstm_threat_forecaster.pt` | — |
| Living System (all four cards) — `dashboard.render_living_system_demo` | `lstm_threat_forecaster.pt`; `dqn_s1.pt`, `dqn_s2.pt` (trained-DQN cards) | `data/raw` CSV (S1 uses real replay) |
| Dose-Response + S3 Comparison — `dashboard.render_results_demo` | `lstm_threat_forecaster.pt`; `dqn_s2.pt`, `soft_reward_baseline_s2.pt` (S5); `s3_masked_seed{1,4,7}.pt`, `s3_soft_reward_seed{1,4,7}.pt` (S3) | `data/raw` CSV (S3 uses real replay) |
| Budgeting Brain — `dashboard.render_budgeting_brain_demo` | `lstm_threat_forecaster.pt`, `s3_masked_seed1.pt` | `data/raw` CSV |
| Migration Wave — `dashboard.render_migration_wave_demo` | `lstm_threat_forecaster.pt`, `dqn_s1.pt` | `data/raw` CSV |
| National Backbone — `dashboard.render_national_backbone_demo` | `lstm_threat_forecaster.pt`, `dqn_s1.pt` | `data/raw` CSV |
| Posture Validation — `dashboard.render_posture_validation_demo` | `lstm_threat_forecaster.pt`, `dqn_s1.pt` | `data/raw` CSV, `tshark`, and `scratchpad/sample.pcap` — a gitignored, locally captured pcap that is **not** in the repo (capture your own; the committed panel was rendered from the original) |
| `/upload`, `/capture` | `lstm_threat_forecaster.pt`, `dqn_s1.pt` | `tshark` (+ Npcap for `/capture`) |

Each driver prints the file(s) it wrote, e.g.:

```bash
python -m dashboard.render_explain_demo
python -m dashboard.render_living_system_demo
python -m dashboard.render_results_demo
python -m dashboard.render_budgeting_brain_demo
python -m dashboard.render_migration_wave_demo
python -m dashboard.render_national_backbone_demo
python -m dashboard.render_posture_validation_demo
```

### Regenerating the inputs

**Raw dataset** (for the replay provider; needs network):

```bash
python data/get_data.py --dataset rt_iot2022     # -> data/raw/rt_iot2022/RT_IOT2022.csv
```

**LSTM threat forecaster** (`checkpoints/lstm_threat_forecaster.pt`),
trained from the committed windowed data:

```bash
python -m forecaster.train_threat_lstm           # defaults: committed .npz -> checkpoints/lstm_threat_forecaster.pt
```

Retraining is seeded (`--seed`, default 0) but is not guaranteed to
reproduce the original checkpoint bit-for-bit, and the dashboard
numbers in this README were produced with the original. If you need to
rebuild the windowed `.npz` itself from the raw CSV:
`python -m forecaster.rt_iot_features`.

**DQN checkpoints.** `python -m experiments.train` always trains the
masked DQN on S1 from `configs/default.yaml`'s `training:` block and is
not scenario-selectable from the CLI. **It does not inject the real
forecaster** — an unwrapped run uses the EWMA placeholder for threat
(see below), so a checkpoint trained that way is *not* equivalent to
the ones the panels use. The panel checkpoints were trained (Addition E
Phase 4b, 2026-09-14) by calling the real functions directly with a
`forecast_provider_factory`:

| Checkpoints | Scenario | Forecaster injected | Seeds / steps |
|---|---|---|---|
| `dqn_s1_seed{1,4,7}.pt`, `soft_reward_s1_seed{1,4,7}.pt` (`dqn_s1.pt` = seed 1) | S1 | `RTIoT2022ReplayProvider` | 1, 4, 7 / 75,000 each |
| `s3_masked_seed{1,4,7}.pt`, `s3_soft_reward_seed{1,4,7}.pt` | S3 | `RTIoT2022ReplayProvider` | 1, 4, 7 / 75,000 each |
| `dqn_s2.pt`, `soft_reward_baseline_s2.pt` | S2 | `LSTMForecastProvider` | 0 / 25,000 |

```python
from experiments.train import train, train_soft_reward_baseline, load_full_config
from forecaster.model import LSTMForecastProvider

config = load_full_config("configs/scenarios/s2_hndl.yaml")  # use the matching file in configs/scenarios/
agent, record = train(
    config, scenario="S2",
    forecast_provider_factory=lambda seed: LSTMForecastProvider("checkpoints/lstm_threat_forecaster.pt"),
)
print(record.checkpoint_path)
```

(The exact driver scripts used for Phase 4b lived in the gitignored
`scratchpad/` and are not in the repo; the table above and
`PROGRESS.md`'s Phase 4b entry record the conventions they followed.
Seeds/steps override via `training_overrides`.) `train()` refuses to run
(`ValueError`) if the config sets `train_eligible: false` — currently
only `configs/scenarios/s6_migration.yaml`, because S6 is held-out
evaluation only, never a training scenario (Hard Rule 8).

**Evaluate a policy** — `experiments/harness.py`'s
`run_scenario(policy, scenario, config, seed)` runs one episode;
`evaluate_multi_seed(policy, scenario, config, eval_seeds)` runs it
across several seeds and reports mean/std, the way every real result in
this repo was measured. Both (and the attack variants) accept the same
optional `forecast_provider_factory`:

```python
from experiments.harness import evaluate_multi_seed
from experiments.train import load_full_config
from agents.baselines import StaticThresholdPolicy

config = load_full_config("configs/scenarios/s3_degradation.yaml")
result = evaluate_multi_seed(
    StaticThresholdPolicy(0.5), "S3", config, eval_seeds=[900, 901, 902],
)
print(result.total_reward_mean, result.total_reward_std)
```

Every real demo-panel figure was produced by reloading a real
checkpoint and re-running a real eval, never transcribed by hand (see
`dashboard/render_results_demo.py`'s docstring for the fullest example).

## Headline results (current: real forecaster + recalibrated floor table)

All from the 2026-09-14 Phase 4b campaign (checkpoint-averaged: 3
training seeds × 8 eval seeds, `StaticThresholdPolicy` tuned via grid
search), sourced from `PROGRESS.md`/`SESSION_LOG.md` and
`dashboard/samples/results_data.json`:

- **Gate W3 — met on both scenarios** (masked DQN's `total_reward` vs.
  the tuned static-threshold baseline): **5.61×** on S1 (was ~12.6×
  under the old placeholder forecaster — the margin genuinely shrank)
  and **3.58×** on S3 (was ~3.78×). `floor_violations_total = 0` for
  every policy and seed.
- **S3, masked vs. soft-reward:** below-floor rate **0.0000** vs.
  **0.0802**; regret events **0.00** vs. **4.42**; masked DQN is ahead
  on every axis measured.
- **S5 steering-attack dose-response** (masked DQN, V(π) vs. α):
  0.0000 at α ≤ 0.1, creeping up to 0.0072 by α = 0.5, then a **step to
  a 0.2984 plateau at α ≥ 0.6** (soft-reward baseline: 0.3912 → 0.4272,
  the same two-plateau shape off a much higher baseline).
- **Posture validation (pcap + real held-out spans, 100 decisions):**
  CALM 66 / ELEVATED 4 / HIGH 30; zero floor-monotonicity violations.

## Repository structure

| Path | What it implements |
|---|---|
| `env/contracts.py` | The frozen interface contracts everything else builds against — `Action`/`KeyType`/`SensitivityClass`/`ThreatPosture` enums, `Request`/`StateDict` shapes, the `ForecastProvider` ABC, event-log TypedDicts. Read this first to understand the state/action space. |
| `env/environment.py` | `SmartKeyNetEnv` — the Gym-style MDP: wires pool sim + deferral queue + masking + forecaster + reward into `reset()`/`step()`. Also where all S1–S6 scenario dispatch logic lives (`config["scenario"]`). Accepts an optional `forecast_provider_factory` to inject a real forecaster. |
| `env/masking.py` | `PolicyTable` (the (sensitivity class × threat posture) → minimum-tier floor table, a one-way ratchet within an episode) + `compute_mask()` (turns a floor + feasibility checks into the legal-action mask). This is where the "security is masking, not reward" guarantee is structurally enforced. Each floor cell's basis is labeled in code (`FloorCalibrationBasis`) — see the floor-table note below. |
| `env/pool_sim.py` | `PoolSim` + `SyntheticSKRQBERTrace` — the QKD pool's refill/drain/exhaustion physics. |
| `env/deferral_queue.py` | `DeferralQueue` — what happens to a hybrid-mandatory request when the pool can't cover it right now (Hard Rule 9: deferred and logged, never silently downgraded). |
| `env/forecast_provider.py` | `MovingAverageForecaster` (EWMA over `[qber, load]`) — the default threat stand-in when no provider is injected, and the **pool-forecast** component the real provider composes. |
| `env/request_generator.py` | `build_tenant_graph`/`RequestGenerator` — the real NetworkX tenant graph and graph-driven request stream, plus the scenario-specific mechanisms (S4's `flood_override`, S6's `set_tenant_sensitivity_class`), and `random_request_generator` (the independent synthetic tenant/class stream the pcap demos use — see the Demo tenant note). |
| `forecaster/model.py` | `SmartKeyForecaster` (real 1-layer LSTM + threat head) and `LSTMForecastProvider` (a `ForecastProvider`; threat from the LSTM, pool forecast from the EWMA — the pool head is not learned). |
| `forecaster/rt_iot_features.py`, `forecaster/train_threat_lstm.py` | Real RT-IoT2022 ingestion (39-feature extraction, scaler, window/stride windowing) and the leakage-free, class-weighted LSTM training that produces `checkpoints/lstm_threat_forecaster.pt`. |
| `forecaster/rt_iot_replay.py` | `RTIoT2022ReplayProvider` — replays real held-out RT-IoT2022 rows through the LSTM (used for S1/S3/S6). |
| `forecaster/pcap_flow_features.py`, `forecaster/pcap_extract_provider.py` | Real `tshark`-backed pcap → flow → feature extractor and the `PCAPExtractProvider` built on it (the engine behind `/upload`, `/capture`, and the posture validation's pcap leg). |
| `forecaster/dataset.py`, `forecaster/train.py` | **Still stubs** (`NotImplementedError`) — the originally-planned rollout-log windowing/training entry points, superseded in practice by `rt_iot_features.py` + `train_threat_lstm.py` above. |
| `metrics/regret.py` | `compute_episode_metrics()`/`attribute_regret()` — regret/churn accounting (forced-rekey ratio, deferral counts). |
| `agents/dqn.py` | `DQNAgent`/`QNetwork` — the masked DQN itself. Masking applied at both action-selection and bootstrap-target time. |
| `agents/baselines.py` | The four Hard-Rule-7-mandated tuned non-RL baselines: `AlwaysPQCPolicy`, `AlwaysHybridPolicy`, `StaticThresholdPolicy` (grid-searchable), `RandomPolicy`. |
| `agents/soft_reward_baseline.py` | The reward-shaped comparison agent: reuses `DQNAgent` unmodified, but trains with `security_masking: false` and a reward that includes a `w_sec * security_score(tier)` term — the "what if we didn't mask" control. |
| `attack/trace_generator.py`, `attack/attacking_provider.py` | The steering attack: `generate_adversarial_window()` implements the paper's equation 7 input-shaping; `AttackingForecastProvider` wraps a real forecaster to inject it live into an episode. |
| `experiments/harness.py` | `run_scenario`/`run_grid`/`evaluate_multi_seed`/`evaluate_attack_multi_seed` — the evaluation harness every real number in this repo was measured through. |
| `experiments/train.py` | `train()`/`train_soft_reward_baseline()` — training entry points, both scenario-parameterized, both accepting a `forecast_provider_factory`. |
| `dashboard/app.py` | The local server: the 11-panel registry (`PANELS`), the `/v2` tabbed shell, `/classic`, and the live `/upload` and `/capture` routes. |
| `dashboard/upload_demo.py`, `dashboard/live_capture.py`, `dashboard/render_upload.py` | The upload/capture pipeline driver, the bounded `tshark` live-capture wrapper, and the pure renderer for their result/form/error pages (including the three caveats). |
| `dashboard/explain.py` | `explain_decision()` — the real, policy-agnostic "why did it pick this action" backend (reads the mask/floor/cost tables directly, zero drift by construction). |
| `dashboard/render_*.py` / `dashboard/render_*_demo.py` | The pure (dependency-free HTML/SVG) panel renderers and the drivers that produce the real data each needs. |
| `dashboard/samples/` | The committed, ready-to-open output of every driver above. |
| `dashboard/mockups/` | The original static, fabricated-data mockups the real renderers were styled to match visually — **never a data source**, only a visual reference. |
| `configs/default.yaml`, `configs/scenarios/*.yaml` | Scenario and hyperparameter configs — one standalone, directly-loadable file per scenario (including `s_narrative_national_backbone.yaml`, the National Backbone panel's config). |
| `docs/v2_gameplan.md` | The phase plan for the pcap/live-capture/validation/backbone work. |
| `docs/smartkeynet_ieee_paper_5.tex` | A paper draft — **an older draft than the submitted paper**, last committed 2026-08-30, i.e. *before* the Phase 4b retrain; its numbers are not kept in sync with this README or with `dashboard/samples/*.json`. Treat the submitted paper, not this file, as authoritative. |
| `PLAN.md` / `PLAN2.md` | The frozen Hard Rules (non-negotiable design constraints) and full original design spec — read these for the *why* behind any of the above. |
| `PROGRESS.md` / `SESSION_LOG.md` | The project's own running status tracker and session-by-session narrative log — the authoritative "what's real, what's next" source, kept current every session. |

**Floor table — what is and isn't grounded.** No standard defines a
4-class × 3-posture grid; that structure is this project's own design.
Per `env/masking.py` and `PROGRESS.md`'s 2026-09-14 recalibration, **9 of
the 12 cells are labeled engineering `judgment`** (informed by NIST FIPS
203, SP 800-57 Rev. 5, and ETSI GS QKD 014, but not dictated by them),
and **the three S3 cells are labeled `citation_with_judgment`**, resting
on one shared basis: NSA/CNSA 2.0's 2030 full-quantum-resistance
deadline, which supports making S3's floor independent of observed
posture. Only one cell's *value* changed in that recalibration —
(S3, CALM): `SERVE_PQC` → `SERVE_HYBRID` — and CNSA 2.0 itself only
requires quantum-resistance (PQC suffices), so choosing *hybrid*
specifically is the project's judgment, not a CNSA requirement.

## Scenarios (S1–S6)

| Scenario | Tests | Train or held-out? |
|---|---|---|
| S1 | Steady-state baseline — no scripted stress. | Train + eval. |
| S2 | HNDL (harvest-now-decrypt-later): a scripted threat-posture elevation partway through the episode. | Train + eval. |
| S3 | QKD pool degradation: a recalibrated SKR-collapse spike that genuinely exhausts the pool within one episode (unlike S1). | Train + eval. |
| S4 | DDoS / noisy-neighbor: one tenant gets an additive flood of extra requests. | Eval only (not used in any Gate W3 training run). |
| S5 | Steering attack dose-response: the real threat signal is progressively replaced with an adversarial one (`α` from 0 to 1) via `attack/trace_generator.py`. Run on top of S2. | Eval only (attack sweep, not a trainable scenario in its own right). |
| S6 | Migration wave: a scripted, exogenous schedule of per-tenant sensitivity-class ratchets (a compliance-style migration), applied entirely upstream in request generation. | **Held-out evaluation only** — `train_eligible: false` is enforced at the code level (Hard Rule 8); an agent trained on S1 is evaluated against S6's schedule, never trained on it. |

Which forecaster drives which scenario (Phase 4b's documented choice):
S1/S3/S6 use `RTIoT2022ReplayProvider` (real held-out flow data); S2/S5
use bare `LSTMForecastProvider`, because the replay provider ignores the
`threat_features` channel that S2's scripted schedule and S5's attack
both operate through.

## What's implemented vs. what's future work (honest)

**Genuinely real, tested, and part of the green suite:**
- The full MDP: pool sim, deferral queue, action masking, reward, all
  wired end-to-end in `SmartKeyNetEnv`.
- All six scenarios (S1–S6), each independently dispatched and tested.
- The masked DQN agent, all four tuned non-RL baselines (Hard Rule 7),
  and the soft-reward reproduction baseline — retrained under the real
  forecaster and recalibrated floor table.
- **The real threat forecaster**: RT-IoT2022 ingestion, a trained LSTM
  (leakage-free temporal split, class-weighted loss; validation
  macro-F1 0.986, non-degenerate), the held-out replay provider, and the
  cold-start fixes — wired in through the `forecast_provider_factory`
  seam.
- Real pcap ingestion (`tshark`-backed extractor) and the `/upload` and
  `/capture` demos built on it.
- The steering attack (equation-7 input shaping) and its full 11-alpha
  dose-response sweep on S2, re-run under the real forecaster.
- Gate W3 (5.61× S1, 3.58× S3), the S3 masked-vs-soft-reward
  comparison, the posture/floor-monotonicity validation, and the
  National Backbone illustrative scenario.
- Eleven dashboard panels, each rendered from real measured data with
  dedicated "honesty guard" tests (see
  [How to verify it works](#how-to-verify-it-works)).
- `dashboard/explain.py`'s per-decision explainability backend.

**Not done / still stubs:**
- `api/main.py` (the AWS-KMS-flavored REST facade, Addition G) — a stub
  only.
- `forecaster/dataset.py` and `forecaster/train.py` — still
  `NotImplementedError` stubs (superseded in practice, see the repo
  map).
- The LSTM's pool head: only the **threat** head is trained; the pool
  forecast is the EWMA (no pool-head training data exists).
- Feeding real features through `env/environment.py`'s *own* tick path
  (Addition F, PLAN2.md §8B). Real signals reach the env only via the
  injected provider; the env's built-in threat feature is still the
  2-dim `[qber, load]` placeholder (see caveats).
- A continuously-streaming live demo shell (a running `SmartKeyNetEnv`
  streamed in real time). `/upload` and `/capture` are bounded,
  run-to-completion requests, not that.
- `docs/report.md` — a section-header skeleton only (`TODO`
  placeholders). `docs/smartkeynet_ieee_paper_5.tex` is an *older*
  paper draft, not a current write-up (see the repo map).

## Known limitations / caveats

These are understood, investigated findings — reported in
`SESSION_LOG.md` and `PROGRESS.md`, not gaps discovered after the fact:

- **S2 and S5 feed the LSTM out-of-distribution, synthetic inputs.**
  The LSTM was trained on 39 real RT-IoT2022 flow features, but the
  environment's own threat feature is still the 2-dim `[qber, load]`
  placeholder, zero-padded into the model's input. The resulting
  `threat_score` is a real model output and responds to that input, but
  on data the model was never trained on — so S2/S5 results exercise
  the masking/ratchet machinery, not real-world detection quality.
  S1/S3/S6 instead replay *real* held-out RT-IoT2022 rows (benign span
  followed by attack span, wrapping by modulo) — real data, but replay,
  not live ingestion. Fixing the underlying gap is Addition F (not done).
- **Cold-start artifact — found and fixed, but history matters.** Both
  providers originally zero-padded their LSTM window at episode start,
  spiking `threat_score` within the first ticks regardless of true
  class; Hard Rule 2's one-way ratchet then locked that spike in.
  `RTIoT2022ReplayProvider` (warm-start with real preceding rows) and
  bare `LSTMForecastProvider` (pad with the earliest real buffered row)
  were both fixed 2026-09-14. Before the second fix, the S5 sweep was
  perfectly flat at every α (uninformative); the re-run is the step curve
  reported above.
- **The S5 dose-response is a two-plateau step, not a smooth rise, and
  the masked agent is not flat-zero.** Masked V(π) jumps to ~0.298 at
  α ≥ 0.6. This is a real boundary of the ratchet's defense — it stops a
  detected threat from being walked back down, but cannot manufacture a
  detection the (attacked) sensor suppressed for the whole episode — not
  a `compute_mask` defect. Zero attacked-floor violations was verified
  for the original 2026-08-26 sweep; it is not separately re-reported
  for the 2026-09-14 re-run.
- **Posture is mostly CALM; ELEVATED is only transient.** On real
  data, a steady-calm span stays CALM and a steady-attack span goes
  HIGH; ELEVATED appears only during a genuine benign→attack
  transition (CALM 66 / ELEVATED 4 / HIGH 30 in the validation). The
  old placeholder-era behavior (posture ratcheting to ELEVATED within
  the first 1–3 decisions from a load-term artifact) no longer applies
  to the real-forecaster runs.
- **`regret_events` and `pool_exhaustion_events` are identical by
  construction**, not two independently-informative metrics — every
  regret event *is* a pool-exhaustion event in the current environment
  (see `experiments/harness.py::run_scenario`'s own docstring).
- **`p99_latency` is a discrete-cost-model artifact above a low
  threshold**, not a meaningful discriminator between policies: it
  saturates at exactly `1.5000` once ≥1.6% of an episode's decisions
  cost `SERVE_HYBRID` (root-caused against `np.percentile`'s
  interpolation). Use `total_reward` or `below_floor_rate` to compare
  policies.
- **S1's training-seed spread collapsed under the real forecaster**
  (pooled `total_reward` std 0.13 on -11524.57). Investigated and not a
  bug: different Q-networks agree on the argmax policy along real
  trajectories, plausibly because each 75,000-step run cycles the same
  real held-out span ~3 times. It means S1's seed-variance is a weak
  robustness signal here.
- **National Backbone demonstrates reclassification, not pool
  pressure** (`pool_fill` = 1.00 throughout), and Defense and Health are
  both S3 in the real seed-30 assignment, so they look identical.
- **The `/upload` and `/capture` threat scores are real model outputs on
  out-of-domain traffic**, and their tenant/service/class labels are
  synthetic — see the three result-page caveats above.
- **Table V (paper) is not re-verified here.** The earlier
  (placeholder-forecaster-era) observation that a re-tuned static
  threshold ties or beats the masked agent on most *displayed* columns
  while losing on `total_reward` was not re-measured column-by-column
  under Phase 4b; the sourced current comparison is the `total_reward`
  Gate W3 result above. The `.tex` file in `docs/` predates Phase 4b.

## Getting a fuller picture

For the frozen design constraints (Hard Rules 1–11) and full original
architecture spec, read [`PLAN.md`](./PLAN.md) and
[`PLAN2.md`](./PLAN2.md). For exactly what's real as of right now,
per-file, [`PROGRESS.md`](./PROGRESS.md) is the single source of
truth — its "Next task" section is kept deliberately current.
[`SESSION_LOG.md`](./SESSION_LOG.md) has the full session-by-session
narrative, investigation writeups, and numbers behind every claim
above.
