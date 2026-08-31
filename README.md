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
itself to talk the floor down — the masking layer's guarantee holds
throughout (verified: zero attacked-floor violations across the full
alpha sweep), even though the attack does measurably suppress
detection at high attack strength.

Six of the paper's seven planned scenarios (S1 steady-state through S6
scripted compliance migration) are real, dispatched, and independently
tested; a masked-DQN-vs-tuned-baseline "Gate W3" comparison and an
11-point steering-attack dose-response sweep are both complete with
real, honestly-reported results (including a nuanced one: see
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
external service is needed to run the test suite or the dashboard
demos below.

**Known gotchas:**
- `checkpoints/*.pt` (trained agent weights) are gitignored and **not
  in the repo**. The test suite and two of the six dashboard demos
  (Explain Decision, Living System) don't need them — they run a
  baseline policy fresh. The other four demos (Budgeting Brain,
  Migration Wave, the S3-comparison-table/dose-response pair) reload
  specific checkpoints by filename and will fail on a fresh clone until
  you train them yourself — see
  [How to reproduce / regenerate results](#how-to-reproduce--regenerate-results).
- `data/raw/` is gitignored too (the RT-IoT2022 dataset has no
  redistribution license); this doesn't block anything above, since
  nothing in `env/`, `agents/`, or `dashboard/` reads it yet (the
  request generator and forecaster are both synthetic/placeholder —
  see [What's implemented vs. future work](#whats-implemented-vs-whats-future-work-honest)).

## How to verify it works

```bash
pytest
```

This is the real command (`pytest.ini` pins `testpaths = tests`,
`pythonpath = .`) — no extra flags needed. Running it now:

```
649 passed, 1 xfailed
```

The green suite is the project's proof its machinery is correct, not
just its demos: it includes exact-match "honesty guard" tests that
assert a rendered value is character-for-character identical to the
real computed field it claims to show (e.g.
`tests/test_render_explain.py`, `tests/test_render_comparison_table.py`,
`tests/test_render_budgeting_brain.py`'s
`test_conserving_agent_shows_zero_exhaustion_markers` /
`test_exhausting_baseline_shows_real_exhaustion_markers_and_banner`,
`tests/test_render_migration_wave.py`'s
`attribute_floor_change` unit tests) — these exist specifically to
catch a dashboard panel that looks right but was quietly seeded with a
nicer-than-real number.

## How to see the results (the dashboard)

The simplest path: open any of these six files directly in a browser.
They're **fully static, self-contained HTML** (inline CSS/SVG, no JS
framework, no build step, no server) — just double-click or
`file://` them, already committed at:

| File | What it shows |
|---|---|
| `dashboard/samples/living_system_01_first_decision.html`, `_02_graph_fully_populated.html`, `_03_final_decision.html` | Three real snapshots of a tenant service graph under S2, nodes/edges colored by the tier each tenant was most recently served (real `NetworkX` graph, real per-decision tier resolution). |
| `dashboard/samples/01_first_decision.html`, `02_floor_driven_only_hybrid_clears.html`, `03_real_cost_tradeoff.html` | "Explain Decision" traces — the floor lookup, the action mask (and *why* each action is legal/illegal), the cost comparison, and the final decision, for three real, structurally-different S2 decisions. |
| `dashboard/samples/dose_response_chart.html` | The steering-attack headline result: V(π) (the below-floor-service rate under true posture) vs. attack strength α, masked agent vs. soft-reward baseline, real 11-point sweep. |
| `dashboard/samples/s3_comparison_table.html` | Masked DQN vs. soft-reward baseline on S3 (QKD degradation), side by side on p99 latency, total reward, forced-rekey ratio, regret/exhaustion events, and below-floor rate. |
| `dashboard/samples/budgeting_brain.html` | Real S3 pool-trajectory comparison, masked DQN vs. `AlwaysHybridPolicy` — same seed, same exogenous conditions, real exhaustion-event markers. |
| `dashboard/samples/migration_wave.html` | S6's three scripted floor-ratchet events (a compliance-style migration) against a real held-out episode from an agent that was **never trained on S6** — with an honesty gate that only credits the schedule for a floor change it can actually verify from the observed data. |

Each `*.html` panel has a matching `*_data.json` (or, for Explain
Decision/Living System, is reproducible via its own demo driver) —
the raw numbers/trajectory the panel was rendered from, if you want to
check a rendered value against source data yourself.

**Not yet buildable:** a seventh panel (Threat Input, live
pcap-style feature visualization) is blocked on the real forecaster —
see below. `dashboard/app.py` (a single live, wired 4-beat demo shell
tying all panels together) is a stub, not started — the static panels
above are the real demo-able artifact today.

## How to reproduce / regenerate results

All commands below are run from the repo root, after `pip install -r
requirements.txt`.

**Regenerate a dashboard panel** (each is a self-contained driver
module, run with `-m` so relative imports resolve; each prints the
file(s) it wrote):

```bash
# No checkpoint needed — runs a baseline policy fresh:
python -m dashboard.render_explain_demo
python -m dashboard.render_living_system_demo

# Needs checkpoints/dqn_s2.pt, soft_reward_baseline_s2.pt,
# s3_masked_seed{1,4,7}.pt, s3_soft_reward_seed{1,4,7}.pt (train first, below):
python -m dashboard.render_results_demo

# Needs checkpoints/s3_masked_seed1.pt:
python -m dashboard.render_budgeting_brain_demo

# Needs checkpoints/dqn_s1.pt:
python -m dashboard.render_migration_wave_demo
```

**Train an agent** — `experiments/train.py`'s CLI entry point
(`python -m experiments.train`) always trains the masked DQN on S1
using `configs/default.yaml`'s `training:` block, and is not
scenario-selectable from the command line:

```bash
python -m experiments.train
```

To train on another scenario or the soft-reward baseline, call the
real functions directly — both take an explicit `scenario` parameter
(defaults to `"S1"`), matching `experiments/harness.py`'s own
`run_scenario`/`run_grid` convention:

```python
from experiments.train import train, train_soft_reward_baseline, load_full_config

# masked DQN, S3:
config = load_full_config("configs/scenarios/s3_degradation.yaml")
agent, record = train(config, scenario="S3")
print(record.checkpoint_path)

# soft-reward baseline (reward-shaped, unmasked security):
config = load_full_config("configs/soft_reward_baseline.yaml")
agent, record = train_soft_reward_baseline(config)
```

`train()` refuses to run (`ValueError`) if the config sets
`train_eligible: false` — currently only
`configs/scenarios/s6_migration.yaml`, because S6 (the migration wave)
is held-out evaluation only, never a training scenario (Hard Rule 8).

**Evaluate a policy** — `experiments/harness.py`'s
`run_scenario(policy, scenario, config, seed)` runs one episode and
returns a `ScenarioResult`; `evaluate_multi_seed(policy, scenario,
config, eval_seeds)` runs it across several seeds and reports
mean/std, the way every real result in this repo was actually
measured:

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

Checkpoints referenced above live in `checkpoints/` (gitignored — you
have to train them yourself; none are pre-supplied). Every real
demo-panel figure in this repo was produced this exact way — reloading
a real checkpoint and re-running a real eval, never transcribed by
hand (see `dashboard/render_results_demo.py`'s docstring for the
fullest example of this pattern).

## Repository structure

| Path | What it implements |
|---|---|
| `env/contracts.py` | The frozen interface contracts everything else builds against — `Action`/`KeyType`/`SensitivityClass`/`ThreatPosture` enums, `Request`/`StateDict` shapes, the `ForecastProvider` ABC, event-log TypedDicts. Read this first to understand the state/action space. |
| `env/environment.py` | `SmartKeyNetEnv` — the Gym-style MDP: wires pool sim + deferral queue + masking + forecaster + reward into `reset()`/`step()`. Also where all S1–S6 scenario dispatch logic lives (`config["scenario"]`). |
| `env/masking.py` | `PolicyTable` (the (sensitivity class × threat posture) → minimum-tier floor table, a one-way ratchet within an episode) + `compute_mask()` (turns a floor + feasibility checks into the legal-action mask). This is where the "security is masking, not reward" guarantee is structurally enforced. |
| `env/pool_sim.py` | `PoolSim` + `SyntheticSKRQBERTrace` — the QKD pool's refill/drain/exhaustion physics. |
| `env/deferral_queue.py` | `DeferralQueue` — what happens to a hybrid-mandatory request when the pool can't cover it right now (Hard Rule 9: deferred and logged, never silently downgraded). |
| `env/forecast_provider.py` | `MovingAverageForecaster` (real, EWMA-based) — the current stand-in for the threat/pool forecaster. **`LSTMForecastProvider` does not exist yet** (see below). |
| `env/request_generator.py` | `build_tenant_graph`/`RequestGenerator` — the real NetworkX tenant graph and graph-driven request stream (each tenant has a persistent sensitivity class, traffic rate, etc.), plus the scenario-specific mechanisms (S4's `flood_override`, S6's `set_tenant_sensitivity_class`). |
| `metrics/regret.py` | `compute_episode_metrics()`/`attribute_regret()` — regret/churn accounting (forced-rekey ratio, deferral counts). |
| `agents/dqn.py` | `DQNAgent`/`QNetwork` — the masked DQN itself. Masking applied at both action-selection and bootstrap-target time. |
| `agents/baselines.py` | The four Hard-Rule-7-mandated tuned non-RL baselines: `AlwaysPQCPolicy`, `AlwaysHybridPolicy`, `StaticThresholdPolicy` (grid-searchable), `RandomPolicy`. |
| `agents/soft_reward_baseline.py` | The reward-shaped comparison agent: reuses `DQNAgent` unmodified, but trains with `security_masking: false` and a reward that includes a `w_sec * security_score(tier)` term — the "what if we didn't mask" control. |
| `attack/trace_generator.py`, `attack/attacking_provider.py` | The steering attack: `generate_adversarial_window()` implements the paper's equation 7 input-shaping; `AttackingForecastProvider` wraps a real forecaster to inject it live into an episode. |
| `experiments/harness.py` | `run_scenario`/`run_grid`/`evaluate_multi_seed`/`evaluate_attack_multi_seed` — the evaluation harness every real number in this repo was measured through. |
| `experiments/train.py` | `train()`/`train_soft_reward_baseline()` — training entry points, both scenario-parameterized. |
| `dashboard/explain.py` | `explain_decision()` — the real, policy-agnostic "why did it pick this action" backend (reads the mask/floor/cost tables directly, zero drift by construction). |
| `dashboard/render_*.py` / `dashboard/render_*_demo.py` | The six real panel renderers (pure, dependency-free HTML/SVG generators) and their drivers (which produce the real data each renderer needs). |
| `dashboard/samples/` | The committed, ready-to-open output of every driver above. |
| `dashboard/mockups/` | The original static, fabricated-data mockups the real renderers were styled to match visually — **never a data source**, only a visual reference. |
| `configs/default.yaml`, `configs/scenarios/*.yaml` | Scenario and hyperparameter configs — one standalone, directly-loadable file per scenario. |
| `docs/smartkeynet_ieee_paper_5.tex` | The real, current paper draft — Table V and the headline numbers are kept in sync with `dashboard/samples/*.json`. |
| `PLAN.md` / `PLAN2.md` | The frozen Hard Rules (non-negotiable design constraints) and full original design spec — read these for the *why* behind any of the above. |
| `PROGRESS.md` / `SESSION_LOG.md` | The project's own running status tracker and session-by-session narrative log — the authoritative "what's real, what's next" source, kept current every session. |

## Scenarios (S1–S6)

| Scenario | Tests | Train or held-out? |
|---|---|---|
| S1 | Steady-state baseline — no scripted stress. | Train + eval. |
| S2 | HNDL (harvest-now-decrypt-later): a scripted threat-posture elevation partway through the episode. | Train + eval. |
| S3 | QKD pool degradation: a recalibrated SKR-collapse spike that genuinely exhausts the pool within one episode (unlike S1). | Train + eval. |
| S4 | DDoS / noisy-neighbor: one tenant gets an additive flood of extra requests. | Eval only (not used in any Gate W3 training run). |
| S5 | Steering attack dose-response: the real threat signal is progressively replaced with an adversarial one (`α` from 0 to 1) via `attack/trace_generator.py`. Run on top of S2. | Eval only (attack sweep, not a trainable scenario in its own right). |
| S6 | Migration wave: a scripted, exogenous schedule of per-tenant sensitivity-class ratchets (a compliance-style migration), applied entirely upstream in request generation. | **Held-out evaluation only** — `train_eligible: false` is enforced at the code level (Hard Rule 8); an agent trained on S1 is evaluated against S6's schedule, never trained on it. |

## What's implemented vs. what's future work (honest)

**Genuinely real, tested, and part of the green suite:**
- The full MDP: pool sim, deferral queue, action masking, reward, all
  wired end-to-end in `SmartKeyNetEnv`.
- All six scenarios (S1–S6), each independently dispatched and tested.
- The masked DQN agent and all four tuned non-RL baselines
  (Hard Rule 7), plus the soft-reward reproduction baseline.
- The steering attack (equation-7 input shaping) and its full 11-alpha
  dose-response sweep on S2.
- Gate W3 (masked DQN beats a properly-tuned static-threshold baseline
  on `total_reward`): met on both S1 (~12.6x) and S3 (~3.78x).
- Six of seven dashboard panels, each rendered from real measured
  data with dedicated "honesty guard" tests (see
  [How to verify it works](#how-to-verify-it-works)).
- `dashboard/explain.py`'s per-decision explainability backend.

**A deliberate, documented stub — not the real design:**
- The threat forecaster (`env/forecast_provider.py::MovingAverageForecaster`)
  is a placeholder EWMA over `[qber, load]`, **not** the dual-head LSTM
  forecaster (Addition A) the architecture describes. Every result in
  this repo runs on top of this placeholder — see the caveats below
  for what that specifically affects.

**Not started:**
- `forecaster/model.py`/`forecaster/dataset.py`/`forecaster/train.py`
  (the real LSTM dual-head forecaster) — all three are stubs behind
  1-test import-smoke checks only.
- `api/main.py` (an AWS-KMS-flavored REST facade) — stub only.
- `dashboard/app.py` (a live, wired 4-beat demo shell tying the six
  panels together into one running app) — stub only; the static panels
  in `dashboard/samples/` are the real, working demo artifact today.
- The seventh dashboard panel (Threat Input / live feature
  visualization) — blocked on the real forecaster above, not a
  rendering gap.
- `docs/report.md` — a section-header skeleton only; the real,
  current write-up is `docs/smartkeynet_ieee_paper_5.tex`.

## Known limitations / caveats

These are understood, investigated findings — reported directly in
the paper and in `SESSION_LOG.md`, not gaps discovered after the fact:

- **The placeholder forecaster saturates early.** Because
  `MovingAverageForecaster`'s threat signal mixes an ordinary system-load
  term into what's meant to represent threat, posture ratchets up to
  ELEVATED within the first 1–3 real decisions of most episodes across
  every scenario checked — often *before* a scenario's own scripted
  threat schedule fires. It has never been observed to reach HIGH under
  normal (non-attack) conditions. This affects how "responsive to a
  scripted threat" any given panel or metric can look; it does not
  affect the masking guarantee itself (the floor computed is always
  the floor enforced, regardless of what triggered it).
- **`p99_latency` is a discrete-cost-model artifact above a low
  threshold**, not a meaningful discriminator between policies. Its
  saturation at exactly `1.5000` once ≥1.6% of an episode's decisions
  cost `SERVE_HYBRID` (root-caused directly against `np.percentile`'s
  interpolation formula) is real and expected — use `total_reward` or
  `below_floor_rate` to compare policies instead.
- **`regret_events` and `pool_exhaustion_events` are identical by
  construction**, not two independently-informative metrics — every
  regret event *is* a pool-exhaustion event in the current
  environment (see `experiments/harness.py::run_scenario`'s own
  docstring).
- **The re-tuned static-threshold baseline ties or beats the masked
  agent on every column Table V displays** (p99, exhaustion, regret,
  below-floor all tie; forced-rekey ratio is in the threshold's favor,
  8.0% vs. 15.6%) — read naively, this looks like a failure of the
  project's premise. It is not the full picture: on `total_reward`,
  the actual scalar being optimized, the masked agent wins by **~3.8x**
  (-10,214.82 vs. -38,566.87). The mechanism: `StaticThresholdPolicy`
  never returns `REUSE`, so it pays a fresh-key rekey cost on nearly
  every decision — its low *forced*-rekey ratio reflects that almost
  none of those very frequent rekeys are the forced kind, not that it
  rekeys rarely. Both the paper's Table V footnote and `SESSION_LOG.md`
  state this explicitly.

## Getting a fuller picture

For the frozen design constraints (Hard Rules 1–11) and full original
architecture spec, read [`PLAN.md`](./PLAN.md) and
[`PLAN2.md`](./PLAN2.md). For exactly what's real as of right now,
per-file, [`PROGRESS.md`](./PROGRESS.md) is the single source of
truth — its "Next task" section is kept deliberately current.
[`SESSION_LOG.md`](./SESSION_LOG.md) has the full session-by-session
narrative, investigation writeups, and numbers behind every claim
above.
