# SmartKeyNet: RL for Hybrid Cryptography

> *"SmartKeyNet is the decision layer for a multi-tenant cloud KMS in
> the hybrid era — classical, post-quantum, and quantum-backed keys
> served per request by a DQN that budgets scarce quantum resources,
> with security floors it structurally cannot violate."*
> — PLAN.md §9

**Status:** scaffolding only. No trained models, no working DQN yet —
see [PLAN.md](./PLAN.md) §10 for the kickoff order this repo follows.

**Team:** 4 people · **Duration:** 2 months / 8 weeks · Read
[PLAN.md](./PLAN.md) for the full concept, Hard Rules, and experiment
grid, then [split.md](./split.md) for who builds what and the weekly
schedule.

**Week 1 start:** `____` (fill in at kickoff) · **Deadline:** start + 8 weeks.

## Architecture

```
                        ┌──────────────────────────────────────────┐
   CV-QKD traces ──────▶│  QKD POOL SIM (SKR/QBER-driven refill)   │
   (real dataset)       └───────────────┬──────────────────────────┘
                                        │ pool level
 NetworkX tenant graph                  ▼
 (~10-50 service nodes,  ┌──────────────────────────────────────────┐
  edges = flows w/       │            ENVIRONMENT (Gym-style)       │
  sensitivity class,     │  request stream ← sampled from graph     │
  traffic rate) ────────▶│  key ages, latency model, energy model   │
                         └──────┬───────────────────────┬───────────┘
 RT_IoT2022-derived             │ state                 │ mask
 network features               ▼                       ▼
   ┌──────────────┐     ┌──────────────┐       ┌──────────────────┐
   │  DUAL-HEAD   │────▶│ POLICY TABLE │──────▶│  ACTION MASKING  │
   │  FORECASTER  │thrt │ (class ×     │ floor │  (structural,    │
   │ threat head +│head │  posture →   │       │   inviolable)    │
   │ pool/demand  │     │  min tier)   │       └────────┬─────────┘
   │ head         │     └──────────────┘                │ safe action set
   └──────┬───────┘                                     │
          │ pool head → DQN state (foresight features)  ▼
                                               ┌─────────────────┐
                                               │   DQN AGENT     │
                                               │ reward: latency,│
                                               │ energy, fresh-  │
                                               │ ness, QKD       │
                                               │ scarcity price  │
                                               │ (NO security    │
                                               │  term, ever)    │
                                               └─────────────────┘
```

See [PLAN.md §4](./PLAN.md#4-how-abstract-architecture--design-details-on-the-way)
for the full state/reward spec, and [PLAN.md §4A](./PLAN.md#4a-required-additions-core-scope--build-these-implementation-level-spec)
for the dual-head forecaster (Addition A) and regret/churn accounting
(Addition C) that this diagram assumes.

## Repo layout

| Path | Owner (split.md §1) | Purpose |
|---|---|---|
| `env/` | B (contracts/environment/pool/masking/metrics-adjacent), A (`request_generator.py`, `forecast_provider.py`) | The MDP: pool sim, deferral queue, masking, Gym env. `env/contracts.py` is the frozen interface everyone builds against. |
| `agents/` | C | Masked DQN, tuned baselines, soft-reward baseline (steering-attack target). |
| `forecaster/` | A | Dual-head LSTM (threat head + pool head), trained offline. |
| `metrics/` | B | Regret & churn accounting (Addition C). |
| `api/` | D | AWS-KMS-flavored REST facade. |
| `dashboard/` | D | Live 4-beat demo. |
| `experiments/` | C | Scenario comparison harness (S1–S6). |
| `attack/` | D | Adversarial threat-trace generator for the steering attack. |
| `data/` | A | Dataset download instructions + samples (see `data/README.md` for licensing). |
| `configs/` | shared (danger zone — ping the team before editing) | Reward weights, `use_foresight` flag, scenario config. |
| `docs/` | D (+ everyone, week 7) | Report skeleton. |
| `tests/` | everyone | Import-smoke tests today; real unit tests as modules land. |

## Hard Rules (do not violate — PLAN.md §4)

1. No security term in the reward. Ever.
2. Floors are enforced by action masking, not reward penalties.
3. One agent, one MDP — the migration wave is a scripted, exogenous schedule.
4. No invented security constants — cite NIST PQC / SP 800-57 / CNSA 2.0 / BSI-ANSSI / ETSI GS QKD 014.
5. No free mid-session algorithm switching.
6. QKD stays architecturally honest — it's a backbone resource behind the KMS.
7. Tuned non-RL baselines are mandatory (always-PQC, always-hybrid, static threshold, random).
8. Train/eval split for migration: S6 is held-out evaluation only.
9. Pool exhaustion never causes a downgrade — it causes a deferral, logged as a regret event.

Full detail and rationale in [PLAN.md](./PLAN.md).

## Getting started

```bash
pip install -r requirements.txt
pytest        # should be green on the current skeleton
```

See `data/README.md` before downloading or loading any dataset —
Q-OPSEC has no LICENSE file and two of its CSVs are verified
degenerate.

## Anti-patterns (banned up front — split.md §4)

- The 3000-line PR. Split it.
- Long-lived branches that diverge from `main` for weeks.
- Editing `env/contracts.py` unilaterally.
- "I'll add a small security term to the reward just to stabilize training." (Rejected on sight — Hard Rule 1.)
- Training on `rl_experiment_*` logs because they're big and convenient.
- Loading `context_dataset_basic.csv` / `context_dataset_advanced.csv`.
- Everyone waiting on Person B — work against stubs instead.
