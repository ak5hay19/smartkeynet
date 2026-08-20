# SmartKeyNet: RL for Hybrid Cryptography

> *"SmartKeyNet is the decision layer for a multi-tenant cloud KMS in
> the hybrid era — classical, post-quantum, and quantum-backed keys
> served per request by a DQN that budgets scarce quantum resources,
> with security floors it structurally cannot violate."*
> — PLAN2.md §13

**Status:** feature-complete. All seven dashboard panels, S1–S6 scenario
dispatch, a trained dual-head forecaster, the steering attack, and the
API/dashboard surface are implemented and tested (`pytest`: 629 passed).

**Headline result holds; the secondary one does not.** Read
[`docs/report.md`](./docs/report.md) §5 before anything else:

| | |
|---|---|
| **S5 steering attack** | **Holds.** Soft-reward arm 14.0% → 27.8% of key establishments below the sensitivity-class floor as the attack strengthens; the masked architecture is **0.0% at every dose**, structurally. |
| **Gate W3 (DQN vs tuned threshold)** | **Failed.** The tuned threshold beats the masked DQN on both S1 and S3, checkpoint-averaged across 5 training seeds. Reported as measured — see `docs/report.md` §5.1 and `PROGRESS.md`'s "Next task". |

**Team:** originally scoped for 4 people, currently executed solo across
all roles · **Duration:** 2 months / 8 weeks · Read
[PLAN2.md](./PLAN2.md) for the full concept, Hard Rules, and experiment
grid, and [PROGRESS.md](./PROGRESS.md) + [SESSION_LOG.md](./SESSION_LOG.md)
for current state and how it got there.

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

See [PLAN2.md §5](./PLAN2.md) for the full state/reward spec and the
eleven Hard Rules, and §4A/§7 for the dual-head forecaster (Addition A),
regret/churn accounting (Addition C) and the seven-panel dashboard.

## Repo layout

| Path | Owner (split.md §1) | Purpose |
|---|---|---|
| `env/` | B (contracts/environment/pool/masking/metrics-adjacent), A (`request_generator.py`, `forecast_provider.py`) | The MDP: pool sim, deferral queue, masking, Gym env, tenant graph, scenario dispatch, and `decision_trace.py` (the Hard Rule 10 six-step trace — one source of truth for both `api/` and `dashboard/`). `env/contracts.py` is the frozen interface everyone builds against and is **untouched**. |
| `agents/` | C | Masked DQN, tuned baselines, soft-reward baseline (steering-attack target). |
| `forecaster/` | A | Dual-head LSTM (threat head + pool head), trained offline. |
| `metrics/` | B | Regret & churn accounting (Addition C). |
| `api/` | D | AWS-KMS-flavored REST facade. |
| `dashboard/` | D | All seven PLAN2 §7 panels, from real runs. Dash app + self-contained static export. |
| `experiments/` | C | Scenario comparison harness (S1–S6), plus `campaign.py` (multi-seed checkpoint-averaged comparisons — the only supported way to produce a DQN number), `ablation.py` (E-A) and `results_table.py` (PLAN2 §7.7). |
| `attack/` | D | Adversarial threat-trace generator and `run_attack.py`, the S5 dose-response experiment. |
| `data/` | A | Dataset instructions (see `data/README.md` for licensing). RT-IoT2022 is operator-placed and gitignored. |
| `configs/` | shared (danger zone — ping the team before editing) | Reward weights, `use_foresight`, `threat_input`, S1–S6 scenario dispatch, the S6 migration schedule, and a `pool:` block that carries the measured demand bracket its sizing was chosen inside. |
| `docs/` | D (+ everyone, week 7) | `report.md` — written, with real numbers. |
| `tests/` | everyone | 629 behavioural tests. Every Hard Rule that can be checked mechanically has one. |

## Cryptographic honesty

`GET /Health` on the API facade publishes exactly which primitives are
real, and it is worth stating here too:

| tier | status |
|---|---|
| classical (T0) | **real** — X25519 ECDH + HKDF-SHA256 + AES-256-GCM via `cryptography` |
| PQC (T1, ML-KEM-768) | **simulated** — `liboqs` is an optional dependency and is not installed. Not quantum-resistant, and every API response says so. |
| hybrid (T2/T3) | **partial** — the HKDF combiner is real; `pool_sim` models key *availability* faithfully but holds a bit count, not bytes, so the QKD material is locally generated |

## Hard Rules (do not violate — PLAN2 §5.4)

1. No security term in the reward. Ever.
2. Floors are enforced by action masking, not reward penalties.
3. One agent, one MDP — the migration wave is a scripted, exogenous schedule.
4. No invented security constants — cite NIST PQC / SP 800-57 / CNSA 2.0 / BSI-ANSSI / ETSI GS QKD 014.
5. No free mid-session algorithm switching.
6. QKD stays architecturally honest — it's a backbone resource behind the KMS.
7. Tuned non-RL baselines are mandatory (always-PQC, always-hybrid, static threshold, random).
8. Train/eval split for migration: S6 is held-out evaluation only.
9. Pool exhaustion never causes a downgrade — it causes a deferral, logged as a regret event.

Full detail and rationale in [PLAN2.md](./PLAN2.md) §5.4–§5.5 (which adds Hard Rules 10 and 11: the Explain Decision panel may only display computed values, and live capture is replay-only).

Every one of these that can be checked mechanically has a test. Two Hard Rule 2 violations were found by measurement this session and closed — see [SESSION_LOG.md](./SESSION_LOG.md) §2.

## Getting started

```bash
pip install -r requirements.txt
pytest -m "not slow"      # 629 passed, ~13s
pytest                    # adds the end-to-end steering-attack runs
```

Tests that need RT-IoT2022 skip cleanly when it is absent (it is
gitignored — place `RT_IOT2022.csv` under `data/raw/` or
`data/raw/rt_iot2022/`).

### Reproducing the results

```bash
python -m forecaster.train          # dual-head forecaster (~40s)
python -m attack.run_attack         # S5 steering attack -> results/steering_dose_response.json
python -m experiments.campaign      # Gate W3: DQN vs tuned threshold, S1 and S3
python -m experiments.ablation      # E-A foresight ablation -> results/foresight_ablation.json
python -m experiments.results_table # closing table -> results/closing_table.json
python -m dashboard.app             # renders dashboard/index.html from those artefacts
uvicorn api.main:app                # the KMS facade; GET /Health for the primitive matrix
```

Every DQN number is produced through `experiments/campaign.py`, which is
checkpoint-averaged, eval-seed-averaged and multi-seed by construction —
see its module docstring for why that is not optional here.

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
- Reporting a DQN number from a single checkpoint or a single training seed (see `experiments/campaign.py`).
- Reporting an accuracy without its base rate (see `forecaster/train.py`).
- Putting a plausible-looking number in a dashboard panel that no experiment produced (see `dashboard/data.py`).
