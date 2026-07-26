# SmartKeyNet — Team Split & GitHub Workflow (4 people)

> Companion to `PLAN.md`. Read PLAN.md first for the concept and Hard Rules. This file is *only* about who builds what, on what weekly schedule, and how you avoid stepping on each other. **Scope: 2 months / 8 weeks, 4 people.** Optimized so all four work in parallel from week 1 without blocking. Stretch B (multi-pool) is OUT at this timeline.

---

## 0. The core principle: split by INTERFACE, not by file

The project has four natural seams. If everyone agrees on the **interfaces** (the function signatures / data shapes crossing each seam) in week 1, all four people can build behind their own interface in parallel and integrate later without merge hell. **The interface contract is sacred; the code behind it is yours.**

The four owners map onto the four PLAN.md layers:

```
  ┌────────────────┐   requests    ┌────────────────┐   state+mask   ┌──────────────┐
  │  A: WORLD      │──────────────▶│  B: ENV +      │───────────────▶│  C: AGENT +  │
  │  (data, graph, │               │  POOL + REWARD │◀───────────────│  BASELINES   │
  │  forecaster)   │◀──────────────│  (the MDP)     │   action       └──────────────┘
  └────────────────┘  forecasts    └───────┬────────┘
                                           │ event logs
                                           ▼
                                   ┌────────────────┐
                                   │  D: EXPERIMENTS│
                                   │  DASHBOARD,    │
                                   │  API, PAPER    │
                                   └────────────────┘
```

**The two interface contracts to freeze in the week-1 kickoff meeting:**
1. **`state_dict` schema** — the exact keys/shapes B hands to C each step, and the mask format. (Write it as a Python `TypedDict` + a dummy generator so C can start before B is done.)
2. **`ForecastProvider` interface** — the methods A exposes (threat head + pool head) that B calls. (PLAN.md Addition A already sketches this; B builds against a `MovingAverageForecaster` stub until A's LSTM lands.)

Freeze those two, commit them as `env/contracts.py` on day 1, and nobody is blocked.

---

## 1. Who owns what

### Person A — "The World" (data + forecaster + graph)
**Owns:** `data/`, `env/request_generator.py`, `env/forecast_provider.py`, `forecaster/` (all).
**Builds:**
- RT-IoT2022 ingestion + feature pipeline → the LSTM threat head.
- Pool-head forecasting (SKR/demand trajectory) — Addition A.
- Sensitivity classifier from `confidentiality_train` → maps requests to classes.
- NetworkX tenant graph generator (`sensitivity_class`, `traffic_rate`, `pqc_capable`) + the request stream it emits.
- The QKD SKR/QBER trace loader (or documented synthetic generator).
**Ships a stub first:** `MovingAverageForecaster` + a random request generator so B and C aren't blocked. Real LSTM comes in month 2.
**Depends on:** the `ForecastProvider` and request-schema contracts (co-owns them with B).

### Person B — "The MDP" (environment + pool + reward + masking)
**Owns:** `env/environment.py`, `env/pool_sim.py`, `env/deferral_queue.py`, `env/masking.py`, `metrics/regret.py`, `configs/`.
**Builds:**
- The Gymnasium-style env: `reset()`, `step(action)`, `state`, `action_mask`.
- **Pool simulator** (trace-driven refill, drain, exhaustion) — the scarcity engine.
- **Deferral queue + regret accounting** (Addition C, Hard Rule 9).
- The **policy table** (class × threat posture → floor) and **action masking**.
- The full reward function (latency, energy, freshness, QKD scarcity, `R_starve`, load-scaled rekey).
**This is the spine — start day 1.** Everyone else's work is meaningless if the env is wrong. B is the natural **tech lead / integrator**.
**Depends on:** the state schema + ForecastProvider contracts.

### Person C — "The Brain" (agent + baselines + experiments harness)
**Owns:** `agents/` (DQN + all baselines), `experiments/harness.py`.
**Builds:**
- Vanilla DQN → Double/Dueling if needed, consuming B's state + mask.
- **All baselines** (always-PQC, always-hybrid, static-threshold grid-searched, random) — Hard Rule 7, build these *before* tuning the agent.
- The **soft-reward baseline agent** reproducing Noetzold (for the steering attack) — the thesis centerpiece.
- Training loop, checkpointing, seed management, the comparison harness that runs any policy across S1–S6.
**Ships a stub first:** a random-action agent so B can test `step()` end-to-end on day 2.
**Depends on:** the state schema contract (can start against the dummy generator immediately).

### Person D — "The Story" (attack, dashboard, API, paper)
**Owns:** `attack/` (steering-trace generator), `dashboard/`, `api/` (FastAPI KMS facade), `docs/` (report/paper), `metrics/` plotting.
**Builds:**
- The **adversarial threat-trace generator** for the steering attack + dose-response sweep.
- The live **dashboard** (graph view, pool gauge, regret counter, forecast strip) — the 4-beat demo.
- The AWS-KMS-style **REST facade** (cloud framing).
- The **report/paper** (starts week 1 — methodology + related work don't need code), figures, viva deck.
**Ships first:** a static dashboard mockup from fake logs so the demo shape is agreed early.
**Depends on:** the event-log schema from B's `metrics/regret.py` (freeze it early too).

---

## 2. Weekly plan — 8 WEEKS / 2 MONTHS (hard deadline)

> 2 months is tight. **Stretch B is deleted, not deferred.** The bar is: a working masked-DQN pool-budgeting agent + the steering-attack result + a live demo + a report. Everything else is negotiable. Each week ends with a Friday merge-and-demo off `main`. Gates are hard — if a gate slips, cut immediately per §2.1, don't push the gate.

**Set your dates now:** fill in `Week 1 start = ____`, deadline = start + 8 weeks. Put these in the repo README.

### Week 1 — Setup & contracts (everyone, together)
- **All:** kickoff meeting → create repo, add collaborators, folder skeleton, branch protection, PR template with Hard-Rules checklist, `.gitignore` + data-download script.
- **All (the critical hour):** write `env/contracts.py` together — state schema, mask format, `ForecastProvider` interface, event-log schema.
- **A:** download RT-IoT2022 + Q-OPSEC `confidentiality_train`; write `data/README.md` + `get_data.py`; commit 100-row samples.
- **B:** skeleton `pool_sim.py` (refill/drain math only).
- **C:** random-action agent stub against the dummy state generator.
- **D:** report skeleton in `docs/` (title, intro, related-work bullets incl. Noetzold critique) + dashboard mockup from fake logs.
- **Gate W1:** `pytest` runs green on empty stubs; contracts frozen and committed. Anyone can now work unblocked.

### Week 2 — The spine (B critical path)
- **B:** full `pool_sim` + `deferral_queue` + `regret.py` with unit tests (Addition C, Hard Rule 9); minimal `environment.py` (`reset`/`step`/`state`/`mask`) + policy table + masking.
- **C:** DQN wired to B's `step()` the moment it exists; overfit S1 on purpose to prove the loop.
- **A:** `MovingAverageForecaster` (real, not stub) + NetworkX graph generator + request stream with `pqc_capable`.
- **D:** FastAPI KMS-facade skeleton; start live dashboard scaffold (graph + pool gauge).
- **Gate W2:** env `step()` runs end-to-end with random agent across a full S1 episode; regret events logged.

### Week 3 — Learning works + baselines (B+C own the gate)
- **C:** all four baselines (always-PQC, always-hybrid, static-threshold **grid-searched**, random) + comparison harness; tune DQN on S1.
- **B:** scenarios S2 (HNDL) and S3 (QKD degradation) in the env.
- **A:** RT-IoT2022 feature pipeline → start training the LSTM threat head offline.
- **D:** dashboard shows live decisions + regret counter on S1.
- **🚩 Gate W3 (make-or-break):** **DQN beats the tuned threshold baseline on S1 and S3.** If it doesn't, STOP feature work — fix the environment/reward first. This is the whole premise.

### Week 4 — Thesis build starts + MIDPOINT (C+D own the gate)
- **C:** build the **soft-reward baseline agent** (reproduces Noetzold reward structure).
- **D:** build the **adversarial threat-trace generator** + dose-response sweep harness.
- **B:** scenario S4 (DDoS / noisy neighbor).
- **A:** finish LSTM threat head; begin pool head (Addition A).
- **🚩 MIDPOINT GATE (end W4):** spine solid, DQN wins on S1–S4, both attack pieces exist. **If behind here, invoke cut order now** — you have 4 weeks left.

### Week 5 — The steering attack (thesis centerpiece — NEVER CUT)
- **C+D:** run the steering attack: soft-reward agent bends under the adversarial trace, masked agent's floor only ratchets up. Produce the split-screen result + dose-response curves.
- **A:** integrate LSTM dual-head via `use_foresight: lstm`; run **E-A ablation** (off/ewma/lstm) on S3.
- **B:** freeze the env; only bug-fixes after this.
- **Gate W5:** steering-attack figure exists and is convincing. This is your paper's headline — bank it.

### Week 6 — Migration story + full results
- **B:** S6 migration-wave scenario (scripted schedule, held-out eval).
- **C:** final training runs across all scenarios, all seeds; fill the master results table.
- **A:** finalize E-A ablation numbers; case-study episode on a real trace segment (optional, cheap).
- **D:** dashboard 4-beat demo complete (living system → budgeting → attack → migration).
- **Gate W6:** every number for the report exists. **Code freeze for features.**

### Week 7 — Write & polish (all hands on report + demo)
- **All:** each person writes their section (A: data/forecaster; B: environment/method; C: agent/baselines/results; D: attack/related-work/intro/figures).
- **D:** all figures final, viva deck built, demo rehearsed on `main` at least twice.
- **Gate W7:** full report draft done; demo runs clean start-to-finish.

### Week 8 — Buffer & finalize
- Buffer for slippage (there will be slippage — that's why W8 exists; do NOT plan new features here).
- Report proofread + references; viva Q&A rehearsal using PLAN.md §8; tag a `v1.0` release off `main`.
- **Final deadline:** everything committed, report submitted, demo frozen.

### 2.1 Cut order (invoke the instant a gate slips)
1. **S6 migration wave** (W6) → lose one experiment, keep thesis.
2. **LSTM head / E-A ablation** (W5) → keep EWMA foresight; ablation shrinks but survives.
3. **S4 or S2** → drop one traffic scenario.
4. **Dose-response sweep** → keep the single-point steering result.

**NEVER cut:** pool sim, action masking, deferral/regret, the four baselines, the steering attack. If these four weeks can only produce *one* headline, it is the steering attack on a working pool-budgeting agent — protect that above all.

---


## 3. GitHub workflow (keep it simple — you're 4 people, not 40)

### Branching model: trunk-based with short-lived feature branches
- `main` is always green (the demo must run off `main` at any time).
- Nobody commits to `main` directly. Everything goes through a Pull Request.
- Branch naming: `area/short-desc` → `env/pool-sim`, `agent/dqn`, `data/lstm-forecaster`, `dash/regret-panel`.
- Keep branches **short-lived** (merge within ~2-3 days). Long branches = merge pain. Split big work into small PRs.

### The PR rule
- Every PR needs **1 review** from someone else before merge. (You're a team — no self-merging into `main`.)
- Reviews check ONE thing above all: **does this respect the Hard Rules in PLAN.md?** (No security in reward, masking not penalties, one MDP, don't-train-on-RL-logs, degenerate files not loaded.) Put this as a checkbox in the PR template.
- Small PRs get reviewed fast. A 2000-line PR won't get a real review — split it.

### Ownership = fewer conflicts
Because the split is by directory (A=`data/forecaster/`, B=`env/metrics/`, C=`agents/`, D=`attack/dashboard/api/docs/`), most PRs touch only one owner's files → almost no merge conflicts. The shared files (`env/contracts.py`, `configs/`) are the danger zone — change those only via a PR that pings everyone.

### First-day repo setup (do this together, live, in the kickoff meeting)
1. One person creates the repo (private), adds the other 3 as collaborators.
2. Add `PLAN.md`, `split.md`, a `README.md`.
3. Create the folder skeleton (empty `__init__.py` in each) from PLAN.md §10.
4. **Write `env/contracts.py` together** — the state schema, mask format, ForecastProvider interface, event-log schema. This is the single most important hour of the project.
5. Add a `.gitignore` (Python, `data/*.csv` if datasets are large/licensed — see below, `*.pt` checkpoints, `__pycache__`).
6. Add a PR template with the Hard-Rules checklist.
7. Protect `main`: require 1 approval, no direct pushes (Settings → Branches → branch protection).

### Data handling on GitHub (important given licensing)
- **Do NOT commit the Q-OPSEC CSVs to a public repo** (no license — see PLAN.md). Options: keep repo private, or `.gitignore` the data and share it via a shared drive, or use Git LFS in a private repo.
- Big datasets (RT-IoT2022) shouldn't live in git at all — add a `data/README.md` with download instructions + a `scripts/get_data.py`, and `.gitignore` the actual files. Everyone downloads locally.
- Commit a tiny **sample** (e.g., 100 rows) for tests so CI/other people can run without the full download.

### Keep it green: minimal CI
- One GitHub Action: on every PR, run `pytest` (the unit tests PLAN.md asks for — pool sim, deferral queue, regret counting, mask correctness, forecaster shapes). If tests fail, PR can't merge. This alone prevents 80% of "it worked on my machine" pain.

### Weekly rhythm
- **Monday:** 30-min sync. Each person: what I shipped, what I'm blocked on, what interface I need frozen. Update the cut-line status.
- **Continuous:** small PRs, fast reviews.
- **Friday:** merge-and-demo — whatever's on `main` gets run end-to-end so regressions surface weekly, not in month 3.

---

## 4. Anti-patterns to ban up front (write these in the repo README)

- **The 3000-line PR.** Nobody reviews it; Hard Rules slip through. Split it.
- **The long-lived branch** that diverges from `main` for two weeks. Merge small, merge often.
- **Editing `env/contracts.py` unilaterally.** That's everyone's interface — ping the team.
- **"I'll add a small security term to the reward just to stabilize training."** This is the single most likely way the thesis dies, and it *will* be tempting when vibe-coding. Reviewers must reject it on sight (Hard Rule 1).
- **Training on `rl_experiment_*` logs** because they're big and convenient. They're the baseline's outputs, not training data.
- **Loading `context_dataset_basic/advanced`** — degenerate, do-not-use.
- **Everyone waiting on B.** If B slips, A/C/D work against stubs. Stubs first, always.

---

## 5. If you're stuck on how to start a coding session
Point Claude Code at `PLAN.md` + `split.md` + your area, and say e.g.:
> "I'm Person B. Per PLAN.md §10 and split.md, build `env/pool_sim.py` (trace-driven refill/drain/exhaustion) and `env/deferral_queue.py` with the unit tests specified in Addition C. Respect Hard Rule 9. Don't touch `agents/` or `data/`."

Scoped, interface-respecting prompts = clean parallel work.
