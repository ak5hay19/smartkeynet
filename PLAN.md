# SmartKeyNet: RL for Hybrid Cryptography
### Final Project Plan — Bootstrap Document

> **How to use this file:** Paste or load this into any Claude Code session to bootstrap work on the project. It contains the full concept, the non-negotiable design rules, the abstract architecture, scenarios, demo plan, timeline, and cut lines. Detailed module design is intentionally deferred — design it on the way, but never violate the Hard Rules section.

---

## 0. One-Paragraph Summary

SmartKeyNet is the decision layer for a **multi-tenant cloud Key Management Service (KMS) operating in the hybrid-cryptography era**. Tenants request cryptographic keys through an AWS-KMS-style API (backed conceptually by ETSI GS QKD 014 key delivery). For every request, a **DQN agent** decides how to serve it — classical, post-quantum (ML-KEM-768), or hybrid (ML-KEM ⊕ QKD-sourced key material) — and when to reuse or rekey. The hard part: quantum key material comes from a **finite pool that refills slowly** (kbps, driven by real CV-QKD traces), so the agent must learn to *budget* it across competing tenants. Security is **never** in the reward: an LSTM threat forecaster + policy table set a per-request **minimum tier floor**, and all actions below the floor are **masked out** before the DQN sees them. We demonstrate that prior soft-reward RL-crypto designs can be *steered* by an adversarially shaped threat signal, while our masked agent's protections can only ratchet upward. That steering-attack-plus-structural-defense is the headline research contribution. Two core mechanisms elevate the agent beyond a per-request recommender: a **dual-head forecaster** (threat posture + pool/demand trajectory) that makes the agent *anticipatory* rather than reactive, and **regret/churn accounting** that quantifies the cost of spending the pool wrong in either direction.

**Team:** 4 people · **Duration:** 3 months · **Weight:** 50% of grade · **Build style:** ~90% AI-assisted codegen with human integration/review
**Target venues (stretch):** IEEE QCE / QCNC; report + viva demo are the primary deliverables.

---

## 1. What the Project Is (Plain English + Analogies)

### The elevator pitch
Networks are entering a decades-long transition where three kinds of cryptographic keys coexist: classical (fast, quantum-vulnerable), post-quantum (quantum-resistant, effectively unlimited), and quantum-distributed (QKD — information-theoretically strong, but **scarce**). Somebody has to decide, request by request, which key material each connection gets. Today that's static config files. SmartKeyNet replaces the static rules with a learning agent that budgets the scarce resource intelligently — while being *structurally incapable* of under-protecting anything.

### Analogy 1 — The hospital blood bank
QKD key material is like **O-negative blood**: universally powerful, arrives in a slow trickle from donors (the QKD link, filling the pool at kbps), and if you transfuse it into every patient with a scraped knee, you have none left when the trauma case arrives. PQC keys are like **synthetic plasma** — manufactured on demand, unlimited, good enough for most patients. Classical keys are a **band-aid** — fine for minor cases only. SmartKeyNet is the triage nurse who learned, from experience, which patients genuinely need O-neg — and hospital policy (the tier floor) makes it *impossible* for the nurse to give a trauma patient a band-aid, no matter what.

### Analogy 2 — The steering attack, or "don't pay the guard by mood"
Imagine a security guard whose bonus depends partly on "how relaxed the office feels." A clever burglar plays soothing music, the guard's metrics say everything is calm, and the guard props the door open. That's what happens when security is a **soft term in an RL reward** — an adversary who can shape the input signals can talk the agent into weaker choices. SmartKeyNet's guard has no such bonus: a locked policy table decides the *minimum* door security, the guard only optimizes things like electricity and patrol effort, and any signal manipulation can only make the doors lock *tighter*. We prove this by attacking both guards on camera.

### Analogy 3 — Migration as a rolling office move
An enterprise going quantum-safe doesn't flip a switch; departments move floor by floor over years (compliance deadlines ratchet policy floors up, subsystem by subsystem). SmartKeyNet doesn't plan the move (that's a different problem — we say so and cite it); it's the **utilities service that keeps every floor powered correctly throughout the move**, even as demand shifts in waves it was never trained on.

### What the agent literally does per request
For each incoming key request `(tenant, service, sensitivity class)` the agent picks one of:
- `SERVE_CLASSICAL` — X25519 / AES-256-GCM class. Lowest tier (T0).
- `SERVE_PQC` — ML-KEM-768 (NIST Category 3). The free workhorse (T1).
- `SERVE_HYBRID` — ML-KEM-768 ⊕ QKD pool material via HKDF. Premium tier (T2/T3). **Consumes pool.**
- `REUSE` — keep the existing session key if age limits allow.
- `REKEY_NOW` — force refresh (e.g., ahead of a forecast threat spike).

Actions below the request's policy floor, or infeasible ones (pool empty, key age exceeded), are **masked** — removed from the action set before the DQN evaluates it.

---

## 2. Why (Motivation, Three Layers)

1. **World problem — Harvest Now, Decrypt Later (HNDL).** Adversaries record classically-encrypted traffic today to decrypt once quantum computers mature. NIST PQC standards, NSA CNSA 2.0 deadlines, and BSI/ANSSI guidance have kicked off a global, decades-long migration. During it, classical / PQC / QKD-backed keys **coexist** — the hybrid era is the deployment reality, not a thought experiment.

2. **Operational problem — somebody must budget scarce quantum keys.** QKD pools refill at kbps under variable secret-key rate (SKR) and rising QBER; requests arrive in bursts across tenants with wildly different data lifetimes (patient records: decades; telemetry: hours). Static rules fail in both directions: *always-hybrid* drains the pool before critical requests arrive; *fixed thresholds* waste quantum material or starve it at the wrong times. Pool dynamics + bursty non-stationary demand = a genuine sequential decision problem where RL earns its place. **(Disqualification rule: if a simple threshold policy matches the DQN in evaluation, the project premise fails — we must include tuned threshold baselines and beat them.)**

3. **Research problem — soft security rewards are an attack surface.** Published RL-for-crypto work (e.g., Q-learning adaptive encryption for WSNs; threat-score-driven PQC selection; the Noetzold 2026-style reward structure) places security in the reward. We reproduce that design as a baseline and show an adversarially shaped threat trace **steers** it into serving weaker keys. Our masked architecture is immune by construction: the threat signal can only raise floors. Contribution in one sentence: *"Security as constraint (action masking), not reward — demonstrated against a live steering attack on a reproduced soft-reward baseline."*

---

## 3. Where (Real-World Use Cases)

- **Cloud KMS with quantum backhaul (primary framing).** A cloud region whose KMS backhaul includes a metro QKD link — the "what AWS KMS looks like ~2030" story. Grounding: JPMorgan × Toshiba/Ciena QKD-secured production link; BT/Toshiba London metro network; SK Telecom; EuroQCI build-out. Tenants = hospital, fintech, logging service, etc., each with per-tenant key policies (exactly how cloud KMS key policies work today).
- **Telecom backbone trusted nodes.** The canonical ETSI GS QKD 014 deployment: QKD as backbone infrastructure delivering keys via a KMS REST API. SmartKeyNet is the policy brain such nodes currently lack (today: static rules).
- **Hospital / financial data centers on a metro QKD ring.** Mixed criticality on shared infrastructure; per-request triage is exactly the need.
- **Enterprises mid-migration (evaluation scenario, not a second system).** Staged CNSA-2.0-style timelines ratchet tenant policy floors upward in waves; SmartKeyNet operates *underneath* whatever migration sequence the org chooses. Viva scoping line: *"Migration sequencing is an orthogonal planning problem (cited); we manage the key-service layer beneath it and show our policy stays stable across an arbitrary staged schedule."*

---

## 4. How (Abstract Architecture — design details on the way)

```
                        ┌──────────────────────────────────────────┐
   CV-QKD traces ──────▶│  QKD POOL SIM (SKR/QBER-driven refill)   │
   (real dataset)       └───────────────┬──────────────────────────┘
                                        │ pool level
 NetworkX tenant graph                  ▼
 (~50 service nodes,     ┌──────────────────────────────────────────┐
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

**State (per step):** threat score + threat forecast (k steps ahead), QBER, SKR, pool fill level, arrival rate, load, avg latency, current key age, current key type (one-hot), request sensitivity class / tenant policy floor, **plus foresight features (Addition A):** projected pool level at t+H, forecast SKR trend, forecast hybrid-mandatory demand over horizon H, and a recent-regret-event flag (Addition C).

**Reward (within the safe set only):**
`r = − w_lat·latency − w_en·energy + w_fr·freshness − w_qkd·(pool bits consumed) − R_starve·(deferred_critical_steps) − c_rekey(load)·1[rekey]`
where `c_rekey(load) = c0·(1 + β·load)` so rekeys that collide with load spikes cost more, and `R_starve` is large (≈10× the base latency weight). *(Scarcity price teaches budgeting; starvation penalty teaches foresight; load-scaled rekey cost teaches timing. There is still no security term — starvation is a latency/availability failure, never a downgrade.)*

**API surface:** AWS-KMS-flavored REST (`GenerateDataKey`, `Encrypt`, key policies, tenants), conceptually backed by ETSI GS QKD 014 key delivery. Real primitive calls (liboqs / cryptography lib) used where cheap, for authenticity — the crypto payloads are real even though the network is simulated.

### Hard Rules (violating any of these kills the thesis — enforce in code review)
1. **No security term in the reward. Ever.** Not even a small one, not even temporarily "to help training." This is the entire point of the paper.
2. **Floors are enforced by action masking in the environment**, not by penalties. Threat signals may only *raise* floors.
3. **One agent, one MDP.** The migration wave is a **scripted, exogenous schedule** (a config file of timed floor changes). The agent never chooses migration order, never sees the graph, never allocates crews. **Test:** deleting the NetworkX graph and replacing it with a plain arrival process must not change one line of agent code.
4. **No invented security constants.** Tiers map to citable artifacts only: NIST PQC categories, SP 800-57 key lifetimes, CNSA 2.0, BSI/ANSSI guidance, ETSI GS QKD 014.
5. **No free mid-session algorithm switching.** Key-type changes happen only at rekey boundaries; rekeying has explicit latency/energy cost.
6. **QKD stays architecturally honest.** Backbone resource behind the KMS, pool semantics, ETSI-style delivery. Endpoints never "do QKD."
7. **Tuned non-RL baselines are mandatory:** always-PQC, always-hybrid, static threshold (grid-searched), random. If a tuned threshold ties the DQN, report it honestly and investigate environment design first.
8. **Train/eval split for migration:** train on stationary scenarios; the migration-wave scenario is **held-out evaluation only**. Training on the schedule = memorizing the timeline = experiment proves nothing.
9. **Pool exhaustion never causes a downgrade.** If a hybrid-mandatory request arrives and the pool can't cover it, the request is **deferred** (queued, accruing latency) until the pool refills — it is *never* served below its floor. Each deferral onset is logged as a **regret event**. Exhaustion is an availability failure the agent pays for in reward; it is never a security failure.

### Datasets & Provenance (READ BEFORE LOADING ANY DATA)

The project needs *external* data in exactly **three** slots. Everything else is simulator physics we write ourselves — do **not** go hunting for datasets for the pool drain, key sizes, tenant graph, or migration schedule; those aren't dataset problems.

**Golden rules:**
- **One real-network slot only** (the threat forecaster). Don't let dataset-collection eat the months that belong to the pool and the masking — those are what the project stands on. RT-IoT2022 alone is sufficient to ship; a second intrusion set is a *paper* nice-to-have, not a requirement.
- **NEVER train the agent on the `rl_experiment_*` logs.** They are *outputs* of the Noetzold agent (the soft-reward baseline we critique). Training on them = imitation-learning the exact flawed policy we're attacking. They are for **baseline reproduction + feedback calibration only.**
- **QKD scarcity is NOT in any borrowed dataset.** It comes from a QKD SKR/QBER trace (or documented synthetic) driving pool refill + simulator drain arithmetic. That absence in Q-OPSEC is precisely the gap we fill.

**Slot-to-source map:**

| Component | Primary source | Optional companion | Notes |
|---|---|---|---|
| **Threat forecaster (LSTM)** | **RT-IoT2022** (real IoT intrusion flows, labeled) | one of CICIDS2017 / UNSW-NB15 / TON_IoT | Real telemetry replaces Noetzold's synthetic macro-signals — a citable upgrade. Two datasets max (primary + one generalization check). |
| **Sensitivity classifier** (content → confidentiality class) | **`confidentiality_train/valid`** (Q-OPSEC; 320/80 rows, balanced 4-class) | synthetic labeled text (stretch) | Reuse directly. Makes us *comparable* to Noetzold on the confidentiality axis (a feature). Tiny + authored → starter classifier, not robust. |
| **QKD pool refill (SKR/QBER)** | **CV-QKD experimental trace** (published QKD testbed) | documented synthetic SKR process (mean kbps, QBER-driven dips for S3) | Source separately. Synthetic fallback is fully acceptable for a capstone if the generation procedure is stated + rate ranges cited. ETSI GS QKD 014 gives key sizes (256-bit). |
| **Policy-table calibration** | **`synthetic_context_dataset`** (Q-OPSEC; 939 rows, 6 balanced classes) | — | Use *only* to sanity-check (risk, conf) → tier mapping. Fully synthetic. |
| **Baseline reproduction + feedback calibration** | **`rl_experiment_*` / `synthetic_rl_*`** logs (Q-OPSEC) | — | Reproduce Noetzold agent for the steering attack; calibrate latency/resource/success distributions. **Never train our agent on these.** |
| Pool drain, key sizes | ETSI GS QKD 014 (spec) | — | Simulator arithmetic, no dataset. |
| Primitive latency/energy costs | published **liboqs / pqm4** benchmarks | — | Measured, not invented. |
| Tenant graph | **NetworkX synthetic** (documented generator) | — | Edge attrs: `sensitivity_class`, `traffic_rate`, `pqc_capable` (legacy endpoints where classical is the only interoperable option → masking makes classical mandatory). S6 flips `pqc_capable`→true as subsystems upgrade. |
| Migration schedule (S6) | **config file** (scripted, exogenous) | — | Not a dataset. Never agent-controlled (Hard Rule 3). |

**DO-NOT-USE (verified degenerate):** `context_dataset_basic.csv` and `context_dataset_advanced.csv` — 422 rows but only 4 unique feature rows; `security_level_label` is 100% "critical" and `encryption_script_label` is 100% one value. No label variety to learn from. A Claude Code session must not load these as training data.

**Licensing / attribution:** the Q-OPSEC repo has **no LICENSE file** → all rights reserved by default. For this capstone: (1) cite Q-OPSEC / Noetzold explicitly in the report, (2) do **not** redistribute their CSVs inside our public repo without permission, (3) send the author a one-line email requesting license clarification (open issues tab) — the reply is citable. Reusing their *inputs* while replacing their *environment* is the cleanest framing for the steering-attack contribution: "same context inputs as prior work, our pool + masking architecture, here's the delta."

### Tech stack (suggested, negotiable)
Python · Gymnasium-style env · PyTorch (DQN: start vanilla, upgrade to Double/Dueling if needed) · LSTM in PyTorch · NetworkX · FastAPI for the KMS API facade · liboqs-python / `cryptography` for real primitive calls · Plotly Dash or a small React dashboard for the demo · Docker compose ("regions"/"tenants" containers for cloud flavor — **no real AWS deployment**, zero research value, high time risk).

---

## 4A. REQUIRED ADDITIONS (core scope — build these; implementation-level spec)

These are not optional polish. They are what elevates the agent from "recommender with a pool check" to an anticipatory resource controller. Claude Code sessions should treat the specs below as buildable requirements.

### Addition A — Pool Foresight (dual-head forecaster)

**Goal:** the agent acts on *predicted* pool trajectory and demand, not just current pool level. This is the structural answer to "isn't a threshold rule enough?" — no static threshold can act on a forecast.

**Modules & files:**
- `forecaster/model.py` — `SmartKeyForecaster(nn.Module)`: shared LSTM encoder (input window W=64 timesteps) with two heads:
  - **Threat head** (existing role): class distribution over threat posture for next k=5 steps → feeds the **policy table** (floors).
  - **Pool head** (new): regression outputs → `pool_level_hat[t+H] for H ∈ {10, 25, 50}`, `skr_mean_hat[H]`, `hybrid_demand_hat[H]` (count of hybrid-mandatory arrivals expected in horizon) → feeds the **DQN state only** (never the policy table — forecasts must not lower floors, Hard Rule 2).
- `forecaster/dataset.py` — builds sliding-window supervised datasets from logged environment rollouts (inputs: QBER, SKR, per-class arrival counts, hybrid serves, pool level, threat features; targets computed from the same logs shifted by H).
- `forecaster/train.py` — offline supervised training: MSE for pool head, cross-entropy for threat head. Trained on rollouts generated by **baseline policies** across S1–S4 seeds. **Frozen during DQN training** (no end-to-end gradients — simpler, stabler, and keeps the ablation clean).
- `env/forecast_provider.py` — `ForecastProvider` interface with two implementations: `LSTMForecastProvider` and `MovingAverageForecaster` (fallback: EWMA of SKR and arrivals). The env must run with either, selected by config flag `use_foresight: {off, ewma, lstm}`. This means the env is buildable in month 1 before the LSTM exists.

**Integration:** foresight outputs are appended to the DQN state vector (see State spec). State-vector length changes with the flag — cover with a unit test.

**Experiment E-A (foresight ablation):** identical agent trained/evaluated with `use_foresight = off / ewma / lstm` on S3 and S6. Report deltas in regret events, pool-exhaustion events, p99 latency. **Success criterion:** LSTM foresight measurably reduces regret events on S3 vs `off`. If EWMA ties LSTM, report honestly (still beats `off`; the claim becomes "foresight matters," not "LSTMs matter").

**Unit tests:** output shapes per head; provider interchangeability; state-length under each flag; no gradient flow from DQN loss into forecaster.

### Addition C — Regret & Churn Accounting

**Goal:** make the cost of misbudgeting *measurable in both directions* — over-spending the pool (starving a later critical request) and mistimed rekeying (churn) — and surface it as the project's headline operational metric.

**Deferral semantics (implements Hard Rule 9):**
- `env/deferral_queue.py` — priority queue (by sensitivity class, FIFO within class) for hybrid-mandatory requests the pool cannot currently cover. Each queued request accrues latency per step; served automatically when pool covers its draw. Never downgraded.
- Event log entries: `regret_event` (deferral onset), `deferred_critical_step` (each waiting step), with timestamps and request metadata.

**Reward wiring (already reflected in the reward formula above):**
- `− R_starve · deferred_critical_steps_this_step`, with `R_starve ≈ 10 × w_lat` (config: `reward.r_starve`).
- Rekey cost `c_rekey(load) = c0·(1 + β·load)` (config: `reward.c_rekey_base`, `reward.c_rekey_load_beta`).
- **Staleness is NOT a reward term:** key age hitting the SP 800-57-derived cap `L` triggers a **forced rekey** — `REUSE` is masked at age ≥ L, the rekey happens automatically, pays `c_rekey(load)`, and is logged as `forced_rekey`. (Keeps Hard Rule 1 clean: no security-flavored shaping; the agent learns to rekey *early at cheap moments* purely from cost.)

**Metrics module:** `metrics/regret.py` — per-episode: `regret_events`, `deferred_critical_steps`, `rekeys_per_100_requests`, `forced_rekey_ratio` (forced / total rekeys), `discretionary_hybrid_serves`. Plus a **retrospective attribution log**: for each regret event, which earlier *discretionary* hybrid serves consumed the pool bits that would have covered it (analysis/plots only — **never** enters reward or state).

**Surfacing:** live regret counter on the dashboard (Demo Beat 2); attribution plot in the report; `regret_events` and `forced_rekey_ratio` columns in the closing comparison table.

**Unit tests:** queue priority/FIFO ordering; regret counting on synthetic exhaustion; forced-rekey trigger at age cap; attribution log consistency (bits attributed ≤ bits spent).

---

## 4B. STRETCH GOALS (only after E-A and S5 are green; first thing cut if behind)

### Stretch B — Multi-Pool Replenishment Allocation

Real trusted nodes don't feed one bucket — they **allocate the QKD link's output across per-tenant reserves**. Give the agent a second, slower decision: every K steps (allocation tick), choose how incoming key material is split across tenant pools.

- **DQN-friendly design (keep one agent, one MDP):** allocation is chosen from a small preset set (e.g., `{balanced, hospital-heavy, fintech-heavy, drain-protect}`) and only on tick steps — the action space becomes `serve-actions ∪ allocation-presets`, with allocation actions masked except at ticks. No hierarchical RL, no second agent.
- State additions: per-tenant reserve levels, per-tenant forecast demand (reuses Addition A's pool head, extended per-tenant).
- Payoff: turns the problem from single-reservoir inventory into a small supply chain — banking the hospital tenant's reserve ahead of its compliance window (great synergy with S6) while the fintech tenant runs lean.
- Effort estimate: ~1.5 weeks. **Do not start before month 3, and never at the expense of S5 or E-A.**

---

## 5. Scenarios (the experiment grid)

| # | Scenario | What changes | What it tests | Train/Eval |
|---|----------|--------------|---------------|-----------|
| S1 | Benign baseline | Steady mixed traffic | Basic budgeting vs baselines | Train |
| S2 | HNDL posture | Threat elevates → floors ratchet up | Floor mechanics + demand shift | Train |
| S3 | QKD degradation | QBER↑, SKR↓, pool refill collapses | Scarcity budgeting under stress | Train |
| S4 | DDoS / noisy neighbor | One low-sensitivity tenant floods API | Protecting critical tenants' pool share | Train |
| S5 | **Steering attack** | Adversarially shaped threat trace vs soft-reward baseline agent AND masked agent | **Headline contribution** | Eval experiment |
| S6 | **Migration wave** | Scripted CNSA-2.0-style timeline ratchets tenant cohorts' floors in phases | Policy robustness under non-stationarity (Option-2 story) | **Held-out eval only** |
| E-A | **Foresight ablation** | Same agent with `use_foresight = off / ewma / lstm` on S3 + S6 | Value of anticipation (Addition A) — regret & exhaustion deltas | Eval experiment |

---

## 6. The Demo (what the mentor/examiner sees — one dashboard, four beats, ~12 min)

1. **The living system (2 min).** Tenant service graph pulsing with requests; edges flash colored by key type served (grey classical / amber PQC / green hybrid). Side panels: QKD pool gauge, threat forecast strip, latency chart. Click one request → tooltip shows class, floor, chosen action, and why.
2. **The budgeting brain (3 min).** Agent vs always-hybrid baseline on S3 (QKD degradation). Baseline drains the pool → red **POOL EXHAUSTED** event when a critical hospital request arrives, its **live regret counter** ticks up, and the deferred request visibly waits in queue. Agent had shifted routine flows to PQC and *saved* the quantum material — its regret counter stays at 0 and the same critical request is served hybrid instantly. Money plots: two diverging pool-level curves + two regret counters. Point at the foresight strip: "it started conserving *here*, when the SKR forecast turned down — a threshold rule can't do that."
3. **The steering attack (4 min — thesis moment).** Split screen, same adversarial threat trace into both agents. Soft-reward baseline's served-tier histogram slides *downward* (talked into weaker keys). Masked agent's floor line only steps *up*. Caption: **"Security isn't in our reward, so it isn't for sale."**
4. **The migration wave (3 min).** Timeline slider: Phase 2 hits, the patient-records tenant cohort's floors ratchet to hybrid-mandatory, the subgraph shifts color, hybrid demand surges, pool dips — and the agent (never trained on this schedule) visibly tightens budgeting and holds latency.

**Closing slide:** one table — Agent (± foresight) vs always-PQC vs always-hybrid vs static-threshold vs random, across S1–S4 + S6: p99 latency, pool-exhaustion events, **regret events**, **forced-rekey ratio**, floor violations (agent column: **0, structurally guaranteed**).

---

## 7. Timeline & Cut Lines (pre-agreed — do not relitigate in week 8)

## 7. Timeline & Cut Lines (2 MONTHS / 8 WEEKS — see split.md for the weekly breakdown)

> Deadline is 2 months. Stretch B is **out of scope**. The three non-negotiable deliverables: a working masked-DQN pool-budgeting agent, the steering-attack result, a live demo + report. Week-by-week owner assignments live in `split.md §2`.

**Weeks 1–2 — the spine.** Repo/contracts setup; pool sim + deferral/regret + minimal env + masking; DQN wired via stubs; data ingestion begins. *Gate: env runs a full S1 episode end-to-end with regret logging.*
**Weeks 3–4 — learning + attack build (midpoint).** Baselines + tuned DQN; S2–S4; soft-reward baseline agent + adversarial-trace generator built; LSTM threat head trained. *🚩 Make-or-break gate (W3): DQN beats tuned threshold on S1 & S3. Midpoint gate (W4): wins on S1–S4, both attack pieces exist.*
**Weeks 5–6 — thesis + results.** Steering attack run (the headline — never cut); LSTM dual-head + E-A ablation; S6 migration wave; all result tables filled; **feature freeze end of W6**.
**Weeks 7–8 — write, polish, buffer.** Everyone writes their report section; dashboard demo finalized + rehearsed; W8 is slippage buffer only (no new features); tag `v1.0`, submit.

**Cut order the instant a gate slips:** (1st) S6 migration wave → lose an experiment, keep the thesis. (2nd) LSTM head of E-A → keep the EWMA foresight variant. (3rd) one of S2/S4. (4th) dose-response sweep → keep single-point steering result. **Never cut:** the pool, the masking, the deferral/regret semantics, the four baselines, the steering attack.
**Scope-creep tripwires (say no immediately):** agent choosing migration order · security term "just to stabilize training" · real AWS deployment · second agent/hierarchical RL · Stretch B · per-flow crypto benchmarking side-quests.
**Scope-creep tripwires (say no immediately):** agent choosing migration order · security term "just to stabilize training" · real AWS deployment · second agent/hierarchical RL · per-flow crypto benchmarking side-quests.

---

## 8. Anticipated Examiner Questions (rehearse these)

- *"Why RL instead of a threshold rule?"* → Show the baseline table AND the E-A ablation: a threshold reacts to the *current* pool level; our agent acts on the *forecast* (pool trajectory + incoming demand), and the ablation quantifies exactly how much anticipation is worth in regret events and exhaustions.
- *"Isn't this just a recommender system?"* → A recommender optimizes each request in isolation; ours can't, because serving hybrid now removes an option ten minutes from now — decisions are coupled through the pool. The per-request-greedy baseline *is* the recommender version of this system, and Beat 2 shows it exhausting the pool and racking up regret events exactly when it matters.
- *"Why would the agent ever serve classical when PQC is nearly free?"* → Two honest reasons, both in the environment: legacy endpoints flagged `pqc_capable: false` (migration-era realism — classical is the only interoperable option there), and measured marginal cost under load (extra handshake bytes/CPU matter at p99 during S4). Policy floors — not the agent — decide *whether* classical is acceptable for a flow; the agent only economizes above the floor.
- *"Isn't the threat score manipulable?"* → Yes — that's Beat 3. Manipulation only raises floors. The soft-reward baseline is the cautionary tale, on screen.
- *"Real migrations need dependency ordering — you ignore it?"* → Deliberately: sequencing is an orthogonal planning problem (cited); we operate beneath any chosen sequence and demonstrate stability across a staged schedule (S6).
- *"Where do the security tiers come from?"* → NIST PQC categories, SP 800-57 lifetimes, CNSA 2.0 / BSI timelines, ETSI GS QKD 014 — no invented constants; the mapping table is in the appendix with citations.
- *"Is the QKD realistic?"* → Pool-behind-KMS with ETSI-style delivery, refill driven by real CV-QKD traces; matches BT/Toshiba, Madrid, SK Telecom, EuroQCI architectures.
- *"Why not deploy on real AWS?"* → No research value; every result runs identically locally; cloud framing is about the deployment story (multi-tenant KMS with quantum backhaul), demonstrated via containerized tenants.

---

## 9. Naming & Framing Cheatsheet

- **Project:** SmartKeyNet: RL for Hybrid Cryptography ✔ (system's whole job = choosing among classical/PQC/hybrid key material per request).
- **Mentor-pitch mapping:** "ML model on network features predicts a threat score" = LSTM forecaster · "assign it a key" = tier policy table · "DQN for optimal boundaries" = masked DQN optimizing within floors. Same sentence, rigorous machinery. Mention the cloud skin to the mentor in one sentence at the next meeting (cheap insurance; core elements unchanged).
- **One-sentence pitch:** *"SmartKeyNet is the decision layer for a multi-tenant cloud KMS in the hybrid era — classical, post-quantum, and quantum-backed keys served per request by a DQN that budgets scarce quantum resources, with security floors it structurally cannot violate."*

---

## 10. First Claude Code Session — Suggested Kickoff Order

1. Scaffold repo: `env/` (Gym env, pool sim, request generator, `deferral_queue.py`, `forecast_provider.py`), `agents/` (DQN, baselines), `forecaster/` (`model.py`, `dataset.py`, `train.py`), `metrics/` (`regret.py`), `api/` (FastAPI KMS facade), `dashboard/`, `experiments/`, `data/`, `configs/`.
2. Build the **pool simulator** first (trace-driven refill, draw-down, exhaustion events) with unit tests.
3. Build the **deferral queue + regret accounting (Addition C)** next — the env's exhaustion semantics depend on it (Hard Rule 9). Unit-test regret counting before any RL exists.
4. Build the request generator from a small NetworkX graph (10 nodes to start, with `pqc_capable` flags).
5. Wire a minimal env (state incl. EWMA foresight features via `MovingAverageForecaster`, mask, full reward formula incl. `R_starve` and `c_rekey(load)`) + vanilla DQN; overfit S1 on purpose to prove the loop works, then generalize.
6. Implement the four baselines + the comparison harness *before* tuning the agent (Hard Rule 7).
7. Only then: **LSTM dual-head forecaster (Addition A)** — generate baseline rollouts, train offline, swap in via `use_foresight: lstm`, run E-A.
8. Then remaining scenarios, dashboard, and (if green) Stretch B.

> Re-read the **Hard Rules** at the start of every session. They are the project.
