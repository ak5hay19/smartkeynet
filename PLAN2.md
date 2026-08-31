# SmartKeyNet — PLAN2.md
### Extended Plan: Dashboard v2 (Threat Input + Explainability) — Bootstrap Document for Paper Drafting

> **How to use this file.** This document is fully self-contained — it does not assume you have access to `PLAN.md`, `split.md`, `SESSION_LOG.md`, `PROGRESS.md`, or any `HANDOFF_*.md` file from the working repo. Everything needed to understand the project's concept, its non-negotiable design rules, its architecture, and its (new, expanded) dashboard is reproduced here.
>
> **This file's specific purpose:** to be fed into a fresh session, alongside the attached dashboard HTML mockup (`smartkeynet_dashboard_mockup_v2.html`), to draft a **rough base research paper / report** on the project.
>
> **Three grounding rules for whoever drafts that paper, non-negotiable:**
> 1. **The attached HTML file is a UI/UX mockup only.** Every number, chart, threat score, sensor reading, and "why this decision" sentence in it was hand-authored for layout demonstration. **None of it is a real experimental result.** Do not cite, quote, or reproduce any value from the mockup as a finding, a measurement, or evidence of system behavior. It illustrates *what the interface would show*, not *what the system has shown*.
> 2. **§10 below ("Current Implementation Status") is the honest state of the underlying code as of the last verified session.** A "rough base paper" drafted from this file should confidently describe *design*, *architecture*, and *methodology* — those are settled. It must describe *results* and *evaluation* as planned/future work, not completed findings, until real experiments exist. See §14 for an explicit section-by-section writing guide reflecting this split.
> 3. **§5's Hard Rules are non-negotiable design constraints**, not stylistic preferences. Treat them as ground truth for any system-design or methodology section — they are the actual reason this project's central claims (no security-reward, structural floors, an honest steering-attack result) are defensible at all.

---

## 1. One-paragraph summary

SmartKeyNet is the decision layer for a **multi-tenant cloud Key Management Service (KMS) operating in the hybrid-cryptography era**. Tenants request cryptographic keys through an AWS-KMS-style API (backed conceptually by ETSI GS QKD 014 key delivery). For every request, a **DQN agent** decides how to serve it — classical, post-quantum (ML-KEM-768), or hybrid (ML-KEM ⊕ QKD-sourced key material) — and when to reuse or rekey. The hard part: quantum key material comes from a **finite pool that refills slowly** (kbps, driven by real or synthetic CV-QKD traces), so the agent must learn to *budget* it across competing tenants. Security is **never** in the reward: a threat forecaster + policy table set a per-request **minimum tier floor**, and all actions below the floor are **masked out** before the DQN sees them. The project demonstrates that prior soft-reward RL-crypto designs can be *steered* by an adversarially shaped threat signal, while this masked agent's protections can only ratchet upward. That steering-attack-plus-structural-defense is the headline research contribution. Two mechanisms elevate the agent beyond a per-request recommender: a **dual-head forecaster** (threat posture + pool/demand trajectory) that makes the agent *anticipatory* rather than reactive, and **regret/churn accounting** that quantifies the cost of spending the pool wrong in either direction. A live dashboard (**detailed in §7 below, "Dashboard v2"**) makes every one of those mechanisms — the threat signal's origin, the floor computation, the masking, the cost tradeoff, the pool budgeting, and the steering attack itself — directly inspectable, not just claimed in a report.

**Team:** originally scoped for 4 people, currently executed solo across all roles. **Duration:** 2 months / 8 weeks. **Build style:** ~90% AI-assisted codegen with human integration/review. **Target venues (stretch):** IEEE QCE / QCNC; report + viva demo are the primary deliverables.

---

## 2. What the project is (plain English + analogies)

### The elevator pitch

Networks are entering a decades-long transition where three kinds of cryptographic keys coexist: classical (fast, quantum-vulnerable), post-quantum (quantum-resistant, effectively unlimited), and quantum-distributed (QKD — information-theoretically strong, but **scarce**). Somebody has to decide, request by request, which key material each connection gets. Today that's static config files. SmartKeyNet replaces the static rules with a learning agent that budgets the scarce resource intelligently — while being *structurally incapable* of under-protecting anything.

### Analogy 1 — The hospital blood bank

QKD key material is like **O-negative blood**: universally powerful, arrives in a slow trickle from donors (the QKD link, filling the pool at kbps), and if you transfuse it into every patient with a scraped knee, you have none left when the trauma case arrives. PQC keys are like **synthetic plasma** — manufactured on demand, unlimited, good enough for most patients. Classical keys are a **band-aid** — fine for minor cases only. SmartKeyNet is the triage nurse who learned, from experience, which patients genuinely need O-neg — and hospital policy (the tier floor) makes it *impossible* for the nurse to give a trauma patient a band-aid, no matter what.

### Analogy 2 — The steering attack, or "don't pay the guard by mood"

Imagine a security guard whose bonus depends partly on "how relaxed the office feels." A clever burglar plays soothing music, the guard's metrics say everything is calm, and the guard props the door open. That's what happens when security is a **soft term in an RL reward** — an adversary who can shape the input signals can talk the agent into weaker choices. SmartKeyNet's guard has no such bonus: a locked policy table decides the *minimum* door security, the guard only optimizes things like electricity and patrol effort, and any signal manipulation can only make the doors lock *tighter*.

### Analogy 3 — Migration as a rolling office move

An enterprise going quantum-safe doesn't flip a switch; departments move floor by floor over years (compliance deadlines ratchet policy floors up, subsystem by subsystem). SmartKeyNet doesn't plan the move (that's a different problem, cited as out of scope); it's the **utilities service that keeps every floor powered correctly throughout the move**, even as demand shifts in waves it was never trained on.

### What the agent literally does per request

For each incoming key request `(tenant, service, sensitivity class)` the agent picks one of:
- `SERVE_CLASSICAL` — X25519 / AES-256-GCM class. Lowest tier (T0).
- `SERVE_PQC` — ML-KEM-768 (NIST Category 3). The free workhorse (T1).
- `SERVE_HYBRID` — ML-KEM-768 ⊕ QKD pool material via HKDF. Premium tier (T2/T3). **Consumes pool.**
- `REUSE` — keep the existing session key if age limits allow.
- `REKEY_NOW` — force refresh (e.g. ahead of a forecast threat spike).

Actions below the request's policy floor, or infeasible ones (pool empty, key age exceeded), are **masked** — removed from the action set before the DQN evaluates it.

---

## 3. Why (motivation, three layers, plus a fourth this document adds)

1. **World problem — Harvest Now, Decrypt Later (HNDL).** Adversaries record classically-encrypted traffic today to decrypt once quantum computers mature. NIST PQC standards, NSA CNSA 2.0 deadlines, and BSI/ANSSI guidance have kicked off a global, decades-long migration. During it, classical / PQC / QKD-backed keys **coexist** — the hybrid era is the deployment reality, not a thought experiment.

2. **Operational problem — somebody must budget scarce quantum keys.** QKD pools refill at kbps under variable secret-key rate (SKR) and rising QBER; requests arrive in bursts across tenants with wildly different data lifetimes (patient records: decades; telemetry: hours). Static rules fail in both directions: *always-hybrid* drains the pool before critical requests arrive; *fixed thresholds* waste quantum material or starve it at the wrong times. **(Disqualification rule: if a simple threshold policy matches the DQN in evaluation, the project premise fails — tuned threshold baselines are mandatory, and beating them is the bar.)**

3. **Research problem — soft security rewards are an attack surface.** Published RL-for-crypto work (e.g. Q-learning adaptive encryption for WSNs; threat-score-driven PQC selection; the Noetzold 2026-style reward structure) places security in the reward. This project reproduces that design as a baseline and shows an adversarially shaped threat trace **steers** it into serving weaker keys. The masked architecture is immune by construction: the threat signal can only raise floors. Contribution in one sentence: *"Security as constraint (action masking), not reward — demonstrated against a live steering attack on a reproduced soft-reward baseline."*

4. **Legibility problem (this document's addition — motivates Dashboard v2, §7–8).** RL-based security systems are usually opaque: a black-box policy makes a call and there's no cheap way to audit *why*. SmartKeyNet's masking architecture happens to make this tractable for free — the floor is a lookup table, the mask is a deterministic function, and the final pick is "cheapest legal option." Dashboard v2's Explain Decision panel (§7.3) is not a new algorithm; it is a claim that **this specific design is auditable in a way soft-reward designs structurally cannot be**, made concrete by exposing the real computed values behind every decision rather than a generated narrative. This is a legibility/interpretability contribution riding on top of the existing masking design, not a separate research thread.

---

## 4. Where (real-world use cases)

- **Cloud KMS with quantum backhaul (primary framing).** A cloud region whose KMS backhaul includes a metro QKD link — the "what AWS KMS looks like ~2030" story. Grounding: JPMorgan × Toshiba/Ciena QKD-secured production link; BT/Toshiba London metro network; SK Telecom; EuroQCI build-out. Tenants = hospital, fintech, logging service, etc., each with per-tenant key policies (exactly how cloud KMS key policies work today).
- **Telecom backbone trusted nodes.** The canonical ETSI GS QKD 014 deployment: QKD as backbone infrastructure delivering keys via a KMS REST API. SmartKeyNet is the policy brain such nodes currently lack (today: static rules).
- **Hospital / financial data centers on a metro QKD ring.** Mixed criticality on shared infrastructure; per-request triage is exactly the need.
- **Enterprises mid-migration (evaluation scenario, not a second system).** Staged CNSA-2.0-style timelines ratchet tenant policy floors upward in waves; SmartKeyNet operates *underneath* whatever migration sequence the org chooses.
- **Security operations reviewing agent behavior after the fact (new, motivates §7.3).** A SOC analyst or auditor asking "why was this hospital session served PQC and not hybrid at 14:02?" needs an answer traceable to a rule, not a black box. Dashboard v2's Explain Decision panel is designed for exactly this use case.

---

## 5. Architecture, State, Reward, and Hard Rules

### 5.1 Architecture (updated to show threat-input flexibility)

```
                    ┌─────────────────────────────────────────────┐
                    │           THREAT INPUT (§7.1 / §8)          │
                    │  offline RT-IoT2022  |  uploaded .pcap  |    │
                    │        replayed .pcap (real-time pace)       │
                    └───────────────────┬───────────────────────┘
                                        │ raw traffic features (same
                                        │ extraction path regardless
                                        │ of source — Hard Rule 11)
                                        ▼
   CV-QKD traces ──────▶┌───────────────────────────┐
   (real or synthetic)  │  QKD POOL SIM (SKR/QBER)  │
                        └───────────┬───────────────┘
                                    │ pool level
 NetworkX tenant graph              ▼
 (tenants, edges w/     ┌──────────────────────────────────────────┐
  sensitivity class,    │            ENVIRONMENT (Gym-style)       │
  traffic rate) ───────▶│  request stream ← sampled from graph     │
                        │  key ages, latency model, energy model   │
                        └──────┬───────────────────────┬───────────┘
   threat features            │ state                 │ mask
   (from Threat Input) ─┐     ▼                       ▼
   ┌──────────────┐     │┌──────────────┐       ┌──────────────────┐
   │  DUAL-HEAD   │─────┴│ POLICY TABLE │──────▶│  ACTION MASKING  │
   │  FORECASTER  │thrt  │ (class ×     │ floor │  (structural,    │
   │ threat head +│head  │  posture →   │       │   inviolable)    │
   │ pool/demand  │      │  min tier)   │       └────────┬─────────┘
   │ head         │      └──────────────┘                │ safe action set
   └──────┬───────┘                                      │
          │ pool head → DQN state (foresight features)   ▼
                                                ┌─────────────────┐
                                                │   DQN AGENT     │
                                                │ reward: latency,│
                                                │ energy, fresh-  │
                                                │ ness, QKD       │
                                                │ scarcity price  │
                                                │ (NO security    │
                                                │  term, ever)    │
                                                └────────┬────────┘
                                                         │ chosen action
                                                         ▼
                                        ┌──────────────────────────────┐
                                        │  EXPLAIN DECISION (§7.3/§8)  │
                                        │  templates the six values    │
                                        │  above into a human-readable │
                                        │  trace — never generates a   │
                                        │  narrative that isn't one of │
                                        │  those real computed values  │
                                        └──────────────────────────────┘
```

### 5.2 State spec (per step)

Threat score + threat forecast (k steps ahead), QBER, SKR, pool fill level, arrival rate, load, avg latency, current key age, current key type (one-hot), request sensitivity class / tenant policy floor, plus foresight features: projected pool level at t+H, forecast SKR trend, forecast hybrid-mandatory demand over horizon H, and a recent-regret-event flag.

### 5.3 Reward (within the safe set only)

```
r = − w_lat·latency − w_en·energy + w_fr·freshness
    − w_qkd·(pool bits consumed)
    − R_starve·(deferred_critical_steps)
    − c_rekey(load)·1[rekey]

where c_rekey(load) = c0·(1 + β·load)
```
`R_starve` is large (≈10× the base latency weight). There is no security term, anywhere, ever.

### 5.4 Hard Rules 1–9 (original, unchanged, non-negotiable)

1. **No security term in the reward. Ever.** Not even a small one, not even temporarily "to help training." This is the entire point of the project.
2. **Floors are enforced by action masking in the environment**, not by penalties. Threat signals may only *raise* floors.
3. **One agent, one MDP.** The migration wave is a **scripted, exogenous schedule**. The agent never chooses migration order, never sees the graph, never allocates crews. Test: deleting the tenant graph and replacing it with a plain arrival process must not change one line of agent code.
4. **No invented security constants.** Tiers map to citable artifacts only: NIST PQC categories, SP 800-57 key lifetimes, CNSA 2.0, BSI/ANSSI guidance, ETSI GS QKD 014.
5. **No free mid-session algorithm switching.** Key-type changes happen only at rekey boundaries; rekeying has explicit latency/energy cost.
6. **QKD stays architecturally honest.** Backbone resource behind the KMS, pool semantics, ETSI-style delivery. Endpoints never "do QKD."
7. **Tuned non-RL baselines are mandatory:** always-PQC, always-hybrid, static threshold (grid-searched), random. If a tuned threshold ties the DQN, report it honestly and investigate environment design first.
8. **Train/eval split for migration:** train on stationary scenarios; the migration-wave scenario is **held-out evaluation only**.
9. **Pool exhaustion never causes a downgrade.** A hybrid-mandatory request the pool can't cover is **deferred** (queued, accruing latency) until the pool refills — never served below its floor. Each deferral onset is logged as a **regret event**. Exhaustion is an availability failure the agent pays for in reward; it is never a security failure.

### 5.5 Hard Rules 10–11 (new, this document's additions — govern Dashboard v2 specifically)

10. **The Explain Decision panel (§7.3) may only display values the pipeline actually computed** — `threat_score`, `posture_probs`, the resolved policy floor, the action mask, per-action cost lookups, and the final chosen action. It must never synthesize an explanation via a generative model standing in for these values. Any human-readable sentence shown must be templated deterministically from the values above — swap in the real numbers, don't invent new reasoning. If this rule is ever violated, the panel becomes exactly the kind of black-box "trust me" system this project argues against.
11. **Live threat input is scoped as replay-only for this project's timeline.** The guaranteed deliverable is: a previously captured `.pcap`/`.pcapng` file streamed at (or faster than) its original packet pace, through the **exact same** feature-extraction path used for offline RT-IoT2022 training data (no second, ad hoc pipeline for "live" data). **True live capture from a real network interface is explicitly out of scope for the guaranteed deliverable** — it is a stretch item only (§11, cut-order), never allowed to compete for time against the steering attack, the masked agent, or the four baselines.

---

## 6. Datasets & Provenance

The project needs *external* data in exactly **three** slots. Everything else (pool drain, key sizes, tenant graph, migration schedule) is simulator physics, not a dataset problem.

**Golden rules:**
- **One real-network slot only** (the threat forecaster). RT-IoT2022 alone is sufficient to ship.
- **NEVER train the agent on `rl_experiment_*` logs.** They are *outputs* of the soft-reward baseline being critiqued — training on them is imitation-learning the exact flawed policy under attack. They exist for **baseline reproduction + calibration only.**
- **QKD scarcity is NOT in any borrowed dataset.** It comes from a QKD SKR/QBER trace (real or documented synthetic) driving pool refill.
- **Pcap ingestion (uploaded or replayed, §7.1/§8) is not a new dataset slot — it's an alternate *input path* into Slot 1.** The feature-extraction function that turns RT-IoT2022 rows into training windows is the same function that turns a pcap's packets into inference-time windows. This keeps Hard Rule 11 honest: no parallel pipeline, no second dataset requirement.

**Slot-to-source map:**

| Component | Primary source | Notes |
|---|---|---|
| **Threat forecaster (LSTM)** | **RT-IoT2022** (real IoT intrusion flows, labeled) | Offline training source. Real telemetry replaces prior work's synthetic macro-signals — a citable upgrade. |
| **Threat forecaster, inference-time (uploaded/replayed pcap)** | Same feature-extraction path as above, fed a `.pcap`/`.pcapng` instead of the training CSV | Not a new dataset — an alternate input to the same trained, frozen model. |
| **Sensitivity classifier** | `confidentiality_train`/`valid` (Q-OPSEC dataset; 4-class) | Starter classifier, not final. |
| **QKD pool refill (SKR/QBER)** | CV-QKD experimental trace, or documented synthetic SKR process | Synthetic fallback fully acceptable if the generation procedure is stated and rate ranges cited. |
| **Policy-table calibration** | `synthetic_context_dataset` (Q-OPSEC dataset) | Sanity-check only for (risk, confidentiality) → tier mapping. |
| **Baseline reproduction + calibration** | `rl_experiment_*` / `synthetic_rl_*` logs | Reproduce the soft-reward baseline for the steering attack. **Never used as training data for the real agent.** |
| Pool drain, key sizes | ETSI GS QKD 014 (spec) | Simulator arithmetic, no dataset. |
| Primitive latency/energy costs | published liboqs / pqm4 benchmarks | Measured, not invented. |
| Tenant graph | NetworkX synthetic (documented generator) | Edge attrs: `sensitivity_class`, `traffic_rate`, `pqc_capable`. |
| Migration schedule (S6) | config file (scripted, exogenous) | Not a dataset. Never agent-controlled (Hard Rule 3). |

**DO-NOT-USE (verified degenerate):** any dataset file found to have near-zero label variety (e.g. a small number of unique feature rows dominating the whole file, or a single label covering ~100% of rows) must not be used as training data — verify before loading anything new.

---

## 7. Dashboard v2 — full panel spec

This supersedes the earlier "4 demo beats" framing with a 7-panel dashboard. Panels 4–7 below are the original 4 beats, renumbered; panels 1–3 are new. Every panel is described with: what it shows, what real system state it's meant to display, and current implementation status (cross-reference §10).

| # | Panel | Purpose | Real data source (design intent) | Status |
|---|-------|---------|-----------------------------------|--------|
| 1 | Threat Input | Choose and inspect the threat-signal source | `ForecastObservation` → `ForecastProvider.update()` → `ThreatForecast` | Not implemented |
| 2 | Living System | Live tenant graph + last-N decisions | `StateDict`, per-request event log | Env logic implemented; dashboard not implemented |
| 3 | Explain Decision | Full reasoning trace for one decision | `ThreatForecast.threat_score`/`posture_probs`, `PolicyTable.floor()`, `compute_mask()`, per-action cost table, chosen `Action` | Underlying values implemented; dashboard/template not implemented |
| 4 | Budgeting Brain | Agent vs. always-hybrid baseline, side by side | `ScenarioResult`, `EpisodeMetrics` (regret events, pool exhaustion, forced-rekey ratio) | Harness/metrics implemented; S3 scenario dispatch not implemented; dashboard not implemented |
| 5 | Steering Attack | Split-screen soft-reward vs. masked agent under an adversarial trace | `attack/trace_generator.py`, `attack/run_attack.py` outputs, dose-response sweep | Not implemented (headline deliverable, never cut) |
| 6 | Migration Wave | Scripted floor-ratchet timeline, held-out eval | `migration_schedule` config, S6 scenario dispatch | Not implemented |
| 7 | Results | Closing comparison table across policies/scenarios | `experiments/harness.run_grid()` output | Harness implemented; most scenarios feeding it are not |

### 7.1 Panel 1 — Threat Input

Three source modes, presented as an explicit user choice:
- **Offline dataset** — RT-IoT2022, batch-processed. Training-time only, not for live decisions.
- **Uploaded pcap** — a single capture, analyzed in one pass; feature windows extracted and scored, no streaming state retained.
- **Replayed pcap (real-time pace)** — a capture streamed at its original packet timing (or a configurable multiple, e.g. 1×/4×), so every downstream panel updates exactly as it would from a genuinely live interface, without the operational risk of depending on live traffic actually occurring during a demo. **This is the guaranteed "looks and behaves live" mode** (Hard Rule 11).

Below the mode selector: a file/stream control (drop a `.pcap`/`.pcapng`, playback transport, speed selector), a rolling table of extracted feature windows (packets/sec, bytes/sec, unique ports touched, SYN ratio, or equivalent — whatever the real feature-extraction step produces), and a visualization of the model pipeline itself (anomaly/reconstruction stage → attack-type classifier stage → fusion stage → posture output), so the *mechanism*, not just the end score, is visible. A closing note makes explicit that this output feeds `env/masking.py`'s floor computation **only in the raise direction** (Hard Rule 2) and is frozen during agent training (no gradients flow back from the DQN's loss into this pipeline).

### 7.2 Panel 2 — Living System

A live view of the tenant network: nodes are tenant/service pairs, edges are colored by whichever key tier was most recently served to that pair (three-way color coding, one color per tier). A scrollable "recent decisions" list lets a viewer select any past decision; selecting one updates a decision-detail card (tenant/service, sensitivity class, policy floor, action taken, one-line reason) and offers a direct link into Panel 3 for the same decision's full trace. Side panels show current pool level (with refill rate and QBER), the current threat forecast (score + posture badge + short-horizon sparkline), and rolling p99 latency — the three inputs that jointly determine what "recent decisions" will look like next.

### 7.3 Panel 3 — Explain Decision

The interpretability centerpiece (Hard Rule 10). Not prose generated after the fact — a **six-step trace**, each step a real, separately-inspectable computation, for whichever decision is selected (shared selection state with Panel 2):

1. **Threat signal ingested** — the raw `threat_score` value and its source (which Panel 1 mode produced it, and which window).
2. **Posture classified** — the three-way posture probability distribution (calm / elevated / high), with the resolved (argmax) posture highlighted.
3. **Policy floor lookup** — the actual (sensitivity class × posture) → floor table, rendered as a grid, with the specific cell that fired for this decision highlighted. This step is the load-bearing one: it is a deterministic lookup, not a judgment call, and Hard Rule 2 guarantees it can only ever move in the "stricter" direction as posture worsens.
4. **Action mask computed** — all five possible actions listed; illegal ones shown struck through with their specific reason (below floor / pool cannot cover the draw / key age exceeded its cap / no existing key yet), matching exactly the three legality rules `compute_mask()` implements.
5. **Cost comparison among legal actions** — for whatever remains legal after step 4, a direct comparison of the per-action cost (latency, energy) actually used by the reward function, with the cheapest option marked.
6. **Final decision** — the actual action chosen, plus one templated sentence combining steps 3–5 (e.g. "floor required X; Y was the cheapest legal option that cleared it" — or, honestly, for cases where the trained policy's own learned preference (not a hard rule) decided among several legal options, a sentence that says exactly that, rather than papering over it with false certainty).

This panel should be built so that it degrades honestly: if a decision was purely floor-driven (only one legal tier existed), step 5 should say so plainly ("no cost tradeoff existed here") rather than presenting a comparison that didn't actually happen.

### 7.4 Panel 4 — Budgeting Brain (formerly "Beat 2")

Side-by-side comparison, same scenario (S3 — QKD degradation), two policies: the masked DQN and the always-hybrid baseline. Each side shows a pool-level trend chart, a regret-event counter, pool-exhaustion count, p99 latency, and forced-rekey ratio, pulled directly from `ScenarioResult`/`EpisodeMetrics`. The baseline side is expected to show a pool-exhaustion event (a visible "exhausted" state) and a nonzero regret count; the agent side is expected to show proactive conservation (visible in the pool trend dipping and recovering before exhaustion) and stay at zero on both.

### 7.5 Panel 5 — Steering Attack (formerly "Beat 3" — headline result, never cut)

A dose-response chart (attack strength 0.0 → 1.0 on the x-axis, share of decisions served below the intended tier on the y-axis) comparing the soft-reward baseline against the masked agent — the baseline's line should rise with attack strength, the masked agent's should stay flat at zero by construction. Below it, a split-screen view: served-tier histograms for both policies at a few attack-strength checkpoints, so the *shape* of the degradation (soft-reward baseline sliding toward weaker tiers) versus its absence (masked agent's floor only ever stepping up) is visually obvious without reading numbers. A single caption line should state the thesis plainly: security that isn't in the reward can't be steered out of it, because it was never a preference to begin with.

### 7.6 Panel 6 — Migration Wave (formerly "Beat 4")

A phase selector (Phase 1/2/3 of a scripted, config-driven floor-ratchet schedule) showing which tenant cohort's floor changed at that phase, and the resulting pool-level response (a dip as newly-mandatory-hybrid demand appears, followed by recovery as the agent reallocates from discretionary spend elsewhere). This scenario is **held-out evaluation only** (Hard Rule 8) — the agent is never trained on this specific schedule.

### 7.7 Panel 7 — Results

The closing comparison table: one row per policy (masked DQN, static-threshold, always-hybrid, always-PQC, random), one column per metric (p99 latency, pool-exhaustion events, regret events, forced-rekey ratio, floor violations). The masked agent's floor-violations column should read zero with an explicit "structural" label — not a result that happened to come out well, a guarantee the architecture makes impossible to violate.

---

## 8. Addition D — Threat Input Flexibility & Explainability (new Required Addition)

Alongside the project's existing required additions (a dual-head forecaster for anticipatory behavior, and regret/churn accounting for measuring misbudgeting), this document adds a third:

**Goal:** make the threat signal's origin swappable (offline dataset / uploaded pcap / replayed pcap) without touching anything downstream, and make every decision's reasoning inspectable without resorting to a generated narrative.

**Modules implied (naming illustrative, not prescriptive of final file layout):**
- A feature-extraction function shared between offline training-data ingestion and pcap ingestion — one implementation, two callers.
- A pcap replay controller that streams packets at real (or scaled) pace and feeds windows into the same forecaster update path used by the live environment loop.
- A decision-trace assembler that reads the same six values described in §7.3 directly off the environment/masking/agent objects at decision time and formats them — no separate model, no separate source of truth.

**Ownership implication (if returning to a multi-person split):** the dashboard-owning role absorbs both new panels; the underlying values they display are already owned by the forecaster and environment/masking roles respectively — this addition is mostly a *visualization* layer over existing outputs, not a new subsystem competing for algorithmic-design time.

**Unit tests implied:** feature-extraction output is identical in shape whether sourced from the training CSV or a pcap window; the decision-trace assembler's step-3 floor lookup always matches `PolicyTable.floor()`'s actual return value for the same inputs (no drift between what's displayed and what's real); the trace assembler never emits a sentence containing a value not present in the six inputs it was given.

**Explicit non-goals (Hard Rule 11):** a real, continuously-running packet capture against a live network interface. Cut before anything in §11's list below if time is short.

---

## 8A. Addition E — Real Threat Forecaster Build-Out (RT-IoT2022)

Formalized 2026-08-31, following a session that built and ran the full
RT-IoT2022 feature-extraction pipeline (Phase 1 below) — see
SESSION_LOG.md's 2026-08-31 "full RT-IoT2022 feature-extraction
pipeline" entry for the complete real numbers and design writeup. This
is now the project's **top named priority for the next several
sessions**, ahead of dashboard polish, the API facade, and any other
stretch work (see §11's updated note below).

**Goal:** replace the placeholder `MovingAverageForecaster` with a
real, trained, dual-head LSTM forecaster (PLAN.md Addition A), in four
phases, each a separate session:

1. **Full RT-IoT2022 feature-extraction pipeline — DONE 2026-08-31.**
   A real download/loading mechanism (`data/get_data.py::_download_rt_iot2022()`,
   via the official `ucimlrepo` client against UCI ML Repository
   dataset id 942), label-degeneracy screening (real result: the full
   file is NOT degenerate — dominant label `DOS_SYN_Hping` at 76.89%
   of 123,117 rows, well under the 98% discard threshold), and a real
   feature-extraction function (`forecaster/rt_iot_features.py::extract_flow_features()`
   — 39 features, packet/byte rates, flow statistics, TCP flags,
   protocol/service distribution, reasoning documented in the module
   itself) windowing raw flow-level data into an LSTM-consumable
   representation (window=64, stride=32). Implemented as ONE shared
   extraction function usable by two future callers (PLAN2.md
   Addition D's "one implementation, two callers": the offline
   training path, run this session against the full dataset; a future
   single-window live-inference path, not yet built). Run against the
   FULL dataset (not a sample): 123,117 rows -> 3,846 windows, saved
   to `data/processed/rt_iot2022/rt_iot2022_windows.npz`. Real,
   honestly-reported finding: the resulting label distribution is
   heavily imbalanced (binary benign/attack windows split
   390/3,456 — 10.1%/89.9%; the 12-class majority-vote label all but
   erases the thinnest raw classes, e.g. `NMAP_FIN_SCAN`'s 28 raw rows
   win 0/3,846 window-level majority votes) — a real constraint Phase
   2's training must design around per Hard Rule 4 (class-weighted
   training, per-class F1, never accuracy). 25 new real behavioral
   tests, full suite 688 passed/1 xfailed. Zero changes to
   `env/forecast_provider.py`/`env/environment.py` (verified via
   `git diff --stat`).
2. **Build and train the dual-head LSTM model** — not started.
   `forecaster/model.py::SmartKeyForecaster` (threat head + pool head)
   and `forecaster/train.py` remain real `NotImplementedError` stubs.
   Will need to decide how Phase 1's RT-IoT2022-direct windowed output
   relates to `forecaster/dataset.py`'s separate rollout-log-based
   `RolloutWindowDataset` design (PLAN.md Addition A's original spec:
   the pool head trains on logged environment rollouts; the threat
   head's real-data source is Phase 1's output).
3. **Validate the trained model** — not started. Per-class F1 (never
   accuracy, given Phase 1's real measured class imbalance); confirm
   it isn't a degenerate always-predict-majority shortcut.
4. **Wire it in via `LSTMForecastProvider` and re-run the FULL
   experimental campaign** — not started. Gate W3 on S1+S3, the
   masked-vs-soft-reward S3 comparison, the full 11-point dose-response
   sweep, all six dashboard panels, and the paper's Table V — since
   every existing result was measured against the placeholder
   forecaster and does not transfer automatically to a real one. This
   is a later, separate, deliberate session per the project owner's
   own instruction — never bundled into the same session as Phases 2-3.

**Until Phase 4 (wire-in + full re-validation) is complete, the
`MovingAverageForecaster` placeholder remains the system's active,
authoritative forecaster, and every existing experimental result in
this document and the paper remains a valid, current measurement of
the system as it stands. Phase 4 will require re-running the full
experimental campaign, since results measured under the placeholder do
not transfer to results under a real forecaster.**

---

## 9. Scenarios (the experiment grid)

| # | Scenario | What changes | What it tests | Train/Eval | Dashboard panel |
|---|----------|--------------|---------------|-----------|------------------|
| S1 | Benign baseline | Steady mixed traffic | Basic budgeting vs baselines | Train | 2, 3, 7 |
| S2 | HNDL posture | Threat elevates → floors ratchet up | Floor mechanics + demand shift | Train | 2, 3 |
| S3 | QKD degradation | QBER↑, SKR↓, pool refill collapses | Scarcity budgeting under stress | Train | 4 |
| S4 | DDoS / noisy neighbor | One low-sensitivity tenant floods API | Protecting critical tenants' pool share | Train | 7 |
| S5 | **Steering attack** | Adversarially shaped threat trace vs soft-reward baseline agent AND masked agent | **Headline contribution** | Eval experiment | 5 |
| S6 | **Migration wave** | Scripted floor-ratchet timeline | Policy robustness under non-stationarity | **Held-out eval only** | 6 |
| E-A | **Foresight ablation** | Same agent, threat input source varied (off / ewma / lstm; and, per this document, offline dataset / uploaded pcap / replayed pcap) | Value of anticipation and of realistic threat input | Eval experiment | 1, 7 |

---

## 10. Current Implementation Status (honest snapshot — read before writing anything results-shaped)

This section exists so a paper drafted from this document doesn't accidentally claim more than is true.

**Solid / implemented and tested:** the QKD pool simulator (refill/drain/exhaustion), the deferral queue and regret accounting, the action-masking logic and its policy-table floor lookup, the full Gymnasium-style environment wiring these together with the reward formula, the masked DQN agent itself, all four tuned non-RL baselines, and the scenario comparison harness. This is the project's spine and it works end-to-end on the benign (S1) scenario.

**Partial / stubbed:** the threat forecaster currently only has a simple, non-learned fallback (an exponentially-weighted moving average over a placeholder signal) — the real trained forecaster (offline-trained on RT-IoT2022, with the dual threat/pool heads) does not exist yet. **Updated 2026-08-31 (§8A, Addition E, Phase 1 DONE): the RT-IoT2022 feature-extraction pipeline that will feed that real forecaster's training is now real, tested, and run against the full 123,117-row dataset (3,846 windowed feature examples saved to `data/processed/rt_iot2022/`) — but the LSTM model itself (`forecaster/model.py::SmartKeyForecaster`), its training loop (`forecaster/train.py`), and its integration via `LSTMForecastProvider` remain real `NotImplementedError` stubs / not started, and `env/forecast_provider.py`'s `MovingAverageForecaster` remains the system's active forecaster.** The request stream is currently a random synthetic generator, not yet sampled from a real tenant graph. Scenario dispatch beyond S1 (S2 through S6) is defined in configuration but not yet acted on by the environment.

**Not started:** the soft-reward baseline agent (the steering attack's target), the adversarial threat-trace generator and the steering attack itself, the live dashboard in any form (all 7 panels in §7 are currently only realized as the illustrative HTML mockup, not a working system), the AWS-KMS-style API facade, and the written report/paper content itself.

**Dashboard-specific:** every panel described in §7, including both new ones (Threat Input, Explain Decision), exists today only as a static, hand-authored HTML/CSS/JS mockup with fabricated example data — it demonstrates the intended interface, not working software. None of its numbers, thresholds, or example decisions were computed by the real pipeline.

**Implication for paper drafting:** describe the system's *design* — architecture, reward formulation, masking mechanism, dashboard design intent — in the present/normative tense ("the environment computes...", "the dashboard is designed to..."). Describe anything not yet built as planned/future work, explicitly. Do not write a results or evaluation section with specific numbers; if a placeholder is needed, mark it clearly as `[TODO: pending real experiments]` rather than filling it with a plausible-looking number.

---

## 11. Timeline & Cut Lines

The three non-negotiable deliverables remain: a working masked-DQN pool-budgeting agent, the steering-attack result, and a report. This document's additions (Panels 1 and 3, and the replay-pcap pathway) are scoped as **low-cost, high-narrative-value dashboard work** — they surface values the rest of the system already computes, and should not be allowed to compete for time against the steering attack or the four mandatory baselines.

**Priority note, added 2026-08-31:** §8A's Addition E (the real LSTM threat-forecaster build-out — RT-IoT2022 feature-extraction pipeline DONE, LSTM build/train/validate/wire-in remaining) is now the **top named priority for the next several sessions**, ahead of dashboard polish, the API facade, and any other stretch work described elsewhere in this document. Pick up Addition E's Phase 2 (build + train the LSTM) next unless explicitly redirected.

**Cut order the instant a gate slips (in order, first cut first):**
1. True live network capture (was never a guaranteed deliverable — §5.5, Hard Rule 11).
2. S6 migration wave (Panel 6) → lose one experiment, keep the thesis.
3. The LSTM head of the foresight ablation → keep the EWMA/replayed-pcap-driven variant.
4. One of S2/S4.
5. The dose-response sweep in the steering attack → keep the single-point steering result.

**Never cut:** the pool simulator, action masking, deferral/regret accounting, the four baselines, the steering attack itself, and — per this document — the Explain Decision panel (Panel 3), since it is cheap to build once the underlying values already exist and directly supports the project's legibility argument (§3, motivation layer 4).

**Scope-creep tripwires (say no immediately):** agent choosing migration order; a security term added "just to stabilize training"; real AWS deployment; a second agent / hierarchical RL; per-flow crypto benchmarking side-quests; **building an actual live packet-capture subsystem** (the replay-based approach delivers the same demo experience at a fraction of the risk and engineering cost).

---

## 12. Anticipated Examiner / Reviewer Questions

- *"Why RL instead of a threshold rule?"* → The tuned threshold baseline reacts to the *current* pool level; the agent (once the forecaster is real) acts on the *forecast* — pool trajectory and incoming demand. The foresight ablation quantifies exactly how much anticipation is worth.
- *"Isn't this just a recommender system?"* → A recommender optimizes each request in isolation; this system can't, because serving hybrid now removes an option later — decisions are coupled through the pool.
- *"Why would the agent ever serve classical when PQC is nearly free?"* → Legacy endpoints flagged as not PQC-capable (migration-era realism), and measured marginal cost under load. Policy floors — not the agent — decide *whether* classical is acceptable; the agent only economizes above the floor.
- *"Isn't the threat score manipulable?"* → Yes — that's the steering attack (Panel 5). Manipulation only ever raises floors. The soft-reward baseline is the cautionary tale, on screen.
- *"Where do the security tiers come from?"* → NIST PQC categories, SP 800-57 lifetimes, CNSA 2.0/BSI timelines, ETSI GS QKD 014 — no invented constants (Hard Rule 4).
- *"Why should I trust the dashboard's explanation panel — couldn't it just be telling a good story after the fact?"* → No: Hard Rule 10 constrains it to display only values the pipeline actually computed, templated deterministically, never generated freeform. The panel is a window into real intermediate state (the floor table lookup, the mask, the cost comparison), not a summary written after the decision was already made.
- *"Could this ingest live network traffic?"* → The guaranteed deliverable is a captured file replayed at real-time pace through the identical pipeline a live interface would use — behaviorally indistinguishable to a viewer, but reproducible and safe for a graded demo. True live capture is a named stretch goal, explicitly not promised, for the reasons in §11.
- *"Why not deploy on real AWS?"* → No research value; every result runs identically locally; the API facade is about the deployment *story*, not a research contribution.

---

## 13. Naming & Framing Cheatsheet

- **Project:** SmartKeyNet: RL for Hybrid Cryptography ✔ (system's whole job = choosing among classical/PQC/hybrid key material per request).
- **Mentor-pitch mapping:** "ML model on network features predicts a threat score" = the forecaster (§7.1/§8) · "assign it a key" = the tier policy table (§7.3 step 3) · "DQN for optimal boundaries" = the masked DQN optimizing within floors (§7.3 steps 4–6).
- **One-sentence pitch:** *"SmartKeyNet is the decision layer for a multi-tenant cloud KMS in the hybrid era — classical, post-quantum, and quantum-backed keys served per request by a DQN that budgets scarce quantum resources, with security floors it structurally cannot violate, and a dashboard that shows exactly why every decision was made."*
- **Dashboard tagline (Panel 5's caption, doubles as thesis statement):** *"Security isn't in our reward, so it isn't for sale."*

---

## 14. Guidance for drafting the "rough base paper" from this document

If you are an AI assistant using this file (plus the attached dashboard HTML mockup) to draft a base paper, here is the recommended split:

**Write confidently now (design is settled, cite this document's sections):**
- Abstract — frame as system design + a demonstrated steering-attack result + planned full evaluation, not a completed empirical study.
- Introduction — §2, §3 (HNDL motivation, operational budgeting problem, soft-reward attack surface, legibility argument).
- Related Work — soft-reward RL-for-crypto prior work (the system this project critiques and reproduces as a baseline), QKD-backed KMS deployments, constrained RL / action masking literature.
- System Design / Architecture — §5 (state, reward, architecture diagram), §6 (dataset provenance), §7 (all 7 dashboard panels, described as design intent — the HTML mockup may be used as a **figure showing the interface design**, explicitly captioned as an illustrative mockup, never as a results figure).
- Methodology — §9 (scenario grid), the four mandatory baselines, the dose-response sweep protocol for the steering attack, the metrics that will be reported (regret events, pool-exhaustion events, forced-rekey ratio, floor violations, p99 latency).
- Threat Model / Explainability Design Rationale — §5.5 (Hard Rules 10–11), §7.3, §8.

**Write as planned/future work, explicitly, not as findings:**
- Results / Evaluation section — every number must come from a real experiment; until then, state what *will* be measured and how, per §9's table, rather than presenting any number.
- Any screenshot or description of the dashboard actually running — describe it as the target interface (§7), not as evidence of a working system, until §10 says otherwise.
- Steering attack outcome — describe the experimental design (§7.5, §9's S5 row) precisely; do not pre-state a result.

**Do not do:**
- Do not invent p99 latency numbers, regret counts, forced-rekey ratios, or dose-response curves. Every such number in the attached HTML mockup is fabricated for layout purposes only (see this document's header).
- Do not describe the dashboard, the LSTM forecaster, the soft-reward baseline, the tenant graph, or any of S2–S6 as implemented — see §10 for exactly what is and isn't real as of this document.
- Do not remove, soften, or silently reinterpret any of the eleven Hard Rules in §5 — they are the paper's actual argument, not incidental implementation details.
