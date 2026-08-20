# SmartKeyNet: RL for Hybrid Cryptography

*Report draft. Every number in this document comes from a run in this
repository and is reproducible with the command given beside it. Where an
experiment did not produce the result the design anticipated, the result
is reported as measured — see §5.1, which is a negative finding.*

**Cross-reference:** every claim here must be consistent with PLAN2 §5.4/§5.5's
eleven Hard Rules. Where a result appears to contradict one, the result is
wrong and the environment or agent gets fixed — never the rule. §5.1 and §7
record four occasions this session where that is exactly what happened.

---

## Abstract

SmartKeyNet is a decision layer for a multi-tenant cloud KMS in the hybrid
era: for every key request it chooses among classical, post-quantum
(ML-KEM-768) and hybrid (ML-KEM ⊕ QKD-sourced) key material, budgeting a
finite QKD pool that refills slowly. Its defining architectural commitment is
that **security is never in the reward**. A threat forecaster and a policy
table set a per-request minimum tier floor, and every action below that floor
is removed from the action set before the agent evaluates it.

We reproduce a published-style soft-reward design, in which protection is a
preference priced against cost, and show it can be **steered**: under an
adversarially suppressed threat signal its share of key establishments below
the sensitivity-class floor rises from 14.0% to 27.8%, while the masked
architecture stays at exactly 0.0% at every attack strength — structurally,
not as a measured outcome. That is the headline contribution and it holds.

The secondary claim — that a masked DQN outperforms tuned non-RL baselines at
budgeting the pool — **did not hold**. Across five training seeds with
checkpoint-averaged evaluation, a grid-searched threshold policy beats the
agent on both S1 and S3 by a wide margin. §5.1 reports the numbers and the
diagnosis.

A dual-head LSTM forecaster trained on RT-IoT2022 reaches balanced accuracy
0.931 against a 0.682 base rate — and then, in the foresight ablation, makes the
system dramatically **worse** than the parameter-free EWMA fallback it was meant
to replace (§5.5). Anticipation itself is worth a great deal: `ewma` improves on
no-foresight on every operational metric, reaching zero exhaustion and zero
deferral on the held-out migration scenario. The learned head does not.

Across 25 (scenario × policy) cells of the closing comparison, **floor violations
are 0.00 ± 0.00 everywhere** — the one column the architecture actually
promises (§5.4).

---

## 1. Introduction

Networks are entering a decades-long transition in which classical,
post-quantum and quantum-distributed keys coexist. Classical keys are fast and
quantum-vulnerable; PQC keys are quantum-resistant and effectively unlimited;
QKD key material is information-theoretically strong and **scarce**, arriving
at kbps from a physical link into a finite pool. Somebody has to decide, request
by request, which a connection gets. Today that is static configuration.

Three problems motivate replacing it with a learning agent (PLAN2 §3):

1. **Harvest Now, Decrypt Later.** Adversaries record traffic today to decrypt
   once quantum computers mature. NIST PQC standardisation, CNSA 2.0 and
   BSI/ANSSI guidance have started a global migration in which the three key
   types coexist as deployment reality.
2. **Somebody must budget scarce quantum keys.** Always-hybrid drains the pool
   before critical requests arrive; fixed thresholds waste material or starve it
   at the wrong moment.
3. **Soft security rewards are an attack surface.** Prior RL-for-crypto work
   places security in the reward. If protection is a preference, whoever controls
   the signal it is priced against can bid it down.

**Contribution:** *security as constraint (action masking), not reward —
demonstrated against a live steering attack on a reproduced soft-reward
baseline.* §5.2 is that demonstration.

A fourth, legibility motivation (PLAN2 §3.4) rides on the masking design for
free: the floor is a lookup, the mask is a deterministic function, and the pick
is "cheapest legal option", so every decision is auditable without a generated
narrative. §3.5 and §6.3 describe how that is enforced rather than claimed.

---

## 2. Related Work

- **RL for adaptive cryptography.** Q-learning for WSN key management;
  threat-score-driven PQC selection; the Noetzold-style reward structure that
  places security in the objective. This is the design we reproduce as
  `agents/soft_reward_baseline.py` and attack in §5.2. The reproduction is
  deliberately faithful and deliberately competent (§3.4) — the comparison is
  worthless against a strawman.
- **QKD-backed KMS architectures.** ETSI GS QKD 014 key delivery; production
  metro deployments (JPMorgan × Toshiba/Ciena, BT/Toshiba London, SK Telecom,
  EuroQCI). These set the architectural constraints we hold to: QKD is a
  backbone resource behind the KMS, delivered as 256-bit keys from a finite
  store, and endpoints never "do QKD" (Hard Rule 6).
- **Post-quantum migration guidance.** NIST PQC categories (ML-KEM-768 at
  Category 3), SP 800-57 cryptoperiods, CNSA 2.0 and BSI/ANSSI staged
  timelines. Every tier and lifetime constant traces to one of these; none is
  invented (Hard Rule 4).
- **Constrained RL / action masking.** Masking as a structural safety mechanism
  rather than a penalty. Our contribution is not the technique but the argument
  that in this domain it is the *only* honest place for security to live.

---

## 3. Methodology

### 3.1 Environment and MDP

One `env.step()` is one request decision. The agent chooses among
`SERVE_CLASSICAL` / `SERVE_PQC` / `SERVE_HYBRID` / `REUSE` / `REKEY_NOW`.

**Pool simulator** (`env/pool_sim.py`). A documented synthetic SKR/QBER process
drives refill of a finite store; `PoolSim.draw` refuses to under-draw. Sizing is
the single most consequential modelling choice in the repo and is derived from
*measured key-establishment demand* rather than asserted — see §3.6.

**Deferral queue and regret accounting** (`env/deferral_queue.py`,
`metrics/regret.py`). Hard Rule 9: a hybrid-mandatory request the pool cannot
cover is **deferred**, never downgraded. Each deferral onset is a regret event;
each waiting step is a deferred-critical-step charged to the reward.

**Policy table and masking** (`env/masking.py`). A (sensitivity class × threat
posture) → floor lookup, with a deliberately **one-way** ratchet: a threat
signal can raise a floor and nothing can lower it. `compute_mask` then applies
five legality rules. Rules 4 and 5 were added this session after measurement
showed the first three were insufficient (§7.1).

**Reward** (`env/environment.py`), unchanged from PLAN2 §5.3:

```
r = −w_lat·latency − w_en·energy + w_fr·freshness
    − w_qkd·(pool bits consumed)
    − R_starve·(deferred_critical_steps)
    − c_rekey(load)·1[rekey]
```

There is no security term. There has never been one. The single quarantined
exception in the repository is `agents/soft_reward_baseline.py`, which exists
to be attacked.

### 3.2 Tenant graph and request stream

`env/request_generator.py` builds a NetworkX graph of `(tenant, service)` nodes
with flow edges carrying `sensitivity_class`, `traffic_rate` and `pqc_capable`.
Four tenant profiles (hospital, fintech, logging, iot-telemetry) differ in
sensitivity mix, PQC capability and traffic scale — which is what gives S4
("one *low-sensitivity* tenant floods the API") something to mean. Tenants peer
through a shared observability hub, so pool contention is genuinely
cross-tenant.

Hard Rule 3 is testable here and tested: `RequestGenerator.stream()` and the
graph-free `random_request_generator` are both `Iterator[Request]` at the same
aggregate rate, and no `StateDict` carries tenant identity or any graph
reference (`tests/test_environment.py`).

### 3.3 Dual-head forecaster (Addition A)

One shared LSTM trunk over `[threat_features(16) | pool_signals(4)]`, with a
threat head (a "now" logit plus k=5 horizon logits) and a pool head (three
quantities × three horizons H ∈ {10, 25, 50}).

It is trainable as a *genuinely shared* trunk because
`forecaster/dataset.build_rollout_dataset` injects real RT-IoT2022 feature
windows into baseline rollouts, so every step carries a threat label **and** a
pool target from the same sequence.

Sixteen flow features are extracted by **one implementation with two callers**
(Hard Rule 11): the same `extract_flow_features` serves the training CSV and the
pcap ingestion path. Standardization is **benign-referenced**, which is not a
detail — see §5.3.

The forecaster is trained offline and **frozen**: `eval()` mode,
`requires_grad=False`, every forward under `torch.no_grad()`, no optimizer, no
training entry point. A test drives twenty updates and asserts not one weight
moved. If a gradient could reach these weights, the agent could learn to shape
its own threat signal and therefore its own floors — the exact failure mode the
architecture exists to rule out.

### 3.4 Agent and baselines

**Masked DQN** (`agents/dqn.py`). Masking is applied at both action-selection
and bootstrap-target time, so the network is never even implicitly trained to
value an action the environment would have forbidden.

**Four tuned baselines** (Hard Rule 7): always-PQC, always-hybrid, grid-searched
static threshold, random. All three tier policies are **reuse-aware** — they
reuse a live session key and only choose a tier when key material must actually
be established. §5.1 explains why that correction was necessary and why it cuts
against the agent.

The threshold is grid-searched on `total_reward` (the agent's own objective, not
a coarse latency proxy) averaged over multiple eval seeds. Both choices are
deliberately in the baseline's favour.

**Soft-reward baseline** (`agents/soft_reward_baseline.py`). A tabular
Q-learning agent whose reward is
`w_sec·threat_score·tier_strength − costs`, with **no mask**. It is trained on
an *honest* signal and attacked only at evaluation, so the claim tested is
inference-time steering rather than training-time poisoning — and it arrives at
evaluation having learned the correct threat-to-tier relationship.

### 3.5 Decision trace (Hard Rule 10)

`env/decision_trace.py` assembles PLAN2 §7.3's six steps directly off the live
environment, policy table and mask. There is no model, no heuristic and no
narration anywhere in that path. Both `api/` and `dashboard/` import it; neither
reimplements it, because two routes to a number are two chances to disagree.

It degrades honestly in both directions PLAN2 requires: step 5 says "no cost
tradeoff existed here" when only one action was legal, and step 6 says plainly
when several options cleared the floor and the policy's own learned preference
chose among them.

### 3.6 Environment calibration, and why it is stated here

The pool refill rate decides whether QKD is scarce at all, so it is derived and
published rather than tuned. Measured on the 50-node graph over 2,000-step
episodes with an unlimited pool (these are *demands*, not outcomes):

| quantity | bits/step |
|---|---|
| floor-mandated hybrid demand, S1 (CALM posture) | 4.66 |
| floor-mandated hybrid demand, S2 (HIGH posture) | 12.28 |
| maximal hybrid demand (always-hybrid) | 20.98 |
| **configured refill** | **15.0** |

Refill must sit strictly inside that bracket: at or below mandated demand every
policy drowns in deferrals regardless of skill; at or above maximal demand
nothing can exhaust the pool and budgeting skill is irrelevant. The second case
is not hypothetical — it was this repository's state as received (§7.2).
`tests/test_pool_sim.py` pins the bracket so the premise cannot regress silently.

---

## 4. Experiments

| # | Scenario | What changes | Train/Eval |
|---|---|---|---|
| S1 | Benign baseline | steady mixed traffic | train |
| S2 | HNDL posture | threat ramps, floors ratchet up | train |
| S3 | QKD degradation | QBER↑, SKR↓, refill collapses to 15% | train |
| S4 | DDoS / noisy neighbour | iot-telemetry floods ×14 | train |
| S5 | **Steering attack** | adversarially suppressed threat trace | eval |
| S6 | **Migration wave** | scripted cohort floor ratchet | **held-out eval** |
| E-A | Foresight ablation | `off` / `ewma` / `lstm` | eval |

Distinguishing behaviour was verified per scenario before any result was
reported (250-step probes, seed 0): S2 drives posture CALM → HIGH and lifts mean
floor 0.40 → 1.08; S3 drops mean SKR 0.2012 → 0.0995 and multiplies always-hybrid
exhaustion 6 → 24; S4 lifts iot-telemetry's request share 5.6% → 36.8%.

**Reporting protocol.** `PROGRESS.md` documents an unresolved training-stability
finding, five diagnostic sessions deep: this setup's greedy policy oscillates
checkpoint-to-checkpoint by large margins with an unidentified mechanism. Every
DQN number in §5 is therefore (a) **checkpoint-averaged** over the last four
`eval_every` windows, (b) **eval-seed-averaged** within each snapshot, and (c)
**repeated across multiple training seeds with the spread reported**.
`SeedSpread` makes the spread a required field so a mean cannot be published
without it.

Hard Rule 8 is enforced in code: `experiments/train.py` and
`forecaster/train.py` both refuse S6 outright.

---

## 5. Results

### 5.1 Gate W3 — masked DQN vs tuned threshold. **Negative.**

`python -m experiments.campaign` — 5 training seeds, checkpoint-averaged over
steps 17,500–25,000, 5 eval seeds per snapshot, 2,000-step episodes.

**S1** (`total_reward`, higher is better):

| policy | mean ± stdev | range |
|---|---|---|
| **masked DQN** | **−3820.8 ± 1623.9** | [−6343.1, −2123.1] |
| **static-threshold (tuned)** | **−955.5 ± 7.7** | [−964.7, −948.6] |
| always-PQC | −872.0 ± 7.8 | [−880.7, −862.7] |
| always-hybrid | −8890.8 ± 2680.9 | [−13121.6, −6396.2] |
| random | −61401.9 ± 11422.6 | [−75540.0, −47189.2] |

**S3**:

| policy | mean ± stdev |
|---|---|
| **masked DQN** | **−97475.6 ± 204475.5** |
| **static-threshold (tuned)** | **−945.8 ± 8.6** |
| always-PQC | −872.0 ± 7.8 |
| always-hybrid | −35509.9 ± 14701.0 |

**The tuned threshold beats the masked DQN on both S1 and S3.** PLAN2 §3.2's
disqualification rule ("if a simple threshold policy matches the DQN in
evaluation, the project premise fails") is triggered. This is reported as
measured; no reward term was added, no masking weakened, and no environment
parameter was moved toward the agent afterwards.

**Diagnosis.** Two things are visible in the same table:

- The agent **did** learn proactive rekeying — `forced_rekey_ratio` 0.107–0.211
  against 0.93–1.00 for every baseline. Four prior diagnostic sessions chased
  exactly this behaviour and never obtained it; the environment corrections in
  §7 appear to be why it now appears.
- But it rekeys **~6× too often**: `rekeys_per_100_requests` 66.8 vs 10.6. At
  roughly 2.5 reward units per rekey that accounts for almost the entire S1 gap.

The documented instability is present at full amplitude: within-run
`total_reward` stdev is **1447 ± 1411** on S1 — comparable to the mean itself.
Per the standing instruction this was not investigated further; it remains the
open question in `PROGRESS.md`.

**What this does not show.** It does not show that masking is unnecessary or that
the architecture is wrong — §5.2 is unaffected and is the project's actual
contribution. It shows that *this* DQN, at *this* training budget, under *this*
reward, does not out-budget a well-tuned threshold. A candidate explanation
worth testing next is the reward's heavy tail: with a genuinely scarce pool a
single step can score −442 while a typical step scores −0.4, which is a hard
regression target for an unclipped DQN.

### 5.2 S5 steering attack — the headline result. **Holds.**

`python -m attack.run_attack` — 2,000-step episodes, seed 0, scored at key
establishments (protection is set when a key is established, not on the cache
hits that follow).

| dose | signal suppressed | soft-reward: below class floor | masked: below class floor |
|---|---|---|---|
| 0.00 | 0.0% | 14.0% | **0.0%** |
| 0.25 | 25.0% | 24.2% | **0.0%** |
| 0.50 | 50.0% | 27.8% | **0.0%** |
| 0.75 | 74.9% | 27.8% | **0.0%** |
| 1.00 | 97.9% | 27.8% | **0.0%** |

Served-tier histogram for the soft-reward arm slides **HYBRID 128 → 0**,
CLASSICAL 48 → 49, PQC 60 → 178. The masked arm is identically zero at every
dose — by construction, not by luck.

The measured quantity is the share of establishments **below the
sensitivity-class floor**: the protection level the policy table's CALM row
guarantees, which the one-way ratchet means no threat signal can lower. Both
arms are asked the same question — "you must establish key material for this
request; which tier?" — under different masks: the masked arm under
`mask & tier_only`, the soft-reward arm under `tier_only` alone. That difference
*is* the thesis.

Three qualifications, all reported rather than smoothed:

- The soft-reward arm is **already at 14.0% with a perfectly honest signal**. It
  has no floor, so the attack roughly doubles a failure rate that was never
  zero — a stronger statement than anticipated, not a weaker one.
- The curve **saturates from dose 0.5**, because the tabular reproduction bins
  the threat score into five buckets. That is a property of the reproduction's
  discretization, not of the attack.
- `below_escalated_floor_share` — the share below the floor the *honest* posture
  would have set — is **non-zero for the masked arm too (21.1% at full
  suppression)**. Suppression genuinely does prevent floors from *escalating*;
  it simply cannot push them below the class floor. Scoring the architecture
  only against the claim it makes would be as misleading as scoring it against
  one it does not.

### 5.3 Forecaster

`python -m forecaster.train` — 8 epochs, 10,253 train / 2,563 validation windows.

| head | metric | value |
|---|---|---|
| threat | accuracy | 0.9461 |
| threat | **balanced accuracy** | **0.9312** |
| threat | *majority-class rate (the number to beat)* | *0.6817* |
| pool | validation MAE (unit-RMS-scaled) | 0.189 |

Balanced accuracy and the base rate are reported together deliberately: raw
accuracy alone is uninformative on this label mixture and would flatter the
model.

**Benign-referenced standardization is load-bearing.** RT-IoT2022 is ~90% attack
rows, so standardizing against the whole capture centres the feature space on
attack traffic and **inverts the threat signal** — measured separation Cohen's
*d* = **−0.98** (benign scoring as *more* threatening) versus **+4.43** once
benign is the reference.

### 5.4 Closing comparison table

`python -m experiments.results_table` → `results/closing_table.json`. Agents
trained on **S1 only**, 5 training seeds, checkpoint-averaged; every policy
evaluated on the same 5 eval seeds over 2,000-step episodes. **S6 is held out —
no agent in this table has ever trained on the migration schedule (Hard Rule 8).**

| scenario | policy | pool exhaustion | regret events | forced-rekey ratio | floor violations |
|---|---|---|---|---|---|
| **S1** | masked DQN | 0.96 ± 2.15 | 0.96 ± 2.15 | **0.107 ± 0.10** | **0 — structural** |
| | static-threshold (tuned) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.965 ± 0.01 | 0 — structural |
| | always-hybrid | 104.80 ± 21.21 | 104.80 ± 21.21 | 0.954 ± 0.01 | 0 — structural |
| | always-PQC | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.931 ± 0.01 | 0 — structural |
| | random | 344.40 ± 22.40 | 344.40 ± 22.40 | 0.051 ± 0.00 | 0 — structural |
| **S2** | masked DQN | 211.44 ± 290.46 | 211.44 ± 290.46 | 0.196 ± 0.12 | **0 — structural** |
| | static-threshold (tuned) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.864 ± 0.02 | 0 — structural |
| | always-hybrid | 203.00 ± 29.83 | 203.00 ± 29.83 | 0.942 ± 0.01 | 0 — structural |
| | always-PQC | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.861 ± 0.02 | 0 — structural |
| | random | 773.20 ± 52.68 | 773.20 ± 52.68 | 0.075 ± 0.00 | 0 — structural |
| **S3** | masked DQN | **0.00 ± 0.00** | **0.00 ± 0.00** | **0.050 ± 0.03** | **0 — structural** |
| | static-threshold (tuned) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.965 ± 0.01 | 0 — structural |
| | always-hybrid | 117.60 ± 28.55 | 117.60 ± 28.55 | 0.946 ± 0.01 | 0 — structural |
| | always-PQC | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.931 ± 0.01 | 0 — structural |
| | random | 348.40 ± 25.44 | 348.40 ± 25.44 | 0.051 ± 0.00 | 0 — structural |
| **S4** | masked DQN | **0.00 ± 0.00** | **0.00 ± 0.00** | **0.046 ± 0.03** | **0 — structural** |
| | static-threshold (tuned) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.968 ± 0.01 | 0 — structural |
| | always-hybrid | 87.20 ± 11.65 | 87.20 ± 11.65 | 0.968 ± 0.01 | 0 — structural |
| | always-PQC | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.939 ± 0.01 | 0 — structural |
| | random | 302.20 ± 11.03 | 302.20 ± 11.03 | 0.046 ± 0.00 | 0 — structural |
| **S6** *(held out)* | masked DQN | **0.00 ± 0.00** | **0.00 ± 0.00** | **0.073 ± 0.04** | **0 — structural** |
| | static-threshold (tuned) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.942 ± 0.01 | 0 — structural |
| | always-hybrid | 144.20 ± 19.87 | 144.20 ± 19.87 | 0.950 ± 0.01 | 0 — structural |
| | always-PQC | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.915 ± 0.01 | 0 — structural |
| | random | 514.00 ± 77.83 | 514.00 ± 77.83 | 0.069 ± 0.01 | 0 — structural |

**Floor violations are 0.00 ± 0.00 in all 25 cells.** That column is the one the
architecture actually promises, and the label "structural" is earned rather than
asserted: `env/masking.py`'s five legality rules make a below-floor delivery
unrepresentable, and `experiments/harness.py` counts *delivered tier* rather than
chosen action, so the counter would notice if a rule were removed — which is
exactly how §7.1's two violations were found.

**p99 latency is omitted from the table because it is uninformative here**:
every policy in every scenario scores exactly 1.500. It takes one of four
discrete values and saturates at the top whenever any hybrid serve occurs, which
is essentially always. Reporting five identical columns would imply a comparison
that does not exist. `total_reward` (§5.1's metric) is likewise not in PLAN2
§7.7's column list and is reported there instead.

**What the table shows that §5.1 does not.** On the operational metrics the
agent looks good: it is at **zero pool-exhaustion and zero regret events on S3,
S4 and the held-out S6**, matching the tuned threshold and beating always-hybrid
by two orders of magnitude — while rekeying **proactively** (forced-rekey ratio
0.046–0.196 against 0.86–0.97 for every non-random baseline). Two caveats keep
this from being a win:

- On **`total_reward`, which is what §5.1 compares, the tuned threshold still
  wins.** The agent buys its zero-exhaustion record with ~6× more rekeys than the
  baselines, and rekeys are not free. Both facts are true; neither cancels the
  other, and the table's metric list simply does not price the rekeys.
- **S2 is bad and unstable**: 211.44 ± 290.46 exhaustion events, a standard
  deviation larger than the mean. That is the documented checkpoint/seed
  instability, and S2 (sustained HIGH posture) is where mandated hybrid demand is
  highest, so an over-spending seed has the least headroom.

### 5.5 E-A foresight ablation

`python -m experiments.ablation` → `results/foresight_ablation.json`. Same
architecture with the forecast varied `off` / `ewma` / `lstm`; trained on **S3**,
5 seeds, checkpoint-averaged; evaluated on S3 and the held-out S6. All three arms
run on `threat_input.source: rt_iot2022` so the comparison is of forecasters, not
of their inputs (§7.5).

| evaluated on | metric | `off` | `ewma` | `lstm` |
|---|---|---|---|---|
| **S3** | total_reward | −44,843.6 ± 75,685.2 | **−4,669.1 ± 5,568.7** | −9,766,367.2 ± 1,723,435.5 |
| | pool exhaustion | 23.61 ± 23.30 | **2.00 ± 4.47** | 787.25 ± 99.38 |
| | regret events | 23.61 ± 23.30 | **2.00 ± 4.47** | 787.25 ± 99.38 |
| | deferred critical steps | 4,223.2 ± 7,585.5 | **220.6 ± 493.3** | 976,242.3 ± 172,232.6 |
| | forced-rekey ratio | 0.330 ± 0.27 | 0.329 ± 0.32 | 0.169 ± 0.08 |
| **S6** *(held out)* | total_reward | −5,164.5 ± 7,154.4 | **−2,165.7 ± 1,175.0** | −10,944,541.1 ± 5,066,928.8 |
| | pool exhaustion | 21.88 ± 48.93 | **0.00 ± 0.00** | 890.56 ± 256.69 |
| | regret events | 21.88 ± 48.93 | **0.00 ± 0.00** | 890.56 ± 256.69 |
| | deferred critical steps | 307.2 ± 686.9 | **0.00 ± 0.00** | 1,094,059.8 ± 506,541.1 |
| | forced-rekey ratio | 0.326 ± 0.35 | 0.293 ± 0.33 | 0.136 ± 0.13 |

Deltas against `off`, which is the quantity E-A exists to measure:

| | total_reward | regret events |
|---|---|---|
| S3 `ewma` | **+40,174.5** | **−21.6** |
| S3 `lstm` | −9,721,523.6 | +763.6 |
| S6 `ewma` | **+2,999.0** | **−21.9** |
| S6 `lstm` | −10,939,376.6 | +868.7 |

**Anticipation is worth a great deal — and the *learned* forecaster is a
disaster.** `ewma` improves on `off` on every operational metric, on both the
training scenario and the held-out one, and reaches **zero exhaustion and zero
deferral on S6**. The `lstm` arm is worse than `off` by two to three orders of
magnitude on every one of them.

This is the sharpest negative result in the project, and it is not the one the
plan anticipated (PLAN2 §11's cut-order treats the LSTM head as the *optional*
part and the EWMA as the fallback; on this evidence the fallback is the better
system).

**Hypothesis, explicitly not verified here.** The pattern is consistent with an
interaction already observed and recorded independently (§6.2, last bullet): the
policy table's ratchet is deliberately **one-way**, so *any* posture escalation is
permanent for the rest of an episode, and a single momentary detector peak — one
decision in 2,000 — is enough to trip it. The LSTM threat head is a genuinely
*more sensitive* detector than the EWMA (balanced accuracy 0.931 on real
traffic, §5.3), so it should trip the ratchet more readily and earlier. Under
sustained HIGH posture, mandated hybrid demand rises to ~12.28 bits/step (§3.6)
while S3's degradation collapses refill to ~2.25 bits/step, and the deferral
queue then diverges — which is what 976,242 deferred critical steps looks like.

If that mechanism is the right one, the finding is *"a better detector produces a
worse operational outcome, because the ratchet is one-way and the pool cannot
fund the floors it raises"* — a real and reportable interaction between two
design choices that are each individually defensible. **It has not been
confirmed.** Confirming it needs a posture-trajectory probe per arm, which was
not run; it is the first item under "Next task" in `PROGRESS.md`. Until then the
numbers above stand as measured and the explanation stands as a hypothesis.

---

## 6. Discussion

### 6.1 Anticipated examiner questions

**"Why RL instead of a threshold rule?"** On this evidence, for pool budgeting,
you should not — §5.1 is a negative result and is reported as one, and §5.5 says
the same about the learned forecaster against a parameter-free EWMA. The
architecture's demonstrated value is §5.2: the *masking* is what resists
steering, and that property belongs to the architecture rather than to the
learned policy. A tuned threshold behind the same mask would be equally immune;
that is the point, not a weakness.

**"Isn't the threat score manipulable?"** Yes — that is §5.2, run as a
dose-response experiment rather than asserted. Manipulation only ever raises
floors.

**"Where do the security tiers come from?"** NIST PQC categories, SP 800-57
cryptoperiods, CNSA 2.0 / BSI-ANSSI timelines, ETSI GS QKD 014's 256-bit keys.
The one floor-table change made this session (§7.3) is argued from SP 800-57 and
CNSA 2.0's "protect longest-lived data strongest and soonest", and it is a
*raise*.

**"Why trust the explanation panel?"** Hard Rule 10 is enforced structurally:
one assembler reading live objects, no generative step, and a test asserting that
every number in a step's summary appears in that step's own computed values.

### 6.2 Limitations

- **§5.1 is negative**, and the training instability behind it is unresolved
  after six sessions.
- **§5.5 is also negative, and more sharply so.** The trained LSTM forecaster is
  two to three orders of magnitude worse than the EWMA fallback on every
  operational metric. The mechanism is hypothesised (a more sensitive detector
  trips the one-way ratchet earlier, and the pool cannot fund the floors it
  raises) but **not verified** — the confirming probe was not run.
- **ML-KEM-768 is simulated.** `liboqs` is an optional dependency and is not
  installed; every API response says so, and `GET /Health` publishes the full
  primitive-honesty matrix. No quantum-resistance claim is made for the PQC path.
- **QKD material is simulated.** `pool_sim` models key *availability* faithfully
  — a finite store, drawn down for real — but holds a bit count, not bytes.
- **Live pcap capture is out of scope** (Hard Rule 11, PLAN2 §11 cut-order item
  1). The shared feature extractor exists and is tested; the upload/replay
  endpoints are not built.
- **With a realistic noisy detector, the one-way ratchet saturates.** A single
  momentary peak — one decision in 2,000 — is enough to change the floor regime
  for the rest of an episode. That is Hard Rule 2 working as specified, but it is
  an operational property worth surfacing.

### 6.3 What the negative result does and does not cost

The project's stated contribution (PLAN2 §2.3) is *"security as constraint, not
reward — demonstrated against a live steering attack"*. That is §5.2 and it
stands. §5.1 was a supporting operational claim, and its failure is informative:
it says that in this environment, at this budget, the *learning* adds nothing to
the *masking*. Reporting it that way is more useful than a tuned win would have
been.

---

## 7. Corrections made this session

Four defects were found by measurement rather than by reading, each fixed in the
direction that makes the agent's case harder, and each frozen before any training
run so none could be post-hoc tuning. Full detail in `SESSION_LOG.md` 2026-08-19.

### 7.1 Two Hard Rule 2 violations

`compute_mask` gated `REUSE` on key *age* only, never on the tier the existing
key delivers. With the one-way ratchet, a session that established a PQC key
under CALM kept reusing it after the floor rose to hybrid. **Measured on S2:
275 of 1,788 REUSE decisions — 15.4% — delivered key material below the
request's current floor**, while the harness reported `floor_violations: 0`
because it compared the *chosen action* against the floor and REUSE is not a
tier action. `REKEY_NOW` was a second such path. Closed as masking rules 4 and 5,
with the counter corrected to measure delivered tier.

### 7.2 The scarce resource was not scarce

Refill funded ~793 hybrid keys per key consumed. `AlwaysHybridPolicy` produced 0
exhaustions on S1, and every grid threshold was byte-identical to always-hybrid
(both `total_reward = −64854.6`), so the mandatory tuned baseline was not a
distinct policy. Re-derived from measured establishment demand (§3.6). Two
further defects surfaced underneath: baselines never reused a live session key
(a tier action on 250/250 decisions while REUSE was legal on 244), and the
request queue ran at utilisation ρ = 1.0 — unbounded by definition, and the cause
of a DQN loss excursion from 2.6e4 to 7.9e5.

### 7.3 The floor table contradicted its own documented intent

`env/masking.py`'s comment states S3 gets hybrid "even before any threat
elevation"; the table gave it PQC. Since S1/S3/S4 all run at CALM posture, **no
scenario Gate W3 was measured on ever mandated a QKD draw at all** — and with
security absent from the reward, "never spend the pool" was optimal by
construction. Corrected to hybrid (a raise). Made *after* the first Gate W3 run
came back negative; both the before and after results are reported, and the
verdict did not change.

### 7.4 The EWMA threat squash made CALM unreachable

`sigmoid(mean)` over non-negative placeholder features could never fall below
0.5 — the ELEVATED anchor — so with the one-way ratchet every episode pinned at
ELEVATED from its second tick (249/250 decisions on benign S1) and the floor
table's entire CALM row was dead. S2's "threat elevates → floors ratchet up" is
only demonstrable because of the recalibration.

---

## 8. Conclusion

Security placed in a reward can be bought back out of it. We reproduced that
design faithfully and steered it: its share of below-floor key establishments
roughly doubles under a suppressed threat signal, while an architecture that
enforces the same protection by masking stays at exactly zero, at every attack
strength, by construction.

Two accompanying claims did not survive measurement, and we report them that
way. Reinforcement learning does not budget the scarce quantum pool better than a
tuned threshold on `total_reward` (§5.1), though it does reach zero exhaustion
and zero regret on three of five scenarios while rekeying proactively (§5.4). And
the trained LSTM forecaster is far worse than the parameter-free EWMA it was
meant to replace (§5.5) — the most surprising result here, and the one most
worth chasing next.

Neither touches the masking result. That is the point of putting security in a
constraint rather than a preference: it holds regardless of how well the learned
parts learn.

---

## References

*Cite as built; no security constant in this system is invented (Hard Rule 4).*

- NIST FIPS 203 (ML-KEM) and the NIST PQC security categories.
- NIST SP 800-57 Part 1, cryptoperiods and key lifetimes.
- NSA CNSA 2.0 migration timelines; BSI and ANSSI post-quantum guidance.
- ETSI GS QKD 014, key delivery API and 256-bit key sizes.
- Sharafaldin et al. / RT-IoT2022 (UCI ML Repository), real IoT intrusion flows.
- Published liboqs / pqm4 primitive benchmarks (for latency/energy cost tables —
  see §6.2: the tables currently in `env/environment.py` are explicitly-labelled
  placeholders, not measured figures).
