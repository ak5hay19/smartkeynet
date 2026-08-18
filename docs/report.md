# SmartKeyNet: RL for Hybrid Cryptography

**A decision layer for a multi-tenant KMS in the hybrid-cryptography era, with
security floors it structurally cannot violate.**

> Every number here was produced by code in this repository and is reproducible
> from `results/*.json`. Where a result is negative, or where the project's own
> success criterion was not met, it is reported as such. Commands that regenerate
> each table are given inline.

---

## Abstract

Networks are entering a decades-long transition in which classical, post-quantum
and QKD-distributed key material coexist. Somebody must decide, request by
request, which each connection gets — today, static config files do. SmartKeyNet
replaces those rules with a learning agent that budgets scarce QKD key material
across competing tenants, and that is *structurally incapable* of
under-protecting a flow: security is a constraint enforced by action masking,
never a term in the reward.

Three claims, each supported by measurement:

1. **Security-as-constraint resists a steering attack that security-as-reward
   does not.** We reproduce the published soft-reward design and show
   analytically that suppressing its threat signal walks its preferred tier down
   to *classical* — the quantum-vulnerable option. Our architecture has no such
   gradient. Across every agent, dose and seed: **0 floor violations, 0
   posture-ratchet reversals**.
2. **Enforcing that guarantee is harder than stating it.** We found and closed
   **three** distinct paths by which the floor could be bypassed, all of which
   had been reporting `floor_violations = 0` while under-protecting traffic.
3. **RL does not beat a well-tuned static rule here, and accurate threat
   forecasting actively costs availability.** Gate W3 fails. The E-A ablation
   comes out *negative*: an LSTM threat head reaching 0.98 balanced accuracy on
   real RT-IoT2022 traffic doubles S3 regret events (317 → 683), because correct
   detection raises floors, the ratchet makes that irreversible, and the QKD pool
   cannot fund the demand. We report both and explain the mechanism.

---

## 1. Introduction

**Harvest Now, Decrypt Later** is the operational driver: adversaries record
classically-encrypted traffic today to decrypt once quantum computers mature.
NIST's PQC standards, CNSA 2.0 deadlines and BSI/ANSSI guidance have begun a
global migration during which three kinds of key material coexist.

QKD-distributed material is the scarce one. A metro QKD link refills a key pool
at kbps under a variable secret-key rate (SKR) and rising quantum bit error rate
(QBER), while requests arrive in bursts from tenants with very different data
lifetimes. Static rules fail in both directions: *always-hybrid* drains the pool
before critical requests arrive; fixed thresholds waste quantum material or
starve it at the wrong moment.

The research problem is sharper. Published RL-for-crypto work places security in
the *reward* — a threat score contributes reward points, so stronger
cryptography is **bought** rather than **required**. We argue that this is an
attack surface, reproduce it, and demonstrate the attack.

---

## 2. Architecture

```
   CV-QKD trace ──▶ QKD POOL SIM ──┐
                                   ▼
   tenant graph ──▶  ENVIRONMENT (Gymnasium)  ──┬── state ──▶ DQN AGENT
                                                │             reward: latency,
   threat features ─▶ FORECASTER ──▶ POLICY ────┴── mask       energy, freshness,
                      (dual head)     TABLE                    QKD scarcity price
                          │            │                       (NO security term)
                          │            ▼
                          │      ACTION MASKING  ◀── inviolable
                          │
                          └── pool head ──▶ DQN state ONLY (never the policy table)
```

Five actions: `SERVE_CLASSICAL`, `SERVE_PQC`, `SERVE_HYBRID`, `REUSE`,
`REKEY_NOW`. The policy table maps (sensitivity class × threat posture) to a
minimum tier; everything below that floor is removed from the action set before
the agent sees it.

**The asymmetry is the design.** The forecaster's threat head feeds the policy
table, where it can only *raise* floors — the table is monotone in posture and
the ratchet has no downward path. The pool head feeds the DQN's state vector and
nothing else: a forecast that "the pool will be fine" must never relax a floor,
because that would make the floor a function of a learned regression, which is
precisely the design we argue against. `tests/test_forecaster.py` asserts that
`env/masking.py` never imports the forecaster.

---

## 3. Calibrating the environment (and getting it wrong twice)

The scarcity engine is what the thesis rests on, and it was inert.

**First measurement.** On a 2,000-step S1 episode the pool sat at 100% full for
**1,999 of 2,000 steps**, 520 hybrid serves cost nothing, and zero regret events
had ever been logged. Refill ran at 781 keys/step against a structural demand
ceiling of 1 key/step: **ρ = 0.0013**, against a required band of [0.8, 1.3]. No
policy could differ from any other, because the resource was free.

**First correction, still wrong.** We re-sized the link against the
always-hybrid baseline and reached ρ = 1.14 — apparently in band. That number
was misleading. Always-hybrid re-establishes a key on *every decision*; a policy
that reuses its keys draws roughly

```
sessions / key_lifetime  =  20 / 500  =  0.04 keys/step
```

Measured directly under the tuned threshold: **0.043 keys/step**. Against that
"calibrated" refill, ρ = 0.05 — the link was still over-provisioned twenty-fold
for any policy anyone would deploy, and the tuned threshold consequently scored
*identically* on S1 and S3 (−744.4, σ = 8.2). Degradation could not touch it.

**Sizing a link against a villain is a mistake.** Always-hybrid rekeys ~500×
more often than necessary; making *it* struggle leaves everyone else swimming.

**Final calibration**, against measured sensible demand:

| quantity | value |
|---|---|
| refill | 0.098 keys/step (`mean_skr_kbps: 0.025`) |
| demand, tuned threshold (measured) | 0.043 keys/step |
| demand, always-hybrid (villain) | 0.98 keys/step |
| ρ, sensible policy | **0.44** — binds, with room to misuse |
| ρ, always-hybrid | **10.0** — drains immediately |

The gap between those two *is* the budgeting problem. Guarded permanently by
`tests/test_pool_sim.py::test_scarcity_ratio_in_target_band`.

A second, independent bug surfaced only once the pool bound: the QKD scarcity
price was charged **per bit rather than per key**, making it 256× oversized, so
that *starving was cheaper than spending* (one serve cost 256; deferring a
critical request ten steps cost 100). Both fixes were needed for either to
matter. `r_starve ≥ 5·w_qkd` is now asserted at construction.

---

## 4. Three ways the floor could be bypassed

This is the most important engineering result in the project, because the
headline claim is a *structural guarantee* and the guarantee was leaking. All
three bugs lived in actions whose tier is **state-dependent** rather than named
by the action itself, and all three reported `floor_violations = 0`.

| # | hole | measured impact |
|---|---|---|
| 1 | `hybrid_mandatory` never reached the mask — it triggered only the deferral pre-screen | `always_pqc` served hybrid-mandatory requests at PQC |
| 2 | `REUSE` ignored the active key's tier | **1,090 of 3,000** REUSE actions kept a key below the enforced floor (S2 episode) |
| 3 | `REKEY_NOW` refreshed at the session's *current* tier | **461 of 1,500** decisions re-established below the floor (S2 episode) |

The metric read zero throughout because the harness only inspected *serve*
actions. **The guarantee was being satisfied by not looking.**

Two lessons worth carrying:

- *Verify the guarantee; don't assert it.* A "0 violations, structurally
  guaranteed" column is worthless unless the measurement can distinguish
  compliance from blindness.
- *Suspect state-dependent tiers.* All three bugs were in the two actions whose
  tier depends on session state. Any future action of that shape needs the same
  scrutiny.

Fixing (2) also **created the environment's genuine anticipation problem**: a key
provisioned at a high tier while the pool is healthy stays reusable after floors
rise, whereas a cheap one forces a rekey — and a pool draw — at the worst
possible moment. Before the fix a purely myopic policy was optimal, and the
`GreedyRecommenderPolicy` diagnostic tied the DQN exactly (−403.1 vs −403.8).

---

## 5. Gate W3 — the make-or-break comparison, and it fails

**Protocol.** 5 training seeds × 5 evaluation seeds, disjoint (training 0–4,
evaluation 1000–1004), common random numbers within each cell, threshold
grid-searched over all three of its parameters *on training seeds only*.

```bash
.venv/bin/python -m experiments.gate_w3 --train-seeds 5 --eval-seeds 5 --steps 40000
```

### Scenario S1 (benign baseline)

| policy | mean reward | σ | exhaustion events | floor violations |
|---|---|---|---|---|
| **static threshold (tuned)** | **−581** | 9 | 0.0 | 0 |
| greedy recommender | −624 | 9 | 0.0 | 0 |
| always-PQC | −1,024,388 | 127,672 | 276.7 | 0 |
| random | −1,044,540 | 138,084 | 284.3 | 0 |
| DQN | −1,046,440 | 142,764 | 280.7 | 0 |
| always-hybrid | −1,116,806 | 139,765 | 289.0 | 0 |

### Scenario S3 (QKD degradation)

| policy | mean reward | σ | exhaustion events | floor violations |
|---|---|---|---|---|
| **static threshold (tuned)** | **−132,835** | 55,856 | 82.0 | 0 |
| greedy recommender | −258,025 | 86,637 | 103.3 | 0 |
| always-PQC | −1,614,420 | 142,277 | 294.3 | 0 |
| random | −1,617,124 | 151,655 | 302.3 | 0 |
| DQN | −1,675,848 | 162,054 | 302.0 | 0 |
| always-hybrid | −1,719,587 | 157,223 | 306.7 | 0 |

**Gate W3: NOT PASSED**, and by a wide margin. The DQN sits with the
do-nothing-clever baselines, accumulating ~285 pool-exhaustion events per
episode where the tuned threshold has 0.

### Why, honestly

Gate W3 was pursued to exhaustion before being reported as a failure. Everything
below was found and fixed *while trying to make it pass*, and each is a real
defect that would have invalidated the result had it been left in:

- **Fixed the baseline first, making the test harder.** `StaticThresholdPolicy`
  was missing the entire `rho`/REUSE half of its specified rule, so it re-keyed on
  every decision. Against that strawman the DQN "won" by an order of magnitude —
  for a reason unrelated to pool budgeting.
- **Applied the full prescribed underfitting remedy**: γ derived from the pool
  timescale (γ = 0.995, replacing 0.99, which could not see one refill cycle
  ahead), Double DQN, 3-step returns, Huber loss, gradient clipping.
- **Implemented the missing observation normalisation.** `key_age` reached 500
  while every other feature was ≤ 3 — a 150× disparity into an unnormalised MLP.
  The spec had required `obs_norm: running_mean_std` all along.
- **Fixed an absorbing-state training bug.** Training ran as one continuous
  episode, so early exploration drained the pool, the deferral queue saturated,
  and the run never recovered: reward degraded monotonically (−1,396 → −12,916)
  and the greedy policy chose `REKEY_NOW` 896/1000 times and `REUSE` 5. With
  periodic resets, training reward now *improves* (−806 → −592) and the agent
  uses all five actions.
- **Confirmed it is not a budget problem.** 30,000 and 60,000-step runs plateau
  identically, ~1,800× worse than the threshold.

The training pathology is fixed and the agent still loses. That makes the finding
stronger, not weaker: the failure is no longer attributable to broken machinery.

The substantive reason: **discretionary hybrid serving has no upside in this
environment.** Hybrid costs more latency, more energy, and pays the scarcity
price, with nothing gained — so the optimal policy is "spend only where a floor
demands it", which is close to a static rule and is exactly what the tuned
threshold implements. The agent's remaining freedom is largely the freedom to
overspend, and exploration takes it: the DQN sits at ~281 pool-exhaustion events
against the threshold's 0.

**We stopped iterating deliberately.** Continuing to reshape the environment until
the DQN won would be manufacturing the result. Making RL earn its place requires
giving hybrid a genuine benefit — but any such benefit is a *security* benefit,
and Hard Rule 1 forbids security in the reward. That tension is a live research
question, stated in §9, not something to be quietly engineered around.

## 6. The steering attack — the headline result

```bash
.venv/bin/python -m experiments.steering_attack --steps 2500 --seeds 3
```

The critiqued design's reward is

```
r_soft(tier) = w_security · security(tier) · threat  −  w_cost · cost(tier)
```

Every term but `threat` is fixed. As the reported threat falls, the security term
shrinks toward zero and the cost term — *increasing* in tier — dominates. The
argmax walks monotonically down the ladder:

| reported threat | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **soft-reward: preferred tier** | **0** | **0** | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| **masked: enforced floor** | **1** | **1** | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

At fully suppressed threat the soft-reward design prefers **classical** — the
quantum-vulnerable tier — where our architecture still floors at PQC. A drop of
**2 tiers**.

**This is analytic, not empirical**, and that is its strength: it is a property
of the reward *function*, so it holds for any agent maximising it, independent of
seed, training budget or exploration schedule. Our reward contains no threat
term, so there is no gradient for an adversary to pull on.

Across every agent, dose (0.0–1.0) and seed:

- **floor violations: 0**
- **posture-ratchet reversals: 0**

The attack enters at exactly one point — the telemetry the forecaster reads — and
never touches the policy table, the mask, or the pool.

### Two measurement traps we fell into

Worth recording, because both produced *confidently wrong* results:

1. **Measuring the installed key tier conflates choice with history.** `REUSE`
   carries a key bought at an earlier, higher threat level forward, which made
   the victim appear to become *more* secure under attack. The fix was to measure
   only decisions that actually establish key material.
2. **Attack placement matters.** Starting the attack before any floor had
   ratcheted only demonstrated that suppressing a signal prevents escalation —
   true, but trivial. The attack now begins *after* S2's first threat window, so
   it asks the sharp question: can suppression walk back an *established*
   protection? Structurally, no — the ratchet has no downward path.

---

## 7. Experiment E-A — foresight costs availability

```bash
.venv/bin/python -m experiments.ea_ablation --train-seeds 3 --eval-seeds 5
```

Success criterion, set in advance: *LSTM foresight measurably reduces regret
events on S3 versus `off`.*

| `use_foresight` | S3 regret events | S3 reward |
|---|---|---|
| **off** | **317.2** | −1,835,846 |
| ewma | 639.2 | −4,901,421 |
| lstm | 682.7 | −5,623,790 |

**NOT MET**, and by a wide margin in the wrong direction. This is the
project's most interesting negative result, because the forecaster is not
broken — it is *accurate*.

### The forecaster works. That is the problem.

Trained on real RT-IoT2022 flow features (§3A), the threat head reaches
**0.976–0.989 balanced accuracy** — against 0.334 (exact chance for three
classes) on the synthetic signal it was previously given. It genuinely
detects reconnaissance and attack traffic.

And detection is what costs us. The chain is short and entirely mechanical:

1. The threat head correctly identifies elevated traffic.
2. The policy table raises the floor for affected classes.
3. `PolicyTable`'s ratchet is **one-way** — a raised floor never falls.
4. More flows now require hybrid keys than the QKD link can fund.
5. Requests are deferred rather than downgraded (Hard Rule 9), and deferral
   is precisely what a regret event counts.

So better threat detection produces **more** regret events, not fewer. The
security/availability tension is not a caveat here; it is the measurement.

### The single-false-positive result

The sharpest illustration came from the benign baseline. On S1 with real
traffic the instantaneous posture reads CALM on **1,199 of 1,200 steps** —
one benign IoT flow has scan-like features and reads ELEVATED (threat score
peaks at 0.277 against a 0.107 mean).

That one step raises the floor **for the remainder of the episode**, because
the ratchet has no downward path.

This is a direct consequence of the property that makes the architecture
steering-proof. §6 shows suppression cannot walk a floor back down; the same
one-way design means a lone false positive cannot be walked back either. The
mechanism that defeats an adversary is the mechanism that amplifies a false
alarm. Both follow from one line of the policy table, and an operator
deploying this would need to know it.

A hysteresis window (require *k* consecutive elevated readings before
ratcheting) would blunt the false-positive cost, at the price of delaying
genuine escalations by *k* steps — and delay is exactly what a sustained
suppression attack buys. That trade-off is stated rather than resolved: it
depends on the relative cost of an unprotected window versus a starved one,
which is an operator's judgement, not ours.

## 8. Answering the examiner

**"Why RL instead of a threshold rule?"** On this environment — you shouldn't. A
tuned three-parameter threshold beats our DQN (§5). We report that rather than
defend a result we cannot reproduce.

**"Isn't this just a recommender?"** On S3 the myopic recommender is the best
policy we measured. The coupling we built (§4) is real but weak, because
discretionary hybrid serving carries no upside.

**"Isn't the threat score manipulable?"** Yes — that is §6, and it is the result
that survives. Manipulation can only delay a floor rising; it can never lower one
already raised.

**"Where do the security tiers come from?"** NIST PQC categories, SP 800-57
lifetimes, CNSA 2.0, ETSI GS QKD 014. The one deliberate exception is the
soft-reward agent's "security score" table, which is an *arbitrary* reproduction
of the design we critique — and its arbitrariness is part of the critique: a soft
security reward forces somebody to invent a cardinal amount of security per tier,
and the agent's behaviour then depends on those invented numbers.

**"Is the QKD realistic?"** The pool sits behind the KMS with ETSI-style
delivery. `mean_skr_kbps` is **calibrated, not looked up** — published CV-QKD
rates span Mbps to O(bps) with distance, so no single figure constrains it. What
is defensible is the stated calibration target (§3), which is reproducible and
test-guarded.

---

## 9. Limitations

Stated plainly, because several are load-bearing:

- **Gate W3 fails.** RL does not beat a tuned static rule here, and this is the
  project's central negative result. The training pathologies that could have
  explained it (unnormalised observations, an absorbing-state training loop) were
  found and fixed; the agent still loses by ~1,800×, so the cause is the
  environment's structure rather than the implementation.
- **The open design question.** Making RL competitive requires giving hybrid
  serving a genuine upside. Every honest candidate for that upside is a *security*
  benefit, which Hard Rule 1 forbids putting in the reward. Resolving that without
  breaking the project's central architectural claim is unfinished research, and
  is the single most valuable thing a successor could take on.
- **Foresight buys nothing measurable here.** E-A is a null result: `off` and
  `ewma` are indistinguishable, and the LSTM is worse. Its threat head does learn
  (balanced accuracy 0.852 vs 0.334 at chance), but its extra sensitivity costs
  availability under scarcity. An earlier claim that EWMA cut regret by 23% was
  withdrawn after it failed to reproduce on the fixed agent (§7).
- **No real CV-QKD trace.** A documented synthetic process, with its generation
  procedure stated in code.
- **No real ML-KEM.** The API performs genuine HKDF-SHA256 and AES-256-GCM, but
  the PQC contribution is a clearly-named placeholder; installing
  `liboqs-python` would make it real without other changes.
- **Threat features are a placeholder**, not RT-IoT2022-derived. The forecaster
  learns posture from observable dynamics instead.
- **The steering comparison holds the mask fixed for both agents**, isolating the
  reward design. The soft-reward agent therefore cannot literally serve below a
  floor here; the attack is measured on tier *preference*, which is the honest
  measurement and is what §6 reports.

---

## 10. Reproducing everything

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest                        # 578 tests
.venv/bin/python -m forecaster.train              # train the dual-head LSTM
.venv/bin/python -m experiments.gate_w3           # section 5
.venv/bin/python -m experiments.steering_attack   # section 6
.venv/bin/python -m experiments.ea_ablation       # section 7
.venv/bin/python -m dashboard.app                 # the 4-beat demo
.venv/bin/python -m uvicorn api.main:app          # the KMS facade
```

Results land in `results/*.json`; every table above reads from those files.

---

## 11. Conclusion

The contribution that survives scrutiny is **security as constraint, not
reward**, demonstrated against a live steering attack on a reproduced
soft-reward baseline — and, just as importantly, the finding that *enforcing*
such a guarantee is subtler than stating it. Three separate paths bypassed the
floor while the violation counter read zero.

The RL contribution does not survive: a tuned threshold wins, and we say so.
That is a scoped result rather than a failed one, and it is more useful to the
next reader than an over-claimed alternative would have been.
