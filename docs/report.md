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
3. **RL is not what this environment needs, and we say so with the numbers.**
   Gate W3 fails on both scenarios. The sharpest form of the result is that
   `greedy_recommender` — the myopic baseline built specifically to represent
   the "isn't this just a recommender system?" objection — is the **best policy
   in the table**: zero regret events on both scenarios and the lowest p99
   latency. With security excluded from the reward by design, discretionary key
   spending has no upside to trade off, so the optimal policy needs no planning
   to find. The contribution is the architecture, not the agent.

4. **Foresight helps once the world is forecastable — and that was our bug, not
   a finding.** The E-A ablation was null while the SKR process was
   (incorrectly) i.i.d. Restoring the specified Ornstein–Uhlenbeck process took
   lag-1 autocorrelation from ≈0 to 0.79, and LSTM foresight then cut S3 regret
   events by **73%**. We report the reversal and its cause rather than the
   corrected number alone.

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

The state vector is **29 dims without foresight, 36 with** — spec §4.2 exactly
(13 normalised scalars, three 4-wide one-hots, and the 4-wide posture vector;
plus `pool_hat(3) + skr_trend(1) + hybrid_demand_hat(3)` under foresight).
Every scalar is scale-free, normalised at source: QBER by `qber_abort`, SKR by
its mean, key age by the SP 800-57 cap `L`, latency per 100 ms.

Absolute episode time is **deliberately excluded**. Including it would let the
agent memorise the S6 migration timeline, which is what Hard Rule 8 forbids;
`steps_since_rekey_norm` is relative and therefore fine.

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

## 3. Calibrating the environment (and getting it wrong three times)

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

**Second correction**, against measured sensible demand, set refill to 0.098
keys/step (`mean_skr_kbps: 0.025`), giving ρ = 0.44 for a sensible policy and
ρ = 10.0 for the villain.

**That overshot into permanent deficit, and it took per-term reward logging to
see it.** The ρ figures above are open-loop ratios; the closed loop behaved
differently. Instrumenting the reward per term (spec §S5 point 1 — added late,
which is exactly the mistake) showed:

| term | share of total reward magnitude |
|---|---|
| **starve** | **99.5%** |
| rekey | 0.25% |
| latency | 0.18% |
| qkd | 0.03% |
| freshness | 0.02% |
| energy | 0.02% |

The reward had collapsed into a single function of deferral-queue backlog. The
cause: at 0.098 keys/step the deferral queue grew **monotonically to 303 and
never drained**, the pool sat empty 85% of steps, and a random policy drew
**zero** keys across an entire episode. That is not scarcity, it is famine —
and under famine every policy drowns equally, so the differences between them
are noise on top of a huge constant starvation cost. Spec §S5 point 3 sets the
rule this violated: no term may exceed 60% or fall below 2% of mean absolute
total.

**Final calibration.** Swept refill 0.025→0.18 kbps measuring, per value, the
six term shares, the queue trajectory, empty/full pool fractions, regret and
overflow. `mean_skr_kbps: 0.10` is the unique setting where every term lands
in band:

| quantity | value |
|---|---|
| refill | 0.391 keys/step (`mean_skr_kbps: 0.10`) |
| term shares | starve 41.6%, rekey 25.9%, latency 20.6%, qkd 7.8%, energy 2.1%, freshness 2.0% |
| pool empty | 21% of steps — binds, does not starve |
| deferral queue | bounded at 6 (was unbounded) |
| S1 regret events | 80 per 1,200 steps |
| S3 | still degrades hard: 195 regret events, queue to 177 |

Above ~0.14 the pool stops binding (regret falls to 1, overflow climbs);
below ~0.09 starve re-dominates.

**The behavioural signature the spec predicts finally appeared.** On S1, over
three seeds:

| policy | pool empty | pool full | regret | overflow |
|---|---|---|---|---|
| always-hybrid | 58% | 0% | 137 | 0 |
| always-PQC | 0% | 66% | 1 | **163** |
| tuned threshold | 0% | 92% | 0 | **302** |

Always-PQC wastes the entire link output while causing no regret; always-hybrid
drains it. Overflow now *discriminates* between policies, which is what makes it
the "free extra axis of evidence" §S1 asks for — a good agent must show low
regret **and** low overflow, and the threshold's zero regret is bought by
hoarding 302 keys' worth of wasted quantum material.

**Consequence for the headline result.** The foresight-value gap on S3 —
`static_threshold` minus `mpc_oracle` on regret events, the spec's §7.1
diagnostic 1 — moved from **4% to 91.8%**. The spec requires >10% for Gate W3
to be winnable at all and asks you to iterate to >25%. Under the previous
calibration no policy could have won; that is now no longer the reason.

Guarded by `tests/test_pool_sim.py::test_scarcity_ratio_in_target_band`, which
was itself part of the problem: it asserted a **hardcoded** demand figure
against a live refill rate, so its numerator was frozen while its denominator
tracked the config. It passed throughout the famine it was supposed to catch.
It now measures the environment it tests, and asserts the behavioural
signature above rather than a remembered constant.

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

**Protocol.** 3 training seeds × 10 evaluation seeds, disjoint, common random
numbers within each cell, 250,000 training steps per seed, threshold
grid-searched over all three of its parameters *on training seeds only*.

```bash
.venv/bin/python -m experiments.gate_w3 --train-seeds 3 --eval-seeds 10 --steps 250000
```

**Primary metric is regret events** (§6), lower better. Reward is reported as
descriptive only — §9.7 forbids a scalar return as a headline, because it is
reward-function-specific. This gate scored on reward until 2026-08-19; see
§9.3.

### Scenario S1 (benign baseline)

| policy | regret | overflow | p99 ms | IQM reward | floor violations |
|---|---|---|---|---|---|
| **DQN** | **0.0** | 508.9 | 150.0 | −623.0 | 0 |
| static threshold (tuned) | **0.0** | 490.8 | 150.0 | −571.9 | 0 |
| greedy recommender | **0.0** | 510.6 | 135.1 | −626.5 | 0 |
| always-PQC | 11.6 | 206.7 | 150.0 | −5,241 | 0 |
| always-hybrid | 249.2 | 0.0 | 150.0 | −14,011 | 0 |
| random | 73.4 | 4.3 | 150.0 | −6,270 | 0 |

Paired difference on regret: **0.0, CI [0.0, 0.0]** — an exact tie.

**The primary metric cannot discriminate on S1**, because three different
policies all reach zero regret. They do it the same way: by hoarding. All three
waste ~500 keys of link output. The gate now says so in its own output rather
than reporting a spurious win.

### Scenario S3 (QKD degradation)

| policy | regret | overflow | p99 ms | IQM reward | floor violations |
|---|---|---|---|---|---|
| DQN | 86.2 | 108.0 | 150.0 | −72,716 | 0 |
| static threshold (tuned) | 0.8 | 231.8 | 150.0 | −603.0 | 0 |
| **greedy recommender** | **0.0** | 242.5 | **135.1** | **−626.5** | 0 |
| always-PQC | 262.3 | 95.8 | 150.0 | −918,779 | 0 |
| always-hybrid | 366.9 | 0.0 | 150.0 | −1,059,163 | 0 |
| random | 278.5 | 2.0 | 150.0 | −953,892 | 0 |

Paired difference on regret (threshold − DQN): **−44.7, CI [−66.3, −25.3]** —
the CI excludes zero in the *wrong direction*. Pool exhaustion also regressed
far beyond the +10% secondary bound.

**Gate W3: NOT PASSED**, on both scenarios, unambiguously.

### The finding is sharper than "the DQN loses"

The `greedy_recommender` baseline exists to answer PLAN.md §8's "isn't this
just a recommender system?" objection by turning it into a number. It is
myopic by construction: cheapest legal action, every step, no regard for the
pool's future.

**It is the best policy in the table.** Zero regret on both scenarios, the
lowest p99 latency of any policy, and a reward within 1% of the tuned
threshold's while beating it outright on S3.

That is the honest answer to the objection, and it is not the answer the
project wanted. In this environment the myopic policy is close to optimal,
which means **the decision is not meaningfully coupled across time** — §7.1's
diagnostic 4. The reason is visible in the reward: with security excluded from
the objective by Hard Rule 1, a discretionary hybrid serve has *no upside at
all*. It costs latency, energy and a scarce key, and buys nothing the reward
can see. The optimal policy is therefore "never spend discretionarily", which
requires no planning to discover.

The DQN is worse than myopia because it has to *learn* not to spend, and
250,000 steps of exploration is not enough to converge on abstention when the
penalty is delayed. Its S3 per-seed rewards were −27,806, −56,260 and
−462,384: one seed diverged by an order of magnitude, which is why every figure
here is IQM rather than a mean.

### What this does and does not invalidate

It does not touch the security claims. **Floor violations are zero for every
policy, on every scenario, at every seed** — including the deliberately hostile
ones. The masking layer, the monotone policy table and the deferral semantics
all hold exactly as designed, and §6's steering result is independent of
whether RL wins.

What it does invalidate is the premise that this particular MDP needs an RL
agent. Per §7.1 fix C, the scoped and defensible claim is:

> A tuned static rule and even a myopic per-request rule match or beat a masked
> DQN on this environment, because excluding security from the reward leaves
> discretionary key spending with no upside to trade off. The contribution is
> the architecture — floors as constraints rather than objectives, which we
> show is unsteerable — not the agent.

Making RL genuinely necessary requires giving hybrid serving an upside that is
not a security term, and every honest candidate we found *is* a security term.
That is stated as the open problem in §10 rather than papered over.

---

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

## 7. Experiment E-A — foresight helps, once the world is forecastable

```bash
.venv/bin/python -m experiments.ea_ablation --train-seeds 2 --eval-seeds 5
```

Success criterion, set in advance: *LSTM foresight measurably reduces regret
events on S3 versus `off`.*

| `use_foresight` | S3 regret events | S3 reward | S6 regret events (held out) |
|---|---|---|---|
| off | 31.5 | −98,769 | **3.1** |
| ewma | 37.6 | −56,920 | 1.0 |
| **lstm** | **8.4** | **−5,825** | 16.9 |

**MET on S3**: the LSTM cuts regret events by **73%** against `off`.

### This was a null result until the SKR process was fixed, and that is the finding

Earlier runs of this same ablation reported `off` and `ewma` as
indistinguishable with the LSTM *worse* than both. The cause was not the model.
It was that `SyntheticSKRQBERTrace` drew SKR **i.i.d. Gaussian** rather than as
the log-space Ornstein–Uhlenbeck process the build spec specifies. An i.i.d.
sequence has no temporal structure, so there was nothing for a pool-head
forecaster to learn, and `test_beats_persistence_baseline` failed exactly as it
should have.

Restoring the specified OU process took lag-1 autocorrelation of the SKR series
from ≈0 to **0.79**, and the ablation reversed. The causal chain is short:

1. i.i.d. supply → the best possible forecast is the mean → the LSTM's pool
   head cannot beat persistence → foresight features carry no information →
   E-A null.
2. Mean-reverting supply with a ~50-step correlation time → a low-SKR drought
   is *predictable several steps ahead* → the agent can stop spending before
   the pool empties rather than after → 73% fewer deferrals.

The honest reading is that **the earlier null result was a property of a
mis-specified environment, not a finding about forecasting.** It is stated here
rather than quietly replaced, because "our ablation was null" and "our ablation
was null because our supply process had no autocorrelation" are very different
claims, and only the second is true.

### Two caveats that stop this being oversold

**EWMA does not help — it is slightly worse than `off` (37.6 vs 31.5).** So the
result is not "foresight helps"; it is "a *model* that captures the supply
process's autocorrelation helps, and a moving average does not". That is a
narrower claim than Addition A's framing, and it is the one the numbers support.

**On the held-out S6 migration scenario the LSTM is markedly worse** (16.9
regret events against 3.1 for `off`). S6 is eval-only by Hard Rule 8, so the
forecaster has never seen migration dynamics, and its confident anticipation of
a supply pattern that no longer applies actively hurts. A reviewer should read
the S3 win as scenario-specific rather than general, and any deployment claim
should be scoped to conditions resembling training.

**Variance is high and the seed count is low.** Five evaluation seeds with a
per-seed standard deviation of 9,576 on a mean of −5,825 is not enough to put a
tight interval on the reward difference; the regret-event ordering is the
robust part. §9.6 asks for 10 seeds on any number that reaches a final table,
and this ablation has not yet been re-run at that width.

### The security/availability tension is still real

The mechanism described in earlier drafts has not gone away: the threat head is
accurate, the policy table's ratchet is one-way, and better detection therefore
raises floors that never fall. On S1 with real RT-IoT2022 traffic the
instantaneous posture reads CALM on 1,199 of 1,200 steps — a single benign IoT
flow with scan-like features reads ELEVATED and raises the floor for the rest
of the episode.

That is a direct consequence of the property that makes the architecture
steering-proof: §6 shows suppression cannot walk a floor back down, and the
same one-way door means a false positive cannot be walked back either. The
difference now is that on a forecastable supply process the agent can *budget
around* the raised floor instead of simply drowning under it, which is what the
73% regret reduction measures.

---

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

## 9. Three defects that invalidated earlier numbers

Every number in the previous draft of this report was produced under at least
the first of these. They are documented rather than silently corrected, because
a reader's trust in the remaining numbers depends on knowing what was wrong and
how it was found.

### 9.1 Results were not reproducible across processes

`env/threat_source.py` seeded the train/eval shuffle of the RT-IoT2022 feature
pools with `abs(hash(posture.name))`. **Python randomises string hashing per
process**, so that seed differed on every interpreter launch. Two identical
300-step S3 rollouts — same code, same config, same seed — produced **41 and 13
regret events**. `threat_source: rt_iot2022` is the default, so every
threat-driven figure in the project was affected.

*Why nothing caught it.* `test_seed_reproducibility` constructs two
environments and compares their trajectories — but does so **inside one
process**, where `PYTHONHASHSEED` is fixed for the process's lifetime. The
property it checks is real and worth checking; it is simply strictly weaker
than the property that matters for a reported result.

*What caught it.* The golden fixture, on its first run — because a fixture is
stored on disk and compared in a *later* process, which is precisely the
comparison the in-process test cannot make. This is the argument for golden
tests in one paragraph.

*Fix.* Seeded from the posture's ordinal, plus an AST test
(`test_no_string_hash_is_used_as_a_random_seed`) that fails if builtin `hash()`
appears anywhere in library code.

### 9.2 The event log under-reported regret by 41%

The environment defers a request at two sites: when a hybrid-mandatory arrival
cannot be covered, and when masking leaves no legal action. Only the first
emitted the `defer_onset` event. Since §4.4 defines `defer_onset` as *being*
the regret event, every log-based consumer — including the dashboard — saw 20
events where the internal counter had 34.

Also caught by the golden fixture, which cross-checks the two counters. There
is now an explicit invariant test that they agree in both directions: too few
means the log understates the headline metric, too many means the
"once per request, not once per waiting step" miscount has returned.

### 9.3 Gate W3 was scored on the wrong metric

`experiments/gate_w3.py` both grid-searched the threshold baseline and ran its
paired comparison on **total reward**. The build spec names **regret events**
as W3's primary metric, and separately warns never to headline a scalar return
because it is reward-function-specific and not comparable across agents with
different reward functions.

So the baseline was tuned for one objective and judged on another. The gate now
selects and compares on regret events, checks the secondary constraints (p99
latency and pool exhaustion within +10%), and reports reward as descriptive
only.

The same fix surfaced a second problem: the grid optimum sits at a **grid
boundary** on both scenarios. The grid was extended past both edges, and the
search now reports how many configurations *tie* on the primary metric —
because an edge optimum among many ties means the parameter is unidentified,
which is a different problem from a grid that is too narrow, and calls for a
different remedy.

---

## 10. Limitations

Stated plainly, because several are load-bearing:

- **Gate W3 fails, and a myopic baseline wins.** RL does not beat a tuned static
  rule here, nor a per-request-greedy one. Every training pathology we could
  find was fixed (unnormalised observations, an absorbing-state loop, an
  under-trained agent at 20k steps, a mis-scored gate) and the result held at
  250,000 steps per seed. The cause is the environment's structure: with
  security excluded from the reward, discretionary key spending has no upside,
  so abstention is optimal and requires no planning.
- **The DQN has high seed variance on S3.** Per-seed rewards of −27,806,
  −56,260 and −462,384: one seed in three diverges by an order of magnitude.
  Every figure is IQM for this reason, and three training seeds is too few to
  characterise the tail. §9.6 asks for 10 on any final-table number.
- **The open design question.** Making RL competitive requires giving hybrid
  serving a genuine upside. Every honest candidate for that upside is a *security*
  benefit, which Hard Rule 1 forbids putting in the reward. Resolving that without
  breaking the project's central architectural claim is unfinished research, and
  is the single most valuable thing a successor could take on.
- **Foresight helps on S3 but hurts on held-out S6.** The LSTM cuts S3 regret
  events 73% against `off`, but on the migration scenario it has never seen it
  is markedly worse (16.9 vs 3.1). EWMA does not help at all. The claim the
  numbers support is narrow: *a model that captures the supply process's
  autocorrelation helps, on scenarios resembling its training data.*
- **No real CV-QKD trace.** A documented synthetic process, with its generation
  procedure stated in code.
- **No real ML-KEM.** The API performs genuine HKDF-SHA256 and AES-256-GCM, but
  the PQC contribution is a clearly-named placeholder; installing
  `liboqs-python` would make it real without other changes.
- **The cost model is ordinal, not measured.** Per-tier latency and energy
  encode only the ordering reuse < classical < PQC < hybrid. §S5 suggests
  measuring the primitives on the evaluation host; that was not done, the
  constants are flagged `measured: false`, and a test fails if the report ever
  claims otherwise.
- **The steering comparison holds the mask fixed for both agents**, isolating the
  reward design. The soft-reward agent therefore cannot literally serve below a
  floor here; the attack is measured on tier *preference*, which is the honest
  measurement and is what §6 reports.

---

## 11. Reproducing everything

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

## 12. Conclusion

The contribution that survives scrutiny is **security as constraint, not
reward**, demonstrated against a live steering attack on a reproduced
soft-reward baseline — and, just as importantly, the finding that *enforcing*
such a guarantee is subtler than stating it. Three separate paths bypassed the
floor while the violation counter read zero.

The RL contribution does not survive: a tuned threshold wins, and we say so.
That is a scoped result rather than a failed one, and it is more useful to the
next reader than an over-claimed alternative would have been.
