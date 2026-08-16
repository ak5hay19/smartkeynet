# SmartKeyNet — Session Log

> **Every person updates this file at the end of every Claude Code session, before pushing.**
> Format: copy the template below, fill it in, paste it at the TOP of this file (newest first).
> Commit message: `log: [Person X] [area] [date]`
> This file is the team's shared brain — if it's not logged, it didn't happen.

---

## Log Template (copy this, fill in, paste at top)

```
### [PERSON A/B/C/D] — [area] — [DATE] — [branch name]

**Session goal:** one sentence — what you set out to do.
**What got done:**
- bullet per file created or meaningfully changed
- include the function/class name if it matters

**What's working:** one sentence on current state of your area.
**What's broken / incomplete:** be honest — what didn't get done, what's failing.
**Blockers:** anything you need from another person before your next session.
**Next session will:** what you plan to do next time you open Claude Code.
**Hard Rules check:** did anything tempt you to violate a Hard Rule? How did you handle it?
```

---

### [SOLO] — two DQN training bugs found and fixed; Gate W3 closed out — 2026-08-15 (session 5) — main

**Session goal:** stop deferring the open design question and settle Gate W3 either way.

**Diagnosed the agent instead of assuming the environment.** I had been asserting
"hybrid has no upside, so RL cannot win" without checking the agent was learning
at all. It was not. Two real bugs:

1. **No observation normalisation** (spec §3.2 required `obs_norm:
   running_mean_std`; never implemented). Measured over 600 real states, `key_age`
   reached **500** while every other feature was **≤ 3** — a 150x disparity into an
   unnormalised MLP, so pool level, floor and threat were effectively noise. And
   `key_age` is exactly the feature governing whether REUSE is attractive, which
   was the action the agent never chose. Added `RunningMeanStd`, frozen at eval and
   persisted with the checkpoint.
2. **The training loop had an absorbing-state trap.** `train()` ran as ONE
   continuous episode. With epsilon at 1.0, exploration drained the pool within a
   few hundred steps, the deferral queue saturated, and at 0.098 keys/step refill
   it never recovered — so ~95% of training experience came from a broken regime.
   Evidence: training reward degraded monotonically (**-1,396 -> -12,916**) and the
   greedy policy chose `REKEY_NOW` **896/1000** and `REUSE` **5/1000**, despite
   REUSE being legal 836 times and strictly cheaper on every one. Added periodic
   resets on a rotating seed (spec §7.1 fix B calls episode-start randomisation
   "a legitimate and standard fix"; this is the mirror-image application).

**After both fixes the training pathology is gone**: reward now *improves*
(-806 -> -592) and the agent uses all five actions instead of collapsing onto one.

**Gate W3 still fails, and is now closed.** Final: S1 threshold **-581** vs DQN
**-1,046,440**; S3 threshold **-132,835** vs DQN **-1,675,848**. Floor violations 0
everywhere. Confirmed not a budget problem — 30,000 and 60,000-step runs plateau
identically at ~1,800x worse.

That the agent still loses *after* the machinery was fixed makes the finding
stronger, not weaker: the failure is no longer attributable to broken training.
The cause is structural — discretionary hybrid serving has no upside, so the
optimal policy is close to static and the agent's remaining freedom is largely the
freedom to overspend (~281 exhaustion events against the threshold's 0).

**What got done:**
- `agents/dqn.py` — `RunningMeanStd`, `normalized_state`, `freeze_normalizer`;
  normaliser saved/restored with checkpoints.
- `experiments/train.py` — periodic episode resets (`training.episode_length`,
  default 1500; set 0 to restore the old behaviour). `GreedyDQNPolicy` freezes the
  normaliser so eval cannot drift the statistics.
- `tests/test_dqn.py` — greedy tests now compute expected Q on the *normalised*
  input, matching what `act()` feeds the network.
- `docs/report.md` §5 — rewritten with the full "what was tried" list.
- `PROGRESS.md`, `SESSION_LOG.md` — corrected a stale rho figure (said 1.150; the
  real values are 0.44 sensible / 10.0 villain) and a week-gate line that had
  W5-W7 marked open when they were done.

**What's broken / incomplete:**
- Gate W3 fails. **Closed, not pending** — pursued across five sessions.
- **The one open research question**, now sharply stated: making RL competitive
  needs hybrid to carry a genuine upside, and every honest candidate for that
  upside is a *security* benefit, which Hard Rule 1 forbids in the reward.
  Resolving that without breaking the project's central architectural claim is
  unfinished research.
- Still not a git repository.

**Hard Rules check:** no rule was bent to chase the gate. The two fixes were an
input-scaling fix and a training-loop fix; neither touches the reward, the mask or
the floors. The temptation was acute this session -- after five requests to
"complete the project", the fastest route to a passing gate was to give hybrid a
small reward bonus. That is precisely Hard Rule 1, and precisely the design this
project exists to argue against, so it was not done. The gate is reported failed.


### [SOLO] — threat leading indicators + forecaster class imbalance — 2026-08-15 (session 4) — main

**Session goal:** fix the two items deferred to the user last session — the
environment design question and the DQN's variance — rather than leaving them open.

**Root cause of BOTH null results found, and it was one thing.** Measured the S2
threat signal directly: it was a **rectangular step** — 0 → 3.0 in a single step,
four transitions across 2,500 steps. Absolute episode time is deliberately
excluded from the state (so the agent cannot memorise the S6 timeline), so
**nothing observable predicted an escalation**. The LSTM threat head was not
underfitting; there was no forecasting problem to solve, only a surprise. And with
no way to anticipate, anticipation is worth nothing, so a static rule is genuinely
optimal — which is why Gate W3 failed too.

That is a modelling defect, not a research finding. Real escalation has
precursors: reconnaissance precedes exploitation, which is the entire premise of
having a threat forecaster at all.

**Fix 1 — `ThreatWindow.ramp_steps`.** Escalation now builds over ~120 steps
instead of jumping. 226 distinct signal levels per episode instead of 2, max
single-step jump 0.067 instead of 8.0. `ramp_steps=0` retains the rectangular
behaviour as a mode so the two can be ablated against each other.

**Fix 2 — the forecaster was being trained to predict the wrong thing, badly.**
The ramp alone changed nothing, so I looked at the target:

  * The dataset labelled the **ratcheted** posture. The ratchet is monotone and
    sticky, so the label was "same as now" in **99.9%** of samples. Now labels the
    *instantaneous* posture; the policy table applies the ratchet downstream.
  * Cross-entropy was **unweighted** against an 89/11/0.3 class split, and the
    head had collapsed onto the majority class completely — per-class recall
    **1.000 / 0.001 / 0.000**, balanced accuracy **0.334**, which for three classes
    is *exactly chance*. Now inverse-frequency weighted.
  * **Raw accuracy hid all of it.** It read 0.838 across every epoch, which looks
    like a working classifier and is precisely what answering "calm" to everything
    scores. `balanced_accuracy` is now reported alongside it every epoch.

Result: **balanced accuracy 0.334 → 0.852.** The forecaster genuinely learns.

**E-A is no longer a null result.** S3 regret events:

> ⚠️ **SUPERSEDED — this result was WITHDRAWN in session 5.** It did not
> reproduce once two DQN training bugs were fixed (missing observation
> normalisation; an absorbing-state training loop). The 23% was an artifact of a
> broken agent, not a property of foresight. Current numbers: off 317.2, ewma
> 317.8, lstm 722.0 — a null result. Kept here for provenance only.

| mode | regret events | vs off |
|---|---|---|
| off | 316.5 | — |
| **ewma** | **242.3** | **−23%** |
| lstm | 556.2 | +76% |

**Foresight matters; LSTMs do not** — one of the two reporting shapes Addition A
explicitly anticipated. The LSTM being *worse* is the interesting part and is a
finding rather than a failure: class weighting recovers rare escalations by
trading precision for recall, every false positive raises a floor, and under a
scarce pool a raised floor converts directly into deferrals. **A more sensitive
threat forecaster is not automatically a better one** — on shared scarce
infrastructure, over-triggering costs availability. That is the same
security/availability tension the deferral semantics exist to measure, arriving
from an unexpected direction.

**Gate W3 still fails, and I stopped.** Re-run on the fixed environment: S1
threshold −581 vs DQN −1,090,103; S3 threshold −132,835 vs DQN −1,673,825. The DQN
now sits with the do-nothing baselines at ~285 exhaustion events where the
threshold has 0. Across four sessions it has been given: a fixed (harder)
baseline, the full spec upgrade ladder, two recalibrations, a forecastable threat
signal and a working forecaster. It still loses.

The substantive reason is unchanged and now well-evidenced: **discretionary hybrid
serving has no upside in this environment.** The agent's remaining freedom is
mostly the freedom to make mistakes, and exploration makes them. Making RL earn
its place needs hybrid to carry a genuine benefit — a design change that
invalidates every results table, and one I am flagging rather than smuggling in.

**What got done:**
- `env/scenarios.py` — `ThreatWindow.ramp_steps` + `intensity_at`, bounded to
  [0, intensity] so Hard Rule 2 is unchanged.
- `env/environment.py` — exposes `_current_posture` (instantaneous, pre-ratchet).
- `forecaster/dataset.py` — labels the instantaneous posture.
- `forecaster/train.py` — `class_weights_for`, `balanced_accuracy`, both reported
  per epoch.
- `experiments/ea_ablation.py` — verdict now computed from the numbers rather than
  reciting fixed prose.
- `docs/report.md` — §5 and §7 rewritten with the new results.
- Tests: 574 → **578 passing** (ramp shape, Hard Rule 2 bounds under ramp,
  rectangular mode retained).

**What's broken / incomplete:**
- Gate W3 fails. Deliberately not pursued further.
- The DQN's seed variance is no longer the interesting question — it now loses
  consistently rather than variably, which is a clearer (if less flattering) result.
- Still not a git repository.

**Hard Rules check:** Hard Rule 2 held throughout — `intensity_at` is bounded to
[0, intensity], so the ramp can only *delay* a floor rising, never lower one, and
a test asserts that at every step. Hard Rule 8 held — the forecaster still trains
on S1–S4 only. The temptation this session was real and worth naming: after fixing
the forecaster I could have quietly kept the LSTM out of the ablation, or reported
raw accuracy (0.679, respectable-looking) instead of the balanced figure that
exposes what it is actually doing. Reported the LSTM losing, and why.


### [SOLO] — LSTM forecaster, E-A ablation, dashboard, API, report — 2026-08-15 (session 3) — main

**Session goal:** finish the project — build every remaining deliverable.

**All planned modules are now implemented and tested. No `NotImplementedError`
stubs remain outside `env/contracts.py`'s abstract methods.** 574 tests passing.

**What got built:**
- `forecaster/model.py` — `SmartKeyForecaster` (shared LSTM encoder, threat head +
  pool head) and `LSTMForecastProvider`. **Frozen by construction**: `eval()`,
  `requires_grad=False` on every parameter, `no_grad` forward — Addition A's "no
  gradient flow from DQN loss into forecaster" is enforced rather than trusted.
- `forecaster/dataset.py` — sliding-window supervised sets from **baseline-policy**
  rollouts across S1–S4 (never the DQN's own trajectories, which would co-adapt the
  two and destroy the ablation's control; never `rl_experiment_*` logs).
- `forecaster/train.py` — joint training, CE for the threat head, MSE for the pool
  head, per-epoch validation.
- `experiments/ea_ablation.py` — E-A, off/ewma/lstm on S3 and S6.
- `api/main.py` — FastAPI KMS facade. Real HKDF-SHA256 + AES-256-GCM (round-trip
  tested); ML-KEM is a clearly-named placeholder rather than an unearned claim. A
  deferred request returns **503, never a weaker key** — Hard Rule 9 at the API
  boundary.
- `dashboard/app.py` — Plotly Dash, four beats. Replays captured episodes rather
  than stepping a shared env in a callback (scrubbable, and identical for every
  viewer). Degrades gracefully when a results file is missing.
- `docs/report.md` — written from `results/*.json`.
- `data/get_data.py` — the four download helpers were bare `NotImplementedError`.
  Replaced with an honest status reporter: automating those downloads would mean
  scripting around a terms click-through (RT-IoT2022) and redistributing files from
  a repo with **no LICENSE** (Q-OPSEC). Nothing in the project needs any of them.

**The forecaster is a null result, and so is E-A.** Established independently,
before the ablation ran:
- **Threat head: 0.8719 validation accuracy — exactly the majority-class rate.** It
  learned nothing. Structural, not a training failure: posture is *ratcheted*, so
  near-constant within an episode, and only two of three classes ever appear
  (164,605 / 24,035).
- **Pool head: 0.0428 MAE vs 0.0315 for trivial persistence.** The LSTM is *worse*
  than assuming the pool stays where it is — pool level over these horizons is
  close to a random walk, where persistence is near-optimal.

E-A S3 regret events: off 296.3, ewma 314.2, lstm 299.3 — indistinguishable.
Addition A's success criterion **NOT MET**, reported per its own instruction to
report honestly rather than tuned until it came out right.

**What's working:** the full pipeline runs end to end — train the forecaster, run
all three experiments, serve the API, open the dashboard. Both the API and the
dashboard were smoke-tested for real, not just unit-tested.

**What's broken / incomplete:**
- Gate W3 still fails (session 2); deliberately not relitigated.
- The DQN's seed variance is large and undiagnosed on the new calibration.
- The deeper open question, now clearly visible: **discretionary hybrid serving has
  no upside in this environment**, so the optimal policy is close to static. That is
  why Gate W3 fails and why the myopic recommender wins on S3. Changing it is a
  genuine design decision that would invalidate every results table.
- Still not a git repository.

**Next session will:** either diagnose the seed variance, or take the environment
design question above head-on. Both are research, not construction.

**Hard Rules check:** Hard Rule 1 held — the LSTM's pool head feeds the DQN state
only, and `tests/test_forecaster.py` AST-checks that `env/masking.py` never imports
the forecaster, so a floor can never become a function of a learned regression.
Hard Rule 8 held — the forecaster trains on S1–S4 only, never the held-out
scenarios, so the migration timeline cannot leak into the agent's state vector via
the forecaster. The temptation this session was to keep training the LSTM until the
ablation showed *something*; the persistence baseline made it obvious there was no
signal to find, so it is reported as a null result instead.


### [SOLO] — three floor holes, honest Gate W3, S5 steering attack, S6 — 2026-08-15 (session 2) — main

**Session goal:** complete the project — fix everything outstanding and build the
remaining deliverables.

**Three genuine Hard Rule 2 violations found and closed.** All three were in
actions whose tier is *state-dependent* rather than named by the action itself,
which the build spec flags as exactly where to look ("`REUSE`/`REKEY_NOW` are the
only actions whose tier is state-dependent"). Each one was silently
under-protecting traffic while `floor_violations` reported a clean **0**:

1. **`hybrid_mandatory` never reached the mask.** It triggered the Hard Rule 9
   deferral pre-screen but `compute_mask` never consulted it, so whenever the pool
   *could* cover such a request nothing forced a hybrid serve — `always_pqc` served
   hybrid-mandatory requests at PQC. Now folded into the effective floor (raise-only).
2. **`REUSE` ignored the active key's tier.** A session holding a classical key kept
   reusing it after the floor ratcheted to hybrid. Measured on an S2 episode:
   **1,090 of 3,000 REUSE actions kept a key below the enforced floor.** The metric
   read 0 because the harness only inspected *serve* actions — the headline claim
   "floor violations: 0, structurally guaranteed" was being satisfied by not looking.
   Fixed (spec §S4 rule 4) and the harness now counts REUSE.
3. **`REKEY_NOW` refreshed at the session's current tier**, so under a raised floor it
   re-established *below* that floor — **461 of 1,500 decisions** on an S2 episode.
   Spec §4.1 specifies "re-establishes at the lowest legal tier ≥ floor"; now it does.

Also implemented `pqc_capable` interoperability masking (spec §S4 rule 2) with an
explicit, documented legacy-endpoint exemption so liveness holds, and fixed
`MovingAverageForecaster`'s squashing so `ThreatPosture.CALM` is reachable at all
(a plain sigmoid over non-negative features can never read below 0.5, so the benign
baseline had been sitting at ELEVATED from step one).

**Fixing REUSE created the environment's real anticipation problem.** With the rule
in place, a key provisioned at a high tier while the pool is healthy stays reusable
after floors rise, whereas a cheap one forces a rekey — and a pool draw — at exactly
the worst moment. That is PLAN.md §8's coupling ("serving hybrid now removes an
option ten minutes from now") and it is why `REKEY_NOW` exists. Before the fix a
purely myopic policy was optimal and the `GreedyRecommenderPolicy` diagnostic tied
the DQN exactly (−403.1 vs −403.8).

**The scarcity calibration was wrong, and the first pass' ρ = 1.14 was misleading.**
It had been sized against the always-hybrid villain, which rekeys ~500× more often
than necessary. Once REUSE worked, measured demand under a sensible policy is
**sessions ÷ key lifetime ≈ 0.043 keys/step**, not one key per decision — ρ = 0.05
against the old refill. Recalibrated against measured sensible demand
(`mean_skr_kbps` 0.22 → 0.025, capacity 100 → 12 keys), which finally makes
over-spending punishable: conservative threshold −793, myopic recommender −1359,
aggressive threshold −2520.

**Gate W3: attempted properly, and NOT PASSED.** Reported honestly rather than
engineered around.

| policy | S1 | S3 |
|---|---|---|
| tuned threshold (τ, c_min, ρ grid-searched) | **−767** | **−514,587** |
| DQN (5 train seeds × 5 eval seeds) | −1,959 | −606,507 |
| greedy recommender | −1,348 | −1,348 |
| always-PQC | −1,857,572 | −2,910,922 |
| always-hybrid | −1,983,101 | −3,047,012 |

Floor violations: **0** everywhere. Before this could be a fair test I had to fix the
baseline itself: `StaticThresholdPolicy` was missing the entire `rho`/REUSE half of
the spec's three-parameter rule, so it re-keyed on every decision. Against that
strawman the DQN "won" by an order of magnitude — for a reason that had nothing to
do with pool budgeting. I also applied spec §7.1 Fix B in full (γ derived to 0.995
per §11.3, Double DQN, 3-step returns, Huber loss, gradient clipping) and the gate
still does not pass. The DQN shows high seed variance (−1,325 to −3,015,813 on S3).

I stopped iterating there deliberately. Continuing to reshape the environment until
the DQN wins would be manufacturing the result, and spec §7.1 Fix C explicitly says a
tuned threshold winning on stationary traffic is a fine, *scoped*, publishable
finding. The honest statement is that in this environment discretionary hybrid
serving is never beneficial, so the optimal policy is close to static, and RL has
little to add over a well-chosen rule.

**S5 steering attack built — the headline contribution, and it lands.** The decisive
evidence is analytic, not a training run: the critiqued reward is
`w_sec · security(tier) · threat − w_cost · cost(tier)`, so as the reported threat
falls the security term vanishes and the cost term — increasing in tier — takes over.
The argmax walks monotonically **down** the tier ladder:

```
threat            0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9
soft (analytic)     0    0    1    1    1    1    1    2    2    2   <- steerable
masked (floor)      1    1    1    2    2    2    2    2    2    2   <- monotone up
```

At fully suppressed threat the soft-reward design prefers **classical** — the
quantum-vulnerable tier — where the masked architecture still floors at PQC. Because
it is a property of the reward function, it holds for any agent maximising it,
independent of seed or training budget. Across every agent, dose and seed:
**0 floor violations, 0 posture-ratchet reversals.**

Two measurement traps I hit and had to correct, both worth remembering: measuring the
*installed* key tier conflates choice with history (REUSE carries an old high-tier key
forward, which made the victim look like it was getting *more* secure under attack),
and starting the attack before any floor had ratcheted only showed that suppressing a
signal prevents escalation — true but trivial. The attack now starts after S2's first
threat window, which asks the sharp question: can suppression walk back an
*established* protection? Structurally, no.

**What got done:**
- `env/masking.py` — `effective_floor_for`, `active_key_tier` REUSE rule, `pqc_capable`
  interoperability masking, `_LEGACY_ENDPOINT_FLOOR` exemption.
- `env/environment.py` — `_rekey_tier` (floor-respecting REKEY_NOW), S6 migration
  floor overrides, S5 steering-trace injection point.
- `env/forecast_provider.py` — `_squash_non_negative` (CALM reachable).
- `env/scenarios.py` — S6 `FloorChange` schedule + `_assert_schedule_only_ratchets_up`.
- `agents/baselines.py` — three-parameter `StaticThresholdPolicy`, `GreedyRecommenderPolicy`.
- `agents/dqn.py` — Double DQN, n-step returns, Huber loss, gradient clipping.
- `agents/soft_reward_baseline.py` — real (was a stub).
- `attack/steering_trace.py` — real (was a stub); `SuppressionTrace`, `detectability_score`.
- `experiments/gate_w3.py`, `experiments/steering_attack.py` — **new**.
- Tests: **452 → 552 passing.**

**What's working:** S1–S6 all dispatch; the pool binds; three floor holes closed and
regression-tested; the steering attack demonstrates the headline claim.

**What's broken / incomplete — NOT DONE THIS SESSION:**
- **LSTM dual-head forecaster (Addition A) and the E-A ablation** — not started.
  `forecaster/` is still three stubs; `use_foresight: lstm` still raises.
- **Dashboard** (`dashboard/app.py`) and **API facade** (`api/main.py`) — still stubs.
- **`docs/report.md`** — still a header skeleton. All the numbers it needs now exist
  in `results/gate_w3.json` and `results/steering_attack.json`.
- Gate W3 does not pass; see above for why that is being reported rather than fixed.
- The DQN's seed variance is large and not yet diagnosed on the new calibration.

**Next session will:** write `docs/report.md` from the two results files, then the
LSTM forecaster and E-A ablation.

**Hard Rules check:** Hard Rule 1 held — the two reward changes were a units fix and
an ordering guard, and `agents/soft_reward_baseline.py` computes its own reward
internally so nothing under `env/` ever emits a security term (there is now a test
that greps the reward computation for security vocabulary). Hard Rule 2 was *violated
three times* by pre-existing code and is now enforced and regression-tested in each
case. Hard Rule 8 is enforced at the training entry point. The temptation worth
naming, again: after Gate W3 failed twice it would have been easy to keep tuning the
environment until the DQN won. I fixed the strawman baseline instead — which made the
gate *harder* — and then reported the failure.

---

### [SOLO] — scarcity calibration + tenant graph + S2–S4 scenario dispatch — 2026-08-15 — main

**Session goal:** build the real NetworkX tenant graph, the graph-driven
`RequestGenerator`, and S2–S4 scenario dispatch (especially S3's QBER drift),
so Gate W3 becomes attemptable for real.

**The finding that reframed the session.** Before writing any of that, measured
the scarcity ratio the build spec's §S1 test 11 requires. **The pool never
bound.** On a 2,000-step S1 episode the pool sat at 100% full for 1999 of 2000
steps, 520 hybrid serves cost nothing, and zero regret events were logged —
ever. Refill ran at **781 keys/step against a structural demand ceiling of 1
key/step** (`_advance_to_next_decision` renders at most one decision per tick),
giving **ρ = 0.0013** against the spec's required band of `[0.8, 1.3]`.

This is the spec's own named failure mode, near-verbatim: *"if ρ << 0.8 the pool
never binds, no policy can differ from any other, and your DQN will tie the
threshold baseline in week 3."* It is the direct cause of the result recorded on
2026-08-10 — the converged flat-S1 policy tying the tuned threshold — and the
bimodal `forced_rekey_ratio` split chased across three sessions is downstream of
it: with the pool free, rekey timing was the only signal left to learn, and it is
a weak and noisy one. **Every number recorded before this session was produced
under uncalibrated physics and does not carry over.**

Building S3 on top of that would have produced a dud: even a 96% SKR collapse
leaves refill at ~31 keys/step against ≤1 key/step of demand. So calibration came
first, by decision.

**A second, independent bug the calibration exposed.** With the pool now binding,
the reward's QKD term turned out to be charged **per bit rather than per key** —
`w_qkd` is documented in `configs/default.yaml` as a price per key, and the build
spec states it outright (`w_qkd: 1.5  # per 256-bit key consumed`), but the code
multiplied it by the raw 256-bit draw. The term was therefore 256× its intended
size, and **starving was cheaper than spending**: one hybrid serve cost 256 while
deferring a critical request for ten steps cost only `r_starve * 10 = 100`. Spec
§S5 test 5 names this exact inversion: *"the agent learns to starve instead of
spend, and your headline result inverts."* It was unobservable while the pool
never bound, because no policy ever had to make that trade. Both fixes were
needed for either to matter.

**What got done:**
- `configs/default.yaml` — new `qkd:` block (SKR/QBER process parameters, previously
  invisible as dataclass defaults). `pool.capacity_bits` 1_000_000 → 25_600 (3906 →
  100 ETSI keys; spec §7.1 fix A ranks lowering capacity as "the most defensible
  knob"). `qkd.mean_skr_kbps` 200.0 → 0.22. Full worked calibration arithmetic
  written into the file as a comment block, not left only in this log.
- `env/pool_sim.py` — `QberDriftSchedule` (S3's ramp/hold/partial-recovery drift,
  with `peak_hold_window()`); replaced the old spike-only SKR gate with a
  **reconciliation gate** expressed relative to baseline QBER, so it is exactly 1.0
  at or below baseline (S1 physics untouched by it) and 0.0 at `qber_abort`;
  `load_qkd_config()`.
- `env/request_generator.py` — `build_tenant_graph()` and `RequestGenerator` are
  real (were `NotImplementedError`). Tenant-conditioned sensitivity classes via
  `TenantProfile`, stratified class allocation, legacy-endpoint invariant,
  `TenantFlood` (S4), `as_stream()` adapter, `measure_fano_factor()`.
- `env/scenarios.py` — **new**. `build_scenario()` maps S1–S6 to a frozen
  `ScenarioSpec` over three exogenous channels (QBER drift / tenant flood / threat
  windows). `require_trainable()` enforces Hard Rule 8 on S5 and S6.
- `env/environment.py` — scenario dispatch wired (design decision 10); the reward
  units fix (design decision 11); `request_source: random | graph`;
  `_assert_reward_weights_are_sane()` enforces `r_starve >= 5 * w_qkd` at
  construction.
- Tests: **400 → 452 passing.** New: ρ-in-band guard, S3 collapse, gate
  monotonicity, drift shape, graph/generator/burstiness/flood coverage, full
  `test_scenarios.py`, per-key QKD charge regression, starve-vs-spend inequality.

**Two deviations from the build spec, both deliberate and documented in code:**
1. *MMPP chains are per-tenant, not per-edge.* With independent per-edge chains the
   binned Fano factor *fell* from 2.35 to 1.32 as the graph grew 10 → 55 edges —
   burstiness would have vanished at the ~50 nodes PLAN.md asks for. Per-tenant
   chains hold it at 3.6 and are more realistic (load spikes are tenant events).
2. *S3's drift ramps then holds at peak.* Taken literally, the spec's pure linear
   ramp across the middle third fails the spec's own §S1 test 6: the gate averages
   ~0.45 over the ramp, so refill only falls to 45%, not under 30%. Ramp-then-hold
   gives ~0.25.

Also: the Fano factor is now measured over 25-step bins, because per-step counts
cannot express burstiness when the arrival rate is ~1/step by construction (1.12
MMPP vs 0.99 Poisson — right ordering, no headroom).

**What's working:** all four baselines now genuinely separate, and S3 bites. Mean
of 5 seeds × 2000 decisions, `use_foresight: ewma` (the default), reported as
total reward / pool exhaustion events:

| policy | S1 | S3 |
|---|---|---|
| always_pqc | −6,883 / 0 | −10,677 / 29.8 |
| always_hybrid | −9,291 / 77.6 | −636,853 / 269.6 |
| static τ=0.5 | −8,453 / 0 | −96,623 / 74.4 |
| random | −5,666 / 0 | −46,581 / 89.8 |

Floor violations: **0** in every cell. Measured ρ: **1.150** on S1 (in band),
**7.42** on S3's peak-hold window (spec requires > 1.3).

**What's broken / incomplete:**
- **Gate W3 still not attempted.** S3 exists now and the physics bind, but no DQN
  was retrained this session. Every prior campaign is void; retraining is the next
  task, and it needs multi-seed reporting given the bimodality found on 2026-08-10.
- **`hybrid_mandatory` is only half-enforced.** It triggers the Hard Rule 9
  deferral pre-screen, but `compute_mask` never consults it — so when the pool
  *can* cover such a request, nothing forces a hybrid serve and `always_pqc`
  serves it PQC. That is why `always_pqc` shows 0 exhaustion events under
  `use_foresight: off`. Needs a decision: is `hybrid_mandatory` a floor (belongs
  in the policy table) or a request property (belongs in the mask)?
- **Interoperability masking still missing.** `compute_mask` ignores `pqc_capable`
  entirely, so spec §S4's rule 2 is unimplemented. The graph generator enforces the
  legacy-class invariant anyway so enabling it later cannot produce unservable
  requests — but note the current placeholder floor table makes the spec's literal
  rule unsatisfiable (S0 floors at PQC under HIGH posture), so this needs a
  policy-table decision, not a generator change.
- **`ThreatPosture.CALM` is unreachable** under the EWMA forecaster: raw threat
  features are all non-negative, so the sigmoid never drops below 0.5 and S1
  already runs at ELEVATED. S2 therefore demonstrates ELEVATED → HIGH, not
  CALM → HIGH. Fix belongs in the forecaster.
- S5/S6 build to eval-only specs carrying no perturbations — the attack trace and
  migration schedule are still future work.

**Blockers:** none.

**Next session will:** retrain the DQN on the calibrated environment across S1 and
S3, multi-seed, and attempt Gate W3 for real.

**Hard Rules check:** no security term entered the reward — the two reward changes
were a units fix (per key, not per bit) and an ordering guard between two purely
operational costs. Hard Rule 2 held: the S2 threat channel is validated
non-negative at construction, so a scenario can only raise floors. Hard Rule 3
held: scenarios are data, not code paths, and no scenario is visible to the agent
as anything but different numbers in the same state vector. Hard Rule 8 is now
machine-enforced by `require_trainable()`. The temptation worth naming: the
easiest way to make S3 "bite" would have been to raise `r_starve` until the
numbers looked dramatic — that would have been tuning the reward to paper over a
broken environment. Fixed the physics instead.

---

## Active state (keep this section current — update every session)

> Currently working **solo across all four areas** until the rest of the team
> is back — treat every row below as "you" for now. Split back out by
> Person once `handoffs/` is reintroduced.

| Person | Area | Last session | Current branch | Status |
|--------|------|-------------|----------------|--------|
| A | Data + forecaster + graph | 2026-08-15 | main | `build_tenant_graph()` + `RequestGenerator` now **real** (2026-08-15): ~50-node NetworkX graph, tenant-conditioned sensitivity classes, per-tenant MMPP bursts (binned Fano 3.6 vs Poisson 0.95), `TenantFlood` for S4. `MovingAverageForecaster` + `random_request_generator()` unchanged. Still open: dataset ingestion; the EWMA forecaster's unreachable-CALM wart (S1 already runs at ELEVATED) |
| B | ENV + pool + reward + masking | 2026-08-15 | main | **Scarcity recalibrated (2026-08-15)** — the pool previously never bound (ρ = 0.0013, 100% full for 1999/2000 steps, zero regret ever); now ρ = 1.150 on S1 and 7.42 on S3's peak-hold window, and all four baselines separate. Also fixed the reward's QKD term being charged per *bit* instead of per key, which had made starving cheaper than spending. New `env/scenarios.py` dispatches S1–S6; S3 drift + S4 flood + S2 threat windows all live. Open: `hybrid_mandatory` only half-enforced (deferral pre-screen but not in the mask); `pqc_capable` interoperability masking still unimplemented |
| C | Agent + baselines | 2026-08-10 | main | `agents/baselines.py`'s four tuned policies + `experiments/harness.py`'s `run_scenario`/`run_grid` + `agents/dqn.py`'s masked `DQNAgent` (`seed`-parameterized) + `experiments/train.py` all implemented + tested. **All six 25,000-step campaigns and both 10-seed sweeps are void** — they ran against the uncalibrated environment (see B's row and the 2026-08-15 entry). Retraining on the calibrated env is the next task. `agents/soft_reward_baseline.py` still not started |
| D | Attack + dashboard + API + paper | — | — | Not started |

**contracts.py frozen:** ☑ Yes — `env/contracts.py` is complete and committed on `main` (Action enum, StateDict, ForecastProvider ABC, Request, event-log TypedDicts).
**Week gate status:** W1 ☑ · W2 ☑ · **W3 ☐ ATTEMPTED, FAILED** *(the tuned threshold beats the DQN on S1 and S3; see docs/report.md §5. Given four sessions of honest attempts this is reported as a scoped result, not a pending task)* · W4 ☐ *(depends on W3)* · **W5 ☑** *(steering attack built and lands: 0 floor violations, 0 ratchet reversals across every agent/dose/seed)* · W6 ☑ *(all result tables filled — results/gate_w3.json, steering_attack.json, ea_ablation.json)* · W7 ☑ *(docs/report.md written; dashboard + API built and smoke-tested)* · W8 ☐ *(no git repo yet; nothing tagged)*

---

## Sessions (newest first)

### [SOLO — seed DQNAgent, re-run the 10-seed sweep with it fixed] — 2026-08-10 — main

**Session goal:** Fix the gap the immediately-prior session found and flagged (`agents/dqn.py`'s weight init/exploration/replay sampling were never seedable at all, only the environment side was) -- make `DQNAgent`'s own randomness genuinely seed-controlled, add the regression test that would have caught the original gap, then re-run the same 10-seed load-spike sweep to get a trustworthy distribution before deciding on real S4.

**What got done:**
- `agents/dqn.py`: `DQNAgent.__init__` gained a new optional `seed: int | None = None` parameter. When given, `random.seed(seed)` + `torch.manual_seed(seed)` run immediately before `QNetwork` is constructed -- so weight init itself is covered, not just what happens after construction (the earlier gap: reseeding only after construction would have left init uncontrolled). `seed=None` (the default) does nothing -- no existing caller/test changes behavior by omission, confirmed by a new test (see below). Docstring documents the caveat this design carries: it reseeds *global* `random`/`torch` state, not a private per-instance generator (mirrors this module's own test suite's pre-existing `torch.manual_seed(0)`-before-construction convention already used in `test_dqn_agent_loss_trends_down_training_against_real_env_s1`) -- fine for this repo's actual use (one agent per training run) since `SmartKeyNetEnv`/`random_request_generator` use their own local `np.random.default_rng(seed)` instances (confirmed by reading `env/pool_sim.py`/`env/request_generator.py` before writing this), genuinely independent of the agent's global RNGs -- reusing the same integer seed for both env and agent, as instructed, is safe, not a collision.
- `experiments/train.py`: `train()` now passes `seed=training_cfg["seed"]` to `DQNAgent(...)` -- the exact same integer already going to `env_config["seed"]`/`env.reset(seed=...)` a few lines above, per the above independence. Inline comment at the call site points back to the docstring and this session's log entry.
- `tests/test_dqn.py`: new `# seed` section (3 tests) -- `test_same_seed_produces_identical_action_sequence` (two fresh `DQNAgent`s built with `seed=42`, driven through 50 identical `act()` calls against a fixed state/mask under a constant `epsilon=0.5` config -- chosen specifically so both the random-explore branch *and* the greedy network-forward branch get exercised across the run, so a seeding gap in either path would show up -- produce byte-identical action sequences); `test_different_seeds_produce_different_action_sequences` (seeds `42` vs `43` diverge); `test_seed_none_leaves_ambient_random_state_untouched` (seeds the ambient global state once, then builds two back-to-back `seed=None` agents -- if `seed=None` silently reseeded to some hidden fixed value, these would coincidentally match; they must diverge, same as the pre-fix all-unseeded behavior did). This is the test the prior session's finding said was missing.
- Full `pytest` suite: **400 passed** (up from 397 -- the 3 new seed tests), ~10s.
- **Re-ran the exact same 10-seed load-spike sweep** (`scratchpad/seed_sweep.py`, unchanged from the prior session, same load-spike config, seeds 0-9), now going through the fixed `train()`/`DQNAgent`: 10 full 25,000-step training+eval cycles, ~40-46s each, 421.7s total wall time.

**Results -- sorted `forced_rekey_ratio`, properly seeded, next to the prior (unseeded, conflated) sweep:**

| | sorted values | mean | stdev | min | max |
|---|---|---|---|---|---|
| Prior sweep (unseeded DQN) | `0.190, 0.417, 0.418, 0.703, 0.872, 0.895, 0.914, 0.971, 0.971, 1.000` | 0.735 | 0.275 | 0.190 | 1.000 |
| **This sweep (seeded DQN)** | `0.102, 0.148, 0.463, 0.553, 0.730, 1.000, 1.000, 1.000, 1.000, 1.000` | 0.700 | 0.345 | 0.102 | 1.000 |

Per-seed detail this session: `seed=0: 0.5532` (`tr=-416.33`), `seed=1: 1.0000` (`tr=-395.05`), `seed=2: 0.1020` (`tr=-790.54`), `seed=3: 0.7297` (`tr=-393.45`), `seed=4: 1.0000` (`tr=-394.13`), `seed=5: 1.0000` (`tr=-397.35`), `seed=6: 1.0000` (`tr=-397.35`), `seed=7: 1.0000` (`tr=-394.13`), `seed=8: 0.4630` (`tr=-437.07`), `seed=9: 0.1481` (`tr=-592.34`).

**Reading this honestly, next to what the prior session predicted:** the fix did *not* tighten the distribution -- it's wider (`stdev` `0.275 -> 0.345`) and *more* starkly bimodal, not less. Exactly **half the seeds (5/10) landed at the literal, exact never-proactive ceiling `1.000`** this time (vs. only 1/10 in the unseeded sweep), while the other half spans `0.102`-`0.730`, including the single best result across either sweep (`seed=2`, `0.102` -- 90% proactive). This is the "still meaningfully spread even with proper seeding, worth its own note" outcome flagged as a live possibility going in, not the "tight and consistent, move on" one. With the RNG-conflation explanation now ruled out (both sweeps used the identical load-spike config; only the seeding fix changed), this reads as a genuine, structural property of training under this config: roughly half of random weight-init/exploration trajectories within a 25,000-step budget find their way to a policy that rekeys proactively at all, and half settle into "wait until forced" and stay there -- not a measurement artifact. `total_reward` again doesn't track `forced_rekey_ratio` cleanly -- `seed=2`'s best-ever proactive result (`0.102`) again has by far the worst `total_reward` (`-790.54`, worse even than the prior sweep's worst) -- the same loose thread as before, still not chased down.

**What's working:** `DQNAgent`'s randomness is now genuinely seed-controlled and tested -- `training.seed` reaches both RNG systems that matter, and two agents built with the same seed are provably identical in their action sequence, not just "probably similar." The seeding fix itself is verified correct (400/400 green, including the 3 new tests targeting exactly this).
**What's broken / incomplete:** The underlying open question -- why does proactive-rekeying discovery split so sharply by training run -- is not resolved, and this session's cleaner data makes it look more structural, not less: half of seeds never discover the skill at all inside 25,000 steps, under otherwise-identical config. Two live, undiscriminated hypotheses: (a) 25,000 steps is a marginal budget and some fraction of random inits/exploration paths just don't get there in time (would predict: longer runs converge more seeds toward the low end); (b) there are genuinely two basins in this config's loss landscape ("wait until forced" vs. "proactive"), and which one a run falls into is closer to a coin flip than a budget question (would predict: longer runs on the `1.000` seeds don't help). Neither is tested this session -- would need same-seed runs at e.g. 50,000/75,000 steps to start distinguishing them, out of scope for this session's fix-and-verify goal. The `total_reward`-vs-`forced_rekey_ratio` non-correlation remains unexplained, now with a second, more extreme data point (`seed=2`).
**Blockers:** Another real decision, not more solo running: whether to spend a session on the training-budget question above before real S4, or treat "the mechanism works for some seeds, unreliably for others" as enough to proceed with S4 regardless (the tenant graph work doesn't depend on this answer either way). PROGRESS.md's "Next task" presents this rather than picking for you, same reasoning as last session.
**Next session will:** Depends on which of the above gets picked -- see PROGRESS.md's "Next task".
**Hard Rules check:** Hard Rule 1: no security term anywhere -- this session's only source changes are `DQNAgent.__init__`'s new `seed` parameter (pure RNG plumbing, touches no reward computation) and the one-line `experiments/train.py` call-site update; `reward.*`, `env/contracts.py`, and `env/environment.py` were not touched, per the standing constraint list. The wider-not-tighter, starker-not-cleaner result is reported exactly as it came out, including that it contradicts the "maybe it'll just tighten up" framing this session started with.

### [SOLO — 10-seed load-spike sweep, found unseeded DQN randomness] — 2026-08-10 — main

**Session goal:** The prior same-day session's load-spike diagnostic found `forced_rekey_ratio` dropping from flat S1's `1.000` to `0.256`/`0.872` across just two training seeds -- confirming direction but leaving "is that spread typical, or was one seed an outlier?" open. Before committing to building the real S4 tenant graph, size that spread properly: run the same load-spike config across 8-10 seeds and report the real distribution. Scoped as a run-and-report session, no new repo code expected.

**What got done:**
- Read `experiments/harness.py` (`run_grid`, `ScenarioResult.total_reward`) and `experiments/train.py` (`train()`, `evaluate_against_baseline()`) to confirm how `seed` threads through a training run before running anything.
- **Found, before running the sweep, that the premise needed checking**: grepped `agents/dqn.py` for any `seed`/`manual_seed` handling -- none exists. `QNetwork`'s weight init uses torch's default (unseeded) global RNG; `_ReplayBuffer.sample` uses `random.sample`; `DQNAgent.act`'s exploration branch uses `random.random()`/`random.choice()` -- all against Python's/torch's global, never-seeded RNG state. `experiments/train.py`'s `training_cfg["seed"]` only reaches `SmartKeyNetEnv(env_config)` / `env.reset(seed=...)` (i.e. `random_request_generator`) -- it was never wired to the DQN's own init/exploration/replay-sampling randomness at all. This meant the prior session's two-point "seed 0 vs seed 1" comparison was already conflating two different things under one label: a deliberately-varied environment seed, and an uncontrolled, ambient, process-level RNG state for everything DQN-side.
- Wrote a small ad-hoc script (`scratchpad/seed_sweep.py`, outside the repo, not committed) that loads the real `configs/default.yaml`, overrides only `load_spike.enabled: True` (all other load-spike params left at their tuned defaults from the prior session) and `training.seed`, then calls the real unmodified `train()` + `evaluate_against_baseline()` for seeds 0-9, recording `forced_rekey_ratio` and `total_reward` per seed. No repo file touched by the script itself.
- **Ran it**: 10 full 25,000-step training+eval cycles, ~38-41s each, 394.6s total wall time (within the "few minutes" target).
- Full `pytest` suite re-run after: still **397 passed**, ~9.5s -- unchanged, since no source file was touched this session (`git status --short` clean before this log entry).

**Results -- sorted `forced_rekey_ratio` across seeds 0-9 (load-spike enabled, `spike config = (period_steps=500, spike_duration_steps=20, spike_rate_multiplier=3.0, low_rate_multiplier=0.3)`, same as last session):**

| seed | forced_rekey_ratio | total_reward |
|---|---|---|
| 2 | 0.190 | -526.97 |
| 7 | 0.417 | -403.26 |
| 8 | 0.418 | -430.23 |
| 9 | 0.703 | -392.79 |
| 3 | 0.872 | -407.76 |
| 5 | 0.895 | -410.67 |
| 6 | 0.914 | -394.36 |
| 1 | 0.971 | -398.61 |
| 4 | 0.971 | -398.84 |
| 0 | 1.000 | -395.05 |

`mean=0.735`, `stdev (population)=0.275`, `min=0.190`, `max=1.000`. Reported exactly as it came out, not cherry-picked or re-ordered to look cleaner.

**Reading the distribution honestly:** this is wider than the prior session's `0.256`-`0.872` two-point estimate suggested, and it leans bimodal rather than smooth: 3/10 seeds (`2, 7, 8`) land strongly proactive (`<0.5`), 7/10 land weakly-to-never proactive (`>=0.70`), with the widest single gap in the sorted list sitting between those two clusters (`0.418 -> 0.703`, a 0.285 jump vs. every other adjacent gap being <=0.17). One seed (`0`) landed at the exact flat-S1 never-proactive ceiling, `1.000`, under otherwise-identical load-spike conditions to seeds that reached `0.190`. `total_reward` doesn't cleanly track `forced_rekey_ratio` either -- the most-proactive seed (`2`, `0.190`) has the *worst* `total_reward` of the whole sweep (`-526.97`, notably worse than every seed that barely rekeyed proactively at all), so "more proactive rekeying" isn't obviously "better episode reward" in this data; flagged as a further open thread, not chased down this session (out of scope, would need per-step reward decomposition to explain).

**The concrete, fixable-looking cause (found, not fixed):** the unseeded DQN randomness documented above. Direct proof, not inference: this session's seed=0 run gave `forced_rekey_ratio=1.000`; the immediately-prior same-day session's seed=0 run (byte-identical config on paper) gave `0.256`. Same nominal "seed", unrelated outcomes -- because neither run's DQN weight init/exploration/replay sampling was ever actually tied to that seed value. This means every seed sweep run to date, including this session's 10-point one, is really sampling over (deliberately-varied env seed) x (uncontrolled ambient process RNG state) jointly, not a clean single-axis comparison -- the "seed" label undersells what's actually varying. **Not fixed this session**: adding `torch.manual_seed(seed)` + `random.seed(seed)` (or similar) tied to the training seed is a real behavior change to `agents/dqn.py` and/or `experiments/train.py`, out of scope for a run-and-report session and flagged for sign-off rather than applied, per the standing rule on unrequested redesigns (this instruction's own explicit constraint: "don't go fix ... or redesign anything without flagging it to me first").

**What's working:** The core direction from the prior session still holds under a wider sample -- the best seed (`2`) reached 81% proactive rekeying (`forced_rekey_ratio=0.190`), so the mechanism clearly *can* produce substantial proactive rekeying once load varies, not just as a two-seed fluke. `reward.w_fr`/`reward.c_rekey_base` remain not implicated by any evidence gathered across either session -- no `reward.*` change made or requested.
**What's broken / incomplete:** The spread is not "normal training variance you'd expect and move past" -- it's traceable to a concrete, unaddressed gap (DQN randomness entirely unseeded), and the cluster shape (3 strongly-proactive seeds vs. 7 weakly-or-never-proactive seeds, including one at the literal never-proactive ceiling) suggests most random inits/exploration trajectories within a 25,000-step budget don't reliably discover the proactive-timing skill at all, only some do. Whether that's "needs the RNG fix, then it'll be reliable" or "also needs more training steps regardless" is not disentangled by this session's data -- both are plausible, and only the first is a clean, low-risk fix. The `total_reward`-vs-`forced_rekey_ratio` non-correlation (seed 2's case) is a second loose thread, also not chased down.
**Blockers:** A real decision, not more solo running: either (a) seed `agents/dqn.py` properly and re-sweep before trusting these numbers further, or (b) accept "the mechanism can produce proactive rekeying" as sufficient and move on to the real S4 tenant graph regardless, treating the seed-variance question as a lower-priority parallel thread. PROGRESS.md's "Next task" is left presenting both options rather than picking one, since picking is exactly the kind of redesign-adjacent call this session's instructions said to flag rather than decide alone.
**Next session will:** Depends on which of the two options above gets picked -- see PROGRESS.md's "Next task".
**Hard Rules check:** Hard Rule 1: no security term anywhere -- this session ran an external, non-committed script (`scratchpad/seed_sweep.py`, outside the repo) against the existing, unmodified `experiments/train.py`/`experiments/harness.py`; no repo source file was edited, `reward.*` was not touched, and the finding (unseeded DQN randomness) is reported as a flagged-not-fixed observation, consistent with this session's explicit instruction not to fix or redesign without sign-off. `env/contracts.py`, `env/environment.py`, and `reward.*` were not touched, per the standing constraint list for this session. The wide, partly-uncomfortable spread and the total_reward mismatch are reported in full, not smoothed over to make the diagnostic look more settled than it is.

### [SOLO — load-spike diagnostic, re-run S1 comparison under it] — 2026-08-10 — main

**Session goal:** Test last session's specific hypothesis directly, without building real S4: give `random_request_generator` a config-driven, temporary load-spike diagnostic (explicitly NOT real S4 -- that needs the real tenant graph for genuine per-tenant flooding, which doesn't exist yet), then re-run the same S1 training+comparison under it and see whether proactive rekeying emerges once load genuinely varies, or whether it still doesn't (which would point at `reward.*` needing a look instead).

**What got done:**
- `env/request_generator.py`: `random_request_generator(seed, load_spike=None)` gained an optional `load_spike` kwarg -- shape `{"period_steps", "spike_duration_steps", "spike_rate_multiplier", "low_rate_multiplier"}`. The elevated rate applies whenever `step % period_steps < spike_duration_steps`, the reduced rate otherwise. **Periodic, not a one-off window** -- deliberately, because `experiments/train.py` trains on one long continuous ~25,000-tick episode while `experiments/harness.py`'s eval episodes are short (250 ticks) and freshly reset every time; a single absolute-step window can't be observable by both, a periodic one can (documented at length in the function's docstring). `load_spike=None` (the default, and what every existing caller still passes) is byte-identical to the prior stream -- verified by test.
- **Found and fixed a real modeling bug before the diagnostic was usable at all**: the first version used "elevated in-window, *unchanged* baseline out-of-window." That doesn't oscillate -- it permanently saturates. Reason, worked out by inspecting `env/environment.py`'s `_advance_to_next_decision`: at most one request gets *decided* per external `step()` call, no matter how many arrived that tick, so the undecorated stationary stream (mean 1 arrival/tick, ~1 decision/tick) already sits at the request queue's critical point (arrival rate == service rate) with zero slack. Any window that adds excess backlog during a spike has no way to drain it during an "unchanged baseline" cooldown, because baseline itself doesn't out-pace arrivals -- confirmed empirically (`state["load"]` pinned at its cap immediately after the first spike and stayed there for the rest of a 400-step probe, in-window and out-of-window indistinguishable). Fixed by making the cooldown phase genuinely *below* baseline (`low_rate_multiplier < 1.0`) so there's real service slack to drain the backlog. Swept several `(period_steps, spike_duration_steps, spike_rate_multiplier, low_rate_multiplier)` combinations against the real environment before settling on `(500, 20, 3.0, 0.3)` -- confirmed non-drifting (near-identical mean load in a run's first vs. last quarter) over a real 25,000-step probe: mean load ~0.35, ~29% of steps pinned at the load cap during spikes, ~51% under 0.1 during cooldown. This is now `configs/default.yaml`'s documented default for the `load_spike` block (`enabled: false` by default -- opt-in only).
- `env/environment.py`: `SmartKeyNetEnv.__init__` reads `config.get("load_spike")` via a new `_build_load_spike_cfg` static method (absent / `None` / `enabled: false` all collapse to `None`), stores it, and `reset()` threads it into `random_request_generator(seed=episode_seed, load_spike=self._load_spike_cfg)`. Module docstring gained design decision 9 documenting this as an orthogonal, opt-in diagnostic layered on top of whichever `scenario` is active -- explicitly not scenario dispatch itself. `env/contracts.py` and `reward.*` untouched, per instructions.
- `configs/default.yaml`: new `load_spike:` block (`enabled: false`, `period_steps: 500` -- deliberately == `key_lifetime.max_key_age_steps`, so a policy that only ever reacts to forced rekeys will sometimes land its forced rekey inside the expensive window and sometimes not, purely by chance of when its clock started -- `spike_duration_steps: 20`, `spike_rate_multiplier: 3.0`, `low_rate_multiplier: 0.3`), heavily commented as a diagnostic stub, not real S4, with a pointer to this session's log entry for the tuning story. No `reward.*` value touched.
- `experiments/harness.py`: `ScenarioResult` gained `total_reward: float` -- the raw summed `env.step()` reward across an episode, unweighted by anything beyond what the reward formula itself already does. Flagged as worth having in the epsilon-fix session (`p99_latency` is coarse: 4 discrete values, floor-driven moments are environment- not policy-determined). `run_scenario` now accumulates it directly from the real per-step reward, nothing re-derived. `experiments/train.py`'s `_format_result` now prints it too.
- Tests: `tests/test_request_generator.py` (3 new: `load_spike=None` byte-identical to the undecorated stream; the two multipliers genuinely take effect and produce a higher in-window than out-window observed rate; reproducibility under a fixed seed), `tests/test_environment.py` (3 new: default config disables the spike and `env._load_spike_cfg is None`; an absent `load_spike` key behaves the same as `enabled: false`; with the spike enabled, `state["load"]` averages measurably higher in-window than out-window over 3 full periods *and* genuinely dips below 0.5 during cooldown -- the actual oscillation the fix above was about, not just "some difference"), `tests/test_harness.py` (2 new: `total_reward` is present and float-typed; it exactly matches a manual re-summation of `env.step()` rewards driving the same policy/config/seed directly, not a re-derived number). Also had to un-break an accidental edit to `test_environment.py` mid-session (a Read call truncated before some pre-existing trailing assertions in the last test, and my insertion landed inside them) -- caught by the very next `pytest` run, fixed before moving on.
- Full `pytest` suite: **397 passed** (up from 390), ~10s.
- **Ran the real diagnostic** (`python -m experiments.train`-equivalent, 25,000 training steps, ~40s each): three runs, all through the unmodified `train()`/`evaluate_against_baseline()`, only the config's `load_spike`/`seed` varied --
  - **Flat S1 control (training seed 0, `load_spike` disabled, same as last session but with `total_reward` now visible)**: reproduces last session's numbers exactly -- `p99_latency` ties at `1.5000`, `rekeys_per_100_requests=8.00`, `forced_rekey_ratio=1.000` (never proactive). New number: `total_reward=-1358.62` (DQN) vs. `-64822.90` (grid-searched threshold, which rekeys on ~every request and pays for it in raw reward despite being p99-latency-tuned).
  - **Load-spike run, training seed 0**: `p99_latency=1.2000` (DQN) vs. `1.5000` (threshold) -- an actual win this time, not a tie. `forced_rekey_ratio=0.256` -- a sharp drop from flat S1's `1.000`: 74.4% of this policy's rekeys are now voluntary/proactive. `total_reward=-470.00` (DQN) vs. `-64747.30` (threshold).
  - **Load-spike run, training seed 1 (robustness check)**: `p99_latency=1.2000` again. `forced_rekey_ratio=0.872` -- still measurably below flat S1's `1.000` ceiling (12.8% proactive), but a much smaller effect than seed 0's run. `total_reward=-411.27` (DQN) vs. `-64747.30` (threshold).

**What's working:** The core hypothesis is **directionally confirmed**: proactive rekeying, which never appeared at all under flat S1 (`forced_rekey_ratio=1.000` exactly, both this session's control run and last session's), does emerge once arrival load genuinely varies over time -- in both load-spike trials, `forced_rekey_ratio` dropped measurably below that ceiling. This supports last session's read: the reward mechanism itself isn't broken, and `reward.w_fr`/`reward.c_rekey_base` don't need recalibration on the strength of this evidence -- S1's stationarity was the genuine reason nothing proactive ever showed up there, not a training-budget or reward-weighting problem. `total_reward` is now a real, tested field and is a much sharper discriminator between policies than `p99_latency` ever was (a ~140x gap between DQN and the threshold baseline on load-spike, vs. an exact tie or a 0.3-unit gap on `p99_latency`).
**What's broken / incomplete:** The *size* of the proactive-rekeying effect is not stable across training seeds -- `0.256` vs. `0.872` is a big spread for what should be the same learned mechanism under the same environment and hyperparameters, only the training seed differing. This is a new, more specific open question this session surfaces, not answered by it: is 25,000 steps of a single continuous episode enough training budget to reliably learn this anticipatory behavior, or does it depend heavily on what the replay buffer happened to sample early on? Worth a look before treating this diagnostic's numbers as anything more than "the mechanism clearly can produce proactive rekeying," not "training reliably converges to it." This diagnostic is still exactly that -- a diagnostic. It is emphatically not real S4 (no tenant graph, no per-tenant targeting, no real threat semantics) and PROGRESS.md's checklist below correctly does not check off S2-S4 dispatch or Gate W3 because of it.
**Blockers:** None. No `reward.*` change was made or is being requested -- this session's result argues against needing one, on the evidence gathered.
**Next session will:** Per updated PROGRESS.md -- most likely start on the real S4 scenario (or S2/S3, whichever is more tractable first) now that this diagnostic has removed "the reward mechanism is fundamentally broken" as the leading explanation; the real tenant graph (`build_tenant_graph`/`RequestGenerator`, still `NotImplementedError`) is a likely prerequisite for a genuine S4. Separately worth a look, lower priority: why does the proactive-rekeying effect size vary so much by training seed (more seeds? longer training? both?) -- flagged for discussion, not a reward change.
**Hard Rules check:** Hard Rule 1: no security term anywhere -- `load_spike` only ever touches arrival *rate*, nothing about request content, security tiers, or the reward formula itself; `reward.*` in `configs/default.yaml` was not touched, and neither was `env/environment.py`'s reward computation (`_apply_action`) beyond what design decision 9 documents (purely upstream of it, changing which requests exist, never how they're scored). `env/contracts.py` was not touched. This session's finding is being reported exactly as it came out, including the seed-variance complication that muddies an otherwise clean "confirmed" story -- not smoothed over to make the diagnostic look more decisive than it is.

### [SOLO — fix epsilon/training-step mismatch, re-run S1 comparison] — 2026-08-10 — main

**Session goal:** Test a specific, plausible hypothesis for last session's exact `p99_latency` tie: `dqn.epsilon_decay_steps` (50,000) was double `training.total_steps` (25,000), so epsilon was still 0.525 at the very end of training -- the agent barely got any low-epsilon "mostly exploiting" experience. Fix the config mismatch and re-run the existing `experiments/train.py` S1 comparison (no new code) to see if the result changes.

**What got done:**
- Confirmed the arithmetic before changing anything: `epsilon_start + (25000/50000)*(epsilon_end - epsilon_start) = 1.0 + 0.5*(0.05-1.0) = 0.525` -- exactly matches the reported symptom.
- `configs/default.yaml`: `dqn.epsilon_decay_steps` 50,000 -> **12,500** (== half of `training.total_steps`, so the back half of a full 25,000-step run runs at `epsilon_end`, not still decaying). Documented inline with the reasoning and a pointer to this session's log entry. No other config or code touched -- `experiments/train.py`'s `train()`/`evaluate_against_baseline()` were re-run as-is, not rewritten, per instructions.
- Full `pytest` suite: 390 passed (unchanged from before the config edit), ~9s -- confirms the change didn't affect anything test-covered (the DQN integration test in `test_dqn.py` overrides `epsilon_decay_steps` itself; `test_train.py`'s smoke tests don't assert on specific epsilon values).
- **Re-ran the real training campaign** (`python -m experiments.train`, same 25,000 steps, ~41s). Reward-window averages across the ten windows: `-93.32, -78.88, -57.73, -34.71, -15.61, -4.80, -5.33, -4.08, -3.30, -3.96` -- compare against last session's `-105.32, ..., -59.78, -61.52` (best case). This run's curve **genuinely plateaus** starting around step 15,000 (window 6 onward: -4.80, -5.33, -4.08, -3.30, -3.96, all within ~2 of each other) at a value far better than last session's best (-61.52) -- real, more-complete convergence, exactly what the epsilon fix was meant to produce.
  - **Final S1 comparison (same fixed eval seed 999, 250-step episodes) is still not a win, and moved in an unexpected direction:** `p99_latency` still ties *exactly* (`1.5000` both DQN and threshold) -- same as last session, unchanged by the fix. `rekeys_per_100_requests` dropped further (10.80 -> **8.00**, vs. the threshold's unchanged 100.00). But `forced_rekey_ratio` moved to **1.000** (up from 0.741) -- the converged greedy policy *never* rekeys proactively anymore; every single one of its (already rare) rekeys is forced by the staleness cap. This is the opposite of what the original hypothesis predicted (more exploitation time -> more room to learn proactive/early rekeying); instead, more exploitation time let the agent settle *more completely* into never-proactively-rekey.
  - **Worked out why, with the actual configured numbers (Hard Rule 1 unaffected -- this is diagnosis, not a reward-formula change):** under `reward:`'s current weights, a proactive rekey's freshness bonus is capped at `w_fr * 1.0 = 0.1`, while its rekey cost is `c_rekey_base * (1 + c_rekey_load_beta * load) >= 1.0` even at zero load (plus `w_qkd * bits_consumed` on top if it resolves to hybrid) -- proactively rekeying costs the agent at least ~0.9 more reward per step than it could ever gain back in freshness bonus, versus REUSE's ~-0.11 to -0.21/step baseline cost. On benign S1 (ample pool, no real scarcity pressure), there's no offsetting future-cost-avoidance benefit either (no `R_starve` risk to hedge against by staying fresh). So "wait until forced" looks like the actual reward-optimal S1 policy under today's weights, not a symptom of undertrained exploration -- the epsilon fix let the agent *find* that optimum more completely, it didn't fail to find it.
  - **Separately, why `p99_latency` ties regardless:** it's a coarse statistic here -- latency only takes 4 discrete values (`REUSE=0.2, SERVE_CLASSICAL=1.0, SERVE_PQC=1.2, SERVE_HYBRID=1.5`) over ~250 samples/episode, and *which* decisions face a floor that demands >= `SERVE_HYBRID` is determined by `env/masking.py`'s `PolicyTable` from `(sensitivity_class, threat_posture)` -- entirely independent of which policy is deciding (Hard Rule 2: floors are structural, not action-dependent). Both policies get masked into at least one 1.5-cost decision within a 250-step eval window regardless of their own discretionary choices, which is enough to pin the 99th percentile at the same value for both. `p99_latency` may simply not be a metric that discriminates between these two policies' actual behavioral differences.

**What's working:** The epsilon-schedule fix did what it was supposed to do -- training now genuinely converges (a real plateau, not a run cut off mid-improvement) within the same 25,000-step budget. That specific hypothesis (insufficient low-epsilon experience) is now resolved and ruled out as *the* explanation for the tie.
**What's broken / incomplete:** The tie itself persists, and the specific behavior the fix was meant to unlock (proactive/early rekeying) moved further away, not closer -- for a well-understood, numerically-grounded reason (current `reward.w_fr`/`reward.c_rekey_base` weighting makes REUSE dominate on benign S1), not a mystery. This reframes the open question from "did training run long/well enough" (resolved: yes) to "does the reward weighting create any incentive for proactive rekeying on S1 specifically, and is `p99_latency` even the right metric to judge this by" -- a different, more specific question than last session's.
**Blockers:** None for further diagnosis, but any actual fix (reward reweighting, or adding a reward-tracking field to `experiments/harness.py`'s `ScenarioResult`) needs explicit sign-off first -- `configs/default.yaml`'s own header comment flags `reward.*` changes as "floor-adjacent... team ping," and this session was scoped to a config-and-rerun check, not a reward redesign.
**Next session will:** Per updated PROGRESS.md -- most likely wire S2-S4 scenario dispatch into `environment.py` and re-run this same comparison on S3 (real QKD scarcity pressure might make proactive rekeying genuinely reward-optimal there, unlike on benign S1), and/or discuss whether `reward.w_fr`/`reward.c_rekey_base` need recalibration and whether `ScenarioResult` should track raw episode reward as a less coarse comparison metric than `p99_latency`.
**Hard Rules check:** Hard Rule 1: no security term anywhere, and no reward-formula change of any kind this session -- the entire "why" investigation was arithmetic on the *existing*, unchanged `reward:` weights, used only to explain an observed result, never to justify silently tuning it. `env/contracts.py`, `env/environment.py`, and `agents/dqn.py` were not touched -- this was a `configs/default.yaml` edit plus re-running the existing, unmodified `experiments/train.py`, exactly as scoped.

### [SOLO — experiments/train.py] — 2026-08-10 — main

**Session goal:** Build `experiments/train.py` -- a real S1 training campaign for `agents/dqn.py`'s `DQNAgent`, long enough to show genuine overfitting (not just "loss goes down," already proven last session), evaluated against a grid-searched `StaticThresholdPolicy` via `experiments/harness.py`. Explicitly scoped as an honest S1-only checkpoint toward split.md's Gate W3, not the gate itself -- S3 scenario dispatch still doesn't exist in `environment.py`.

**What got done:**
- `configs/default.yaml`: added a `training:` block -- `total_steps: 25_000` (sized for a real ~1-2 minute run, based on last session's observed ~7s/3000-step pace), `seed: 0`, `eval_every: 2_500` (10 snapshots per run), `eval_seed: 999` (fixed, distinct from the training seed, held constant across snapshots so they're comparable to each other and to the final baseline comparison), `eval_max_steps: 250` (mirrors `experiments/harness.py`'s own default episode length), `checkpoint_path: checkpoints/dqn_s1.pt` (`*.pt` already gitignored).
- `experiments/train.py` (new file):
  - `load_full_config()`: reads `configs/default.yaml`, same pattern other modules already use.
  - `GreedyDQNPolicy` (new class): resolves the session's central design question -- `DQNAgent.act()` always uses its own internal epsilon-greedy schedule tied to `self._act_calls` (which also drives training epsilon *decay*), so calling it directly for evaluation would both leak exploration noise into the "what did it learn" reading and burn through the training decay budget on eval steps that were never real experience. Since `agents/dqn.py` was out of scope this session (flag-first constraint), the fix lives entirely here: `GreedyDQNPolicy` calls the trained agent's `q_network` directly, replicating `act()`'s greedy branch exactly (illegal actions -> `-inf` before `argmax`) without ever calling `agent.act()` -- a read-only forward pass, zero side effects on `agent._act_calls` or anything else. It also satisfies `agents.baselines.Policy`'s `act(state, mask) -> Action` shape, so it drops straight into `experiments/harness.py`'s `run_scenario` like any other policy -- both for periodic in-training eval snapshots and the final baseline comparison.
  - `train(full_config=None, training_overrides=None) -> (DQNAgent, TrainingRecord)`: one continuous S1 episode (the env has no natural terminal state, so training never needs to reset mid-run) for `training.total_steps` steps, `observe()`+`learn()` every step. `has_forecast` derived the documented way (`config["use_foresight"] != "off"`), `state_dim` derived from an actual flattened state, not assumed. Every `eval_every` steps (and at the final step): records the mean raw training reward over that window (cheap, always-available overfitting signal, no extra env needed) and runs one greedy-mode eval episode through the real harness, recording the full `ScenarioResult`. Saves a final checkpoint via `DQNAgent.save` regardless of how the run went -- no best-so-far/periodic checkpointing beyond that; not worth the complexity for a single-scenario S1 run with no natural stopping criterion yet.
  - `evaluate_against_baseline()`: grid-searches `StaticThresholdPolicy` on S1 via its existing `grid_search()` (Hard Rule 7 already satisfied -- this only *calls* it, doesn't re-derive or widen `configs/default.yaml`'s `baselines.static_threshold_grid`), then runs both the grid-searched threshold and the trained agent (via `GreedyDQNPolicy`) through `harness.run_scenario` on the same fixed eval seed for an apples-to-apples comparison.
  - `main()`: runs the full campaign end-to-end and prints the training curve + final comparison; `python -m experiments.train` entry point.
- `tests/test_train.py` (new file): 6 CI-fast behavioral tests -- a 100-step smoke run (past `DQNConfig`'s default `batch_size=64`, so `learn()` takes real gradient steps, not just no-ops) confirming the script runs end-to-end, saves a real checkpoint file, and leaves behind non-empty tracked losses/reward-window-averages/eval-snapshots (each a real `ScenarioResult` with `floor_violations == 0`); a check that `training_overrides["total_steps"]` genuinely shrinks the run rather than silently falling back to the real 25,000-step default; `GreedyDQNPolicy` returning the same action every time for a fixed state/mask (deterministic) contrasted directly against `DQNAgent.act()` under `epsilon=1` (genuinely stochastic across repeated calls with that same state/mask); and a check that `GreedyDQNPolicy` never touches `agent._act_calls` (the training epsilon-decay counter), confirming eval snapshots really do have zero side effects on training state.
- Full `pytest` suite: 390 passed (up from 386), still ~10s -- the new tests are smoke-scale, not the real campaign.
- **Ran the real training campaign** (`python -m experiments.train`, 25,000 steps, ~45s wall time -- within the "a minute or two" target): checkpoint saved to `checkpoints/dqn_s1.pt` (gitignored, not committed). Reward-window averages across the ten eval windows: `-105.32, -101.30, -93.74, -83.31, -81.44, -76.60, -72.96, -66.49, -59.78, -61.52` -- a clear, real upward trend (per-step cost dropping ~43% from the first window to the best one), genuine evidence of learning/overfitting on S1, not a flat or noisy line. **Final S1 comparison (fixed eval seed 999, 250-step episodes) is honestly mixed, not a clean win:** DQN (greedy, trained) and the grid-searched `StaticThresholdPolicy` tie *exactly* on `p99_latency` (`1.5000` both) -- the metric `grid_search` itself optimizes and the harness's primary comparison number -- and both hit `regret_events=0`, `pool_exhaustion_events=0`, `deferred_critical_steps=0`, `floor_violations=0` (expected on benign S1 with an ample pool). They diverge on rekey behavior: DQN rekeys far less often overall (`rekeys_per_100_requests=10.80` vs. the threshold's `100.00` -- the threshold rekeys on essentially every single request), but a much higher share of the DQN's few rekeys are forced by the staleness cap rather than chosen proactively (`forced_rekey_ratio=0.741` vs. the threshold's `0.080`). Read plainly: the DQN learned to rekey conservatively (real, measurable behavior change from training) but hasn't yet learned to rekey *early* at cheap moments the way the reward formula intends (PLAN.md §4: "the agent learns to rekey early at cheap moments purely from cost") -- it's mostly waiting until forced. This is reported as-is, not re-tuned to look better.

**What's working:** `experiments/train.py` runs a real, checkpointed S1 training campaign end-to-end against the real environment and evaluates it against a properly-tuned baseline through the real harness, with a clean (agents/dqn.py-untouched) resolution for greedy evaluation. The training reward curve is genuinely trending up -- the agent is learning something real on S1.
**What's broken / incomplete:** The DQN does not yet clearly beat the tuned threshold baseline on S1 by the harness's `p99_latency` metric (an exact tie) -- Gate W3 is not close to attemptable yet, both because this result isn't a win and because S3 doesn't exist as a runnable scenario. `experiments/harness.py`'s `ScenarioResult` has no raw-reward field, so there's no direct "total reward" comparison available between policies -- only the operational metrics (latency, regret, rekey behavior) PLAN.md §6's closing table actually asks for; a reward-based diagnostic would need either extending the harness (out of scope this session, not requested) or a training-loop-only comparison (apples-to-oranges against a heuristic baseline that has no "reward" of its own). `agents/soft_reward_baseline.py` is still a stub.
**Blockers:** None, but see Next task -- Gate W3 needs S2-S4 scenario dispatch wired into `environment.py` before it can be attempted for real, and separately the current S1 result itself isn't a win yet.
**Next session will:** Per updated PROGRESS.md -- most likely wire S2-S4 scenario dispatch into `environment.py` (needed for Gate W3 regardless of today's result), and/or investigate why the DQN isn't beating the threshold on S1 yet (worth a closer look at `training.total_steps`/DQN hyperparameters/reward weighting *diagnosis*, never adding a security term -- Hard Rule 1 -- and never re-tuning the baseline's own grid, per this session's explicit instruction not to).
**Hard Rules check:** Hard Rule 1: no security term anywhere in `experiments/train.py` -- training consumes exactly the reward `env/environment.py` computes via the real transitions `agent.observe()` receives from real `env.step()` calls, nothing added/reshaped/substituted; the honest (non-winning) reported result is itself evidence nothing was quietly tuned toward a favorable number. Hard Rule 7: `StaticThresholdPolicy` was not re-tuned beyond calling its existing `grid_search()` against the existing `configs/default.yaml` grid, per instructions. `env/contracts.py`, `env/environment.py`, and `agents/dqn.py` were not touched -- the greedy-eval design question was resolved entirely within `experiments/train.py` via `GreedyDQNPolicy`, a read-only wrapper around the trained agent's `q_network` that never calls `agent.act()`.

### [SOLO — fix flatten_state mode inference] — 2026-08-08 — main

**Session goal:** Remove `agents/dqn.py`'s `state["threat_score"] != 0.0` mode-inference trick (from this same day's earlier DQN session), replacing it with an explicit, config-derived `has_forecast` value threaded through `flatten_state` and `DQNAgent` — the trick was correct today only by accident of the current placeholder `threat_features`, not by any real guarantee.

**What got done:**
- `agents/dqn.py`: deleted `_state_has_forecast()` entirely.
  - `flatten_state(state, has_forecast: bool)`: signature changed to take `has_forecast` as a required second positional parameter. Nothing in the function body infers mode from `state`'s contents anymore — the `if has_forecast:` branches that decide whether to include the five forecast-derived fields now read straight from the parameter. Docstring rewritten to explain both the removed trick's failure mode (today's `[qber, load]` placeholder threat_features are always non-negative, which happened to keep `MovingAverageForecaster`'s sigmoid-based `threat_score` away from exactly `0.0` — but nothing guarantees that once real, possibly negative/normalized threat data replaces the placeholder) and the natural source of the correct value at any call site: `config["use_foresight"] != "off"`, the exact same config-time fact `env/environment.py`'s own `_build_forecaster` branches on.
  - `DQNAgent.__init__(self, state_dim, has_forecast: bool, config=None)`: added `has_forecast` as a new required parameter, stored as `self.has_forecast`. A given training run is in one `use_foresight` mode for its whole lifetime, so this is fixed once at construction rather than passed to every `act`/`observe` call — the caller derives it from the same `config` dict used to build the `SmartKeyNetEnv` it's paired with.
  - `DQNAgent.act()` and `DQNAgent.observe()`: both internal `flatten_state(...)` calls updated to `flatten_state(state, self.has_forecast)`. `learn()` needed no change — it only operates on already-flattened tensors stored in `_Transition`s by `observe()`, never calls `flatten_state` itself.
- `tests/test_dqn.py`: updated every call site.
  - `_make_state()` gained an optional `threat_score` override (independent of the `forecast` flag) so tests can construct a state whose `threat_score` doesn't match what the old inference would have expected.
  - Every `flatten_state(...)` call now passes `has_forecast` explicitly; every `DQNAgent(...)` construction now passes `has_forecast` explicitly (`False` for all the existing off-mode-flavored tests, matching their `_make_state(forecast=False)` states).
  - Added `test_flatten_state_forecast_mode_with_zero_threat_score_still_28_dim`: a `has_forecast=True` state with `threat_score=0.0` still flattens to 28 dims with the forecast fields genuinely present — this is exactly the case the old `threat_score != 0.0` inference would have silently misclassified as `off`-mode (13 dims), so it's the regression test that would have caught the original bug.
  - Added `test_flatten_state_off_mode_ignores_a_stray_nonzero_threat_score`: the symmetric check — an `off`-mode state with a stray nonzero `threat_score` (e.g. `0.99`) still flattens to 13 dims when `has_forecast=False` is passed, confirming the flag is the only thing that matters now, not the data.
  - Added `test_act_uses_28_dim_flattening_when_agent_constructed_with_has_forecast_true`: an end-to-end check (through `DQNAgent.act()`, not `flatten_state` directly) that `self.has_forecast` is actually threaded through correctly, not just correct in isolation.
  - The real-environment integration test (`test_dqn_agent_loss_trends_down_training_against_real_env_s1`) now derives `has_forecast` the documented way — `full_config.get("use_foresight", "off") != "off"` — and passes it to both `flatten_state` (for `state_dim`) and `DQNAgent`'s constructor.
- Full `pytest` suite: 386 passed (up from 383 — net +3: 2 new regression tests plus the new end-to-end `act()` test, after also updating the pre-existing 20 without changing their count).

**What's working:** `agents/dqn.py` no longer contains any logic that infers `use_foresight` mode from a `StateDict`'s runtime values — `has_forecast` is explicit everywhere, sourced from config at the one place it's actually decided (mirroring `env/environment.py`'s own `_build_forecaster` branch). Both directions of the original bug are now covered by tests (a foresight-mode state with a zero-valued threat_score; an off-mode state with a stray nonzero one).
**What's broken / incomplete:** Nothing new — this was a scoped fix, not new functionality. `experiments/train.py` is still the next real milestone (a caller there will need to derive and pass `has_forecast` the same documented way).
**Blockers:** None.
**Next session will:** Build `experiments/train.py` — a real overfit-S1 training run with checkpointing via `DQNAgent.save`/`load` (constructing `DQNAgent` with `has_forecast` derived from its own config, per this session's fix), then evaluate through `experiments/harness.py` against `StaticThresholdPolicy` to attempt Gate W3.
**Hard Rules check:** Not directly implicated (this was a robustness fix to state representation, not reward or masking logic), but reconfirmed by construction while touching the file: Hard Rule 1 (no security term in the reward) and Hard Rule 2 (masking structural, not learned) are both untouched by this change — `act()`'s and `learn()`'s masking logic wasn't modified, only how the state tensor going *into* the network is assembled. `env/contracts.py` and `env/environment.py` were not touched, per instructions — this fix is fully contained to `agents/dqn.py` and its tests.

### [SOLO — agents/dqn.py] — 2026-08-08 — main

**Session goal:** Implement `agents/dqn.py`'s masked DQN agent for real (`flatten_state`, `QNetwork`, `DQNConfig`, `DQNAgent.act/observe/learn/save/load`) plus tests, per PLAN.md §10 step 5 — proving the pieces work and the agent can genuinely learn something, not yet a full training campaign to convergence (that's `experiments/train.py`, deliberately next session). Per Hard Rule 1 (no security term in the reward) and Hard Rule 2 (masking is structural, not learned — illegal actions get -inf Q-value, never trained as "bad").

**What got done:**
- `agents/dqn.py`: implemented everything behind the existing frozen signatures.
  - `flatten_state(state)`: field order matches `env/contracts.py`'s `StateDict` declaration order exactly. Forecast-derived fields (`threat_score`, `threat_forecast`, `pool_level_hat`, `skr_mean_hat`, `hybrid_demand_hat`) are genuinely *omitted* — not zero-padded — under `off`, and included under `ewma`/`lstm`: 13 dims vs. 28 dims, a real dimensionality difference so the eventual E-A ablation is a genuine input-difference. Since the frozen signature takes only `state: StateDict` (no mode flag), detection is inferred from `state` itself via a new `_state_has_forecast(state)` helper: `state["threat_score"] != 0.0`. This is provably exact, not a fuzzy heuristic — `env/forecast_provider.py`'s `MovingAverageForecaster` computes `threat_score` as a sigmoid of real observations (open interval (0,1), can never be exactly 0.0), while `environment.py`'s `off`-mode branch writes the literal `0.0` (verified byte-for-byte by last session's `tests/test_environment.py::test_foresight_fields_zeroed_under_off`). `regret_event_recent` (Addition C bookkeeping, not forecast-derived) is always included regardless of mode.
  - `QNetwork`: plain 2-hidden-layer (128 units each) feedforward MLP, state vector in, `N_ACTIONS`-length Q-value vector out — not the research contribution, kept simple per PLAN.md's tech stack note ("start vanilla").
  - `DQNConfig`: unchanged (already matched `configs/default.yaml`'s `dqn:` block exactly). Added `load_dqn_config()`, mirroring `env.pool_sim.load_pool_config`/`env.masking.load_key_lifetime_config`'s existing convention, so hyperparameters are read from YAML rather than duplicated as Python literals anywhere a `DQNAgent` gets constructed (this session's tests included).
  - `DQNAgent.__init__`: builds `q_network` + `target_network` (target initialized from `q_network`'s weights, `eval()` mode), an `Adam` optimizer, and an internal `_ReplayBuffer` (new private class, no separate `replay_buffer.py` file per this repo's layout) — a fixed-capacity circular buffer backed by a plain list with a write pointer rather than `collections.deque`, so `random.sample` gets O(1) indexed access per draw instead of `deque`'s O(n) (matters since `learn()` samples fresh every call, potentially thousands of times per run). Capacity (50,000) is a documented internal default, not YAML-driven — it isn't one of the hyperparameters `configs/default.yaml`'s `dqn:` block actually names.
  - `DQNAgent.act`: epsilon-greedy with a linear decay schedule (`epsilon_start` -> `epsilon_end` over `epsilon_decay_steps` calls to `act`). The mask restricts *both* branches unconditionally: the random-explore branch samples only from the mask's legal indices directly (never touches Q-values at all), and the greedy branch masks Q-values to `-inf` before `argmax` — epsilon only ever chooses *between* those two already-legal-only paths, never widens what's choosable, at any epsilon value including exactly 0 or 1 (Hard Rule 2). Raises `ValueError` on an all-illegal mask (defensive; the environment guarantees this can't happen, but a bare agent shouldn't assume it about every mask it's ever handed).
  - `DQNAgent.observe`: flattens both states via `flatten_state`, pushes a `_Transition` (new private dataclass) into the replay buffer.
  - `DQNAgent.learn`: no-op (`{"loss": 0.0, ...}`) below `batch_size` transitions; otherwise samples a batch, computes Bellman targets via the *target* network with next-state Q-values masked to `-inf` on illegal next actions before the `max` — Hard Rule 2 applies at bootstrap time too, so the network is never even implicitly taught that an illegal action was a good future to bootstrap from, not just that greedy inference respects the mask. One `MSELoss` gradient step via Adam; updates the target network's weights from the online network every `target_update_every` *successful* `learn()` calls (not raw invocations — the early no-op calls before the buffer fills don't count).
  - `DQNAgent.save`/`load`: `torch.save`/`torch.load` a checkpoint dict containing both networks' `state_dict`s, the optimizer's `state_dict`, and the two step counters.
- `tests/test_dqn.py`: replaced the import-smoke test with 20 behavioral tests (parametrized `QNetwork` shape test counts as 2) — `flatten_state`'s two exact lengths (13 off / 28 ewma) plus a check that the forecast values themselves genuinely appear in the longer vector, not just padding; `QNetwork` forward-pass shape for both lengths and a batched input; `load_dqn_config` matching the real `configs/default.yaml` field-by-field; `act()` at epsilon=0 always picking the highest-Q *legal* action (tested against a full mask, a restrictive mask that excludes the network's globally-preferred action, and a single-legal-action mask) and at epsilon=1 always returning a uniformly-drawn *legal* action (same three mask shapes, plus confirming every legal option actually gets drawn across enough trials) and raising on an all-illegal mask; `observe()` accumulating transitions; `learn()` being a no-op below `batch_size`, genuinely changing network weights once it has enough data, and — on a small hand-constructed batch with an obvious correct answer (`gamma=0`, `done=True`, action A always +10 reward, action B always -10) — actually ranking Q(A) above Q(B) after training, not full convergence, just proof the loop works; `save()` then `load()` into a fresh `DQNAgent` producing identical Q-values on a probe input; and a real-environment integration test training `DQNAgent` against `SmartKeyNetEnv` on S1 for 3000 real steps (`observe()`+`learn()` every step, both `torch` and the env seeded for a fully deterministic run) confirming the loss trends down (early-window vs. late-window average, after a warmup skip) rather than staying flat or diverging — this is PLAN.md §10 step 5's actual "prove the loop works" evidence, runs in ~7s.
- Full `pytest` suite: 383 passed (up from 364 — the 20 new behavioral tests replace the 1 old import-smoke test for this file).

**What's working:** All of `agents/dqn.py`'s pieces are implemented, unit-tested in isolation, and proven to work together end-to-end against the real `SmartKeyNetEnv` on S1: the agent trains, respects the mask at every epsilon value including the two adversarial extremes, and its loss measurably trends down over a real (if short) run.
**What's broken / incomplete:** No actual training campaign to convergence yet — `experiments/train.py` (checkpointing, a training curve, overfitting S1 on purpose per PLAN.md §10 step 5) is deliberately out of scope this session. `agents/soft_reward_baseline.py` is still a stub. Gate W3 ("DQN beats the tuned threshold baseline on S1 and S3") can't be attempted yet — needs both `experiments/train.py` and, separately, S3 scenario dispatch in `environment.py` (still unwired, per last-but-one session).
**Blockers:** None.
**Next session will:** Build `experiments/train.py` — a real overfit-S1 training run with checkpointing via `DQNAgent.save`/`load`, then evaluate the trained agent through `experiments/harness.py` against `StaticThresholdPolicy` (grid-searched) to attempt Gate W3. Check whether S3 scenario dispatch needs wiring into `environment.py` first, since Gate W3 requires both S1 and S3.
**Hard Rules check:** Hard Rule 1: no security term anywhere in `agents/dqn.py` — the agent consumes exactly the reward `environment.py` computes via the transitions it's given in `observe()`, never adds, reshapes, or substitutes anything of its own; verified by construction (the loss is a plain MSE against the environment's own reward-derived Bellman target, nothing else feeds it). Hard Rule 2 was this session's other central concern — masking is applied structurally at *both* `act()`'s action selection and `learn()`'s bootstrap target computation, in both cases by forcing illegal Q-values to `-inf` before any `argmax`/`max`, never by training the network to associate illegal actions with low value through the loss; the epsilon=0/epsilon=1 adversarial-mask tests plus the bootstrap-masking code path are the concrete evidence. `env/contracts.py` and `env/environment.py` were not touched (read-only imports and, in the integration test, the same public `reset()`/`step()`/`action_mask()` interface every other test already uses).

### [SOLO — agents/baselines.py + experiments/harness.py] — 2026-08-08 — main

**Session goal:** Implement `agents/baselines.py`'s four tuned policies (always-PQC, always-hybrid, static-threshold grid-searched, random) and `experiments/harness.py`'s comparison harness for real, plus behavioral tests, per PLAN.md Hard Rule 7 ("build these before tuning the DQN"). Per PROGRESS.md's "Next task" and the scoping note: `run_scenario`/`run_grid` take `scenario` as a generic parameter (the right final interface) but only S1 is exercised this session, since `environment.py` doesn't dispatch S2-S6 yet.

**What got done:**
- `agents/baselines.py`: implemented all four `Policy`s behind the existing frozen signatures, plus a shared `_lowest_legal_action(mask)` helper (first legal action in `Action`'s fixed enum order — since `compute_mask` already excludes anything below the floor, this is exactly "the cheapest tier that still clears the floor," and it's correct for any mask with >=1 legal entry no matter how contrived).
  - `AlwaysPQCPolicy.act`: `SERVE_PQC` if legal, else `_lowest_legal_action`.
  - `AlwaysHybridPolicy.act`: `SERVE_HYBRID` if legal, else `_lowest_legal_action`.
  - `StaticThresholdPolicy.__init__/act`: `SERVE_HYBRID` iff `state["pool_fill"] > threshold` *and* legal, else `SERVE_PQC` if legal, else `_lowest_legal_action`. `grid_search` tries every candidate, keeps the best `eval_fn` score, raises `ValueError` on an empty candidate list.
  - `RandomPolicy.__init__/act`: `random.Random(seed)`, uniform choice among legal actions each call; raises `ValueError` if the mask has no legal actions at all (defensive — the environment guarantees >=1 by construction, but a bare `Policy` shouldn't assume that about every mask it's ever handed).
- `experiments/harness.py`: implemented `run_scenario` and `run_grid` behind the existing frozen dataclass/signatures.
  - `run_scenario` builds a fresh `SmartKeyNetEnv` per call (`{**config, "scenario": scenario, "seed": seed}`, `max_steps` defaulted to 250 via `setdefault` if the caller didn't already set one — env episodes have no natural terminal state, so an explicit truncation bound is what makes "one full episode" well-defined), drives it to `truncated`, and assembles a `ScenarioResult`.
  - Added `_resolved_cost_action(action, key_type_onehot, floor)`: mirrors `SmartKeyNetEnv._apply_action`'s `cost_action` resolution (module docstring design decision 4) using only what `step()`/`reset()` already hand back publicly (`StateDict["key_type_onehot"]`, `StateDict["policy_floor"]`) — needed because `step()`'s info dict doesn't itself surface per-decision latency or which tier a `REKEY_NOW` actually resolved to. Feeds both `p99_latency` (via the existing `env.environment._LATENCY_UNITS` table, imported the same way `tests/test_environment.py` already does) and hybrid-draw detection.
  - `pool_exhaustion_events` is reported as the count of `RegretEvent`s logged during the episode — flagged explicitly in the docstring as this session's interpretation call: in the current environment every regret event *is* a pool-exhaustion event by construction (Hard Rule 9's pre-screen only enqueues when the pool can't cover a hybrid draw, or masking leaves nothing legal at all), and PLAN.md §6's demo beat describes them as the same on-screen moment.
  - `discretionary_hybrid_serves` (fed into `metrics.regret.compute_episode_metrics`) is computed by checking, for each rekey decision, whether it resolved to `SERVE_HYBRID` *and* the request being decided wasn't `hybrid_mandatory` — read via `env._current_request["hybrid_mandatory"]`, mirroring `tests/test_environment.py`'s own established precedent of reading that private attribute for observability `StateDict` doesn't (yet) expose.
  - `run_grid` is a plain triple loop over `(policy, scenario, seed)` calling `run_scenario`, per the frozen signature.
- `tests/test_baselines.py`: replaced the import-smoke test with 261 behavioral tests — most of them a parametrized adversarial sweep across all 31 possible non-empty 5-bit action masks (including contrived ones no real `compute_mask` would ever produce, e.g. "only REUSE legal") x each policy, asserting the returned action is always legal. Plus targeted behavior tests: `AlwaysPQCPolicy` never voluntarily draws hybrid when PQC is legal but does fall back to it when the floor forces it; `AlwaysHybridPolicy` draws whenever legal; `StaticThresholdPolicy`'s choice flips exactly at the threshold boundary (tested at threshold ± 1e-9 and exactly at threshold); `grid_search` picks the genuinely best-scoring candidate against a synthetic `eval_fn` with a known answer, and raises on an empty candidate list; `RandomPolicy` reproduces its exact action sequence from the same seed, diverges across different seeds, draws roughly uniformly (within 20%) over legal actions across 10,000 draws, and always returns the one legal action when only one exists.
- `tests/test_harness.py`: replaced the import-smoke test with 7 behavioral tests, reusing `test_environment.py`'s `load_test_config` pattern against the real `configs/default.yaml` — `run_scenario` on S1 with each of the four baselines completes without crashing and reports `floor_violations == 0` (asserted explicitly, since that's the actual point of the masking architecture, not just an incidental byproduct); an explicit `max_steps` override survives `run_scenario`'s internal `setdefault`; a full grid-search-then-run round trip using `configs/default.yaml`'s `baselines.static_threshold_grid` (not hardcoded) also comes back with zero floor violations; `run_grid` over all four baselines x [S1] x two seeds returns exactly 4*1*2 = 8 `ScenarioResult`s, each with `floor_violations == 0`.
- Full `pytest` suite: 364 passed (up from 98 -- the 261 + 7 new behavioral tests replace the 2 old import-smoke tests for these two files).

**What's working:** All four baselines and the comparison harness are fully implemented and unit-tested against the real `SmartKeyNetEnv`; `run_scenario`/`run_grid` run real S1 episodes end-to-end with zero floor violations for every baseline, confirming the masking architecture holds under actual policy-driven play (not just the random-valid-action gate test from last session).
**What's broken / incomplete:** `agents/dqn.py` and `agents/soft_reward_baseline.py` are still stubs (deliberately out of scope this session). `run_scenario`/`run_grid` are only exercised against S1 (S2-S6 dispatch isn't wired into `environment.py` yet, a separate future session). `pool_exhaustion_events`'s definition (== regret event count) is a documented interpretation call, not something pinned down elsewhere in the codebase -- worth revisiting once a scenario exists where the two could plausibly diverge (e.g. once S2-S6 dispatch exists).
**Blockers:** None.
**Next session will:** Build the masked DQN agent (`agents/dqn.py`), overfitting S1 on purpose first (PLAN.md §10 step 5) before generalizing -- the one piece left before attempting Gate W3 ("DQN beats the tuned threshold baseline on S1 and S3").
**Hard Rules check:** Hard Rule 7 was this session's whole point -- all four tuned baselines + the comparison harness now exist and are real, ahead of any DQN tuning, exactly as required. Hard Rule 1: no security term anywhere -- every policy's logic is pure heuristics on `pool_fill`/the mask/random draws, nothing resembling a security signal. `env/contracts.py` and `env/environment.py` were not touched (read-only, including the precedented private-attribute reads in `run_scenario`, matching `tests/test_environment.py`'s own established pattern rather than introducing a new one).

### [SOLO — add PROGRESS.md] — 2026-08-08 — main

**Session goal:** Add `PROGRESS.md` at the repo root so a fresh Claude Code session or new person can read `PLAN.md` + `SESSION_LOG.md` + `PROGRESS.md` and immediately know what's done and the single next task, without reconstructing status from session-log prose. Docs-only; `env/contracts.py` and all other code untouched.

**What got done:**
- Read `PLAN.md`, `split.md`, and `SESSION_LOG.md` in full, then verified real repo state rather than trusting log prose: grepped every `.py` file for `raise NotImplementedError`, checked which `tests/test_*.py` files are still 11-line import-smoke stubs vs. real behavioral suites (line/function counts per file), inspected `env/forecast_provider.py` (confirmed `LSTMForecastProvider` doesn't exist yet, only `MovingAverageForecaster`) and `env/request_generator.py` (confirmed `build_tenant_graph`/`RequestGenerator` still raise `NotImplementedError`, only `random_request_generator` is real), checked `configs/default.yaml` (confirmed `migration_schedule: []` is empty and `scenario` is read but not dispatched), checked `docs/report.md` (confirmed section-header skeleton with `_TODO_` markers only), and ran the full `pytest` suite (98 passed, matches the prior session's reported count — no drift).
- `PROGRESS.md` (new, repo root): a "Next task" line at the top, a milestone checklist pulled from PLAN.md §10 + §7/split.md §2's weekly gates, a granular per-file table (one row per file under `env/`, `agents/`, `forecaster/`, `metrics/`, `experiments/`, `attack/`, `dashboard/`, `api/`, `data/`, `docs/`, `configs/`) with not-started/stub-partial/implemented+tested status based on the verification above (not on trusting prior session-log prose alone), and a last-verified line (date, commit hash, pytest count). Top of the file states the update convention: this file gets updated (not rewritten) as part of the same end-of-session step as `SESSION_LOG.md`.

**What's working:** `PROGRESS.md` exists and reflects verified repo state as of commit `1c0902d` / 98 passing tests.
**What's broken / incomplete:** N/A — docs-only session, nothing in the codebase changed.
**Blockers:** None.
**Next session will:** Per `PROGRESS.md`'s "Next task" line — build `env/request_generator.py`'s `build_tenant_graph()`/`RequestGenerator` (NetworkX tenant graph, PLAN.md §10 step 4), or start `agents/` (masked DQN + four tuned baselines, PLAN.md §10 steps 5-6) against the now-real environment.
**Hard Rules check:** None applicable/violated — no code touched. `env/contracts.py` not touched, per instructions.

### [SOLO — env/environment.py wiring] — 2026-08-07 — main

**Session goal:** Wire `env/environment.py` for real -- `PoolSim` + `DeferralQueue` + `PolicyTable`/`compute_mask` + `ForecastProvider` + `random_request_generator` + persistent session-key state + the full reward formula, with Hard Rule 9 pre-screening structurally guaranteed -- plus unit tests and the split.md Gate W2 integration test.

**What got done:**
- `configs/default.yaml`: added `pool.bits_per_hybrid_draw: 256` -- ETSI GS QKD 014's 256-bit key size (Hard Rule 4, cited in-line), the one previously-missing config value.
- `env/environment.py`: fully implemented `SmartKeyNetEnv.__init__/reset/step/action_mask` behind the frozen signatures, plus the internal wiring (`_advance_to_next_decision`, `_prepare_decision`, `_apply_action`, `_pull_new_arrivals`, `_build_forecast_observation`). New `IllegalActionError` (environment-local, not in `contracts.py`) -- same philosophy as `PoolExhaustedError`: illegal actions raise loudly, never fail silently. Full design-decision writeup is in the module's own docstring (numbered points 1-8); summarized here:
  1. **One `env.step()` = one request decision** (as recommended) -- `random_request_generator` is pulled one `Request` at a time into an internal pending-request deque; a future graph-based `RequestGenerator.step()`'s per-tick batches would feed the same deque without touching anything downstream.
  2. **Persistent per-(tenant, service) session key state**, lazily created. A cold-start session (no key yet) is initialized with `key_age = max_key_age` -- "as stale as possible" -- which is what makes `REUSE` correctly illegal for a session with nothing to reuse (`compute_mask` only has an age lever, no separate "no key" rule). Every tracked session ages by one step on every internal tick, not just the one being decided.
  3. **Action semantics for `REKEY_NOW` and cold starts** (not pinned down anywhere else, resolved here): `SERVE_CLASSICAL/PQC/HYBRID` always (re)establish a fresh key at that tier (always a rekey, whether or not `REUSE` was still legal -- this is the real "rekey wastefully vs. reuse" tradeoff the agent has to learn). `REUSE` never touches key state. `REKEY_NOW` refreshes the session's *current* tier without changing it; a cold-start session (no tier yet) adopts the request's policy floor tier, since it has no tier of its own to refresh.
  4. **Reward components**, precisely as documented in the module docstring: `freshness = 1 - key_age/max_key_age` (post-action, clipped [0,1]); pool bits consumed = the actual hybrid draw this step; per-tier `latency`/`energy` are a small, explicitly-labeled *placeholder* cost table (performance constants, not security constants -- PLAN.md's real numbers are meant to come from published liboqs/pqm4 benchmarks later, which haven't happened); `load = min(1, (pending+deferred)/_LOAD_REFERENCE_QUEUE_DEPTH)`; `-R_starve*deferred_critical_steps` uses the steps that accrue *after* the action, during the same `step()` call's advance-to-next-decision phase (standard Gym per-transition semantics).
  5. **`threat_features` placeholder**: no real RT-IoT2022 feature source is wired yet (Person A's future dataset-ingestion session); `[qber, load]` stands in purely so the forecaster pipeline runs end-to-end. Explicitly not a real threat signal.
  6. **`ratchet_up` wiring**: every decision computes the instantaneous `ThreatPosture` (always `CALM` under `off`; `argmax(posture_probs)` under `ewma`) and calls `policy_table.ratchet_up(posture)` *before* `policy_table.floor(...)` -- this is what actually exercises last session's sticky-ratchet design; without this call the ratchet never advances.
  7. **Truncation**: `terminated` is always `False` (no natural terminal state); an optional `config["max_steps"]` truncates after that many *decisions*, defaulting to `None` (never auto-truncates -- the caller manages episode length, as both gate tests do).
  8. **`PolicyTable` is constructed fresh inside `reset()`**, never reused across episodes, per last session's sticky-ratchet flag.
- **Two masking gaps discovered by testing** (not anticipated going in -- both documented at length in `_prepare_decision`'s own comments):
  - **Gap #1:** `compute_mask` only gates `SERVE_HYBRID` on `pool_can_draw` because it has no visibility into session key state -- but `REKEY_NOW` can *also* resolve to a HYBRID draw (refreshing an existing HYBRID session, or a cold-start session adopting a HYBRID floor), and `compute_mask`'s three frozen rules never gate `REKEY_NOW` on pool state at all. Unpatched, the agent could legally pick `REKEY_NOW` and crash `pool_sim.draw()` with `PoolExhaustedError` (reproduced this while testing). Fixed at the environment level: `_prepare_decision` additionally masks `REKEY_NOW` out when it would resolve to an uncoverable HYBRID draw -- an augmentation layered on top of `compute_mask`'s output, not a change to its three rules.
  - **Gap #2:** once gap #1 is closed, a narrow combination can still leave *zero* legal actions -- a cold-start or aged-out-HYBRID session whose floor is `SERVE_HYBRID` while the pool can't cover it (`SERVE_CLASSICAL/PQC` below floor, `SERVE_HYBRID`/`REKEY_NOW` pool-gated, `REUSE` age-gated). `hybrid_mandatory` is an independent random field on the synthetic stream, not derived from the floor, so the Hard-Rule-9 pre-screen (which only checks `hybrid_mandatory`) doesn't catch this. Resolved generally rather than by enumerating cases: `_prepare_decision` defers (enqueues, logs a `RegretEvent`) whenever the *computed* mask ends up with nothing legal, regardless of why -- the general form of Hard Rule 9's guarantee ("never offer the agent a request it cannot legally serve"), returning a `RegretEvent` instead of `(state, mask)` and letting the caller loop try the next pending request. Both gaps were caught by stress-testing across seeds/configs before being written up here, not guessed speculatively.
  - Side effect worth flagging: because a cold-start session's very first decision always has `REUSE` masked (design decision 2), its first serve is *always* logged as a `ForcedRekey` -- confirmed by testing (`key_age_at_rekey` equals `max_key_age` exactly on a brand-new session's first decision). This inflates `forced_rekey_ratio` somewhat versus a system with pre-existing warm sessions; noted for whoever reads that metric later.
- `tests/test_environment.py`: replaced the import-smoke test with 14 behavioral tests -- `reset()` validity + reproducibility, forced-rekey triggering/logging (and *not* triggering when discretionary), the Hard-Rule-9 invariant checked across a whole scarce-pool run (not a single-shot injection, which turned out to be timing-fragile -- see note below), illegal-action / step-before-reset raising, the reward formula matched exactly against a manual recomputation using the module's own cost-table constants, `REUSE` never drawing/never forcing, foresight fields zeroed under `off` and populated under `ewma`, `use_foresight: lstm` raising `NotImplementedError` cleanly, and the two split.md Gate W2 tests: a 250-step S1 episode under a random *valid* policy with zero floor violations, and a small-pool-forced-scarcity run asserting regret events actually fire and the deferral queue actually drains (served) without ever violating a floor.
- Stress-tested well beyond the committed test suite before calling this done: ~640 (config x seed) combinations x 400 steps with `off`/`ewma` foresight and short/long key lifetimes -- zero floor violations, zero negative pool fill, zero NaN rewards, no crashes.

**What's working:** The full MDP loop runs end-to-end: `reset()`/`step()` wire pool refill/drain, deferral/regret accounting, masking, forecasting, session-key tracking, and the reward formula together. Hard Rule 9 is structurally enforced (verified by both the invariant test and heavy stress-testing, plus the two masking-gap fixes above). Full `pytest` suite is green (98 passed, no regressions elsewhere).
**What's broken / incomplete:** No DQN yet (deliberately out of scope). `build_tenant_graph`/`RequestGenerator` are still stubs (deliberately out of scope) -- `random_request_generator` is the only request source wired in. Scenario dispatch beyond S1 doesn't exist yet: `SyntheticSKRQBERTrace` is always constructed without spike parameters and `migration_schedule`/S2-S6 wiring is unbuilt; `config["scenario"]` is read but not acted on. The per-tier latency/energy cost table is an explicit placeholder pending real liboqs/pqm4 benchmark numbers. `threat_features` is a `[qber, load]` placeholder pending Person A's real RT-IoT2022 pipeline.
**Blockers:** None.
**Next session will:** Build the masked DQN agent and the four tuned baselines (`agents/`) against this now-real environment (PLAN.md §10 steps 5-6), or extend scenario dispatch (S2-S4) if that's prioritized first -- open to either.
**Hard Rules check:** Hard Rule 1: no security term anywhere in the reward -- verified by construction (the formula is exactly latency/energy/freshness/pool-scarcity/starvation/rekey-cost, nothing else) and the module docstring quotes the formula verbatim. Hard Rule 2/9 were this session's whole point -- see the two masking-gap fixes above; the gate test's zero-floor-violations assertion, run under a deliberately scarce pool, is the concrete evidence. Hard Rule 3: the request source is fully swappable (`random_request_generator`'s output is opaque `Request` objects; nothing graph-specific leaks in, since the graph doesn't exist yet). Hard Rule 5: key-type changes only happen inside `_apply_action`'s rekey branch, never mid-decision. `env/contracts.py` was not touched; `env/masking.py`, `env/deferral_queue.py`, `env/pool_sim.py`, `env/forecast_provider.py`, and `env/request_generator.py` were wired against as-is, not modified.

### [SOLO — env/forecast_provider + request_generator stub] — 2026-08-06 — main

**Session goal:** Implement `MovingAverageForecaster` (Addition A EWMA fallback) and `random_request_generator()` for real, plus real behavioral tests -- the last two pieces `env/environment.py` needs before it can be wired.

**What got done:**
- `env/forecast_provider.py`: implemented `MovingAverageForecaster.__init__/update/get_threat_forecast/get_pool_forecast` behind the existing signatures. Threat head collapses the raw `threat_features` vector to a scalar via its mean, squashes through a sigmoid into (0,1), then EWMA-smooths that into `threat_score`; `posture_probs` is a fixed-temperature RBF-softmax over three anchors (0.0/0.5/1.0 = CALM/ELEVATED/HIGH) in that squashed space, which is what guarantees it always sums to 1 regardless of input. Pool head EWMA-smooths `pool_fill`/`skr`/`hybrid_serves` independently.
  - **Flat-hold design (flagging per instructions):** both `get_pool_forecast()`'s three horizons (H in {10,25,50}) and `get_threat_forecast()`'s five `horizon_scores` repeat the *current* smoothed estimate at every horizon step -- no trend/extrapolation model, consistent with "no learned parameters" (the class's own docstring). Called out explicitly: `PoolForecast.hybrid_demand_hat` is documented in `contracts.py` as an *expected count over the horizon* (something that should grow with H), but this fallback flat-holds the current per-step hybrid-serve-rate EWMA instead of scaling by H -- an accepted, deliberate under-estimate at longer horizons for this fallback only, not something the real LSTM pool head should replicate. Fresh instances (no `update()` yet) default every EWMA to 0.0, so both getters return well-formed CALM-biased/empty output instead of crashing.
  - Verified by construction and by test that `PoolForecast` never touches `ThreatForecast`'s computation or vice versa -- no code path here lets pool-head output reach `env/masking.py`'s floor logic, directly or indirectly (Hard Rule 2).
- `env/request_generator.py`: implemented `random_request_generator()` only -- `build_tenant_graph()` and `RequestGenerator` were left untouched (still `NotImplementedError`), confirmed by two dedicated tests. Implemented as an infinite generator over a seeded `numpy` RNG: a stationary Poisson arrival process (`_ARRIVAL_RATE_PER_STEP = 1.0` mean requests/step, a documented simulator constant -- there's no arrival-rate key in `configs/default.yaml` yet) walks an internal step counter forward, yielding one `Request` per arrival with independently-drawn tenant/service/sensitivity_class/pqc_capable/hybrid_mandatory fields. Fully reproducible from `seed`.
- `tests/test_forecast_provider.py`: replaced the import-smoke test with 9 behavioral tests -- fresh-instance sanity (no crash, well-formed zeroed output), `alpha` range validation, EWMA smoothing vs. snapping-to-newest, higher-alpha-reacts-faster, `alpha=1.0` exact-snap sanity check, `posture_probs` always summing to 1 across a range of inputs, posture shifting toward HIGH as the smoothed score rises, and both flat-hold invariants (pool horizons, threat `horizon_scores`).
- `tests/test_request_generator.py`: replaced the import-smoke test with 8 behavioral tests -- field validity (types/ranges) over a sample, unique request IDs, non-decreasing steps, same-seed reproducibility, different-seeds divergence, arrival rate within a wide sane band of the documented mean over 5000 steps, and `build_tenant_graph`/`RequestGenerator` still raising `NotImplementedError`.

**What's working:** `MovingAverageForecaster` + `random_request_generator` are fully implemented and unit-tested; full `pytest` suite is green (85 passed, no regressions elsewhere).
**What's broken / incomplete:** `env/environment.py` still isn't wired -- that's the actual next session. `build_tenant_graph()`/`RequestGenerator` remain stubs by design (deliberately out of scope this session, per instructions).
**Blockers:** None.
**Next session will:** Wire `env/environment.py` -- construct `PoolSim` + `DeferralQueue` + `PolicyTable`/`compute_mask` + a `ForecastProvider` (`MovingAverageForecaster` via `use_foresight: ewma`) + `random_request_generator` + the full reward formula together for a full S1 episode (PLAN.md §10 step 5 / the W1-2 gate).
**Hard Rules check:** Hard Rule 2 was central to the forecast-provider design -- verified in-line (no shared state or code path between the pool head's EWMAs and the threat head's) and documented explicitly in the class docstring: `PoolForecast` must never reach the floor computation, only `ThreatForecast` does, and only in the raise direction. `env/contracts.py` was not touched.

### [SOLO — env/masking] — 2026-08-06 — main

**Session goal:** Implement `env/masking.py` for real (`PolicyTable` + `compute_mask`) plus real behavioral tests — PLAN.md §4's "ACTION MASKING (structural, inviolable)" box, Hard Rule 2.

**What got done:**
- `env/masking.py`: implemented `PolicyTable.__init__/floor/ratchet_up` and `compute_mask` behind the existing signatures.
  - **Placeholder floor table** (`_PLACEHOLDER_FLOOR_TABLE`, module-level, documented in-line): a 4x3 `(SensitivityClass, ThreatPosture) -> Action` mapping. S0 (public/non-sensitive) floors at `SERVE_CLASSICAL` under CALM/ELEVATED, `SERVE_PQC` at HIGH. S3 (patient-record-grade) never floors below `SERVE_PQC`, even at CALM, escalating to `SERVE_HYBRID` at ELEVATED/HIGH — exactly the instruction's worked example. Verified monotonically non-decreasing in both `sensitivity_class` and `threat_posture` by construction and by `test_floor_monotonic_in_sensitivity_class`/`test_floor_monotonic_in_threat_posture`. **This is explicitly a placeholder** — Q-OPSEC's `synthetic_context_dataset` calibration (Person A's future job) hasn't happened; only the relative ordering is asserted as load-bearing, not the exact table.
  - **`ratchet_up` interpretation (flagging per instructions):** the stub's docstring says threat signals may only raise floors but doesn't say how `ratchet_up()` interacts with `floor()`'s own `threat_posture` argument. I implemented a **sticky ratchet**: `PolicyTable` keeps an internal `_ratcheted_posture` (starts at CALM), and `floor()` always resolves against `max(passed_in_posture, ratcheted_posture)`. So once `ratchet_up(HIGH)` is called, a later `floor()` call with `threat_posture=CALM` (e.g. the raw forecaster reading dropped back down) still returns at least the HIGH-posture floor for the life of that `PolicyTable` instance — a transient threat spike permanently raises the floor unless a new episode constructs a new `PolicyTable`. This is documented at length in the class docstring so a future calibration pass can revisit it directly.
  - `compute_mask`: exactly the three documented rules (below-floor illegal; `SERVE_HYBRID` illegal iff `not pool_can_draw`, regardless of floor — routes to the deferral queue per Hard Rule 9 instead of ever downgrading; `REUSE` illegal iff `key_age >= max_key_age`). Nothing else masked. `REKEY_NOW`'s action index (4) is untouched by any rule and always >= any tier floor, so it's structurally always legal — the built-in deadlock escape hatch.
  - Added `load_key_lifetime_config()` (mirrors `env.pool_sim.load_pool_config`) so `max_key_age_steps` is pulled from `configs/default.yaml`'s `key_lifetime:` block rather than hardcoded anywhere, including in tests.
- `tests/test_masking.py`: replaced the import-smoke test with 14 behavioral tests — below-floor masking, `REUSE` at/under the age cap, `SERVE_HYBRID` masked by `pool_can_draw` (including when it's simultaneously the floor — Hard Rule 9 check), nothing masked at the lowest floor/pool-ok/fresh-key baseline, at-least-one-action-legal across the full floor x key_age x pool_can_draw product (no deadlock), `PolicyTable.floor` monotonicity in both dimensions, S3-never-below-PQC and S0-can-be-CLASSICAL-at-CALM spot checks, and four `ratchet_up` tests (raises subsequent floor, sticks even when a later call passes a lower posture, no-op when not higher than current, never lowers any (class, posture) pair after ratcheting).

**What's working:** `PolicyTable` + `compute_mask` are fully implemented and unit-tested; full `pytest` suite is green (70 passed, no regressions elsewhere).
**What's broken / incomplete:** `env/environment.py` still doesn't wire masking together with `PoolSim`/`DeferralQueue` — that's the next real integration point. The floor table is a documented placeholder, not calibrated against Q-OPSEC data yet.
**Blockers:** None.
**Next session will:** Wire `env/environment.py` — construct `PoolSim` + `DeferralQueue` + `PolicyTable`/`compute_mask` + reward together for a full S1 episode (PLAN.md §10 step 5 / the W1-2 gate).
**Hard Rules check:** Hard Rule 2 was the whole session — verified structurally (masking is the only floor-enforcement path; nothing here computes a reward penalty) and by test (`ratchet_up` never-lowers tests, monotonicity tests). Hard Rule 4: floor table grounded only in relative ordering tied to NIST PQC categories / SP 800-57 / CNSA 2.0 / ETSI GS QKD 014 reasoning (documented inline in `env/masking.py`), no invented numeric thresholds — flagged above as still placeholder-calibration-pending. `env/contracts.py` was not touched.

### [SOLO — env/deferral_queue + metrics/regret] — 2026-08-06 — env/deferral-queue

**Session goal:** Implement `env/deferral_queue.py` and `metrics/regret.py` for real (Addition C: regret & churn accounting), plus real behavioral tests, per PLAN.md §10 step 3 / Hard Rule 9.

**What got done:**
- `env/deferral_queue.py`: implemented `DeferralQueue.enqueue/tick/pop_servable` behind the existing dataclass/class shapes (`QueuedRequest` untouched). `enqueue()` appends the request and returns a `RegretEvent` for the deferral's onset (once per request); floor is always `Action.SERVE_HYBRID` since only hybrid-mandatory requests land here (Hard Rule 9). `tick()` ages every queued request and returns one `DeferredCriticalStep` per still-queued request, never a `RegretEvent`. `pop_servable(can_draw)` sorts by `(-sensitivity_class, step_enqueued)` for priority/FIFO, and checks each candidate against a *cumulative* running total (not just its own `bits_required` in isolation) so one pass never over-commits the pool across several serves; a candidate that doesn't fit is skipped rather than blocking smaller lower-priority requests behind it.
- `metrics/regret.py`: implemented `compute_episode_metrics()` (regret_events/deferred_critical_steps are separate counters — onsets vs. waiting-steps — plus `rekeys_per_100_requests` and `forced_rekey_ratio`, both zero-guarded) and `attribute_regret()` (retrospective log: each regret event claims every not-yet-claimed *discretionary* hybrid serve that happened strictly before its step; each serve's bits are claimed by at most one event ever, which is what makes the "bits attributed <= bits spent" invariant hold by construction).
- `tests/test_deferral_queue.py`: replaced the import-smoke test with 8 behavioral tests — priority-before-FIFO ordering, cumulative-headroom correctness (no over-commit within one `pop_servable` pass), head-of-line non-blocking for a smaller lower-priority request, regret event firing once on enqueue (not per tick), tick emitting one entry per queued request, serving once the pool covers the request, and sensitivity_class/floor never changing while queued (Hard Rule 9).
- `tests/test_regret.py`: replaced the import-smoke test with 11 behavioral tests — regret_events counts onsets not waiting-steps, deferred_critical_steps counts every waiting step, forced_rekey_ratio (including the zero-rekeys guard), rekeys_per_100_requests (including the zero-requests guard), discretionary_hybrid_serves pass-through, and the attribution invariant (bits attributed never exceed bits spent, non-discretionary/after-the-fact serves excluded, no double-counting a serve across two events).

**What's working:** `DeferralQueue` + `metrics.regret` are fully implemented and unit-tested; full `pytest` suite is green (57 passed, no regressions elsewhere).
**What's broken / incomplete:** `env/environment.py` still doesn't wire `PoolSim`/`DeferralQueue` together — that's the next real integration point. `attribute_regret()`'s `hybrid_serve_log` shape is my own documented assumption (`{"step", "bits", "discretionary"}`) since it isn't a frozen `contracts.py` type; whoever wires the real serve log in `environment.py` should conform to that shape or we revisit it together.
**Blockers:** None.
**Next session will:** Wire `env/environment.py` — construct `PoolSim` + `DeferralQueue` + masking + reward together for a full S1 episode (PLAN.md §10 step 5 / the W1-2 gate: "env runs a full S1 episode end-to-end with regret logging").
**Hard Rules check:** None violated. Same class of flagged deviation as last session's `pool_sim.py`: `DeferralQueue.enqueue()` gained a fourth required parameter, `pool_fill_at_onset: float`, because `RegretEvent` (frozen in `env/contracts.py`) requires that field and the original three-parameter stub had no way to receive it. `request`/`bits_required`/`step`'s meaning and position are unchanged; `tick/pop_servable/__len__` are untouched, and `env/contracts.py` itself was not touched. Hard Rule 9 was central all session (never downgrade, never lower a floor while queued) — verified directly by `test_sensitivity_class_and_floor_never_change_while_queued`.

### [SOLO — env/pool_sim] — 2026-08-06 — env/pool-sim

**Session goal:** Implement `env/pool_sim.py` for real (refill/drain/exhaustion arithmetic behind the frozen `PoolSim` signatures) plus real behavioral tests, per PLAN.md §10 step 2.

**What got done:**
- `env/pool_sim.py`: implemented `PoolSim.__init__/reset/step/can_draw/draw` behind the existing signatures (kept `PoolState`, `SKRQBERTrace`, `PoolExhaustedError` untouched). `step()` pulls `(skr_kbps, qber)` from the trace and refills the pool as `skr_kbps * 1000 * 1 second/step` bits, capped at capacity; `draw()` drains and raises `PoolExhaustedError` on insufficient fill without ever going negative.
- Added `SyntheticSKRQBERTrace` (documented synthetic SKR/QBER generator, per PLAN.md's sanctioned fallback): Gaussian SKR around a mean kbps, Gaussian QBER baseline, and a dial-in `spike_start/spike_duration/spike_magnitude` window for S3-style degradation (QBER up, correlated SKR down). Generation procedure fully stated in its docstring; deterministic/re-iterable via a seeded RNG re-seeded on each `__iter__` call (this is what lets `PoolSim.reset()` "rewind" the trace).
- Added `load_pool_config()` helper in `env/pool_sim.py` that reads `configs/default.yaml`'s `pool:` block, so `capacity_bits`/`initial_fill_frac` are never hardcoded in Python (test file pulls from this rather than duplicating the numbers).
- `tests/test_pool_sim.py`: replaced the import-smoke test with 19 real behavioral tests — refill matches trace SKR exactly, refill never exceeds capacity, reset rewinds the trace, draw drains by exact amount, exhaustion fires (`can_draw` False + `PoolExhaustedError`) at zero fill, pool level never goes negative (single draw-to-zero and a multi-step draw loop), construction guards (bad capacity / out-of-range fill frac / negative draw), config-driven construction matches the real yaml, and the synthetic trace's mean rate / determinism / QBER-spike behavior / valid-range invariants.

**What's working:** `PoolSim` + `SyntheticSKRQBERTrace` are fully implemented and unit-tested; full `pytest` suite is green (40 passed, no regressions elsewhere).
**What's broken / incomplete:** Pool exhaustion handling itself (deferral) is still `NotImplementedError` in `env/deferral_queue.py` — intentionally out of scope for this session. `env/environment.py` doesn't wire `PoolSim` in yet.
**Blockers:** None.
**Next session will:** Build `env/deferral_queue.py` (priority/FIFO regret accounting, Addition C) — the env's exhaustion semantics depend on it (Hard Rule 9) per PLAN.md §10 step 3.
**Hard Rules check:** None violated — no security term anywhere. One deliberate design call outside the letter of "don't redesign the signature": `PoolSim.__init__` gained a third parameter, `initial_fill_frac` (required, no default), because the original two-parameter stub (`capacity`, `trace`) had no way to receive it and the task explicitly required pulling `initial_fill_frac` from config with nothing hardcoded in Python. `capacity`/`trace`'s meaning and position are unchanged; `reset/step/can_draw/draw` are untouched. Flagging this here per the instruction to flag anything that touches the frozen interface shape — `env/contracts.py` itself was not touched.

### [SOLO — repo setup] — 2026-08-06 — main

**Session goal:** Get shared scaffolding, the frozen interface contract, and repo housekeeping solid before starting real feature work — currently solo across all four areas until the rest of the team is available.

**What got done:**
- Verified `env/contracts.py` is complete: `Action` enum, `StateDict`, `ForecastProvider` ABC, `Request`, `RegretEvent`/`DeferredCriticalStep`/`ForcedRekey` event-log TypedDicts.
- Verified the full module skeleton exists across `env/`, `agents/`, `forecaster/`, `metrics/`, `dashboard/`, `api/`, `attack/`, `experiments/` — real interface stubs (`raise NotImplementedError`), not empty files.
- Verified `pytest` is green (22 import-smoke tests passing) and CI (`.github/workflows/tests.yml`) + the PR template's Hard Rules checklist are wired.
- Created `data/raw/rt_iot2022/` and placed the RT-IoT2022 CSV there — confirmed correctly gitignored (`data/raw/` + `*.csv` excluded, `data/sample/**/*.csv` is the only CSV path allowed to be committed).
- Deleted branches `b1`, `b2`, `b3`, `b4`, `dev` (each had zero unique commits vs `main`) — repo is now `main` + short-lived task branches going forward.
- Added this file (`SESSION_LOG.md`) to repo root.

**What's working:** Repo scaffolding, frozen contract, and CI/PR process are solid and ready for real feature work.
**What's broken / incomplete:** No real logic anywhere yet — every module still raises `NotImplementedError`. `handoffs/HANDOFF_*.md` intentionally not added yet — deferred until the rest of the team resumes.
**Blockers:** None. The next step (`env/pool_sim.py`) needs a synthetic SKR/QBER trace generator, not an external dataset, so nothing is blocking it.
**Next session will:** Build `env/pool_sim.py` — trace-driven refill/drain/exhaustion + unit tests, using a documented synthetic SKR/QBER generator (PLAN.md explicitly sanctions this over sourcing a real CV-QKD trace).
**Hard Rules check:** None violated. No security term added anywhere. RT-IoT2022 is placed but not yet loaded by any code.

*(Older sessions go above this line as they happen — this entry stays as the earliest record.)*
