# SmartKeyNet — Session Log

> Append-only. Newest entry at the bottom of its date section. This file
> and `PROGRESS.md` are updated together as the last step of every
> session (see PROGRESS.md's update convention).
>
> **Note:** this file was created on 2026-08-19. Earlier sessions
> (2026-08-08 through 2026-08-18) are referenced throughout the codebase
> and `PROGRESS.md` but their log entries were not present in the working
> copy this session started from — the tree had no `.git` directory and
> no `SESSION_LOG.md`/`split.md`. Their findings survive in `PROGRESS.md`
> and in module docstrings; nothing has been reconstructed or invented
> here.

---

## 2026-08-19/20 — full build: scenarios, forecaster, steering attack, demo surface

Single long session. Goal: take the repo from "spine implemented, S1
only" to the full 7-panel demo in `mock.html`, following `PLAN2.md`.

### 0. Repository state as received

The working copy **had no `.git` directory** — it was an extracted
archive, not a clone. The enclosing git repository was the user's home
directory, on an unrelated branch. `git init` was run inside
`smartkeynet-main`, the received tree committed verbatim as a `main`
baseline so all subsequent work is reviewable as a diff, and `dev` cut
from it. There is no remote, so the PR cannot be opened from here.

`SESSION_LOG.md` and `split.md` did not exist. `PLAN2.md`, `PROGRESS.md`,
`README.md` and `mock.html` did. `RT_IOT2022.csv` was at
`data/raw/RT_IOT2022.csv` (not `data/raw/rt_iot2022/`); both paths are
now accepted. `pytest` on arrival: **400 passed**.

### 1. Environment recalibration (pre-registered before any training run)

Before writing feature code, one probe was run against the existing S1
environment. It found that **the scarce resource the whole project is
about was not scarce**, and three further defects behind it. All four
were fixed in the direction that makes the comparison *harder* for the
agent, and all were frozen before any training run so none could be
post-hoc tuning toward a result. Hard Rule 7 directs exactly this
("investigate environment design first").

**(a) The QKD pool was effectively infinite.** Refill was 200 kbps ×
1000 = ~203,000 bits/step against a 256-bit draw with at most one
decision per step — the link funded ~793 hybrid keys per key consumed.
`pool_fill` pinned at 1.0 by step 2 and never moved.

- `AlwaysHybridPolicy` — the baseline whose entire job is to drain the
  pool — produced **0 exhaustion events, 0 regret events, 0 deferrals**
  on S1. PLAN2 §7.4's Panel 4 premise could not occur.
- Every threshold in `baselines.static_threshold_grid` saw
  `pool_fill = 1.0 > t`, so `StaticThresholdPolicy` was **byte-identical
  to `AlwaysHybridPolicy`** (both `total_reward = -64854.6` on S1/seed
  0). The mandatory tuned-threshold baseline was not a distinct policy.
- The reward's `-w_qkd * bits` charged **-256.0 per draw** against a
  -1.5 latency term: hybrid was a cliff to avoid, not a budget to spend.

**(b) The baselines never reused a live session key.** All three tier
policies re-established fresh key material on *every* request — measured
on S1/seed 0: a tier action on 250/250 decisions while REUSE was legal
on 244 of them. A first smoke campaign showed the DQN "beating" every
baseline by >10× on `total_reward`, which on inspection was almost
entirely *"the DQN discovered REUSE"*. A baseline that hands the agent a
free 10× is not a tuned baseline, and it is the wrong physics: under
ETSI GS QKD 014 key material is consumed at key *establishment*, not per
request against a live session key.

**(c) Pool sizing used the wrong denominator.** Once baselines reuse,
establishments are ~40× rarer than requests, so a refill sized against
request volume went slack again. Re-derived from measured
key-establishment demand, bracketed on both sides (50-node graph,
2000-step episodes, seeds 0-2, unlimited pool — demands, not outcomes):

| | bits/step |
|---|---|
| floor-mandated demand, S1 (CALM) | 4.66 |
| floor-mandated demand, S2 (HIGH) | 12.28 |
| maximal demand (always-hybrid) | 20.98 |

`refill_bits_per_step = 15.0` sits inside that bracket (~72% of maximal,
~1.7× mandated on S2). `tenant_graph.n_nodes` 10 → 50 (the plan's own
stated target): node count sets concurrent sessions and therefore
establishment rate, and 6 sessions produced too few events per episode
to carry information.

**(d) The request queue ran at utilisation ρ = 1.0.** The environment
renders exactly one decision per tick, so an arrival rate of 1.0 put the
queue at its critical point — unbounded growth by definition. Measured
on S1/always-PQC over 2,000 steps: pending depth 2 → 14 → 56 and still
climbing, `load` pinned at its cap, and the reward's
`-r_starve * deferred_critical_steps` term diverging with it, which blew
DQN batch loss from ~2.6e4 to ~7.9e5 in one 3,000-step run. This was a
**known defect** — `random_request_generator`'s own `load_spike`
docstring works around it — documented but never fixed. Arrival rate
1.0 → 0.8; the queue is now stable (depth 0–14, ends at 0).

Also: harness episode length 250 → 2,000. At `max_key_age_steps = 500` a
250-step episode is *half of one cryptoperiod*, so sessions almost never
aged out and the entire reuse-vs-rekey tradeoff that `w_fr` and
`c_rekey(load)` exist to create was invisible.

Premise after all four fixes (2,000 steps, seeds 0-2, mean reward):

| | always-hybrid | always-PQC | thr@0.3 | thr@0.9 |
|---|---|---|---|---|
| S1 | 41-75 exhaust, -5614 | 0 exhaust, -782 | -991 | -936 |
| S2 | 90-125 exhaust, -18409 | 0 exhaust, -896 | -993 | -945 |
| S3 | 69-106 exhaust, -17373 | 0 exhaust, -782 | -981 | -930 |
| S4 | 39-56 exhaust, -6033 | 0 exhaust, -806 | -972 | -915 |

### 2. Two Hard Rule 2 violations found and closed

While reading the first Gate W3 output: **`compute_mask` gated `REUSE`
on key *age* only, never on the tier the existing key delivers.** Because
`PolicyTable`'s ratchet is deliberately one-way, a session that
established a PQC key under CALM posture kept reusing it after the floor
ratcheted to SERVE_HYBRID.

> Measured on S2 (2,000 steps, seed 0, always-PQC): **275 of 1,788 REUSE
> decisions — 15.4% — delivered key material below the request's current
> floor.**

It was invisible because `experiments/harness.py`'s `floor_violations`
counter compared the *chosen action* against the floor, and REUSE is not
a tier action, so it reported 0 throughout. `REKEY_NOW` is a second such
path and is reachable even at flat CALM posture (sessions are keyed on
(tenant, service) while the floor is a function of the *request's*
sensitivity class, so two requests on one session can carry different
floors); once the counter was corrected, S1/seed 1 surfaced 2 violations
immediately.

Fixed as masking rules 4 and 5, plus a counter that measures *delivered
tier*. Verified after: 0 below-floor REUSE decisions on S1 and S2, and 0
harness-measured floor violations across 4 scenarios × 4 policies × 3
seeds. `env/contracts.py` untouched.

### 3. Floor table corrected against its own documented intent

`env/masking.py`'s comment block states that S3 (patient-record-grade,
decades-long lifetime) gets SERVE_HYBRID "even before any threat
elevation". The table did not implement it — `(S3, CALM)` gave
SERVE_PQC. **Consequence: no CALM-row entry mandated hybrid anywhere, and
S1/S3/S4 all run at CALM posture, so in every scenario Gate W3 was
measured on, nothing ever mandated a QKD draw at all.** Since Hard Rule 1
keeps security out of the reward, an unmandated hybrid serve is pure
cost — so "never spend the pool" was optimal *by construction*, and
`AlwaysPQCPolicy` was unbeatable for structural reasons rather than
because it budgets well.

Corrected to SERVE_HYBRID. Grounding: S3 is the decades-long-
confidentiality class, i.e. precisely the HNDL target; SP 800-57 and
CNSA 2.0 both say protect longest-lived data strongest and soonest; a
floor that waits for a *current* threat elevation before protecting data
whose exposure window is decades has the HNDL threat model backwards.
This is a floor **raise**, so Hard Rule 2 is unaffected in direction, and
it makes the environment strictly harder for every policy including the
agent. Made *after* Gate W3's first run came back negative, and both the
before and after results are reported.

### 4. Gate W3 — the make-or-break result. **FAILED.**

Run through `experiments/campaign.py`: 5 training seeds, checkpoint-
averaged over the last 4 `eval_every` windows (steps 17,500-25,000), each
snapshot itself averaged over 5 fixed eval seeds, 2,000-step episodes.

**S1** (post-correction):

| policy | total_reward |
|---|---|
| masked DQN | **-3820.8 ± 1623.9** |
| static-threshold (tuned) | **-955.5 ± 7.7** |
| always-PQC | -872.0 ± 7.8 |
| always-hybrid | -8890.8 ± 2680.9 |
| random | -61401.9 ± 11422.6 |

**S3**:

| policy | total_reward |
|---|---|
| masked DQN | **-97475.6 ± 204475.5** |
| static-threshold (tuned) | **-945.8 ± 8.6** |
| always-PQC | -872.0 ± 7.8 |
| always-hybrid | -35509.9 ± 14701.0 |

**Verdict: TUNED THRESHOLD WINS on both S1 and S3.** Reported as-is. No
reward term was added, no masking weakened, no environment tuned toward
the agent.

Two things the agent *did* learn, worth recording:

- `forced_rekey_ratio` **0.107-0.211** against **0.93-1.00** for every
  baseline: the agent rekeys **proactively**, which is exactly the
  behaviour four prior diagnostic sessions were chasing and never
  obtained. The environment fixes in §1 appear to be why.
- But `rekeys_per_100_requests` **66.8** against **10.6** for baselines —
  it rekeys ~6× too often, and at ~2.5 reward units per rekey that
  accounts for almost the whole S1 gap.

The documented checkpoint oscillation is present at full amplitude:
within-run `total_reward` stdev **1447 ± 1411** on S1, comparable to the
mean itself. Not chased, per instruction.

### 5. Dual-head forecaster (Addition A) — trained, real numbers

One shared LSTM trunk over `[threat_features(16) | pool_signals(4)]`,
threat head (now-logit + k=5 horizon logits) + pool head (3 quantities ×
3 horizons). Trainable as a genuinely shared trunk because
`build_rollout_dataset` injects real RT-IoT2022 windows into baseline
rollouts, so every step carries both supervision signals.

8 epochs, 10,253 train / 2,563 validation windows:

- threat head: accuracy **0.9461**, **balanced accuracy 0.9312**, against
  a **majority-class rate of 0.6817**
- pool head: validation MAE **0.189** (unit-RMS-scaled targets)

The first version of this printed 0.9461 against an unstated chance
level, which is not a result; the base rate and balanced accuracy now
travel together.

**Benign-referenced standardization matters and is measured.**
Standardizing against the whole capture centres on attack traffic (90% of
rows) and *inverts* the threat signal: Cohen's d **-0.98** the wrong way,
versus **+4.43** the right way once benign is the reference.

**Prerequisite fix:** the EWMA threat squash was `sigmoid(mean)`, and the
placeholder features `[qber, load]` are both non-negative, so
`threat_score` could never fall below 0.5 — the ELEVATED anchor. With the
one-way ratchet that pinned every episode at ELEVATED from its second
tick (measured: **249/250 decisions** on benign S1) and made the floor
table's entire CALM row unreachable. S2's "threat elevates → floors
ratchet up" is only demonstrable because of the recalibration.

**Also observed, worth flagging:** with a realistic (noisy) detector, a
*single* momentary peak — 1 decision in 2,000, threat score 0.269 at
decision 1205 — is enough to trip the one-way ratchet and change the
floor regime for the remaining 795 decisions. That is Hard Rule 2 working
as specified, not a bug, but it is an operational property a reviewer
should know about.

### 6. S5 steering attack — the headline result. **Holds.**

2,000-step episodes, seed 0, scored at key establishments:

| dose | suppressed | soft-reward: below class floor | masked: below class floor |
|---|---|---|---|
| 0.00 | 0.0% | 14.0% | **0.0%** |
| 0.25 | 25.0% | 24.2% | **0.0%** |
| 0.50 | 50.0% | 27.8% | **0.0%** |
| 0.75 | 74.9% | 27.8% | **0.0%** |
| 1.00 | 97.9% | 27.8% | **0.0%** |

Soft-reward tier histogram slides HYBRID 128 → 0, CLASSICAL 48 → 49, PQC
60 → 178. The masked arm is identically zero at every dose — structural,
not measured luck.

Two honest notes:

- The soft-reward arm is **already at 14.0% with a perfectly honest
  signal**. It has no floor at all, so the attack roughly doubles a
  failure rate that was never zero — a stronger statement than PLAN2 §7.5
  predicts, not a weaker one.
- The curve **saturates from dose 0.5**: the tabular agent bins the threat
  score into 5 buckets, so once suppression reaches the lowest bucket,
  further suppression changes nothing. A property of the reproduction's
  discretization, not of the attack.
- `below_escalated_floor_share` (share below the floor the *honest*
  posture would have set) is **non-zero for the masked arm too — 21.1% at
  full suppression** — because suppression does prevent floors from
  *escalating*; it just cannot push them below the class floor. Reported
  as its own column rather than omitted.

**Three measurement bugs found and fixed while building this**, each of
which had produced a wrong headline number: (1) the soft-reward arm ran
*masked* so it could drive the environment — 0.0% at every dose for both
arms, i.e. measuring the mask rather than the reward design; (2) Q-rows
initialized to zeros, and every real soft-reward value is negative, so
the greedy policy returned whatever it had never tried — a curve flat at
44.0% across all doses; (3) scoring every decision let a REUSE by the
soft-reward arm deliver whichever tier the *driving* policy established
— 0.0% at full suppression, the opposite of the truth. Fixed by scoring
at key establishments with a tier-only mask per arm.

### 7. Demo surface

S6 migration wave (scripted, exogenous, held-out; a lowering schedule
entry is rejected at construction). E-A ablation (all arms on
`rt_iot2022`, since evaluating the LSTM under the scalar `scenario`
source measures a distribution mismatch — observed: -341,702 vs `off`'s
-6,506 on S6 before the fix). Closing results table. Decision-trace
assembler (Hard Rule 10, one source of truth, no generative step). API
facade with the primitive-honesty matrix on `/Health` (ML-KEM is
simulated and says so on every response). Seven-panel dashboard from real
runs, with explicit "not yet run" where an artefact is missing.

`pytest`: **400 → 629 passed** (3 slow deselected).

---

## 2026-08-20 — handoff note (session stopped at a clean point)

Stopped deliberately, not blocked. The working tree is clean, `pytest` is
**632 passed** (629 + the 3 `slow` end-to-end steering runs), and every
unit of work below is committed on `dev`.

### What is done

All eleven `NotImplementedError` stubs the session started with are
implemented and behaviourally tested. `pytest` went 400 → 632.

| area | state |
|---|---|
| Tenant graph + graph-driven `RequestGenerator` | done, Hard Rule 3 swappability asserted by test |
| S1–S6 scenario dispatch | done, each scenario's distinguishing behaviour pinned by test |
| Dual-head LSTM forecaster (Addition A) | trained; balanced accuracy 0.9312 vs a 0.6817 base rate |
| **Gate W3** | attempted for real — **FAILED**, reported as measured |
| **S5 steering attack** | **HOLDS** — masked arm 0.0% at every dose |
| S6 migration wave | done, held-out, lowering schedules rejected at construction |
| E-A ablation / closing table | code done and smoke-tested; **long runs outstanding** |
| Decision trace (Hard Rule 10) | done, one source of truth for `api/` and `dashboard/` |
| API facade | done, primitive-honesty matrix on `/Health` |
| Dashboard (all 7 panels) | done, Dash app + static export, real runs only |
| `docs/report.md` | written; §5.4 awaits the two artefacts above |

Four environment defects and two Hard Rule 2 violations were found by
measurement and fixed — all before any training run, all in the direction
that makes the agent's case harder. Detail in this file's main entry.

### What is next

**The single decision** (also at the top of `PROGRESS.md`): what to do
about Gate W3's negative result. Two pre-committed options are written
out there — (a) test the heavy-tailed-reward hypothesis with standard
reward clipping, with the stopping rule stated in advance, or (b) report
the negative result and reframe around the steering contribution, which
`docs/report.md` §6.3 already does.

**Mechanically next**, before anything else: run the two outstanding
experiment commands listed in `PROGRESS.md` → "Unfinished when this
session stopped", then commit `results/foresight_ablation.json`,
`results/closing_table.json` and `dashboard/index.html`, and fill in
`docs/report.md` §5.4.

### Gotchas for whoever picks this up

1. **There is no git remote.** The working copy as received had no
   `.git` at all (it was an extracted archive; the enclosing repository
   was the user's home directory on an unrelated branch). `main` is a
   verbatim baseline of the received tree and `dev` holds all the work,
   so `git diff main..dev` is the whole session. `git push` cannot work
   until a remote is added.

2. **`configs/default.yaml`'s `pool:` block is load-bearing and is not a
   tuning knob.** `refill_bits_per_step: 15.0` must stay strictly inside
   the measured demand bracket (floor-mandated 4.66–12.28 bits/step,
   maximal 20.98). Outside it, the budgeting problem stops existing in
   one direction or the other — that was the repo's state as received.
   `tests/test_pool_sim.py::test_configured_refill_sits_inside_the_measured_demand_bracket`
   pins it. **If you change the graph size, the arrival rate, the key
   lifetime, or the floor table, re-measure the bracket** (the probe is
   `AlwaysPQC` vs `AlwaysHybrid` with an unlimited pool) and update both
   the config comment and that test's constants.

3. **Never report a DQN number outside `experiments/campaign.py`.** The
   checkpoint oscillation is unresolved after six sessions and is present
   at full amplitude (within-run `total_reward` stdev 1447 ± 1411 on S1,
   comparable to the mean). `evaluate_against_baseline` in
   `experiments/train.py` is single-checkpoint and is kept only for
   `main()`'s human-readable summary and for tests — its docstring says
   so.

4. **`use_foresight: lstm` needs `threat_input.source: rt_iot2022`.**
   The model trained on 16-dimensional flow windows; the `scenario`
   source emits one scalar, which the provider broadcasts, and the result
   is a distribution mismatch rather than a forecast. `experiments/ablation.py`
   sets this for every arm; anything else running the LSTM must too.

5. **RT-IoT2022 is gitignored and operator-placed.** Both
   `data/raw/RT_IOT2022.csv` and `data/raw/rt_iot2022/RT_IOT2022.csv`
   resolve. Tests that need it skip cleanly (verified: 624 passed,
   5 skipped with the file removed), so CI stays green without it.
   **Do not move or rename it while a background experiment is running**
   — doing exactly that killed the first ablation run this session.

6. **`env/contracts.py` is untouched and should stay that way.** Every
   change this session lives in `masking.py`, `environment.py` or a new
   module. Masking gained rules 4 and 5 via an *optional* parameter, so
   every pre-existing call shape is byte-identical.

7. **The forecaster checkpoint (`checkpoints/*.pt`) is gitignored.** Run
   `python -m forecaster.train` (~40 s) before anything that sets
   `use_foresight: lstm`, or the environment raises a `FileNotFoundError`
   that names the command.

---

## 2026-08-20 — both long runs landed; repo feature-complete and green

Continuation of the handoff above. The two outstanding experiment runs
completed; nothing was re-run with changed settings and no figure was
hand-written.

```
[11:57:11] ablation start
           wrote results/foresight_ablation.json
[12:18:31] results_table start
           wrote results/closing_table.json
[12:21:53] dashboard start
           wrote dashboard/index.html
[12:21:54] ALL DONE
```

### Closing comparison (`results/closing_table.json`)

25 cells, 5 policies × 5 scenarios, agents trained on **S1 only**, 5
training seeds, checkpoint-averaged, S6 held out.

**`floor_violations` is 0.00 ± 0.00 in all 25 cells** — the one column
the architecture actually promises.

The agent looks considerably better here than in Gate W3: **zero
pool-exhaustion and zero regret events on S3, S4 and held-out S6**
(matching the tuned threshold, beating always-hybrid's 87–144 by two
orders of magnitude) while rekeying **proactively** — forced-rekey ratio
0.046–0.196 against 0.86–0.97 for every non-random baseline. Two things
keep that from being a win, and `docs/report.md` §5.4 says both: the
agent buys the record with ~6× more rekeys, which `total_reward` prices
and PLAN2 §7.7's column list does not; and **S2 is bad and unstable**
(211.44 ± 290.46 exhaustion events — stdev larger than the mean).

`p99_latency` is omitted from the reported table and the omission is
stated: every policy in every scenario scores exactly **1.500**. It takes
one of four discrete values and saturates whenever any hybrid serve
occurs, which is essentially always.

### E-A foresight ablation (`results/foresight_ablation.json`) — sharp negative

Trained on S3, 5 seeds, checkpoint-averaged; all arms on
`threat_input.source: rt_iot2022`.

| eval | metric | `off` | `ewma` | `lstm` |
|---|---|---|---|---|
| S3 | total_reward | −44,844 ± 75,685 | **−4,669 ± 5,569** | −9,766,367 ± 1,723,435 |
| S3 | exhaustion | 23.61 ± 23.30 | **2.00 ± 4.47** | 787.25 ± 99.38 |
| S6 | total_reward | −5,165 ± 7,154 | **−2,166 ± 1,175** | −10,944,541 ± 5,066,929 |
| S6 | exhaustion | 21.88 ± 48.93 | **0.00 ± 0.00** | 890.56 ± 256.69 |
| S6 | deferred steps | 307.2 ± 686.9 | **0.00 ± 0.00** | 1,094,059.8 ± 506,541.1 |

**Anticipation is worth a lot; the *learned* forecaster is a disaster.**
`ewma` improves on `off` on every operational metric on both scenarios
and reaches zero exhaustion and zero deferral on the held-out one.
`lstm` is two to three orders of magnitude worse than `off`.

This is the sharpest negative in the project and it inverts PLAN2 §11's
cut-order, which treats the LSTM head as the optional part and the EWMA
as the fallback. On this evidence the fallback is the better system.

`docs/report.md` §5.5 records a **hypothesis, explicitly marked
unverified**: the LSTM is a more sensitive detector (balanced accuracy
0.931), the ratchet is deliberately one-way, so it should trip earlier
and hold posture at HIGH — where mandated demand (~12.28 bits/step)
exceeds S3's collapsed refill (~2.25 bits/step) and the deferral queue
diverges. **The confirming posture-trajectory probe was not run**, and is
now the first item under "Next task".

### Wiring and corrections

- `docs/report.md` §5.4 (closing table) and a new §5.5 (ablation) carry
  the real numbers; abstract, §6.1, §6.2 and the conclusion updated to
  carry both negative results.
- `dashboard/index.html` regenerated: **zero "Not yet run" placeholders**,
  all 25 "0 — structural" cells.
- Five test assertions added for the now-populated panels (the
  missing/corrupt-artefact tests are unchanged — they monkeypatch
  `RESULTS_DIR` to an empty directory and still exercise the placeholder
  path).
- **Correction:** an earlier note in `PROGRESS.md` said the ablation
  artefact feeds dashboard Panel 1. It does not — nothing reads
  `results/foresight_ablation.json` in `dashboard/data.py`. The
  artefact-backed panels are 5 and 7. Surfacing the ablation in the
  dashboard would be new work and was not done.
- `results/dashboard_payload.json` gitignored (~700 KB of regenerable
  replay traces; a build input to the HTML, not a result).

### State

**No config, environment or experiment knob was touched.**
`refill_bits_per_step`, graph size, arrival rate, key lifetime and the
floor table are all exactly as they were when the runs were launched, so
the committed numbers correspond to the committed code.

`pytest`: **637 passed**; **632 passed / 5 skipped** without the
gitignored dataset. Nothing left running in the background. Not pushed —
there is still no remote (see the handoff note's gotcha 1).
