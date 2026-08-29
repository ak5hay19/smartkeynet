# SmartKeyNet — Progress Tracker

> **Update convention:** updating this file's checkboxes and the "Next task"
> line is part of the same end-of-session step as updating `SESSION_LOG.md`.
> This file gets *updated*, not rewritten, at the end of every session.
> It exists so a fresh Claude Code session (or a new person) can read
> `PLAN.md` + `SESSION_LOG.md` + this file and know what's done and what
> the single next task is, without reconstructing status from log prose.

---

## Next task

**Read-only repo state verification — DONE 2026-08-29.** Ground-truth
checkpoint: branch/sync confirmed clean after pushing one commit of
drift found on both `main`/`dev21` (approved fast-forward, not a
divergence — all four refs now identical, `227c7e8`); test suite
re-confirmed genuinely green (`568 passed, 1 xfailed`, matches prior
claim exactly); this file's per-file status table and milestone
checklist spot-checked against the filesystem — **zero corrections
needed, every claim checked was accurate**; both standing
instrumentation items (p99_latency saturation, exhaustion==regret
identity) reconfirmed as already resolved and documented directly in
`experiments/harness.py`'s own docstrings, not just in session-log
prose. See SESSION_LOG.md's newest entry for full detail. This did not
change any priority below — resume where the entry underneath left off.

**Paper integration of both addenda — DONE 2026-08-27** (real
`docs/smartkeynet_ieee_paper_5.tex` now exists in the repo; see
SESSION_LOG.md's newest entry, "paper integration: S5 limitations +
soft-reward Fig. 5 correction into the real .tex", for the full
comparison/insertion writeup). Both addenda below are now superseded as
"not yet inserted" — their real findings are in the paper's
`\subsection{Limitations}`. Two concrete items remain open from that
session, neither resolved:

- **Fig. 5's TikZ diagram itself still needs a matching visual fix, not
  yet applied — needs sign-off first.** Fig. 5 (`\label{fig:steering}`)
  is a TikZ diagram, not an image; its node text ("$\hat p_t$ enters the
  reward" -> "security term shrinks; latency outbids it" -> "weaker
  tier served") matches the addendum's own reconstructed paraphrase
  almost verbatim, so the reconstruction was accurate — but that also
  means the inaccurate claim lives in the diagram's own node text, not
  in the caption or body prose (which is generic and not inaccurate).
  The Limitations-section text fix is done; the diagram's `s1`/`s2` node
  text (in the `soft` block, around the `\node[softbox]` lines) still
  visually implies a continuous "posture enters the reward, security
  term shrinks" mechanism that the soft-reward addendum's own finding
  contradicts (only `REKEY_NOW`'s discrete tier-resolution reads the
  floor — no continuous term exists). Left unedited, per instruction,
  pending explicit sign-off on the specific node-text change.
- **Table V's numbers do not match the real measured S3-comparison
  figures — flagged, not touched.** Table V currently reads (Masked /
  Soft-reward): p99 lat `150.0`/`120.0`, Exhaustion `0.0`/`19.5`, Regret
  `0.0`/`19.5`, Rekey% `12.1`/`9.5`, Below-floor `0`/`27,301` — internally
  consistent with the abstract/conclusion's own `27,301`/`42` figures, so
  a real (not placeholder) six-policy result set, but from a different
  run/config than the 2026-08-25 masked-vs-soft-reward S3 comparison
  session's real numbers: p99 `1.5000`/`1.4064`, Exhaustion/Regret
  `0.00`/`3.54`, Rekey% `15.6`/`70.3`, Below-floor rate `0.0000`/`0.1687`.
  Scales don't reconcile cleanly (e.g. table's Exhaustion==Regret==`19.5`
  vs. measured `3.54`; below-floor as a raw count of `27,301` vs. a
  `0.1687` rate). **Not blocking, not urgent, but real**: deciding units
  and re-deriving the full six-policy Table V row set (likely needs
  fresh S3 runs for the four non-RL baselines too, which have never been
  measured against the recalibrated S3 config) is a separate, dedicated
  task.

`docs/report.md` re-confirmed 2026-08-27: still a bare section-outline
skeleton (owners + TODOs only, PLAN.md cross-references), no Fig. 5
prose, no Table V — out of scope for paper integration, the real paper
is the `.tex` file.

Original addenda pointers, kept for history (superseded by the
integration above):
- **`docs/steering_attack_limitations_addendum.md`** (added 2026-08-27, commit
  `81a1a17`, 81 lines) — the masked agent's S5 dose-response findings
  (the one-way-ratchet boundary at alpha>=0.9). **Its "supplementary
  sweep" claim (an honest pre-attack warm-up window as a mitigation) was
  found factually wrong against its own cited source during 2026-08-27's
  integration session — the real data show the opposite (see that
  session's SESSION_LOG.md entry) — the paper was written from the real
  numbers/conclusion, not this file's paragraph verbatim.** This file
  itself was left uncorrected, since editing addendum `.md` files was
  out of scope for that session.
- **`docs/soft_reward_curve_addendum.md`** (added 2026-08-27, commit
  following `9eec891`) — the soft-reward baseline's own curve-shape
  mechanism PLUS a proposed correction to the paper's Fig. 5
  reward-mechanism description. Checked against its cited source during
  integration and found accurate; inserted into the paper with only
  notational conversion. Its own "exact suggested edit" section (a
  reconstructed paraphrase) is superseded — the real Fig. 5 sentence
  was located during integration and matched the reconstruction closely.

**The soft-reward baseline's own S5 `V(pi)` curve shape — RESOLVED
2026-08-27 (see SESSION_LOG.md's newest entry for the full mechanism trace).**
Real, code-verified answers, no retraining needed (reused the still-on-disk
`checkpoints/soft_reward_baseline_s2.pt`):
- `agents/soft_reward_baseline.py::compute_soft_reward` does **not** read
  `threat_score`/`posture_probs` at all — `security_score(tier)` is a fixed
  3-value lookup keyed only on the delivered tier. The one real exception:
  `state["policy_floor"]` (a discrete, one-way-ratchet-processed value, the
  SAME mechanism the masked agent's own floor uses) feeds `REKEY_NOW`'s
  delivery resolution (`max(existing, floor)`) — a narrow, direct,
  non-Q-network-mediated posture channel, but only for that one action type.
- **Flag for whoever does final paper integration**: if the paper draft's
  Fig. 5 narrative describes a continuous `p̂_t` term entering this agent's
  reward with the security term shrinking under attack, **that does not
  match what was actually built** — the real design is a fixed, tier-only
  security term plus one narrow discrete side-channel, not a continuously
  threat-weighted term. Revise Fig. 5's description (or the surrounding
  prose) to match the mechanism above rather than the originally
  anticipated shape.
- The flat-then-step curve shape is the **same discrete-posture-bucket-
  crossing mechanism** already established for the masked agent's own
  alpha>=0.9 finding: a finer-grained re-run (alpha step 0.05 near the
  boundary) pinned the soft-reward agent's own crossing to `alpha 0.80 ->
  0.85` (posture-bucket-mismatch rate jumps `0.0224 -> 0.8112` in that exact
  interval, coincident with `V(pi)_true`'s own step `0.2896 -> 0.3256`).
- A direct, controlled same-state-except-posture comparison (100
  tier-establishing decisions at alpha=0.9, 16 posture-divergent) found the
  Q-network's chosen ACTION never changes (0/16) — the agent reliably picks
  `REKEY_NOW` regardless of perceived posture — but the TIER THAT `REKEY_NOW`
  DELIVERS changes in 6/16 of those cases, purely via the deterministic
  resolution formula reading the attacked `policy_floor`. This corrects the
  prior session's own softer "Q-network sees different inputs and decides
  differently" hypothesis with a more precise, evidence-backed mechanism.
- The already-substantial alpha=0 baseline (`0.2896`, no attack at all) is
  confirmed structural, not attack-related — same root cause as the
  already-documented unmanipulated S3 below-floor rate (`0.1687`): this
  agent's reward has no incentive to proactively rekey once holding a
  tier, so a floor that ratchets up mid-episode (S2's own scripted
  `elevate_at_step=50`) leaves an already-established lower-tier session
  key simply `REUSE`d indefinitely, with or without any attacker.

**The S5 steering-attack dose-response sweep is now real and complete
(2026-08-26) — PLAN.md §5's last scenario row, the paper's headline result.**
`attack/attacking_provider.py::AttackingForecastProvider` (live equation-7
wrapper) + `experiments/harness.py`'s dual-tracking measurement
(`run_scenario_under_attack`/`evaluate_attack_multi_seed`, real `V(pi)` per
paper eq. 4, measured against TRUE posture via a parallel shadow
`PolicyTable`, never the agent's own attacked floor) ran the real 11-alpha
sweep on S2 (masked DQN vs. soft-reward baseline, fresh 25,000-step
checkpoints for both, 5 eval seeds each). **The central, most important
finding: the masked agent's `V(pi)` is NOT flat zero everywhere, contradicting
this thread's own pre-registered prediction** — `0.0000` through alpha=0.4,
rising to `0.0016-0.0096` at alpha 0.5-0.8, then `0.3000` at alpha>=0.9.
Investigated immediately and rigorously, per Hard Rule 7, to a confident
mechanistic conclusion, **not left as an open bug**:
- **Ruled out a `compute_mask`/`PolicyTable` implementation bug directly**:
  the masked agent's delivered tier is below the floor it was actually
  *shown* exactly `0` times, at every one of 5 eval seeds, at every one of
  11 alphas — Hard Rule 2's literal guarantee (the mask enforces whatever
  floor it computes) holds perfectly throughout.
- **The real mechanism, traced per-decision**: `AttackingForecastProvider`
  shapes every window from tick 1 onward, including before S2's own
  `elevate_at_step=50`. At alpha=0.7 the real signal is only diluted, not
  erased — the live environment's own ratchet still detects it and locks
  HIGH within 2 decisions of the schedule firing. At alpha>=0.9 the real
  signal is suppressed so completely that the ratchet **never fires at
  all** for the rest of the episode — there is no already-detected floor
  for the one-way ratchet to defend, because its only channel of truth
  (the same forecaster the attack shapes) never detected anything.
- **This is a real, now-precisely-quantified boundary of the one-way
  ratchet's defense, not a masking defect**: it stops an adversary from
  talking an *already-ratcheted* floor back down; it cannot manufacture a
  detection the sensor was never allowed to make when the attack is strong
  and persistent enough to suppress the true signal for the entire episode.
  A supplementary, scratchpad-only delayed-onset sweep (attack applied only
  from `elevate_at_step` onward, leaving the — mostly irrelevant —
  pre-elevation portion untouched) reproduced nearly identical numbers,
  confirming the gap is about continuous suppression during/after the real
  elevation window, not an artifact of shaping the pre-elevation noise.
- **The soft-reward agent's own `V(pi)` also did not match the paper's
  clean "rises monotonically with alpha" prediction**: already substantially
  nonzero at alpha=0.0 (`0.2896` — this agent has no masking at all, so it
  violates real floors as a baseline property, independent of any attack),
  flat through alpha=0.8, then a step to `0.3256` at alpha>=0.9 — the same
  "needs enough dilution to cross a threshold" shape the masked agent's own
  curve has, just starting from a much higher floor, not the smooth curve
  originally anticipated.

**This finding belongs in the paper's existing Limitations section** — cite
this session's exact figures (masked `V(pi)`: `0.0000` at alpha<=0.4 up to
`0.3000` at alpha>=0.9; 0 attacked-floor violations throughout, confirming
it is a ratchet-scope boundary, not an implementation bug) rather than
presenting the split-screen result as an unconditional, always-zero
guarantee. **A possible mitigation direction is flagged here as future
work only, not built this session, for whoever next picks up
masking-adjacent work**: e.g. a conservative default/startup floor that
doesn't relax until the forecaster has observed some minimum variance in
its input (a suspiciously flat/constant sensor reading for an extended
window could itself be treated as a signal that the input channel may be
compromised, rather than trusted as genuine calm). See SESSION_LOG.md's
newest entry ("S5 steering-attack dose-response sweep: masked DQN vs
soft-reward baseline") for the full numbers, per-decision mechanism trace,
and Hard Rules 2/7 reasoning.

**Remaining lower-priority items, now that the headline result exists in
real form**: the standing `env/forecast_provider.py` posture-ceiling item
(below, unchanged), the real LSTM dual-head forecaster (Addition A), and
Dashboard v2 (Thread 2, below).

---

**Threads 1 and 3 are now both closed, and the soft-reward baseline agent is
now real too (2026-08-25).** Gate W3 is genuinely, fully met on both halves
(2026-08-24), **the full S1-S6 scenario grid is real and dispatched
(2026-08-24)**, and **`agents/soft_reward_baseline.py` — the Noetzold-style
soft-reward baseline agent (PLAN.md §5 S5's future steering-attack target) —
is built and tested (2026-08-25), no longer a stub.** See SESSION_LOG.md's
newest entry ("soft-reward baseline agent (Noetzold-style reproduction)") for
the full design/testing writeup. **This session's real milestone**: the
agent reuses `agents.dqn.DQNAgent` completely unmodified (no new agent
class — only its reward function and config differ); its "no action
masking" property required a signed-off, minimal, additive
`env/environment.py` config flag (`security_masking`, design decision 16)
after this session found the mask is enforced at the environment boundary
(`step()`'s `IllegalActionError`), not just in how an agent reads it — a
genuine surprise, flagged via `AskUserQuestion` and resolved with explicit
user sign-off before being built, per this repo's own "stop and flag before
touching protected files" convention. A real 25,000-step training campaign
confirmed the property this agent exists to demonstrate: `floor_violations >
0` at every one of 10 eval checkpoints (12-184 out of a 250-step episode) —
direct, measured evidence security is genuinely soft and gets traded away,
not inferred from the reward formula alone.

**Follow-up (1) is now DONE (2026-08-25, later session) — the real,
unmanipulated masked-DQN-vs-soft-reward-baseline comparison on S3 exists,**
using the same checkpoint-averaged (3 training seeds x 8 eval seeds)
methodology Gate W3 established. See SESSION_LOG.md's newest entry
("masked DQN vs soft-reward baseline on real S3") for full numbers. **The
real result does NOT match the paper draft's anticipated "soft-reward
wins efficiency, trades away security" tradeoff shape** — reported
honestly per Hard Rule 7, not reshaped to fit: the masked DQN comes out
ahead or comparable on **every** metric measured, not just below-floor
rate:
- **below-floor service rate** (the structural, expected divergence):
  masked `0.0000` vs. soft-reward `0.1687` (1012/6000 decisions across
  all seeds).
- **regret/pool-exhaustion events** (a second, previously-unemphasized
  failure mode): masked `0.00` vs. soft-reward `3.54` mean — the
  soft-reward agent's reward has no pool-scarcity/starvation term at all,
  so on S3's genuinely scarce pool it can fail on availability too, not
  just security.
- **forced_rekey_ratio**: masked `0.156` (mostly proactive, the "better"
  direction) vs. soft-reward `0.703` (mostly forced) — the opposite of
  "soft-reward is more efficient at rekey timing."
- **total_reward variance**: soft-reward's training-seed std (`12380.91`)
  is ~10x the masked agent's (`1303.25`) — the masked agent is both safer
  and far more consistent run-to-run.
- **p99_latency**: close (`1.5000` vs. `1.4064`), and the previously-
  flagged non-discriminating-metric oddity reappears (5/6 policy-seed
  cells tie at exactly `1.5000`), with one genuine exception this session
  (soft-reward seed 1: `1.2191`) — **root-caused 2026-08-25 (later,
  diagnostic session) — NOT A BUG, resolved.** Reloaded all six real S3
  checkpoints and tallied the exact per-decision `cost_action`
  distribution across all 48 eval episodes (3 training seeds x 8 eval
  seeds x 2 agents): `np.percentile`'s linear interpolation on a
  250-length array reads the 246th/247th smallest values (0-indexed),
  so `p99_latency` reports exactly `1.5000` (SERVE_HYBRID's real,
  uncapped per-decision cost — the max of a 4-value discrete set, not a
  ceiling) whenever >=4/250 decisions (>=1.6%) cost SERVE_HYBRID — true
  for essentially every real policy tested on S3, since its
  scarcity-driven floor makes that common. Verified exactly, not
  approximately: every episode over the threshold reported `1.5000`
  precisely; the one boundary episode (SERVE_HYBRID count == 3)
  reported `1.3530`, matching the interpolation formula to 4 decimals —
  this also explains the one-exception-cell mean (`1.2191`) down to the
  last decimal (mean of that checkpoint's 8 eval-seed values, 7 at
  `1.2000` + 1 at `1.3530`). No cap/ceiling/timeout exists anywhere in
  the latency-computation path (`env/environment.py::_apply_action`,
  `env/deferral_queue.py` both confirmed clean) — this is a mechanical
  percentile-over-discrete-data artifact, not a defect, so no code fix
  was made. `experiments/harness.py`'s `ScenarioResult.p99_latency`/
  `MultiSeedEvalResult.p99_latency_mean`/`_std` docstrings gained the
  full mechanism and now point to `total_reward`/`below_floor_rate` as
  the sharper metrics for S3 comparisons. See SESSION_LOG.md's newest
  entry ("root-cause p99_latency saturation on S3, diagnostic") for the
  full verification. **All five Table V metrics from the prior
  session's comparison stand as reported — none needed correction.**

**Follow-up (2)'s first half is now DONE (2026-08-25, later session) — the
attack generator (`attack/trace_generator.py`) is real and tested,
implementing the paper draft's equation 7 input-shaping attack
(`x̃t = (1-α)xt + α·g(xt)`).** See SESSION_LOG.md's newest entry
("adversarial trace generator (steering attack, eq. 7)") for the full
design/testing writeup. `g(xt)` returns an all-zero window (the real floor
of the current placeholder `[qber, load]` forecaster's benign region) —
deliberately targeted at the CURRENT placeholder forecaster, not the future
LSTM one; will need revisiting once that exists. Verified end-to-end
through the real `MovingAverageForecaster`, not assumed from the formula: a
genuinely severe `true_window=[5.0,5.0]` (mirroring how S2's own
`elevated_signal` mechanism already reaches HIGH, since ordinary in-domain
`[qber,load]` values are structurally capped at ELEVATED per the
2026-08-24 posture-ceiling finding below) steady-states to HIGH posture,
and shaping it at `alpha=1.0` steady-states to ELEVATED — a genuine
discrete floor crossing, not just a lower number. The masking-safety
property (this session's most important test) was verified directly with
real numbers: for `sensitivity_class=S2`, `compute_mask()` given only the
attacked, underestimated `SERVE_PQC` floor never legalizes `SERVE_CLASSICAL`
(below even the wrong floor), while `SERVE_PQC` itself becomes legal under
the attack where it wasn't under the true `SERVE_HYBRID` floor — the
attack's real, measured effect, with the safety guarantee intact — plus a
general, exhaustive 12-cell (4 sensitivity classes x 3 postures) sweep of
the same property. Needed **zero changes** to `env/forecast_provider.py`,
`env/masking.py`, or `env/environment.py` (Hard Rule 3), verified via
`git diff --stat`. `dashboard/explain.py` was read and confirmed to already
report a shaped/lowered posture consistently, with no changes needed.

**Superseded 2026-08-26 — the actual S5 dose-response sweep described below
has now been run for real** (on S2, not S3 — S2's own scripted
`elevate_at_step`/`elevated_signal` schedule genuinely reaches HIGH posture,
avoiding the standing posture-ceiling limitation that would have narrowed
S3's informative alpha range). See "Next task" above and SESSION_LOG.md's
newest entry for the full result — the paper prediction was **not**
confirmed as stated (masked agent's `V(pi)` is not flat zero everywhere),
investigated to a real, non-bug mechanistic conclusion, reported honestly
per Hard Rule 7. Original framing, kept for history: PLAN.md §5's last
remaining scenario row, PLAN2.md §7.5's Panel 5, the headline
steering-attack result: running both agents (masked DQN, soft-reward
baseline) against S3 (or S5's own config, TBD that session) across eleven
alpha values, reporting `V(π)` (below-floor service rate,
`experiments/harness.py::MultiSeedEvalResult.below_floor_rate_mean`,
already built) for both, with the explicit paper prediction (soft-reward
rises, masked stays flat at zero) confirmed or honestly reported otherwise
per Hard Rule 7. Both agents being compared are real, their unmanipulated
baseline behavior is measured on both S1 and S3, and the attack generator
that will drive the sweep is now real and proven correct in isolation —
every precondition this thread has named is now satisfied. Explicitly NOT
run yet — this session's own instruction was to build the generator only,
not the sweep.

PLAN.md §5's entire scenario table now has working, tested code AND a real
run behind every row, including S5 (2026-08-26 — see above).

**One thread remains open — Thread 2's Dashboard v2** (blocked on dataset
ingestion for its next concrete step, but has unblocked alternatives — see
Thread 2 below). **`experiments/train.py::train()`'s hardcoded `scenario:
"S1"` override — RESOLVED 2026-08-24, after being flagged three separate
times across three prior sessions (the multi-seed eval probe, the S3 Gate
W3 attempt's scratchpad workaround, and the S6 build session) without being
fixed.** `train()` now takes a real, explicit `scenario: str = "S1"`
parameter, following `experiments/harness.py`'s `run_scenario`/`run_grid`
convention exactly — this directly unblocked the soft-reward baseline
agent's own training entry point (`experiments/train.py::
train_soft_reward_baseline`, additive, 2026-08-25), which will itself need
to run on multiple scenarios for follow-up (1) above. **One standing,
unrelated open item remains flagged, not part of any thread**:
`env/forecast_provider.py`'s placeholder threat-feature formula's posture
ceiling (below).

**The `env/forecast_provider.py` open item, unchanged since 2026-08-24's
earlier S3 session, not part of any thread**:
`env/forecast_provider.py`'s placeholder threat-feature formula
(`_threat_features_placeholder` -> `MovingAverageForecaster`) has a
confirmed structural property — the discrete posture classification
(`argmax(posture_probs)`, the only thing `env/masking.py`'s floor table
reads) can **never** reach HIGH for any real `[qber, load]` input, however
severe, because both features are bounded in `[0,1]` and the resulting
sigmoid-squashed signal tops out at ~0.731, which the RBF-softmax always
resolves toward ELEVATED over HIGH. QBER itself is *not* negligible (it does
move `threat_score`/`posture_probs` measurably) — the ceiling is on the
*discrete* posture only. Read/investigate-only, per that session's
instruction (`env/forecast_provider.py` untouched) — flagged here as a real,
precisely understood, not-yet-fixed property, distinct from S2/S3/S4/S6's own
scenario work.

**Thread 1 — DQN training-instability → Gate W3 (2026-08-19: instability
substantially resolved, real gate genuinely attempted, S1 half MET, S3 half
blocked on an environment-design decision; 2026-08-24: S3 half recalibrated
and genuinely attempted for real — MET. Gate W3 is now fully closed.)**

**2026-08-24 update — S3 half resolved.** A dedicated session recalibrated
`configs/scenarios/s3_degradation.yaml` (S1/`configs/default.yaml` untouched,
byte-for-byte) after finding, precisely, why the 2026-08-19 blocker was
structural rather than a tuning problem: `env/pool_sim.py`'s existing
spike-degradation formula has a hard 50%-SKR-reduction ceiling regardless of
`spike_magnitude` (empirically re-verified: `0.6`/`0.9`/`0.99`/`5.0` all
produce byte-identical in-window SKR), and even a *hypothetical* zero-refill
window couldn't have stressed the prior 1,000,000-bit pool, since one
250-step episode's maximum possible draw volume (`250 * 256 = 64,000` bits)
is under 6.4% of that capacity regardless of refill. Fix: `pool_sim.py`
gained an optional `spike_skr_multiplier` field (additive, `None` preserves
prior behavior exactly), and S3's *own* scenario file (only — zero risk to
S1) dropped `pool.capacity_bits` to `20,000` (computed against the
64,000-bit max-draw-volume figure, not guessed) with
`qkd_degradation.spike_skr_multiplier: 0.0` (grounded in real QKD
security-proof literature: BB84-family protocols provably yield zero
extractable key above a QBER security threshold, ~11% canonically — not an
arbitrary worst case). Verified directly: under the real committed config
files, same seed, S1's pool never drops below ~70% full or produces a
regret event (any seed tested); S3's collapses to near-total exhaustion
(0.16% of its own capacity) with 21-33 real deferral events per 250-step
episode, every seed tested. Gate W3 S3 attempt (same checkpoint-averaged,
multi-training-seed methodology as S1): **DQN `total_reward` -10214.82
(training-seed std 1303.25) beats the tuned `StaticThresholdPolicy`'s
-38566.87 (eval-seed std 1636.12) by ~3.78x — MET**, gap ~21.7x/~17.3x
either side's own spread. Full numbers, the root-cause investigation, and
the posture-saturation investigation (see the new open item above) are in
SESSION_LOG.md's 2026-08-24 "S3 QKD-degradation recalibration, Gate W3 S3
attempt" entry.

A same-day
2026-08-19 session (see SESSION_LOG.md, "close REUSE/REKEY_NOW
floor-enforcement gap, Hard Rule 2") found and fixed a real bug directly
relevant to this thread: `env/masking.py`'s `compute_mask()` let REUSE stay
legal past a floor ratchet, and `env/environment.py`'s REKEY_NOW resolution
could silently refresh a stale below-floor tier — both on S1 too, since S1
already ratchets posture mid-episode under the default `ewma` foresight.
That session's own fork added a third option: re-run the 2026-08-17/18
multi-seed `forced_rekey_ratio` probe with the fix in place. **A later
same-day session did exactly that** (see SESSION_LOG.md, "post-fix re-run of
forced_rekey_ratio multi-seed probe") — identical methodology, same seeds
`1`/`4`/`7`, same load-spike S1 config, same 8-fixed-eval-seed/`eval_every=750`
design, only the underlying masking fix changed. **Result: swings shrank
substantially (mean swing 0.21-0.30 → 0.17-0.18, max swing ~0.89-0.91 →
~0.49-0.54, >0.5 frequency 19-27% → 0-3%) and the ceiling-fraction-by-thirds
drift — the clearest, most consistent pathology across every prior session in
this thread — is completely gone (0% at ceiling across all nine seed×third
cells, versus up to 62% pre-fix).** This is strong evidence the masking bug
was a real, likely-dominant contributor to the checkpoint-to-checkpoint
instability. **It is not a full resolution**: a genuine, smaller residual
remains (max swings still `~0.49`-`0.54`, e.g. seed 1's step
`71250->72000`, mean-of-8 `0.144->0.459`; seed 4 still has 3/99 swings
`>0.5`) — the dominant, most-damaging component (permanent worst-case
ceiling stickiness) is gone, not the phenomenon entirely. Full numbers and
the direct before/after tables are in SESSION_LOG.md's newest entry.

**Fork resolved, 2026-08-19 (same day, later session): moved to attempting
Gate W3 for real rather than the Q-value-margin inspection.** Reasoning: the
pathological failure mode (permanent worst-case ceiling stickiness) was
gone, the residual swinging that remained was ordinary, bounded noise rather
than the dominant effect any of options (a)/(b) were originally framed
against — and the actual project milestone (split.md's Gate W3: "DQN beats
the tuned threshold baseline on S1 and S3") had never been genuinely
attempted at all, for real reasons (S3 didn't exist as a scenario until two
sessions ago; the masking bug made any such comparison meaningless before
that). Attempting the real, defined milestone took priority over a further
diagnostic on an already-substantially-resolved instability. **Result (see
SESSION_LOG.md, "Gate W3 attempt: DQN vs tuned threshold on real S1 and
S3", full numbers there):**
- **S1: gate MET, decisively.** Checkpoint-averaged (3 training seeds x 8
  eval seeds), DQN `total_reward` -5119.71 (training-seed std 47.59) vs. a
  properly `total_reward`-tuned `StaticThresholdPolicy`'s -64616.41
  (eval-seed std 213.22) — DQN wins by ~12.6x, a gap roughly three orders of
  magnitude larger than either side's spread. `p99_latency` ties at exactly
  `1.5000` for both but is confirmed non-discriminating (constant across the
  *entire* threshold grid), not a genuine tie.
- **S3 (as of 2026-08-19): found, before running any training, that it's not
  meaningfully attemptable under current config** — under `configs/
  default.yaml`'s real pool settings, S3's degradation produces byte-identical
  floor/pool/posture trajectories to S1 (verified across seeds 0/1/4/7) —
  refill dwarfs the degraded draw by orders of magnitude, and posture is
  already saturated regardless of the QBER spike. Not run, per instruction
  not to retune S3's config to force a result. This is a real
  environment-design finding (Hard Rule 7: "investigate environment design
  first"), not a DQN or baseline failure. **Superseded 2026-08-24**: a
  dedicated recalibration session (see the "2026-08-24 update" note above and
  SESSION_LOG.md's newest entry) fixed this — S3 now genuinely diverges from
  S1, and the S3 half of Gate W3 was attempted for real and **MET** (~3.78x).
- **A related, second methodological finding**: `experiments/train.py`'s
  existing `evaluate_against_baseline` tunes `StaticThresholdPolicy` via
  single-seed `-p99_latency`, which this session found is constant across
  the *entire* threshold grid under real S1 — meaning `grid_search`'s tie-
  break silently defaults to the grid's first candidate, not a genuinely
  tuned choice. Not fixed this session (that function isn't what produced
  the Gate W3 numbers above — this session's own grid search used
  `total_reward` throughout) — flagged for whoever next touches
  `experiments/train.py`.
- **Real, reusable code added** (this was not a diagnostic-only session):
  `experiments/harness.py` gained `evaluate_multi_seed()` +
  `MultiSeedEvalResult` — runs any `Policy` across multiple eval seeds and
  reports mean/std for the metrics PLAN.md's closing table cares about,
  never a bare point estimate. 6 real tests in `tests/test_harness.py`. See
  that file's per-file row below.

`test_dqn_agent_loss_trends_down_training_against_real_env_s1`'s
`xfail(strict=True)` marker remains untouched — this session's result
doesn't resolve it either; that's still a separate, standing item, not
addressed by a successful Gate W3 S1 result. Do not touch `agents/dqn.py`
for anything short of a deliberate, sign-off'd decision; `experiments/
train.py`/`experiments/harness.py` may gain further real, tested additions
(as this session did) but their existing DQN training/eval logic itself
(not the new `evaluate_multi_seed` addition) should still be changed
deliberately, not incidentally.

**Thread 2 — Dashboard v2 (started 2026-08-19).** `dashboard/explain.py`
(the Explain Decision panel's backend, PLAN2.md §7.3) is now
implemented+tested — see its `dashboard/` per-file row below. The
concrete next Dashboard v2 step is the Threat Input panel (PLAN2.md
§7.1), but that's blocked on Person A's feature-extraction work (the
shared RT-IoT2022/pcap feature-extraction function PLAN2.md §6 and
Hard Rule 11 require) — dataset ingestion hasn't started (see `data/`
below). Until that exists, do not stub a placeholder extraction path for
Threat Input (Hard Rule 11: one shared extraction path, no parallel
pipeline). Other dashboard-adjacent, unblocked options: Panel 2 (Living
System) could start against `dashboard/explain.py`'s real trace output
plus `StateDict`/event-log fields, no new dependency; or start on
`agents/soft_reward_baseline.py` (needed before the steering attack,
PLAN2.md §7.5/§9 S5, can be built).

**Thread 3 — Real scenario dispatch. CLOSED 2026-08-24 — the full S1-S6
grid is now real and dispatched.** S1/S2/S3/S4/S6 scenario dispatch is
real (see `env/environment.py`'s per-file row). S4 (DDoS/noisy-neighbor)
needed a notion of which tenant a request belongs to that persists and
can be targeted — 2026-08-23's session built exactly that
(`build_tenant_graph`/`RequestGenerator`), and **2026-08-24's session
wired real S4 dispatch on top of it**: `config["ddos"]`
(required only under `scenario: S4`) picks a designated low-sensitivity
tenant and floods it for the whole episode via `RequestGenerator`'s new
`flood_override` mechanism — an *additive* second, independent Poisson
arrival stream for that one tenant, not a `traffic_rate` multiply (this
codebase's actual sampling model — a single shared Poisson total split by
weighted multinomial across tenants — would otherwise mechanically dilute
every other tenant's share too; verified, and the additive mechanism
instead leaves every other tenant's own arrival stream byte-for-byte
unaffected). See `env/request_generator.py`'s and `env/environment.py`'s
per-file rows and SESSION_LOG.md's 2026-08-24 S4 entry for the full
empirical findings and mechanism reasoning.

**2026-08-24, same day, later session — S6 (migration wave) wired, closing
this thread.** `config["migration_graph_seed"]`/`config["migration_schedule"]`
(required only under `scenario: S6`) build a tenant graph once, same
pattern as S4, and apply a small, scripted list of `{step, tenant_index,
new_sensitivity_class}` ratchet events via `RequestGenerator`'s new
`set_tenant_sensitivity_class` method — entirely upstream, in request
generation (Hard Rule 3, verified at the code level: masking.py/
contracts.py/the reward calculation have zero mentions of "migration"/
"s6"). Needed **zero `env/masking.py` changes**: both `compute_mask`/
`PolicyTable.floor` and `_prepare_decision` already read a request's
`sensitivity_class` fresh every decision, with no caching anywhere that
assumes it's fixed for an episode (confirmed by reading before designing,
not assumed) — so mutating one tenant's node attribute is sufficient on
its own for the floor to respond correctly. **Hard Rule 8 (train/eval
split) is now a real, code-level guard, not just documentation**:
`experiments/train.py::train()` checks `config["train_eligible"]`
(default `True`, set `False` only in the new `configs/scenarios/
s6_migration.yaml`) before any training work and raises `ValueError` if
selected — keyed on the flag itself, not a hardcoded `scenario == "S6"`
string, so it survived the 2026-08-24 (later session) fix to `train()`'s
then-separately-flagged hardcoded-`"S1"` debt exactly as designed (see
"Next task" above and `experiments/train.py`'s per-file row — that fix
finally exercised this guard through a real, scenario-selectable call).
See `env/request_generator.py`'s and
`env/environment.py`'s per-file rows, `experiments/train.py`'s per-file
row, and SESSION_LOG.md's newest entry ("S6 (migration wave) scenario
dispatch, Hard Rule 8 guard") for the full design reasoning and test
results (14 new tests: 5 in `test_request_generator.py`, 6 in
`test_environment.py`, 3 in `test_train.py`).

---

**2026-08-18's diagnostic recap (Thread 1, unchanged from before the pause):**

**A second 2026-08-18 diagnostic tested both candidate explanations for the
checkpoint-to-checkpoint swings and disfavored both — the mechanism is
still unknown, and even 8-eval-seed averaging doesn't tame it.** See item 6
below for the full result; the standing recommendation to report
`forced_rekey_ratio` "as a distribution across eval seeds" (item 5) is now
known to be insufficient on its own — the distribution's *mean* swings
checkpoint-to-checkpoint almost as hard as a single draw did.

**The 2026-08-18 dense re-probe overturned the 2026-08-17 "mid-training
regression" framing itself.** There is no localized regression event —
`forced_rekey_ratio` oscillates continuously (swings of 0.5+ between
adjacent 1,000-step snapshots, about 1 in 3 of the time) across the
*entire* 1,000-75,000 step range, for every seed tested, including the
one previously read as "flat, never found a good policy." The earlier
3-point sample (25k/50k/75k) was undersampling this noise, not
observing a real found-then-lost event. See SESSION_LOG.md 2026-08-18
for the full data. Five sessions on this thread now (three same-day
2026-08-10, one 2026-08-17, one 2026-08-18); recap in order:

1. First load-spike session: `forced_rekey_ratio` dropped from flat
   S1's `1.000` to `0.256`/`0.872` across two training seeds —
   confirmed direction, spread unquantified.
2. 10-seed follow-up sized the spread (`0.190`-`1.000`, mean `0.735`,
   stdev `0.275`, leaning bimodal) and found why it was worth
   distrusting: `agents/dqn.py` never seeded its own weight
   init/exploration/replay sampling at all — `training.seed` only ever
   reached the environment. Flagged, not fixed.
3. **This session fixed that**: `DQNAgent.__init__` gained a `seed`
   parameter (seeds `random`+`torch` before `QNetwork` construction),
   `experiments/train.py` now threads `training.seed` to both the env
   and the agent, and a new test (`tests/test_dqn.py`) proves same-seed
   agents produce identical action sequences and different seeds
   diverge. Re-ran the identical 10-seed load-spike sweep with the fix
   in place:

   | | sorted `forced_rekey_ratio` | mean | stdev |
   |---|---|---|---|
   | Before fix (unseeded DQN) | `0.190, 0.417, 0.418, 0.703, 0.872, 0.895, 0.914, 0.971, 0.971, 1.000` | 0.735 | 0.275 |
   | **After fix (seeded DQN)** | `0.102, 0.148, 0.463, 0.553, 0.730, 1.000, 1.000, 1.000, 1.000, 1.000` | 0.700 | 0.345 |

   **The fix made the spread wider, not tighter** — exactly half the
   seeds (5/10) now land at the *exact* never-proactive ceiling
   `1.000` (vs. 1/10 before), while the other half ranges `0.102`-
   `0.730`, including the best result seen across either sweep
   (`0.102` — 90% proactive). With RNG-conflation ruled out as the
   explanation (both sweeps used identical config; only the seeding
   fix changed), this looks like a genuine, structural
   learn/don't-learn split: roughly half of random weight-init +
   exploration trajectories discover any proactive rekeying at all
   within a 25,000-step budget, and half settle into "wait until
   forced" and stay there. `reward.w_fr`/`reward.c_rekey_base` remain
   uninvolved in any of this — no `reward.*` change made or requested
   across any of the three sessions.

4. **2026-08-17 budget probe** ran a strict, pre-committed decision
   rule to test the training-budget hypothesis before building
   anything: 3 of the 5
   ceiling-stuck seeds (`1`, `4`, `7`) trained to `50,000` and `75,000`
   steps (up from `25,000`), load-spike enabled, same config. Verdict:
   **DID NOT BUDGE** — no seed cleared `forced_rekey_ratio <= 0.5` at
   75k, and none held a monotonically non-increasing trajectory. This
   isn't just "training was too short": seeds `1` and `4` reached
   genuinely good intermediate values at 50k (`0.1020` and `0.6585`,
   `seed=1`'s matching the best result seen in any sweep) and then
   **regressed back to the exact `1.000` ceiling by 75k**. Seed `7`
   never moved off the ceiling at any budget tested. A real, substantial
   improvement appearing and then being lost mid-training is a
   different, more concerning phenomenon than either of the two
   hypotheses framed on 2026-08-10 (marginal budget vs. static bimodal
   landscape) — it looks more like training instability / policy
   forgetting within a single continuous run than a simple
   convergence-speed question. See SESSION_LOG.md 2026-08-17 for all
   six data points.

5. **2026-08-18 dense diagnostic** re-ran seeds `1`, `4`, `7` to
   75,000 steps with `eval_every=1000` (~75 snapshots instead of 3) to
   localize the 2026-08-17 regression's onset. Found none — max
   single-window swings of `0.90`-`0.92` occur throughout the entire
   run for every seed, not just around the buffer-capacity crossing at
   step `50,000`. **Buffer-capacity hypothesis: ruled out** — loss is
   unremarkable around step 50,000 (matches its neighbors, no spike),
   and swing amplitude doesn't shrink or shift near that step. There
   *is* a real, separate, noisy long-run trend — the fraction of
   snapshots sitting exactly at the `1.000` ceiling roughly doubles
   each third of a run (seed 1: `8% -> 48% -> 64%`) — but it's a
   smooth drift across the whole run, not a step-change tied to
   `12,500` (epsilon-decay-complete) or `50,000` (buffer-capacity)
   specifically, and the swings persist at full amplitude even in the
   run's final third.

6. **2026-08-18's second same-day diagnostic** directly tested both
   candidate explanations item 5 left open, using a measurement-only
   reimplementation of `train()`'s loop (`DQNAgent`/`SmartKeyNetEnv`/
   `GreedyDQNPolicy`/`run_scenario` imported and called exactly as
   `train()` does, not a new training path) run for seeds `1`, `4`, `7`
   to `75,000` steps, `eval_every=750` (deliberately not a multiple of
   `target_update_every=1000`), 8 fixed eval seeds (`900`-`907`) per
   snapshot instead of one. **Hypothesis (a) — single-episode eval
   noise — RULED OUT**: the spread across a checkpoint's own 8 eval
   draws is small (mean `0.05`-`0.07`), but the *mean* of those 8 draws
   still swings 4-6x larger (mean `0.21`-`0.30`, still `>0.5` on
   19-27% of adjacent 750-step snapshots) from one checkpoint to the
   next — averaging away eval-episode randomness does not tame the
   swing. **Hypothesis (b) — eval-cadence/target-sync aliasing —
   evidence against, not a clean rule-out** (between-session
   comparison, not a controlled ablation): the non-aligned
   `eval_every=750` cadence swings with the same character and
   magnitude as 2026-08-18's earlier aligned `eval_every=1000` data
   (max swing `~0.89`-`0.91` both), if anything slightly *less*
   frequently — the opposite of what aliasing inflation would predict.
   The 2026-08-10 ceiling-fraction long-run drift (item 5) was
   confirmed to survive 8-seed averaging too. **Net effect: both
   leading explanations for the swings are now disfavored, and the
   standing "report as a distribution across eval seeds"
   recommendation from item 5 is revised** — it doesn't fix the
   problem, since the distribution's own mean is nearly as unstable
   checkpoint-to-checkpoint as a single draw. See SESSION_LOG.md's
   second 2026-08-18 entry for the full comparison tables.

**Concretely next (pick one — a real decision, not more solo running):**
(a) a direct Q-value-margin inspection at a known swing (e.g. seed 1's
step `71250->72000` transition, `forced_rekey_ratio` mean dropping from
`~0.90` to `0.149`) — check whether the greedy action at the eval
episode's early, trajectory-determining decision points sits at a
near-tie in Q-value between two actions, such that one `learn()` call's
weight update is enough to flip the argmax and cascade into a very
different downstream key-age trajectory for the rest of a short
250-step episode (this would explain instability surviving 8-seed
averaging without the underlying Q-values needing to move much) — not
run yet, no repo code exists for this, would need new (flagged-first)
instrumentation in a copy of the eval loop, not `agents/dqn.py` itself;
or (b) treat "the greedy policy genuinely oscillates checkpoint-to-
checkpoint, mechanism unknown" as a standing, now twice-confirmed
property of this training setup and move on to real S4 regardless,
using **checkpoint-averaged** (e.g. mean of several `eval_every`
windows near the end of training), not single-checkpoint, comparisons
for any future DQN-vs-baseline number. Real S4 still needs the tenant
graph (`build_tenant_graph`/`RequestGenerator`, still
`NotImplementedError`) for genuine per-tenant flooding regardless of
this thread — the load-spike diagnostic remains a cruder, tenant-blind
stand-in, not real S4 itself. Do not mistake `configs/default.yaml`'s
`load_spike:` block or `random_request_generator`'s `load_spike` kwarg
for finished S4 work — both are documented as a diagnostic stub
throughout.

---

## Milestone checklist

Pulled from PLAN.md §10 (kickoff order) and §7 / split.md §2 (weekly gates).

- [x] Repo scaffolded (folder skeleton, `env/contracts.py` frozen, CI + PR template)
- [x] Gate W1 — `pytest` green on stubs; contracts frozen and committed
- [x] Spine wired end-to-end into `environment.py` — pool sim + deferral queue
      + masking + forecaster (EWMA) + request stream (random) + full reward
      formula, `reset()`/`step()` running a complete S1 episode with regret
      logging
- [x] Gate W2 — env `step()` runs end-to-end with a random valid agent across
      a full S1 episode; regret events logged
- [x] Real NetworkX tenant graph + graph-driven `RequestGenerator`
      (`build_tenant_graph`, `RequestGenerator.reset/step`) — PLAN.md §10 step 4
- [x] Masked DQN agent (`agents/dqn.py`)
- [x] Four tuned baselines — always-PQC, always-hybrid, static-threshold
      (grid-searched), random (`agents/baselines.py`) + comparison harness
      (`experiments/harness.py`) — Hard Rule 7
- [x] 🚩 Gate W3 (make-or-break) — DQN beats the tuned threshold baseline on S1 and S3.
      **2026-08-19: genuinely attempted for the first time. S1: MET, decisively (DQN
      `total_reward` -5119.71 vs. tuned threshold's -64616.41, ~12.6x, checkpoint-averaged
      across 3 training seeds x 8 eval seeds — gap dwarfs both sides' spread by ~3 orders of
      magnitude). S3: not meaningfully attemptable under current `configs/default.yaml` pool
      settings at the time — S3's degradation produced byte-identical floor/pool/posture
      trajectories to S1 (verified across 4 seeds).** **2026-08-24: S3 recalibrated
      (`configs/scenarios/s3_degradation.yaml` only — `configs/default.yaml`/S1 untouched,
      byte-for-byte) after a root-cause investigation found the blocker was structural (a
      hard 50%-SKR-reduction ceiling in the existing spike formula, and a pool-capacity-vs.
      max-episode-draw-volume scale mismatch), not a tuning problem. S3 now genuinely diverges
      from S1 (near-total pool exhaustion + real deferral events under a fixed-seed stress
      test, vs. S1's untouched, never-stressed pool). **S3 gate attempt: MET** — same
      checkpoint-averaged/multi-training-seed methodology as S1, DQN `total_reward` -10214.82
      (training-seed std 1303.25) vs. tuned threshold's -38566.87 (eval-seed std 1636.12),
      ~3.78x, gap ~21.7x/~17.3x either side's own spread. **Gate W3 is now genuinely, fully
      met on both halves** — marked `[x]`, no longer `[~]`. See SESSION_LOG.md's 2026-08-24
      "S3 QKD-degradation recalibration, Gate W3 S3 attempt" entry for full methodology,
      numbers, and the separately-flagged posture-saturation finding (see "Next task").**
      Full prior history
      *(Still not attemptable for real — S3 doesn't exist as a scenario yet. 2026-08-10's
      load-spike diagnostics (NOT real S3/S4 — see Next task and `configs/default.yaml`'s
      `load_spike:` block) showed `forced_rekey_ratio` dropping well below flat-S1's
      never-proactive `1.000` once arrival load genuinely varies, directionally confirming
      the reward mechanism isn't broken — but even after fixing `agents/dqn.py`'s previously
      unseeded randomness and re-running the 10-seed sweep with genuinely controlled seeds,
      the spread got wider, not tighter (`0.102`-`1.000`, half the seeds at the exact
      never-proactive ceiling) — a real learn/don't-learn split by training run, not a
      measurement artifact. **2026-08-17: a budget probe on 3 stuck seeds at 50k/75k steps
      found this isn't a training-budget question either** — two of the three reached good
      intermediate values at 50k then regressed back to the exact ceiling by 75k, a
      mid-training instability, not slow convergence. **2026-08-18: a denser (every-1000-step)
      re-probe overturned even that framing** — there's no localized regression at all;
      `forced_rekey_ratio` swings 0.5+ between adjacent 1,000-step snapshots roughly 1 in 3
      of the time, continuously across an entire 75,000-step run, for every seed tested
      (including one previously read as "flat"). **A second same-day diagnostic then tested
      whether this is single-episode eval noise or eval-cadence/target-sync aliasing — both
      disfavored**: averaging 8 fixed eval seeds per checkpoint only shrinks the swing by
      ~4-6x less than the checkpoint-to-checkpoint swing itself (still >0.5 on ~1-in-4
      adjacent snapshots), and a non-aligned eval cadence swings just as hard as an aligned
      one. The mechanism behind the swings remains unknown — future Gate W3 attempts need
      **checkpoint-averaged**, not single-checkpoint or eval-seed-distribution, comparisons,
      on top of multi-seed training reporting. Evidence toward attempting the gate once S3
      exists, not the gate itself.)*
- [x] **Soft-reward baseline agent reproducing Noetzold
      (`agents/soft_reward_baseline.py`) — real and tested since 2026-08-25.**
      Reuses `agents.dqn.DQNAgent` completely unmodified; its own reward
      function (`compute_soft_reward`, a genuine `w_sec*security_score(tier)`
      term, Hard Rule 4-grounded) and config (`configs/soft_reward_baseline.yaml`,
      `security_masking: false`) are the only new pieces. A real 25,000-step
      training run confirmed `floor_violations > 0` at every one of 10 eval
      checkpoints — direct, measured evidence security is genuinely traded
      away. See `agents/soft_reward_baseline.py`'s per-file row and
      SESSION_LOG.md's newest entry for the full design/testing writeup.
- [x] Scenario dispatch S2-S4 wired into `environment.py` — S2 (HNDL) + S3
      (QKD degradation) real since 2026-08-19; **S4 (DDoS/noisy-neighbor)
      real since 2026-08-24** — see `env/environment.py`'s per-file row for
      the full mechanism. The 2026-08-10 `load_spike` diagnostic remains a
      request-rate-only stand-in, unrelated to this real dispatch. (S5/S6
      are separate milestone items, tracked below, not part of "S2-S4.")
- [ ] Real LSTM dual-head forecaster (Addition A) — `forecaster/model.py`,
      `forecaster/dataset.py`, `forecaster/train.py`, `LSTMForecastProvider`
      in `env/forecast_provider.py`
- [ ] E-A foresight ablation (off / ewma / lstm on S3 + S6)
- [x] Steering attack — adversarial threat-trace generator (`attack/trace_generator.py`,
      real and tested since 2026-08-25 — implements paper draft equation 7,
      `x̃t = (1-α)xt + α·g(xt)`, verified end-to-end through the real forecaster
      and masking layer, zero changes needed to `env/`)
      + **the attack run itself, real and complete since 2026-08-26**:
      `attack/attacking_provider.py::AttackingForecastProvider` (live equation-7
      wrapper, injected via `env/environment.py`'s new, flagged/signed-off
      `forecast_provider_factory` param) + `experiments/harness.py`'s dual-tracking
      measurement (`run_scenario_under_attack`/`evaluate_attack_multi_seed`, real
      `V(pi)` per paper eq. 4) produced the real 11-alpha split-screen result on
      S2 (masked DQN vs. soft-reward baseline, 5 eval seeds each) — Gate W5
      **attempted, with a real, more-nuanced-than-predicted result** (masked
      agent's `V(pi)` is NOT flat zero everywhere — see "Next task" below and
      SESSION_LOG.md's newest entry for the full mechanism and honest read).
      Headline contribution never cut; the finding itself belongs in the paper's
      Limitations section, not treated as a failed gate.
- [x] **S6 migration wave (scripted schedule, held-out eval only) — real and
      dispatched since 2026-08-24.** `config["migration_graph_seed"]`/
      `config["migration_schedule"]` (required only under `scenario: S6`)
      build a tenant graph once and apply a small, scripted list of
      `{step, tenant_index, new_sensitivity_class}` ratchet events via
      `RequestGenerator`'s new `set_tenant_sensitivity_class`, entirely
      upstream in request generation (Hard Rule 3) — zero `env/masking.py`
      changes needed, since floor computation already reads
      `sensitivity_class` fresh off every request every decision. Hard Rule
      8 (train/eval split) enforced as a real code-level guard:
      `experiments/train.py::train()` refuses to proceed if
      `config["train_eligible"]` is `False` (set only in `configs/scenarios/
      s6_migration.yaml`). **Completing S1-S6 (bar S5) is itself a real
      milestone** — see `env/environment.py`'s/`env/request_generator.py`'s
      per-file rows and SESSION_LOG.md's newest entry for the full mechanism,
      the three-event schedule chosen and why, and all test results.
- [ ] Live dashboard (`dashboard/app.py`) — 4-beat demo
- [ ] AWS-KMS-style API facade (`api/main.py`)
- [ ] Report (`docs/report.md`) — currently a section-header skeleton with TODOs only

---

## Per-file status

Status values: **not started** (stub, `raise NotImplementedError`, 11-line
import-smoke test only) · **stub (partial)** (some functions real, some
still `NotImplementedError`) · **implemented+tested** (real logic, real
behavioral tests, part of the green `pytest` run).

### env/

| File | Status | Notes |
|---|---|---|
| `env/contracts.py` | implemented+tested | Frozen interface contract — `Action`, `StateDict`, `ForecastProvider` ABC, `Request`, event-log TypedDicts. 5 real tests (`test_contracts.py`). |
| `env/pool_sim.py` | implemented+tested | `PoolSim` (refill/drain/exhaustion) + `SyntheticSKRQBERTrace`. 22 tests (`test_pool_sim.py`, up from 19). **2026-08-19: no code changed here** — `SyntheticSKRQBERTrace`'s existing `spike_start`/`spike_duration`/`spike_magnitude` params (already present, already documented as "the dial-in hook for the S3 'QKD degradation' scenario") are now actually exercised by real S3 dispatch in `environment.py`, not just by `test_pool_sim.py`'s own standalone degradation test. **2026-08-24 (Gate W3 S3 recalibration): `SyntheticSKRQBERTrace` gained an optional `spike_skr_multiplier: float \| None = None` field** (docstring step 4a) — when set, replaces the pre-existing `qber`-derived, hard-capped-at-50%-reduction formula with a direct multiplier during the spike window, decoupling degradation severity from `qber`'s own clip/noise. `None` (the default, and every pre-existing caller/config) reproduces the prior formula byte-for-byte — verified directly (`test_spike_skr_multiplier_none_is_byte_identical_to_prior_formula`). Root cause for adding this: the pre-existing formula's 50% cap is magnitude-independent (re-verified this session: `spike_magnitude` in `{0.6, 0.9, 0.99, 5.0}` all produce byte-identical in-window SKR, 99.7448 kbps exact), so no `spike_magnitude` value could ever produce genuine scarcity — see `configs/scenarios/s3_degradation.yaml`'s row and SESSION_LOG.md's newest entry for the full investigation and the grounding (real QKD security-threshold literature, Hard Rule 4) behind the `0.0` value S3 now uses. 3 new tests. |
| `env/deferral_queue.py` | implemented+tested | `DeferralQueue.enqueue/tick/pop_servable`, priority+FIFO, cumulative-headroom draw. 8 tests (`test_deferral_queue.py`). |
| `env/masking.py` | implemented+tested | `PolicyTable` (placeholder floor table, sticky ratchet) + `compute_mask`. 19 tests (`test_masking.py`, up from 14). Floor table not yet calibrated against Q-OPSEC data. **2026-08-19: closed a real Hard Rule 2 gap** — `compute_mask` gained an opt-in `current_key_type` param and a fourth rule: REUSE is illegal if the session's existing key tier is now below the current floor (previously only key *age* gated REUSE, never whether the established tier still met a since-ratcheted-up floor). Found and fixed via independent review of this repo's own code against Hard Rule 2's text; measured real on a live S2 episode before the fix: 64/279 (22.9%) REUSE/REKEY_NOW decisions delivered below-floor key material; 0/270 after. The prior "REUSE illegal only past key-age cap" description of this file is now out of date — see `env/environment.py`'s row for the REKEY_NOW half of the same fix. |
| `env/forecast_provider.py` | stub (partial) | `MovingAverageForecaster` (EWMA fallback) implemented+tested (9 tests, `test_forecast_provider.py`). `LSTMForecastProvider` does not exist yet (Addition A) — `use_foresight: lstm` currently raises `NotImplementedError` in `environment.py`. |
| `env/request_generator.py` | implemented+tested | `random_request_generator()` implemented+tested (as before, incl. the 2026-08-10 `load_spike` diagnostic — **explicitly not real S4**, see `configs/default.yaml`'s `load_spike:` block). **2026-08-23 (new): `build_tenant_graph(n_nodes, seed) -> nx.Graph` and `RequestGenerator(graph, seed)` are now real, tested implementations**, closing the "real NetworkX tenant graph" milestone item. `build_tenant_graph`: hub-and-spoke topology (one `kms_hub` node, one edge to each of `n_nodes` tenant nodes) — a direct rendering of PLAN.md §4's architecture diagram and a natural future dashboard graph view; each tenant node carries persistent `sensitivity_class` (real `SensitivityClass` value, Hard Rule 4), `traffic_rate` (positive float, `0.2`-`5.0`), `pqc_capable` (bool, same `_PQC_CAPABLE_PROB=0.9` convention as the random generator, Hard Rule 6), and `services` (1-2 names drawn from the existing `_SERVICES` vocabulary, never re-rolled). New `load_tenant_graph_config()` standalone typed loader for `tenant_graph:` (mirrors `load_pool_config`/`load_key_lifetime_config`/`load_dqn_config`'s convention, not auto-invoked inside `build_tenant_graph` itself, same relationship those three have to their own modules). `RequestGenerator`: samples which tenant arrives weighted by that tenant's persistent `traffic_rate` (verified statistically); a sampled request's `sensitivity_class`/`pqc_capable` are read directly from the tenant's node attributes, never re-rolled (verified with zero drift across 500 sampled requests) — this is the property that gives tenants real identity, the point of the session. `hybrid_mandatory` stays an independent per-request draw (contracts.py/PLAN.md never describe it as tenant-persistent). Also directly iterable (`__iter__`) as an `Iterator[Request]` adapter over `.step()`, so it can be swapped in wherever `random_request_generator`'s output is consumed. **Two deviations from the session brief, both because the brief didn't match the real, frozen `env/contracts.py`** (verified by reading that file, not assumed): `Request`'s real field is `service`, not `service_id`; `Request` has no `data_lifetime_days` field at all (nothing added to the frozen contract to manufacture one — flagged, not invented). 22 tests total in `test_request_generator.py` (up from 11: 13 new, minus the 2 removed `NotImplementedError` placeholder tests): node count/topology/attribute validity, byte-identical graphs under a repeated seed, a different seed producing a different graph, `RequestGenerator` field validity, same-seed/`reset()` reproducibility, the zero-drift tenant-attribute check, and the traffic-rate-weighted arrival-share statistical check. **Hard Rule 3 swap test**: `tests/test_environment.py::test_hard_rule_3_graph_driven_generator_is_a_drop_in_replacement` duplicates the existing Gate W2 test (`test_gate_full_s1_episode_random_valid_policy_zero_floor_violations`) verbatim except for one line constructing `SmartKeyNetEnv` with a `request_stream_factory` built from the real graph/generator — a full 250-step S1 episode under a random *valid* policy, zero floor violations, through completely unmodified agent-facing code. See `env/environment.py`'s row for the one narrow, additive `environment.py` change this needed. **2026-08-24 (new): `RequestGenerator` gained an optional `flood_override` constructor parameter** (`{"tenant_id": str, "extra_rate": float}`, default `None`) — S4's (DDoS/noisy-neighbor) real mechanism. Deliberately *additive*, not a `traffic_rate` mutation: every `step()` call's existing base weighted-multinomial draw is completely unmodified code, byte-for-byte, whether or not a flood is active (verified: `flood_override=None` behaves identically to omitting the argument, and every pre-existing test in this file that never passes it still passes unchanged); when set, a **second, independent RNG stream** (`_flood_rng`, separate from the base `_rng`) draws an additional `Poisson(extra_rate)` batch for the one designated tenant, reusing the exact same `_build_request` field-construction logic (refactored out of the base loop this session, verified behavior-preserving by the full pre-existing test suite passing unmodified before any new test was added). This design choice was deliberate, not the literal "multiply the node's traffic_rate attribute" framing initially considered: under this class's real sampling model (one shared `Poisson(_ARRIVAL_RATE_PER_STEP)` total split across tenants by weighted multinomial, not independent per-tenant Poisson processes), a bare weight multiply would mechanically dilute every *other* tenant's expected share too (verified empirically this session: an 8.6%→36% weight shift for one tenant cut every other tenant's realized share to ~70% of its own baseline) — failing the requirement that the flood be isolated to the designated tenant. The additive two-RNG-stream design instead makes that isolation exact: 6 new tests in `test_request_generator.py` (up from 22) confirm the flooded tenant's count rises (>3x), every *other* tenant's count is **byte-for-byte identical** (not just statistically close) between a flood-on and flood-off run, flood-batch requests pass the same field-validity/zero-drift checks as any other request, an unknown `tenant_id` raises `ValueError`, and `flood_override=None`/omitted are indistinguishable. See `env/environment.py`'s row for how `configs/scenarios/s4_ddos.yaml` drives this in practice, and for the honest, separate finding that this exact-isolation guarantee does *not* carry through unchanged to environment-mediated decision *counts* over a fixed external step budget (a downstream FIFO-queue-draining effect, not a leak in this mechanism — see that row). **2026-08-24 (new, same day, later session): `RequestGenerator` gained `set_tenant_sensitivity_class(tenant_id, new_sensitivity_class)`** — S6's (migration wave) real mechanism. Writes directly into `_tenant_attrs_by_id[tenant_id]`, the same live-reference dict object NetworkX stores on the graph node (not a copy), so the underlying graph updates too by construction; validates the new value via the `SensitivityClass` enum constructor (Hard Rule 4). No caching anywhere in this class (or in `env/masking.py`'s floor computation, verified by reading both) assumes a tenant's `sensitivity_class` is fixed for an episode, so this one mutation is sufficient — every subsequent request for that tenant reads the new class starting with the very next `step()` call, base or flood. 5 new tests in `test_request_generator.py` (up from 28): unknown-tenant/invalid-class rejection, immediate effect on subsequent requests (verified request-by-request against a real, high-traffic tenant), byte-for-byte isolation of every other tenant, and the underlying-graph-node update. See `env/environment.py`'s row for how `configs/scenarios/s6_migration.yaml`'s scripted schedule drives this in practice. |
| `env/environment.py` | implemented+tested | `SmartKeyNetEnv.reset/step/action_mask` fully wired (pool + deferral + masking + forecast + reward + session-key state). 30 behavioral tests (up from 25) incl. the split.md Gate W2 tests (`test_environment.py`). **2026-08-19 (design decision 11): `_resulting_key_type()` fixed** — REKEY_NOW now resolves to `max(existing session tier, current floor)`, never lower (previously always kept the existing tier verbatim, a real Hard Rule 2 gap — see `env/masking.py`'s row for the REUSE half of the same fix, found and fixed together). Masking gap #1's pool-draw gate now delegates to `_resulting_key_type` itself instead of a separate, previously-inconsistent copy. 2026-08-10: wired `config["load_spike"]` through to `random_request_generator` (design decision 9) — orthogonal to scenario dispatch, not a substitute for it. **2026-08-19 (design decision 10): real S2 (HNDL) + S3 (QKD degradation) scenario dispatch.** `config["scenario"]` now genuinely gates behavior for S2/S3 (S1, and the still-undispatched S4/S5/S6, are unaffected — confirmed by an explicit regression test). S2: `config["threat_schedule"]` (required only under `scenario: S2`) makes `_threat_features_placeholder()` return a scripted elevated signal from a configured step onward, flowing through the *existing, unmodified* `MovingAverageForecaster` → `PolicyTable.ratchet_up`/`floor` → `compute_mask` chain — `env/masking.py`'s floor table itself was never touched (Hard Rule 2), verified directly by a test cross-checking the env's real per-decision floor against a fresh `PolicyTable().floor()` call across a spread of sensitivity classes. S3: `config["qkd_degradation"]` (required only under `scenario: S3`) threads straight into `SyntheticSKRQBERTrace`'s pre-existing spike params at `reset()` — no new pool-sim code. **Observability finding, documented in `configs/scenarios/s3_degradation.yaml`'s own comments**: under `pool:`'s realistic default-scale numbers (256-bit draws vs. ~200,000-bit/tick mean SKR refill), S3's degradation is real but invisible in practice — refill dwarfs any plausible draw pattern by 2-3 orders of magnitude; the regression test demonstrates the real, measurable effect (higher regret-event count, lower minimum pool fill vs. S1) using the same scarcity-forcing small-pool override `test_environment.py`'s existing Hard Rule 9 gate test already established, not a fabricated number. New standalone scenario configs: `configs/scenarios/s2_hndl.yaml`, `configs/scenarios/s3_degradation.yaml` (both directly loadable via `experiments/train.py`'s `load_full_config(path)`, verified). S4 (DDoS/noisy-neighbor) and S6 (migration wave) deliberately NOT dispatched — both need a tenant-identity concept `env/request_generator.py` doesn't have yet; see "Next task". **2026-08-19 (Gate W3 attempt session): this observability finding is now confirmed to reach further than pool fill alone** — under real default config, S3's floor *and posture* sequences are also byte-identical to S1's (not just pool fill), because posture is already saturated regardless of the QBER spike. This blocked Gate W3's S3 half from being attempted at all this session — see SESSION_LOG.md and Thread 1 above. **2026-08-23 (design decision 12): `__init__` gained an optional `request_stream_factory` parameter** (`episode_seed -> Iterator[Request]`, default `None`) — the one narrow, additive exception needed for `env/request_generator.py`'s new real `RequestGenerator` to be a genuine drop-in replacement for `random_request_generator` (Hard Rule 3's swap test). Confirmed before making this change that `random_request_generator` was hardcoded inline in `reset()` (not already injectable) — reading, not assuming. `None` (the default) reproduces prior behavior byte-for-byte; every existing single-argument `SmartKeyNetEnv(config)` call site is unaffected, verified by the full existing test suite passing unmodified. No other line in this file changed. **2026-08-24 (design decision 13): real S4 (DDoS/noisy-neighbor) scenario dispatch.** `config["ddos"]` (required only under `scenario: S4`, same fail-fast convention as S2/S3) builds a real tenant graph once at construction time (`build_tenant_graph(n_nodes, seed=ddos.graph_seed)` — deliberately a seed dedicated to graph structure, decoupled from `episode_seed`, so which tenant `tenant_index` names stays a fixed structural fact of the config, not something that reshuffles across eval seeds) and, in `reset()`, constructs a `RequestGenerator` with a `flood_override` targeting that tenant for the whole episode. Entirely upstream, in the request-arrival process — confirmed by a code-level grep/read check (not just a behavioral test) that `masking.py`, `agents/dqn.py`, and `_apply_action`'s reward calculation contain zero mentions of "scenario"/"ddos"/branching on tenant identity; `"ddos"` appears in the codebase only in `environment.py`'s own sanctioned dispatch code and in `request_generator.py`'s docstring prose (the actual `flood_override` mechanism itself is scenario-agnostic, generically named). 36 tests in `test_environment.py` (up from 31, incl. renaming the now-outdated `test_s4_and_s6_scenarios_are_not_yet_dispatched` to `test_s5_and_s6_...`), incl.: constructing under `scenario: S4` without `config["ddos"]` raises `KeyError`; the flooded tenant's own decision count is measurably (>3x) higher with the flood active vs. inactive, same seed; a real empirical finding, honestly reported rather than assumed — **the critical (highest-sensitivity) tenant's own decision throughput collapses under the flood** (>50% reduction, confirmed over an equal span of real elapsed simulator time via `env._step_count`, not just an equal external-step budget) even though its *per-decision* regret rate barely moves — this environment's dominant, clearly measurable "noisy neighbor" effect is service-opportunity crowd-out via the shared one-decision-per-tick FIFO queue, not (at this baseline policy, this pool scale) a large per-decision quality hit; regret/pool-exhaustion events are confirmed visible only under the same small-pool scarcity override this suite's other regret tests already use (S4's flood is real but invisible in pool-exhaustion terms at `pool:`'s default scale, the same pre-existing calibration-headroom property S3's config already documented — not a new finding, a consistent one). New standalone scenario config `configs/scenarios/s4_ddos.yaml` (directly loadable via `load_full_config(path)`, verified) documents the full empirical numbers in its own comments. See `env/request_generator.py`'s row for the `flood_override` mechanism itself and why it's additive rather than a `traffic_rate` multiply. **2026-08-24 (design decision 14, Gate W3 S3 recalibration): `_qkd_degradation_trace_kwargs()` gained an additive, optional read of `qkd_degradation.spike_skr_multiplier`** (`.get(...)`, not `[...]` -- every S3 config/test that predates this session, and doesn't set this key, is completely unaffected) -- threads straight into `env/pool_sim.py`'s new `SyntheticSKRQBERTrace.spike_skr_multiplier` field (see that file's row). No other line in this function or elsewhere in this file changed; S1/S2/S4 dispatch paths are byte-for-byte untouched, confirmed by the full pre-existing test suite passing unmodified. 38 tests in `test_environment.py` (up from 36): 2 new -- a direct, no-test-override comparison of the real, committed `configs/default.yaml` (S1) against the real, recalibrated `configs/scenarios/s3_degradation.yaml` (S3), same seed, proving genuine divergence (near-total pool exhaustion + real regret events under S3 vs. S1's untouched, never-stressed pool); and an explicit Hard Rule 9 check under the new severe S3 config (deferred, never downgraded, despite genuine exhaustion). See `configs/scenarios/s3_degradation.yaml`'s row and SESSION_LOG.md's newest entry for the full root-cause investigation, the design decision's justification, and the separately-flagged posture-saturation finding. **2026-08-24 (design decision 15, new, same day, later session): real S6 (migration wave) scenario dispatch, completing the full S1-S6 grid.** `config["migration_graph_seed"]` (required only under `scenario: S6`, same fail-fast convention as `ddos.graph_seed`) builds a tenant graph once at construction time, same pattern as S4. `config["migration_schedule"]` (already a top-level key in every config, default `[]`; only ever non-empty for S6's own config) is a small, explicit list of `{step, tenant_index, new_sensitivity_class}` entries — scripted and exogenous (Hard Rule 3). `reset()` under S6 keeps a live reference to the `RequestGenerator` instance (`self._request_generator`, unlike S4's discarded `iter(...)` wrapper) so `_advance_to_next_decision`'s tick loop can call its new `set_tenant_sensitivity_class` at each scripted step — a single, unconditional dispatch loop (`self._migration_schedule` is `[]` for every scenario but S6, so it's a guaranteed no-op elsewhere). **Needed zero `env/masking.py` changes**: confirmed by reading before designing that both `compute_mask`/`PolicyTable.floor` and `_prepare_decision` already read a request's `sensitivity_class` fresh every decision, with no caching anywhere that assumes it's fixed for an episode. **Hard Rule 8 (train/eval split) is enforced in `experiments/train.py`, not here** — see that file's row. 6 new tests in `test_environment.py` (up from 38, incl. renaming `test_s4_and_s6_scenarios_are_not_yet_dispatched` to `test_s5_scenario_is_not_yet_dispatched` since S6 is no longer undispatched): `migration_graph_seed` required under S6 (`KeyError` otherwise); a direct, request-by-request check through the real environment confirming the ratcheted tenant's requests carry the old `sensitivity_class` strictly before the scripted step and the new one at/after it; the resulting `policy_floor` cross-checked against a real `PolicyTable().floor()` call (`use_foresight: off` pins posture at CALM so the check isolates exactly this effect); every other tenant's `sensitivity_class` confirmed byte-for-byte unaffected (isolation check, same spirit as S4's); a Hard Rule 3 code-level grep (masking.py/contracts.py/`_apply_action`'s body have zero "migration"/"s6" mentions); and a held-out-eval sanity run via `experiments/harness.py::run_scenario` against the real, committed `s6_migration.yaml` with a `StaticThresholdPolicy` baseline (`floor_violations == 0` holds across the ratchet points too). See `env/request_generator.py`'s row, `configs/scenarios/s6_migration.yaml`'s row, `experiments/train.py`'s row, and SESSION_LOG.md's newest entry for the full design reasoning and schedule justification. **2026-08-25 (design decision 16, soft-reward baseline agent session): `__init__` gained `config.get("security_masking", True)`.** Found and flagged before being built (via `AskUserQuestion`, user sign-off recorded in SESSION_LOG.md's newest entry): `step()`'s `IllegalActionError` enforces the mask unconditionally at the environment boundary, regardless of which agent is calling — meaning `agents/soft_reward_baseline.py`'s "no action masking" property could not be achieved by that agent's own action-selection code alone, the way its reward function could differ purely in its own code. When `False`, `_prepare_decision` calls the same, unmodified `compute_mask()` with `floor=Action.SERVE_CLASSICAL`/`current_key_type=None` instead of the real floor/session key type — the floor-based and REUSE-below-floor rules become no-ops, while `pool_can_draw`/`key_age`/`max_key_age` pass through unchanged (physical/protocol feasibility, not the security-floor restriction this flag targets). Default `True`, zero effect on any pre-existing config/caller, proven byte-for-byte (4 new tests in `test_environment.py`). Only `configs/soft_reward_baseline.yaml` sets this `False`. Zero `env/masking.py` changes. |

### agents/

| File | Status | Notes |
|---|---|---|
| `agents/dqn.py` | implemented+tested | `flatten_state(state, has_forecast)` (genuinely variable-length: 13 dims under `off`, 28 under `ewma`/`lstm`; `has_forecast` is now an **explicit required parameter** — the earlier `state["threat_score"] != 0.0` inference trick was removed 2026-08-08 as fragile-by-accident, see that session's log entry). `QNetwork` (2-hidden-layer MLP), `DQNConfig`/`load_dqn_config` (reads `configs/default.yaml`'s `dqn:` block), `DQNAgent(state_dim, has_forecast, config, seed=None)` (internal circular-buffer replay, `act`/`observe`/`learn`/`save`/`load`, `has_forecast` fixed once at construction and threaded through every internal `flatten_state` call) — masking applied structurally at both action-selection *and* bootstrap-target time (Hard Rule 2), no security term anywhere (Hard Rule 1). **2026-08-10: `seed` parameter added** — reseeds `random`+`torch`'s global RNGs before `QNetwork` construction when given, so weight init/exploration/replay sampling are genuinely reproducible (`seed=None` default is unchanged prior behavior); fixes a gap the same day's earlier sessions found and flagged (see `experiments/train.py`'s row and SESSION_LOG.md). 26 tests (`test_dqn.py`, up from 23), incl. a regression test for a foresight-mode state with `threat_score == 0.0` still flattening to 28 dims, an integration test training against the real `SmartKeyNetEnv` on S1 for 3000 steps with loss trending down, and 3 new seed tests (same-seed reproducibility, different-seed divergence, `seed=None` leaves ambient state alone). Not yet run to convergence — that's `experiments/train.py`. |
| `agents/baselines.py` | implemented+tested | `AlwaysPQCPolicy`, `AlwaysHybridPolicy`, `StaticThresholdPolicy` (incl. `grid_search`), `RandomPolicy` — all real, sharing a `_lowest_legal_action` fallback. 261 tests (`test_baselines.py`), incl. an adversarial parametrized sweep over all 31 non-empty action masks per policy (never returns an illegal action, however contrived the mask). |
| `agents/soft_reward_baseline.py` | implemented+tested | **2026-08-25 (new):** Noetzold-style soft-reward baseline agent — the deliberate, contained reproduction Hard Rule 1's own argument is tested against, not a violation of it (see module docstring's explicit tension/resolution). No new agent class: `agents.dqn.DQNAgent` is reused directly, completely unmodified (masking's absence lives in `env/environment.py`'s new `security_masking` config flag, not agent code — see that file's row). This module's own contribution: `SoftRewardConfig`/`load_soft_reward_config` (reads `configs/soft_reward_baseline.yaml`'s own `soft_reward:` block, deliberately separate from `configs/default.yaml`'s `reward:`), `compute_soft_reward(state, action, cfg) = -w_lat*latency - w_en*energy + w_sec*security_score(tier)` (the reproduced design's own, simpler formula — not "our full reward plus a security term"), `security_score`/`_TIER_SECURITY_SCORE` (`classical=0.2, pqc=0.6, hybrid=1.0`, straight from the reproduced spec, Hard Rule 4), and `resolved_cost_action`/`delivered_tier` (mirror `experiments/harness.py`'s own tier-resolution helpers so REUSE/REKEY_NOW get a security_score grounded in what tier they actually deliver — REUSE's can genuinely be below the real floor now, unlike the masked agent). 13 tests in `tests/test_soft_reward_baseline.py` (rewritten from the prior 1-test import-smoke stub): formula/tier-resolution unit tests plus two Hard Rule 1 boundary checks (direct grep confirming `agents/dqn.py` and `env/environment.py::_apply_action` contain zero `security_score`/`w_sec` mentions). A real 25,000-step training campaign (S1, seed 0) confirmed the property this agent exists to demonstrate: `floor_violations > 0` at every one of 10 eval checkpoints (12-184 out of a 250-step episode) via the unmodified `experiments/harness.py::run_scenario`. See `env/environment.py`'s row (design decision 16) and `experiments/train.py`'s row (`train_soft_reward_baseline`) for the rest of the mechanism, and SESSION_LOG.md's newest entry for the full numbers. |

### forecaster/

| File | Status | Notes |
|---|---|---|
| `forecaster/model.py` | not started | Stub, `test_forecaster_model.py` is 1 import-smoke test. |
| `forecaster/dataset.py` | not started | Stub, `test_forecaster_dataset.py` is 1 import-smoke test. |
| `forecaster/train.py` | not started | Stub, `test_forecaster_train.py` is 1 import-smoke test. |

### metrics/

| File | Status | Notes |
|---|---|---|
| `metrics/regret.py` | implemented+tested | `compute_episode_metrics()` + `attribute_regret()`. 11 tests (`test_regret.py`). |

### experiments/

| File | Status | Notes |
|---|---|---|
| `experiments/harness.py` | implemented+tested | `run_scenario` (one policy x scenario x seed episode → `ScenarioResult`, truncated via `max_steps`, default 250) + `run_grid` (every combination). Recomputes per-decision latency/hybrid-draw resolution from public `StateDict` fields (mirrors `env.environment`'s private cost tables/`REKEY_NOW` resolution, since `step()` doesn't surface them directly). `ScenarioResult` gained `total_reward: float` 2026-08-10 (raw summed episode reward — a sharper policy discriminator than `p99_latency`, see that session's log entry). 13 tests (`test_harness.py`, up from 8), incl. the S1 x four-baselines zero-floor-violations check, a `run_grid` combination-count check, and a `total_reward` check against a manually-summed reference. **2026-08-19: `floor_violations` counting was itself buggy, now fixed** — it only ever checked the three tier-serving actions (`action in _TIER_ACTIONS`), so it silently reported `0` on a real S2 episode that independently measured 64 real below-floor deliveries via REUSE/REKEY_NOW (see `env/masking.py`'s row for the underlying Hard Rule 2 fix this depended on). New `_delivered_tier()` helper checks every action's actually-delivered tier; `_resolved_cost_action()`'s REKEY_NOW resolution fixed to match. **The "floor_violations: 0" guarantee this file's docstring and `ScenarioResult`'s own field comment describe was not actually a settled guarantee for REUSE/REKEY_NOW before this session — it is now, verified directly, not just claimed.** **2026-08-19 (Gate W3 attempt session): gained `evaluate_multi_seed(policy, scenario, config, eval_seeds) -> MultiSeedEvalResult`** — runs any `Policy` across multiple eval seeds, reports mean/std for `p99_latency`/`total_reward`/`forced_rekey_ratio`, mean `regret_events`/`pool_exhaustion_events`, and a summed (not averaged) `floor_violations_total`; keeps every raw per-seed `ScenarioResult`. Built to replace single-eval-seed point estimates for Gate W3 (and any future policy comparison) — 2026-08-18's diagnostics found single-checkpoint/single-seed `forced_rekey_ratio` measurements unreliable, and this is the fix at the measurement layer. 19 tests (`test_harness.py`, up from 13), incl. means/std verified against a manual `np.mean`/`np.std` of direct `run_scenario` calls, the `n=1` degenerate case reducing exactly to `run_scenario` itself, and `floor_violations_total` summing (not averaging) across seeds. **2026-08-25 (masked-vs-soft-reward S3 comparison session): `MultiSeedEvalResult` gained `below_floor_rate_mean`/`below_floor_rate_std`** — the RATE form of `floor_violations_total` (`floor_violations / effective_max_steps` per seed, then mean/std across seeds; `effective_max_steps` resolved via the same `config.get("max_steps", _DEFAULT_MAX_STEPS)` fallback `run_scenario` itself uses for every call `evaluate_multi_seed` makes, not a re-guessed denominator) — this is PLAN.md's paper-draft "below-floor service rate" (equation 4), needed to compare the masked DQN against `agents/soft_reward_baseline.py` on real S3 (see SESSION_LOG.md's newest entry). Purely additive (no existing field/call site changed). 21 tests (`test_harness.py`, up from 19): a zero-rate check for a masked policy, and a direct manual-computation cross-check using `security_masking: false` + an always-SERVE_CLASSICAL stub policy on S2. |
| `experiments/train.py` | implemented+tested | `train()` (one continuous S1 episode, `total_steps` from `configs/default.yaml`'s new `training:` block, periodic greedy-mode eval snapshots via the harness, final `DQNAgent.save` checkpoint), `GreedyDQNPolicy` (wraps a trained agent's `q_network` directly for deterministic epsilon=0 evaluation without touching `agents/dqn.py` or the agent's training epsilon-decay counter), `evaluate_against_baseline()` (trained agent vs. grid-searched `StaticThresholdPolicy`, same fixed eval seed). **2026-08-10: `train()` now passes `training_cfg["seed"]` to `DQNAgent(..., seed=...)` too**, not just `env.reset(seed=...)` — see `agents/dqn.py`'s row. 6 tests (`test_train.py`), incl. a smoke run (100 steps) and a determinism check contrasting `GreedyDQNPolicy` against `DQNAgent.act()`'s genuine epsilon=1 stochasticity. **Six real 25,000-step campaigns executed 2026-08-10 across four sessions** (~40-46s/run): an epsilon-schedule fix (`epsilon_decay_steps` 50k→12.5k) let training genuinely converge but the converged flat-S1 policy still tied the tuned threshold on `p99_latency` and never rekeyed proactively (`forced_rekey_ratio=1.000`); a load-spike diagnostic (see `env/request_generator.py`'s row) re-ran under it and got `0.256`/`0.872` across two seeds; a 10-seed sweep sized that spread properly (`0.190`-`1.000`, mean `0.735`, stdev `0.275`) and found `agents/dqn.py`'s randomness was never seeded — training seed only reached the environment; a same-day fix session seeded it and re-ran the same 10-seed sweep — the spread got *wider*, not tighter (`0.102`-`1.000`, mean `0.700`, stdev `0.345`), with exactly half the seeds landing at the exact never-proactive ceiling. **2026-08-17: a budget probe (6 more real campaigns, 50k/75k steps, 3 of the 5 stuck seeds) found the stuck seeds don't respond to more training budget** — 2 of 3 reached good intermediate `forced_rekey_ratio` values at 50k steps (`0.102`, `0.659`) and then regressed back to the exact `1.000` ceiling by 75k, read at the time as a mid-training instability. **2026-08-18: a denser re-probe (3 more real 75,000-step campaigns, `eval_every=1000` — ~75 snapshots each instead of 3) overturned that framing** — there is no localized regression: `forced_rekey_ratio` swings 0.5+ between adjacent 1,000-step snapshots roughly 1 in 3 of the time, continuously across the entire run, for every seed including one previously read as "flat, never found a good policy." Buffer-capacity crossing at step 50,000 (`agents/dqn.py`'s `_REPLAY_BUFFER_CAPACITY`) is ruled out as the cause — no loss anomaly or swing-amplitude change near that step. A real but separate long-run drift toward the ceiling exists (noisy, not step-changed at any specific step) layered on top of the noise. **A second 2026-08-18 diagnostic (3 more real 75,000-step campaigns, `eval_every=750`, 8 fixed eval seeds/snapshot instead of 1) tested the two remaining hypotheses that session left open**: single-episode eval noise (RULED OUT — 8-seed averaging only shrinks the checkpoint-to-checkpoint swing to ~4-6x smaller than the swing itself, nowhere near enough to explain it) and eval-cadence/target-sync aliasing (evidence against, not a clean rule-out since it's a between-session comparison — a non-aligned cadence swings just as hard as the aligned one did). The ceiling-fraction drift was confirmed to survive 8-seed averaging. The actual mechanism behind the swings is still unidentified. See SESSION_LOG.md's two 2026-08-18 entries for the full data. **2026-08-19: the Hard Rule 2 masking fix (see env/masking.py's row) substantially reduced the checkpoint-oscillation itself (ceiling-drift eliminated, swings down ~40-45%) — see SESSION_LOG.md's post-fix probe entry. Later the same day, Gate W3 was genuinely attempted for real** (this file's own `train()`/`GreedyDQNPolicy`/`evaluate_against_baseline` re-confirmed unmodified and correctly read-only first): 3 training seeds x 75,000 steps each on real S1, evaluated checkpoint-averaged via the new `experiments/harness.py::evaluate_multi_seed()` (8 eval seeds) rather than `evaluate_against_baseline()`'s existing single-seed methodology. **S1 result: DQN beats a properly-tuned `StaticThresholdPolicy` by ~12.6x on `total_reward`** (-5119.71 vs -64616.41), robustly across both training- and eval-seed spread — Gate W3's S1 half is met. **A real methodological finding along the way**: `evaluate_against_baseline()`'s existing `threshold_eval_fn` tunes on single-seed `-p99_latency`, which this session found is constant (`1.5000`) across the *entire* threshold grid under real S1 — `StaticThresholdPolicy.grid_search`'s tie-break then silently defaults to the grid's first candidate, not a genuinely tuned choice. **Not fixed here** (this session's own Gate W3 numbers used a separate, correctly `total_reward`-tuned grid search, not this function) — flagged for whoever next touches this file. `train()` was not generalized to scenarios beyond S1 in that session, since S3 turned out not to be meaningfully attemptable under current config at the time (see env/environment.py's row) — building that generality then would have been speculative. **2026-08-24 (S6 session): `train()` gained a Hard Rule 8 guard** — checks `config.get("train_eligible", True)` at the very top, before any env/agent construction, and raises `ValueError` if `False` (`configs/scenarios/s6_migration.yaml` is the one config that sets this). Deliberately keyed on the flag itself, not a hardcoded `scenario == "S6"` string check, so it would stay correct even once the file's then-separately-flagged hardcoded-`"S1"` override was later generalized — a string-keyed guard would otherwise have been silently defeated by that fix. **2026-08-24 (this session, later): the hardcoded-`"S1"` debt itself is now fixed** — `train()` gained a real, explicit `scenario: str = "S1"` parameter, threaded into `env_config` and into the periodic eval snapshot's `run_scenario` call exactly the way `experiments/harness.py`'s `run_scenario`/`run_grid` already do (an external override, not read from `full_config["scenario"]`) — the same existing convention, not a second parallel one. Defaulting to `"S1"` means every pre-existing call site (both tests and `main()`) is unaffected — proven with a genuine before/after comparison (identical config/seed/overrides, git-stashed the diff and re-ran): reward-window averages, full loss curve (237 real gradient steps), and every eval snapshot's `total_reward`/`p99_latency`/`forced_rekey_ratio` are byte-for-byte identical pre- and post-fix. The Hard Rule 8 guard's own logic/location was not touched (per instruction) — what changed is that it's now reachable through a genuinely scenario-selectable call: `train(s6_config, scenario="S6", ...)` raises `ValueError` before `SmartKeyNetEnv.__init__` is ever called (verified directly via an instrumented call count of exactly 0), where previously the guard fired only because `train()` happened to ignore whatever scenario a config named, never because scenario selection itself was real. `scenario="S3"` was verified two ways to actually dispatch S3's real degradation dynamics, not just be silently accepted: (a) pointing `scenario="S3"` at the S1-only `default.yaml` (which lacks `qkd_degradation`) raises `KeyError`, proving the parameter reaches `SmartKeyNetEnv`'s real dispatch; (b) a direct pool-trajectory stress test (same env-construction line `train()` uses internally, mirrors SESSION_LOG.md 2026-08-24's own S3-vs-S1 divergence methodology) shows S3's pool collapsing to near-total exhaustion (min fill 0.16%) while S1's stays >69% full, under the real committed config files. A control test confirms `scenario="S3"` (where `train_eligible` defaults `True`) does not raise — the guard is scenario-specific, not a blanket block. `evaluate_against_baseline()` is unchanged and still evaluates only against S1 — explicitly out of this session's scope (a separate function, not part of `train()`'s scenario-selection fix). 5 new tests in `test_train.py` (up from 7, total 12): the byte-for-byte before/after comparison, the S3 dispatch-proof pair, the S6 guard end-to-end proof (with an instrumented zero-env-construction check), and the S3 control test. See SESSION_LOG.md's newest entry for the full before/after numbers. **2026-08-25 (this session): gained `train_soft_reward_baseline(full_config, training_overrides=None, scenario="S1") -> (DQNAgent, TrainingRecord)`** — additive, does not touch `train()` (confirmed via `git diff`). Mirrors `train()`'s own loop exactly (reuses `DQNAgent`/`GreedyDQNPolicy`/`run_scenario` unmodified) except for the one thing that must differ: each step's reward is computed via `agents.soft_reward_baseline.compute_soft_reward(state, action, soft_reward_cfg)`, not `env.step()`'s own returned reward (discarded, proven genuinely unused via a poisoned-reward monkeypatch test). Defaults to loading `configs/soft_reward_baseline.yaml`, not `configs/default.yaml`. Same Hard Rule 8 `train_eligible` guard, mirrored. 5 new tests in `test_train.py`: smoke run, `security_masking: False` dispatch proof (instrumented config capture), the poisoned-reward Hard Rule 1 proof, the real 3,000-step training-run proof (loss trends down, `floor_violations > 0`), and the Hard Rule 8 guard mirror. See SESSION_LOG.md's newest entry for the full 25,000-step campaign numbers. |

### attack/

| File | Status | Notes |
|---|---|---|
| `attack/trace_generator.py` | implemented+tested | **2026-08-25 (new file):** the equation-7 adversarial input-shaping generator (PLAN.md §5 S5, PLAN2.md §7.5 Panel 5). `generate_adversarial_window(true_window, alpha) -> shaped_window` implements `x̃t = (1-α)xt + α·g(xt)` exactly (`alpha` an explicit, independent per-call parameter, not config-baked, so a future dose-response sweep can call it across eleven alpha values against the same base window in one run); `g(xt) -> [0.0]*len(xt)` — a deterministic, all-zero "benign region" target chosen because zero qber / zero load are the real lowest legitimate values `env/environment.py::_threat_features_placeholder()`'s current `[qber, load]` window can take, not an arbitrary choice — deliberately targets the CURRENT placeholder `MovingAverageForecaster`, documented as needing revisiting once the real LSTM forecaster (Addition A) exists. Pure input transformation: never touches `ForecastProvider`/`env/masking.py`/`env/environment.py` — a future sweep session substitutes the shaped window for the true one before it reaches `forecaster.update()`. 23 tests in `test_trace_generator.py`: exact (not approximate) boundary equality at `alpha=0`/`alpha=1`; linearity at `alpha∈{0.25,0.5,0.75}` checked against an independently-recomputed formula in the test itself; a no-mutation/new-object check; an end-to-end attack-effectiveness proof through the real forecaster (a genuinely severe `true_window=[5.0,5.0]` — mirroring how `env/environment.py`'s own real S2 `elevated_signal` mechanism already reaches HIGH, since ordinary in-domain `[qber,load]` values are structurally capped at ELEVATED per the 2026-08-24 posture-ceiling finding in `env/forecast_provider.py`'s row — steady-states to HIGH posture at `alpha=0` and genuinely crosses down to ELEVATED at `alpha=1`, with `threat_score`/`posture_probs[HIGH]`/`posture_probs[CALM]` all moving monotonically across the full alpha grid); and the masking-safety property (this session's most important test) verified with real numbers for `sensitivity_class=S2` (`true_floor=SERVE_HYBRID`, `shaped_floor=SERVE_PQC` — `compute_mask()` given only the attacked floor never legalizes `SERVE_CLASSICAL`, while `SERVE_PQC` becomes legal under the attack where it wasn't under the true floor — the attack's real, measured effect, safety guarantee intact), plus a general, exhaustive 12-cell (4 sensitivity classes x 3 postures) sweep of the same property independent of this session's specific numbers. Zero changes to `env/forecast_provider.py`/`env/masking.py`/`env/environment.py`, verified via `git diff --stat`. See SESSION_LOG.md's newest entry for the full design reasoning and verified numbers. |
| `attack/steering_trace.py` | not started | Stub, `test_steering_trace.py` is 1 import-smoke test — an older, PLAN.md-era file name/shape for the eventual dose-response sweep module; superseded by PLAN2.md's `attack/trace_generator.py` + `attack/run_attack.py` naming (see PLAN2.md §7.5's table). Not touched or removed this session — out of scope; the future S5 dose-response sweep session should resolve this naming duplication (most likely: build the sweep as a separate, PLAN2.md-named module and retire this stub, rather than reusing it). |

### dashboard/

| File | Status | Notes |
|---|---|---|
| `dashboard/app.py` | not started | Stub, `test_dashboard_app.py` is 1 import-smoke test. |
| `dashboard/explain.py` | implemented+tested | **2026-08-19 (new file):** Explain Decision panel backend (PLAN2.md §7.3, Addition D; Hard Rule 10) -- Person D's first real code this project. `explain_decision(...)` (pure function, all six inputs explicit) + `explain_decision_from_env(env, state, chosen_action)` (convenience wrapper pulling those inputs off a live `SmartKeyNetEnv`, mirroring `experiments/harness.py`'s established precedent for reaching into a few private env attributes the public Gym API doesn't yet surface). Returns a `DecisionTrace` dataclass covering all six PLAN2.md §7.3 steps: threat score + source, posture probs + resolved posture, floor lookup (+ the full real floor table, imported from `env.masking`, never re-typed), the action mask (calls `env.masking.compute_mask()` directly -- zero possible drift between this module's legal/reason fields and the masking layer's real behavior, by construction, not by convention), cost comparison (reads `env.environment`'s real `_LATENCY_UNITS`/`_ENERGY_UNITS`/`_KEY_TYPE_TO_SERVE_ACTION`, never re-derived), and a deterministically templated final sentence. Policy-agnostic by design (no dependency on `agents/dqn.py`) -- verified in a scratchpad sanity script against `StaticThresholdPolicy` on real S1 steps, which also caught and fixed one real bug: the final-sentence template originally said "a learned preference from the policy" for the cost-tradeoff case, which is wrong for a non-learning baseline; reworded to "the policy's own preference among legal options." 29 tests (`test_explain.py`, up from 26), incl. every (sensitivity_class, posture) floor-table cell checked against a fresh `PolicyTable`, 6 parametrized mask edge cases (pool empty, key age at/over cap, cold start, all-legal, HYBRID-floor-with-empty-pool) each checked against a real `compute_mask()` call, REKEY_NOW cost-resolution cases (existing-tier and cold-start-adopts-floor), and an end-to-end test stepping a real `SmartKeyNetEnv` and cross-checking every trace against a fresh `compute_mask()` call built from the same env state. **2026-08-19 (same day, later session): updated to track `env/masking.py`'s Hard Rule 2 fix** — `_mask_entries()` now passes `current_key_type` into `compute_mask()` and gained the matching stale-tier reason case (would otherwise crash on the new illegality reason), `_resolved_cost_action()` updated to the same `max(existing, floor)` REKEY_NOW resolution (would otherwise display a cost for a tier that was never actually served). Explicitly out of scope this session (per PLAN2.md's Hard Rule 11 and Hard Rule 10's scoping): pcap ingestion / Threat Input panel (§7.1) and any dashboard HTML/frontend -- this is a Python module returning structured data only. |

### api/

| File | Status | Notes |
|---|---|---|
| `api/main.py` | not started | Stub, `test_api_main.py` is 1 import-smoke test. |

### data/

| File | Status | Notes |
|---|---|---|
| `data/get_data.py` | not started | All four `_download_*` helpers `raise NotImplementedError`; `main()` exists as a dispatcher shell. |
| `data/README.md` | present (doc) | Download instructions written. |
| `data/sample/` | empty | Only a `README.md` placeholder — no committed sample CSVs yet for CI/others to run against. |
| `data/raw/rt_iot2022/RT_IOT2022.csv` | placed, not ingested | Real dataset file present locally (gitignored, not committed); no code reads it yet. |

### docs/

| File | Status | Notes |
|---|---|---|
| `docs/report.md` | skeleton only | Section headers (Abstract, Intro, Related Work, Methodology 3.1-3.3, ...) with `_TODO_` markers and owner tags; no written content yet. |

### configs/

| File | Status | Notes |
|---|---|---|
| `configs/default.yaml` | partial | `pool`, `key_lifetime`, `reward`, `use_foresight`, `tenant_graph.n_nodes`, `load_spike`, `dqn`, `baselines`, `steering_attack`, `training` keys all present. **2026-08-25 (new key): `security_masking: true`** — documented, additive-only (see `env/environment.py`'s row, design decision 16); every other config either inherits this default or sets it explicitly, except `configs/soft_reward_baseline.yaml`, the one config that sets it `false`. `migration_schedule: []` — **2026-08-24: no longer a placeholder; this is the correct, load-bearing value for every scenario but S6** (see `configs/scenarios/s6_migration.yaml`'s row). This file's own comment above the key was corrected from a stale, never-implemented `{tenant_cohort, new_floor}` schema description to the real, shipped `{step, tenant_index, new_sensitivity_class}` one (comment-only change, the key itself is unchanged). `scenario: S1` — deliberately does not carry `threat_schedule`/`qkd_degradation`/`ddos`/`migration_graph_seed` (those are S2/S3/S4/S6-only, required only when `scenario` selects them — see `env/environment.py`'s row). `load_spike.enabled: false` by default (2026-08-10 diagnostic stub, NOT real S4 — see `env/request_generator.py`'s row and SESSION_LOG.md). |
| `configs/scenarios/s2_hndl.yaml` | implemented+tested | **2026-08-19 (new file)**: standalone, directly-loadable S2 config (same shape as `default.yaml`, `scenario: S2` + a `threat_schedule` block). Verified to load and run end-to-end (`tests/test_environment.py::test_scenario_config_files_load_and_construct_a_working_env`). |
| `configs/scenarios/s3_degradation.yaml` | implemented+tested | **2026-08-19 (new file)**: standalone, directly-loadable S3 config, `scenario: S3` + a `qkd_degradation` block. Same verification as above. Its own comments originally documented the pool-scale observability finding (see `env/environment.py`'s row). **2026-08-24 (Gate W3 S3 recalibration): `pool.capacity_bits` dropped `1,000,000 -> 20,000`** (scoped to this file only -- `configs/default.yaml`/S1 completely untouched) and `qkd_degradation` gained `spike_skr_multiplier: 0.0`. Both numbers are computed/grounded, not guessed: capacity kept below the 64,000-bit max-possible-episode-draw-volume (`250 steps * 256 bits/draw`) so genuine exhaustion is reachable, and the multiplier reflects real QKD security-proof literature (BB84-family protocols provably yield zero extractable key above a QBER security threshold, ~11% canonically) rather than an arbitrary worst case. The file's header comment was rewritten to document the full root-cause investigation (why the pre-existing `spike_magnitude` dial couldn't fix this at any value) and retire its prior "S1 and S3 describe the same deployment" framing (no longer true by design -- see the file itself). Verified: real, committed S1 vs. S3 config files now produce genuinely different pool trajectories under the same seed (near-total S3 exhaustion + real regret events vs. S1's untouched pool) -- see `env/pool_sim.py`/`env/environment.py`'s rows and SESSION_LOG.md's newest entry. |
| `configs/scenarios/s4_ddos.yaml` | implemented+tested | **2026-08-24 (new file)**: standalone, directly-loadable S4 config, `scenario: S4` + a `ddos` block (`graph_seed`, `tenant_index`, `extra_rate`). Verified to load and run end-to-end (same `test_scenario_config_files_load_and_construct_a_working_env` test, extended). Its own comments document the full empirical numbers this session found (decision-throughput crowd-out effect, and the same pool-scale regret-event observability caveat S3's file already documented). |
| `configs/scenarios/s6_migration.yaml` | implemented+tested | **2026-08-24 (new file, same day, later session)**: standalone, directly-loadable S6 config, `scenario: S6` + `migration_graph_seed`/`migration_schedule` + **`train_eligible: false`** (the Hard Rule 8 guard `experiments/train.py::train()` checks — the one config in the repo that sets this). Schedule: three phased ratchet events across a 250-step episode, each targeting a tenant starting from a different real, verified pre-migration class (via a real `build_tenant_graph(seed=0, n_nodes=10)` call, not guessed) — `step 60: tenant_0 S1->S3`, `step 130: tenant_3 S2->S3`, `step 190: tenant_4 S0->S2`; no event targets an already-S3 tenant (would be a silent no-op). Verified to load and run end-to-end (`test_scenario_config_files_load_and_construct_a_working_env`, extended) and to actually refuse training (`test_train.py`'s Hard Rule 8 tests) and to run cleanly under held-out evaluation via `experiments/harness.py::run_scenario` (`floor_violations == 0` across all three ratchet points). See `env/environment.py`'s and `env/request_generator.py`'s rows and SESSION_LOG.md's newest entry for the full schedule justification. |
| `configs/soft_reward_baseline.yaml` | implemented+tested | **2026-08-25 (new file)**: standalone config for `agents/soft_reward_baseline.py`'s training entry point (`experiments/train.py::train_soft_reward_baseline`), deliberately separate from `configs/default.yaml` (not a variant loaded alongside it). `security_masking: false` (the one config in the repo that sets this); `reward:`/`dqn:` blocks copied verbatim from `default.yaml` (the former inert — present only so `SmartKeyNetEnv` can construct — the latter genuinely identical, since the DQN architecture is meant to be literally the same); its own new `soft_reward:` block (`w_lat: 1.0, w_en: 0.1, w_sec: 1.0`) is the only genuinely new set of weights. `scenario: S1` — the unmanipulated comparison the masked agent's own campaign uses first. Verified to load, construct, and train for real (25,000-step campaign, see SESSION_LOG.md's newest entry). |
| `configs/soft_reward_baseline_s3.yaml` | implemented+tested | **2026-08-25 (new file, masked-vs-soft-reward S3 comparison session)**: the S3 variant of `configs/soft_reward_baseline.yaml` (PROGRESS.md's former "Next task" follow-up (1), now done). `pool:`/`qkd_degradation:` copied verbatim from the real, committed `configs/scenarios/s3_degradation.yaml` (not re-derived or re-tuned — this was a measurement session); `security_masking: false`/`soft_reward:` copied verbatim from `configs/soft_reward_baseline.yaml`. Same standalone-config-per-scenario convention as `configs/scenarios/*.yaml`, kept alongside `configs/soft_reward_baseline.yaml` in `configs/` (not `configs/scenarios/`) since it belongs to this agent, not the masked one. Verified to load, construct (`security_masking is False`, `scenario == "S3"`), and train for real (3 training seeds x 75,000 steps, see SESSION_LOG.md's newest entry for the full comparison numbers). |

---

## Last verified

- **Date:** 2026-08-25 (adversarial trace generator session)
- **Commit:** `36f3b20` ("docs: explain p99_latency saturation as discrete-cost-model artifact, recommend total_reward/below_floor_rate instead -- 2026-08-25") — the commit this session started from; see SESSION_LOG.md for this session's own commit
- **`pytest` pass count:** 554 passed, 1 xfailed (531 prior + 23 new, all in the new `tests/test_trace_generator.py`)
- **Branch:** Confirmed `main`/`dev21` in sync at session start (`git rev-parse main dev21` both `36f3b2092be0a9f773a623972b23abcee4729f93`). This session's commit (real code: `attack/trace_generator.py` (new) + `tests/test_trace_generator.py` (new) — nothing else touched, confirmed via `git status --porcelain` and an empty `git diff --stat` on every protected env/agent file) was ff-merged into `dev21` at the end — see SESSION_LOG.md for the final shared hash. Not pushed to origin this session, per instruction.
