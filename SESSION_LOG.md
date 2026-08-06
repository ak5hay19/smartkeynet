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

## Active state (keep this section current — update every session)

> Currently working **solo across all four areas** until the rest of the team
> is back — treat every row below as "you" for now. Split back out by
> Person once `handoffs/` is reintroduced.

| Person | Area | Last session | Current branch | Status |
|--------|------|-------------|----------------|--------|
| A | Data + forecaster + graph | 2026-08-06 | main | Scaffolding verified, dataset placed — no feature code yet |
| B | ENV + pool + reward + masking | 2026-08-06 | env/deferral-queue | `pool_sim.py` + `deferral_queue.py` + `metrics/regret.py` implemented + tested — `environment.py` wiring is next |
| C | Agent + baselines | — | — | Not started |
| D | Attack + dashboard + API + paper | — | — | Not started |

**contracts.py frozen:** ☑ Yes — `env/contracts.py` is complete and committed on `main` (Action enum, StateDict, ForecastProvider ABC, Request, event-log TypedDicts).
**Week gate status:** W1 ☐ *(contracts freeze ✅ done; A's real data ingestion, B's real pool_sim, C's random-agent stub, D's report skeleton still open)* · W2 ☐ · W3 ☐ · W4 ☐ · W5 ☐ · W6 ☐ · W7 ☐ · W8 ☐

---

## Sessions (newest first)

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
