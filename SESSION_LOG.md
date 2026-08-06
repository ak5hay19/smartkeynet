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
| A | Data + forecaster + graph | 2026-08-06 | main | `forecast_provider.py`'s `MovingAverageForecaster` + `request_generator.py`'s `random_request_generator()` implemented + tested — dataset ingestion, `build_tenant_graph`/`RequestGenerator` still not started |
| B | ENV + pool + reward + masking | 2026-08-06 | main | `pool_sim.py` + `deferral_queue.py` + `metrics/regret.py` + `masking.py` implemented + tested — `environment.py` wiring is next |
| C | Agent + baselines | — | — | Not started |
| D | Attack + dashboard + API + paper | — | — | Not started |

**contracts.py frozen:** ☑ Yes — `env/contracts.py` is complete and committed on `main` (Action enum, StateDict, ForecastProvider ABC, Request, event-log TypedDicts).
**Week gate status:** W1 ☐ *(contracts freeze ✅ done; A's real data ingestion, B's real pool_sim, C's random-agent stub, D's report skeleton still open)* · W2 ☐ · W3 ☐ · W4 ☐ · W5 ☐ · W6 ☐ · W7 ☐ · W8 ☐

---

## Sessions (newest first)

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
