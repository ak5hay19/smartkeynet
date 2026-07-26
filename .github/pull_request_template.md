## What does this PR do?

<!-- One or two sentences. Link an issue if there is one. -->

## Which owner area does this touch?

- [ ] A — World (data/, env/request_generator.py, env/forecast_provider.py, forecaster/)
- [ ] B — MDP (env/environment.py, env/pool_sim.py, env/deferral_queue.py, env/masking.py, metrics/, configs/)
- [ ] C — Brain (agents/, experiments/harness.py)
- [ ] D — Story (attack/, dashboard/, api/, docs/)
- [ ] `env/contracts.py` or `configs/` (danger zone — did you ping the whole team? split.md §3)

## Hard Rules checklist (PLAN.md §4 "Hard Rules") — reviewer, check every box before approving

- [ ] **No security term in the reward, anywhere, ever.** Not even a small one "to help training" (Hard Rule 1).
- [ ] **Floors are enforced by action masking**, not by reward penalties. Any threat-signal change only *raises* floors, never lowers them (Hard Rule 2).
- [ ] **One agent, one MDP.** No migration-order choice, no graph visibility, no crew allocation leaking into the agent (Hard Rule 3).
- [ ] **No invented security constants.** Tiers cite NIST PQC categories / SP 800-57 / CNSA 2.0 / BSI-ANSSI / ETSI GS QKD 014 (Hard Rule 4).
- [ ] **No free mid-session algorithm switching.** Key-type changes only at rekey boundaries, with explicit cost (Hard Rule 5).
- [ ] **QKD stays architecturally honest** — backbone resource behind the KMS, pool semantics, ETSI-style delivery; no endpoint "does QKD" (Hard Rule 6).
- [ ] **Tuned baselines are respected** — no baseline silently weakened to make the DQN look better (Hard Rule 7).
- [ ] **Train/eval split for migration honored** — S6 is held-out eval only, never trained on (Hard Rule 8).
- [ ] **Pool exhaustion never causes a downgrade** — a hybrid-mandatory request that can't be served is deferred and logged as a regret event, never served below floor (Hard Rule 9).
- [ ] Not trained on `rl_experiment_*` / `synthetic_rl_*` Q-OPSEC logs (those are baseline-reproduction/calibration inputs only).
- [ ] Not loading `context_dataset_basic.csv` / `context_dataset_advanced.csv` (verified degenerate — do not use).
- [ ] No Q-OPSEC CSVs committed to the repo (no LICENSE file — see `data/README.md`).

## Tests

- [ ] `pytest` passes locally.
- [ ] Added/updated tests for any new module or changed interface.

## Anything the reviewer should look at closely?

<!-- Optional. -->
