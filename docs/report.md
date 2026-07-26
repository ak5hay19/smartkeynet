# SmartKeyNet: RL for Hybrid Cryptography

> Report skeleton (PLAN.md §10 kickoff step 1; split.md §1, Person D
> "report/paper starts week 1 — methodology + related work don't need
> code"). Section owners noted per split.md §1/§2 week 7 plan. Fill in
> as results land; do not remove the Hard Rules cross-reference below.

## Abstract

_TODO (all, week 7-8)._ One paragraph — condense PLAN.md §0.

## 1. Introduction

_Owner: Person D (split.md §2, Week 7)._

- Motivation (PLAN.md §2: HNDL, operational scarcity, soft-reward attack surface)
- Contribution statement (PLAN.md §2.3: "Security as constraint... demonstrated against a live steering attack")
- Roadmap of the report

## 2. Related Work

_Owner: Person D (split.md §2, Week 7)._

- RL for adaptive cryptography / WSN key management (incl. the
  Noetzold soft-reward design we reproduce and critique — PLAN.md §2)
- QKD-backed KMS architectures (ETSI GS QKD 014; BT/Toshiba, JPMorgan
  x Toshiba/Ciena, SK Telecom, EuroQCI — PLAN.md §3)
- Post-quantum migration guidance (NIST PQC, NSA CNSA 2.0, BSI/ANSSI — PLAN.md §2, Hard Rule 4)
- Positioning: how SmartKeyNet differs (masking vs. soft reward; dual-head foresight; regret accounting)

## 3. Methodology

_Owner: Person B, with A/C subsections (split.md §2, Week 7)._

### 3.1 Environment & MDP (Person B)
- Pool simulator (`env/pool_sim.py`)
- Deferral queue & regret accounting (`env/deferral_queue.py`, Addition C)
- Policy table & action masking (`env/masking.py`, Hard Rule 2)
- Reward formula (PLAN.md §4)

### 3.2 Data & Forecaster (Person A)
- RT-IoT2022 threat forecaster pipeline
- Dual-head LSTM (Addition A): threat head + pool head
- Tenant graph generator

### 3.3 Agent & Baselines (Person C)
- Masked DQN (`agents/dqn.py`)
- Tuned baselines (Hard Rule 7)
- Soft-reward baseline reproduction (`agents/soft_reward_baseline.py`)

## 4. Experiments

_Owner: Person C, with D's attack subsection (split.md §2, Week 7)._

- Scenario grid S1-S6 (PLAN.md §5)
- Foresight ablation E-A (Addition A)
- Steering attack + dose-response sweep (Person D; PLAN.md §5 S5)

## 5. Results

_Owner: Person C (split.md §2, Week 7)._

- Closing comparison table (PLAN.md §6): Agent (± foresight) vs.
  always-PQC vs. always-hybrid vs. static-threshold vs. random, across
  S1-S4 + S6 — p99 latency, pool-exhaustion events, regret events,
  forced-rekey ratio, floor violations (agent column: 0, structurally
  guaranteed)
- Steering-attack figures (Person D)
- Regret attribution plot (Addition C)

## 6. Discussion

_Owner: all (synthesize week 7-8)._

- Anticipated examiner questions (PLAN.md §8) — pre-answer the obvious ones here
- Limitations, cut lines actually invoked (split.md §2.1)

## 7. Conclusion

_Owner: Person D (week 7-8)._

## References

_Owner: all — cite as you build; do not invent security constants (Hard Rule 4)._

---

**Cross-reference:** every claim in this report must be consistent
with PLAN.md's Hard Rules (§4 "Hard Rules" — no security term in
reward, floors via masking only, one agent/one MDP, no invented
security constants, no free mid-session algorithm switching, QKD
architecturally honest, tuned baselines mandatory, train/eval split
for migration, pool exhaustion never downgrades). If a result seems to
contradict a Hard Rule, the result is wrong — fix the environment/agent,
don't soften the rule.
