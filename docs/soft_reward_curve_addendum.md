# SmartKeyNet — Soft-Reward Baseline Curve-Shape Addendum (Fig. 5 correction)

> **Purpose:** ready-to-paste content correcting the paper's description of
> the soft-reward baseline agent's reward-posture dependence, and supplying
> the real, evidence-backed mechanism behind its S5 dose-response curve
> shape. Drafted from `SESSION_LOG.md`'s "root-cause soft-reward baseline's
> S5 dose-response curve shape, diagnostic" entry (commit `9eec891`, both
> `main`/`dev21`) — every number below is quoted from that entry exactly,
> not re-derived or rounded.
>
> **Status:** drafted, not yet inserted into any paper source. **A real
> gap, flagged rather than papered over:** no paper source file
> (`smartkeynet_ieee_paper_5.pdf` or its source) exists anywhere in this
> repository — confirmed by a full-repo glob for `*.pdf` and `*ieee*`
> (zero hits) and by reading `docs/report.md` in full, which is a
> section-outline skeleton (owners + TODOs only) with no Fig. 5 prose, no
> reward-mechanism description, and no figure captions of any kind. The
> "existing language" referenced below is therefore reconstructed from
> this thread's own prior session's hedged paraphrase (`SESSION_LOG.md`,
> 2026-08-27 entry: *"if the paper draft's Fig. 5 narrative describes
> `p̂_t` ... entering this agent's reward with the security term
> shrinking ... that does not match what was actually built"*) — **not a
> verbatim quote from an actual paper file**, since none could be located.
> See the Open Items section below; whoever holds the real paper source
> must verify the exact existing sentence(s) before applying the suggested
> edit.
>
> **Where this belongs:** alongside `docs/steering_attack_limitations_addendum.md`'s
> own "Open item" section (which already flags that the soft-reward
> curve's mechanism was "not yet resolved" as of that file's own drafting)
> — that item is now resolved, and this file supersedes it. In the paper
> itself, insert wherever Fig. 5 (or its surrounding text) describes the
> soft-reward baseline's reward as a function of estimated threat/posture.

---

## Paragraph to insert

> The soft-reward baseline's reward function does not read the estimated
> threat probability or discrete posture directly: `compute_soft_reward`
> is computed from exactly two state fields, `policy_floor` and
> `key_type_onehot`, and its security term, `security_score(tier)`, is a
> fixed three-value lookup keyed only on the tier actually delivered —
> there is no continuous threat-weighted term for the attack to shrink.
> The one real exception is narrow and discrete: `REKEY_NOW`'s delivered
> tier resolves to `max(existing tier, floor)`, where `floor` is the same
> one-way-ratchet-processed `policy_floor` the masked agent's own action
> mask reads — so an attack that pulls this upstream value down can lower
> what a `REKEY_NOW` decision delivers, without the agent's policy
> changing its mind about which action to take. We verified this
> directly: across every tier-establishing decision (100 total, `SERVE_*`
> or `REKEY_NOW`) at α = 0.9 across 5 eval seeds, 16 decisions had a
> true/attacked posture-bucket divergence. In none of the 16 did the
> Q-network's chosen action change under a true-posture substitution
> (holding every other state field fixed) — the agent picks `REKEY_NOW`
> regardless. But in 6 of those 16, the tier that same, unchanged
> `REKEY_NOW` decision delivered did change, because the resolution
> formula reads the attacked floor directly. The curve's flat-then-step
> shape follows the same discrete posture-bucket-crossing mechanism
> already established for the masked agent's own high-α finding: a
> finer-resolution sweep of the existing checkpoints pinned the crossing
> to α = 0.80 → 0.85, where the posture-bucket-mismatch rate jumps from
> 0.0224 to 0.8112 in exact step with V(π) rising from 0.2896 to 0.3256.
> Both agents' only real point of leverage for this attack, despite
> having completely differently structured reward functions — one with no
> security term at all, one with a fixed tier-only term — turns out to be
> the same shared upstream quantity: the one-way-ratchet-processed
> posture/floor computed by the (attackable) forecaster. Neither agent's
> reward is what the attack actually reaches; both are only ever reached
> through this one shared channel.

---

## Supporting numbers

**Controlled same-state-except-posture comparison** (α = 0.9, 5 eval seeds, real, already-trained `checkpoints/soft_reward_baseline_s2.pt`, no retraining):

| Quantity | Count |
|---|---|
| Total tier-establishing decisions (`SERVE_*`/`REKEY_NOW`) | 100 |
| Of those, true/attacked posture-bucket divergent | 16 |
| Of the 16, Q-network chosen action changed | 0 |
| Of the 16, delivered tier changed (same, unchanged `REKEY_NOW` action) | 6 |

**Discrete-threshold correlation** (finer alpha grid than the original 11-point sweep, same 5 eval seeds, S2):

| α | V(π)_true (mean, 5 seeds) | Posture-bucket-mismatch rate |
|---|---|---|
| 0.00 | 0.2896 | 0.0000 |
| 0.50 | 0.2896 | 0.0032 |
| 0.70 | 0.2896 | 0.0080 |
| 0.75 | 0.2896 | 0.0112 |
| 0.80 | 0.2896 | 0.0224 |
| **0.85** | **0.3256** | **0.8112** |
| 0.88 – 1.00 | 0.3256 | 0.8112 |

Every mismatch across the whole grid is an *understatement* (mismatch rate == understatement rate exactly, at every α) — the attack never causes the attacked posture to read higher than the true one.

**Alpha=0 baseline (`0.2896`) — pre-existing, NOT attack-caused.** At α = 0.0 the posture-bucket-mismatch rate is exactly `0.0000` (no attack is possible, attacked == true trivially), yet `V(π)_true` is already `0.2896`. This is the same structural property already measured, independently, in the earlier masked-vs-soft-reward **unmanipulated S3 comparison** (no attacker present at all): below-floor service rate `0.1687` — this agent's reward has no incentive to proactively rekey once holding a tier, so once a floor ratchets up mid-episode, an already-established lower-tier key just keeps getting `REUSE`d. Do not present the α = 0 value as an attack effect in the paper.

---

## Exact suggested edit

**Existing language this corrects (reconstructed paraphrase — see the Status note above: no verbatim source sentence could be located in this repo):**

> "...`p̂_t` enters the reward, [the] security term shrinks, latency outbids it, [a] weaker tier [is] served..."

**Proposed minimal correction**, to be checked against and substituted into the real sentence once located:

> "...the delivered tier's security term is fixed once a tier is chosen; the attack instead reaches this agent only through `REKEY_NOW`'s delivery resolution, which reads the same one-way-ratcheted floor the masked agent's mask reads — so a suppressed floor can lower what a `REKEY_NOW` decision delivers, without the agent's own policy choosing differently or any continuous threat term entering the reward at all..."

This is deliberately scoped as a targeted phrase-level substitution, not a rewrite of the surrounding section — the goal is to remove the "`p̂_t` enters the reward and shrinks a continuous security term" claim and replace it with the real, narrower, discrete mechanism above.

---

## Open items — not resolved by this file

- **The exact existing sentence could not be verified.** No paper source
  file exists anywhere in this repository (glob for `*.pdf`/`*ieee*`:
  zero hits; `docs/report.md` is a skeleton with no Fig. 5 prose).
  The "existing language" quoted above is a reconstruction from this
  thread's own hedged description, not a verified quote — whoever holds
  the actual `smartkeynet_ieee_paper_5.pdf` source must locate the real
  sentence(s) and confirm the proposed correction still fits before
  applying it verbatim.
- **Whether Fig. 5's diagram itself (not just its caption/surrounding
  prose) draws an arrow or block implying the reward reads `p̂_t`
  directly** is an open question this file cannot answer without seeing
  the actual figure — if the diagram depicts that mechanism graphically,
  it needs a matching correction (e.g. re-routing the arrow through
  `policy_floor` → `REKEY_NOW` resolution specifically, not into the
  reward block generally), not just a text edit. Flagged here for
  whoever does final paper integration to decide.
- No numeric gaps: every figure in this file was quoted directly from
  `SESSION_LOG.md`'s 2026-08-27 entry: none needed to be invented,
  approximated, or reconstructed from summary.
