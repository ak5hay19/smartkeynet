# SmartKeyNet — Limitations Addendum (S5 Dose-Response Sweep)

> **Purpose:** ready-to-paste content for the paper's Limitations section,
> drafted from the real S5 dose-response sweep results. Self-contained —
> traceable to `SESSION_LOG.md` (commit `4669278`, both `main`/`dev21`) if
> the underlying numbers ever need re-verifying.
>
> **Status:** drafted, not yet inserted into `docs/report.md` /
> `smartkeynet_ieee_paper_5.pdf`'s source. Insert after the existing
> sentence: *"masking survives because under-estimation costs the masked
> system nothing, not because it prevents under-estimation."*

---

## Paragraph to insert

> We measured this boundary directly. Sweeping attack strength α across
> eleven steps on S2, the masked agent's below-floor service rate V(π)
> holds at exactly zero for α ≤ 0.8, then rises to 0.30 at α ≥ 0.9.
> Tracing the mechanism confirmed the masking logic itself is not at
> fault: across every seed and every α, zero decisions fell below the
> floor the agent was actually shown — the guarantee after (2)–(3) held
> without exception. What changes at high α is the floor computation
> upstream of it. Our floor-ratchet is one-way by construction, but a
> one-way ratchet can only hold a threat level it has been allowed to
> observe at least once; when the attack suppresses the true signal from
> the first tick of the episode onward, the ratchet is never given a
> genuine reading to lock onto. A supplementary sweep in which the attack
> begins only after the scripted threat elevation (rather than from tick
> 1) confirms this precisely: V(π) returns to zero once the estimator
> receives even one honest observation before suppression starts. The
> guarantee, stated precisely, is therefore: masking cannot be talked out
> of a threat it has already registered, but it cannot register a threat
> that was never allowed to arrive undisguised. Whether this residual gap
> matters operationally depends on whether a deployment can guarantee an
> honest warm-up window — a weaker but still meaningful claim than the
> reward-shaped baseline offers, whose degradation is present at every α
> rather than confined to sustained attacks from session start.

---

## Supporting numbers (for a table/footnote, or reviewer questions)

**Primary sweep** — attack active from tick 1, S2, masked DQN, 5 eval seeds/α:

| α range | Masked DQN V(π) | Mechanism |
|---|---|---|
| 0.0 – 0.8 | 0.0000 | Clean — floor computed correctly, zero violations |
| 0.9 – 1.0 | up to 0.30 | Floor computation under-estimates; masking enforcement itself still 0 violations against the (wrong) floor it was shown |

**Supplementary sweep** — attack begins only at the scripted `elevate_at_step` (not tick 1), same scenario/checkpoints:

| α range | Masked DQN V(π) | Interpretation |
|---|---|---|
| 0.0 – 1.0 | 0.0000 throughout | One honest reading before suppression starts is sufficient for the ratchet to lock and hold |

**Root cause, confirmed by direct code check (not inferred):** `compute_mask()` / `PolicyTable` enforcement showed **zero attacked-floor violations at every seed, every α, in both sweeps** — the masking layer is not the failure point. The gap lives entirely in threat-signal detection, one layer upstream of masking.

---

## Open item — not yet resolved, don't fold into the paragraph above

The soft-reward baseline's own V(π) curve did **not** match the paper's original clean-rise prediction — it showed a flat-then-step pattern off an already-substantial unattacked baseline, rather than a smooth increase with α. This is reported honestly in `SESSION_LOG.md` but the *mechanism* behind that specific shape has not been investigated the way the masked agent's finding was. Recommend either:
- a short follow-up session isolating why (same rigor as the masked-agent investigation), if time allows before the deadline, or
- stating in the paper only what's measured ("the soft-reward baseline's degradation does not follow a smooth dose-response curve; see Fig. X") without claiming an explanation not yet established.

Do **not** present this curve as clean/predicted-shape in the paper — that would misrepresent what was actually measured.

---

## Project status snapshot (for reference, dated to this session)

| Area | Status |
|---|---|
| Environment / pool / masking / reward (spine) | ~95% — full S1–S6 grid real and tested, Gate W3 met |
| Agent / baselines / experiments | ~90% — masked DQN, 4 baselines, soft-reward baseline, multi-seed + attack-aware eval harness all real and tested |
| Steering attack (S5) | Essentially done — trace generator, attacking forecast provider, real 11-α sweep, both findings run to real mechanisms |
| Data / forecaster / tenant graph | ~30% — tenant graph real; threat-detection pipeline still a placeholder EWMA stub (not required for the attack's validity — it targets the estimator's *input*, not its internals) |
| Dashboard / API / paper | Paper draft strong; dashboard is one backend panel with nothing rendered; API facade not started |

**Overall: ~65–70% complete.** The shift since the last snapshot is qualitative, not just numeric — the project's actual headline thesis result (the steering attack) now exists and has been honestly interrogated, rather than being the one major piece still missing. Remaining work is weighted toward presentation/packaging (dashboard, real forecaster, final report integration) rather than open research risk.
