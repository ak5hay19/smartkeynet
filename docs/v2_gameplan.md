# SmartKeyNet — v2 Gameplan (Live PCAP Stretch Goal + Final Presentation Steps)

Branch: `v2` (all technical work below happens here; `main` stays
frozen at `BASE PROJECT DONE`, hash `3e2c3ad`, as the safe fallback
for the review regardless of how `v2` goes).

**Hard rule:** set a hard stop date for Phases 1-3 before starting —
no more than roughly a third of remaining runway. The moment that
date hits, move to Phase 4-5 regardless of where the branch is.
A working, rehearsed deck without the pcap demo beats an unfinished
pcap demo and no rehearsal.

---

## Phase 0 — Safety net (done)

- `main` frozen at `3e2c3ad` (BASE PROJECT DONE).
- `v2` branch created, feasibility investigation complete and
  committed (`950d8a8`): tshark can supply the `service` field
  (real dissectors for all 9 non-`-` RT-IoT2022 categories) and
  payload-only byte counts (`tcp.len` confirmed excludes TCP header).
  One caveat found: modern traffic (QUIC/TLS 1.3) doesn't map
  cleanly onto RT-IoT2022's 2022-era category set — needs an
  explicit mapping policy, not a blocker.

---

## Phase 1 — Build the real extractor (next)

Turn the feasibility findings into a working, tested per-flow
feature extractor.

- Build the per-flow aggregation module (tshark gives packets, not
  flows — this is the main new work).
- Decide and document the service-mapping policy (how TLS/QUIC/etc.
  map onto RT-IoT2022's 10 categories, including an explicit
  "unmapped/other" bucket).
- Match RT-IoT2022's flow-timeout conventions so flows are cut the
  same way the training data was.
- Validate on a real captured pcap: confirm output shape/ranges are
  sane, confirm it plugs into the existing `forecast_provider_factory`
  seam the way `RTIoT2022ReplayProvider` does.
- **Done when:** a real `.pcap` file produces real, structurally
  correct feature vectors the trained LSTM can consume.

## Phase 2 — Upload demo

Wire Phase 1's extractor into an upload-and-run flow: person uploads
a `.pcap`, extraction runs live, the trained forecaster + masked DQN
make real decisions, streamed to the screen as they're computed.

- **Done when:** a real file can be uploaded and produces genuine,
  live-computed decisions on screen.

## Phase 3 — Live capture (optional, only if time allows)

Same extractor, triggered by a live capture session on the person's
own laptop instead of an uploaded file — a small delta on top of
Phase 1/2, not new work. Explicitly skippable without guilt.

---

## Phase 4 — Use case + slides

- Pick use case(s) from `KMS_usecases.md`.
- Drop real Phase 4b numbers (Gate W3: 5.61x/3.58x; S3 comparison:
  below-floor 0.0 vs 0.080, regret 0.0 vs 4.42; S5 dose-response:
  real two-plateau step curve, ~0.2984 ceiling) into the drafted
  slide content.
- Optionally build the named South Korea (or chosen use case)
  narrative scenario into the dashboard, if time allows.

## Phase 5 — Rehearsal

Out loud, ideally to someone who doesn't know the project, timed.
Non-negotiable — this is the step most likely to get silently
dropped if a technical phase runs long. Don't let that happen.

---

## Known honest caveats to carry into any demo of this stretch work

- Live/uploaded traffic will differ from RT-IoT2022's training
  distribution, both in content (attack vs. benign) and in protocol
  mix (modern traffic skews QUIC/TLS 1.3, training data doesn't).
  State this plainly if the pcap demo is shown — don't imply
  live-traffic-equivalent confidence to what the model was actually
  validated on.
- This entire branch is stretch-goal work layered on top of a
  complete, real, already-working base project. If it doesn't
  finish, that's a fine outcome — say so plainly, don't apologize for
  it.
