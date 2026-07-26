# SmartKeyNet — Data

> Read PLAN.md §"Datasets & Provenance" before touching anything in
> this folder. That section is authoritative; this README is a
> practical how-to on top of it.

This project uses external data in exactly **three** slots: the threat
forecaster (RT-IoT2022), the sensitivity classifier + policy-table
calibration + baseline reproduction (Q-OPSEC), and a QKD SKR/QBER
trace (or documented synthetic). Everything else — pool drain, key
sizes, the tenant graph, the migration schedule — is simulator physics
we write ourselves. Do not go hunting for datasets for those.

## Golden rules (see PLAN.md for full detail)

- **NEVER train the agent on the `rl_experiment_*` / `synthetic_rl_*`
  Q-OPSEC logs.** They are *outputs* of the Noetzold soft-reward agent
  we critique in the steering attack (PLAN.md §2, §5 S5). They exist
  here for **baseline reproduction + feedback calibration only**.
- **DO NOT load `context_dataset_basic.csv` or
  `context_dataset_advanced.csv`.** Verified degenerate: 422 rows but
  only 4 unique feature rows, `security_level_label` is 100%
  "critical", `encryption_script_label` is 100% one value. No label
  variety to learn from.
- **QKD scarcity is not in any borrowed dataset.** It comes from a
  CV-QKD SKR/QBER trace (or a documented synthetic generator) driving
  the pool sim's refill/drain arithmetic (`env/pool_sim.py`).

## Licensing / attribution — Q-OPSEC has NO LICENSE file

The Q-OPSEC repo ships with no `LICENSE` file, which means all rights
are reserved by default. For this capstone:

1. Cite Q-OPSEC / Noetzold explicitly in the report (`docs/`).
2. **Do not redistribute their CSVs inside this repo**, public or
   private-but-shared-widely. Keep them `.gitignore`d locally (see the
   root `.gitignore`); commit only small derived samples we author
   ourselves (e.g. a 100-row sample) if needed for CI.
3. Send the author a one-line email / open a GitHub issue requesting
   license clarification — the reply is citable in the report.

## Slot-to-source map (PLAN.md "Datasets & Provenance")

| Component | Primary source | Optional companion | Notes |
|---|---|---|---|
| Threat forecaster (LSTM) | **RT-IoT2022** | CICIDS2017 / UNSW-NB15 / TON_IoT (pick one, max) | Real IoT intrusion flows, labeled. |
| Sensitivity classifier | **Q-OPSEC `confidentiality_train`/`valid`** (320/80 rows, 4-class) | synthetic labeled text (stretch) | Reuse directly; comparable to Noetzold on the confidentiality axis. |
| QKD pool refill (SKR/QBER) | **CV-QKD experimental trace** | documented synthetic SKR process | Synthetic fallback is fine for a capstone if the generation procedure is stated and rate ranges are cited. |
| Policy-table calibration | **Q-OPSEC `synthetic_context_dataset`** (939 rows, 6 balanced classes) | — | Sanity-check (risk, confidentiality) -> tier mapping only. |
| Baseline reproduction + feedback calibration | **Q-OPSEC `rl_experiment_*` / `synthetic_rl_*`** | — | Reproduce the Noetzold agent for the steering attack. **Never train our agent on these.** |
| Pool drain, key sizes | ETSI GS QKD 014 (spec, not a dataset) | — | Simulator arithmetic. |
| Primitive latency/energy costs | published liboqs / pqm4 benchmarks | — | Measured, not invented. |
| Tenant graph | NetworkX synthetic generator (`env/request_generator.py`) | — | Not a dataset. |
| Migration schedule (S6) | config file (`configs/`), scripted, exogenous | — | Not a dataset. Never agent-controlled (Hard Rule 3). |

**DO-NOT-USE:** `context_dataset_basic.csv`, `context_dataset_advanced.csv` (see above).

## Layout

```
data/
  README.md          # this file
  get_data.py         # download/prep stub — see below
  sample/             # small, repo-committable samples for CI (see PLAN.md licensing note)
  raw/                # (gitignored) full downloaded datasets — created by get_data.py
```

## Downloading

```
python data/get_data.py --dataset rt_iot2022
python data/get_data.py --dataset qopsec-confidentiality
python data/get_data.py --dataset qopsec-synthetic-context
python data/get_data.py --dataset qkd-trace
```

`get_data.py` is a stub (PLAN.md scaffolding phase) — it does not yet
fetch anything. Fill in each `_download_*` function with the real
source URL/instructions once you have followed the Q-OPSEC licensing
step above and located the RT-IoT2022 UCI/Kaggle mirror and a citable
CV-QKD trace source.
