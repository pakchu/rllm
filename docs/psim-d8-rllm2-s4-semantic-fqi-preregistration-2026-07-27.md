# PSIM-D8-RLLM2-S4 semantic fitted-Q preregistration

## Purpose

S3 proved that the pinned `google/gemma-4-E4B-it` source representation can be
extracted deterministically for every 2020–2023 daily PSIM state. It did not
prove profitability. S4 is the first economic stage and is limited to:

1. opening the exact 2020 BTCUSDT 5-minute and funding outcomes;
2. fitting the frozen semantic fitted-Q family on 2020 only; and
3. sealing every 2021 target schedule before any 2021 outcome is opened.

No 2020 training metric may select a primary candidate. Both frozen candidates
must transfer into 2021, where the preregistered gate will make the first
algorithm choice.

## Frozen primary family

The two primary candidates inherit the original RLLM1 semantic gate without a
new hyperparameter search:

- `semantic_ridge_fqi`: PCA32 plus current-position one-hot, Ridge alpha 100;
- `semantic_extra_trees_fqi`: PCA32 plus current-position one-hot, 512 trees,
  depth 6, `sqrt` features, leaf 12, split 24, seed 20260727.

Both use 25 Bellman iterations, discount 0.99, and actions
`FLAT / SHORT / LONG = 0 / -0.5 / +0.5` account gross. PCA is fit only on the
366 source rows from 2020 with full SVD, no whitening, and deterministic
component-sign canonicalization.

The source representation is the exact S3 2,560-dimensional float32 embedding.
Calendar fields, split labels, hashes, market data, returns, rewards, and PnL
are excluded from model features. Relation logits are reserved for controls and
diagnostics; they are not appended to either primary.

## Controls

The sealed 2021 family contains constants, persistence, exact source-payload
memory, metadata-only, path-size-only, cadence/topology-only, shuffled relation
and old/new pairing controls, status-scrubbed and protocol-side ablations,
current-position-only and masked-semantic controls, plus fixed reward,
direction, and action-code ablations for both primaries.

All shuffles are deterministic source-only within-month permutations using seed
20260727. The full family is fixed before any 2020 outcome is opened and later
forms one shared weekly max-stat family.

## Economics

S4 reuses the strict BCTP accounting implementation:

- decisions at 12:05 UTC daily and exact next-available 5-minute execution;
- 6 bp per changed notional, with a separate 10 bp stress result;
- exact funding, conservative boundary debit, and terminal flatten;
- full-calendar CAGR including flat time;
- one global strict-MDD high-water mark with favorable-then-adverse intrabar
  path and virtual liquidation cost;
- reward:

```text
log(max(E_end/E_pre, 1e-12))
  - (1/3) * held_path_downside_fraction
  - 0.001 * abs(target_new - target_old)
```

The only authorized outcome slice is `[2020-01-01T00:00:00Z,
2021-01-01T00:00:00Z)`: 105,408 five-minute rows and 1,098 eight-hour
funding rows. The shared 2020–2023 parent files may be streamed only by
`training/bctp_stage_sources.py`, which must stop after the exact 2020 row
count without reading the first 2021 row. Full parent payload hashing is
forbidden; only the isolated 2020 copies may be hashed and parsed. Existing
2021-or-later stage-local outcome copies remain forbidden.

## Chronology

S4 succeeds only after deterministic 2020 transition, PCA, base-schedule,
delayed-schedule, and schedule-manifest artifacts are hash-bound and the 2021
schedule manifest records zero market/funding access. The authorizing result is
published last.

On success, the only next authorization is a one-shot 2021 evaluation. QLoRA,
2022, and 2023 remain forbidden. On any training or sealing failure, S4 rejects
without opening 2021 outcomes.

Canonical machine-readable preregistration:

`results/psim_d8_rllm2_s4_semantic_fqi_preregistration_2026-07-27.json`
