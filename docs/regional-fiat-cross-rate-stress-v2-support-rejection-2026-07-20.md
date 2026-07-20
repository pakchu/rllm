# RFXS2-576 source-support rejection — 2026-07-20

## Verdict

**Reject RFXS2-576 before opening outcomes.**

The frozen regional-fiat cross-rate clock had ample and balanced incidence in
both preregistered periods. It nevertheless failed the immutable source-novelty
gate because its train-period common stress score remained too closely related
to the BTC return shadow:

- train absolute Spearman correlation: **0.5178017801**;
- preregistered maximum: **0.50**.

The selection-period value was 0.3900353632 and the timestamp/exposure overlap
checks against FQPR and SDDR all passed. Those passes do not override the single
failed conjunctive gate. No execution OHLC, funding, future return, label, PnL,
absolute return, CAGR, or MDD was opened. The threshold, direction, baseline,
and hold are not repaired after incidence.

## Frozen support result

| Period | Events | LONG / SHORT | Largest month | Required coverage | Result |
|---|---:|---:|---:|---|:---:|
| Train, 2021–2022 | 106 | 45 / 61 | 8.49% | all frozen annual, quarterly, side, month, and contributor gates | pass |
| Selection, 2023 | 75 | 36 / 39 | 10.67% | all frozen half-year, quarterly, side, month, and contributor gates | pass |

Train incidence was 42 events in 2021 and 64 in 2022. Selection incidence was
41 events in 2023H1 and 34 in 2023H2. Every required train quarter and every
2023 quarter cleared its minimum. EUR, TRY, and BRL each contributed to more
than 40% of accepted events in both periods.

## Frozen novelty result

| Gate | Train | Selection / 2023 | Limit | Result |
|---|---:|---:|---:|:---:|
| abs Spearman, common z vs BTC-return z | **0.517802** | 0.390035 | at most 0.50 in each period | **fail** |
| FQPR exact-entry Jaccard | — | 0.058577 | at most 0.20 | pass |
| FQPR abs signed-exposure Pearson | — | 0.026008 | at most 0.40 | pass |
| SDDR exact-entry Jaccard | — | 0.020202 | at most 0.10 | pass |
| SDDR abs signed-exposure Pearson | — | 0.010115 | at most 0.40 | pass |

RFXS2 is temporally distinct from the two comparator strategies, but the
underlying train-period score does not clear the preregistered return-shadow
independence requirement. It is therefore not authorized for an outcome
evaluator and is retired as a candidate family on this branch.

## Outcome-blind controls

The committed evaluator opened exactly seven whitelisted source-only inputs:
the frozen four-book daily panel and manifest, three mechanism/source-decision
documents, and the frozen FQPR and SDDR clocks. Its result records:

- `source_only = true`;
- `outcomes_opened = false`;
- `execution_ohlc_opened = false`;
- `funding_opened = false`;
- `future_return_opened = false`;
- `pnl_cagr_mdd_opened = false`;
- `post_2023_source_opened = false`;
- `next_stage_authorized = null`.

Source-only controls were emitted only as diagnostics and were not used to
select a replacement rule. No same-family variant advances.

## Integrity anchors

- source commit: `e576d22c6f2d567d4b40358f755bef4b27c188d4`
- frozen evaluator commit: `cdd11ad1dec81209273be6130b4d99119979b49d`
- evaluator SHA-256:
  `dc9f2237160de4db843a32b053d0ebdd46dc44aa2074e07d1579088199ea41a4`
- source-panel SHA-256:
  `5dbc697c8299ac892295a01302e9f2d883a6e252c8d3d85a8f60f3a369b533d3`
- result SHA-256:
  `0b18ef3ab0a8b057e8966dcd5f358ea30320069ea5ba4ffdda3519526e5b0986`
- clock SHA-256:
  `180181a7f95308a6fe5bac3d829dbb49c8e1e6aae8e84e69e6558146bee32413`

Two complete executions were byte-identical for both result and clock. The
second execution took 1.31 seconds with 119,864 KiB maximum resident memory.
