# ORPB-21 source-support verdict

## Verdict

**Retire ORPB-21 unchanged before novelty and economic outcomes.**

The frozen 21-operation participant-breadth residual produced **137** primary
train events in `[2021-01-01, 2023-01-01)`, exceeding the preregistered maximum
of **130**. It missed only `train_events_max`; the lower count bound, yearly
coverage, side balance, month concentration, 2023 support, and all source/clock
integrity checks passed.

This is a source-support rejection, not an unprofitability claim. No comparator
clock values were loaded for novelty and no BTC, funding, future return, PnL,
absolute return, CAGR, or strict-MDD field was opened.

## Frozen support statistics

| Split | Events | LONG | SHORT | Largest month share |
|---|---:|---:|---:|---:|
| 2021–2022 train | 137 | 78 | 59 | 8.03% |
| 2023 selection | 73 | 31 | 42 | 13.70% |

Train yearly counts were 62 in 2021 and 75 in 2022. Selection half-year counts
were 35 in 2023H1 and 38 in 2023H2. All counts include only clocks whose origin,
decision, entry, and next-operation exit are fully contained in the declared
split.

The source evaluator read all 1,498 bound operations, retained 1,489 complete
rows, preserved nine clock-only quarantines, and built 1,297 eligible feature
rows after exact quarantine resets. It wrote 1,588 primary/control clock rows.

## Why no repair is allowed

Reducing density after seeing 137 events would require changing the lookback,
tail, regression, calendar, amount floor, side, or another selection rule after
incidence was known. The preregistration explicitly forbids that repair. ORPB-21
therefore cannot proceed to comparator overlap or economic testing and cannot
be rebranded as a weaker threshold variant.

## Reproducibility

- preregistration SHA-256:
  `62855414b6926ff3e0f2bc37fe3c4c5c6f46f78803c66d6da564ec65de937b30`;
- evaluator SHA-256:
  `f743b12a28e4928c2b9bcb996d8086ed6c65b91de40f8a6b798f7f3fd3222ae8`;
- clock ledger SHA-256:
  `ef21323229801f11557e0c2d9d4465f7d58b13569552d656d64fdb7d440622ed`;
- support report SHA-256:
  `cb341310436e5de2cc578dd8232f99f0e78efd50a6c8110a9e5c549dc60c5d0b`;
- support manifest hash:
  `62cc70768d70964ce46f65a9d6025b589bf7b7937e18d88fa79ef0271ea8804d`.

The next research unit must start from a genuinely independent mechanism; it
may not tune ORPB-21 using this failed incidence result.
