# FQPR-3 evaluator freeze

## Decision

The sequential evaluator for `FQPR-3 — Fiat-Quote Participation Rotation` is
frozen **before opening any price, funding, or strategy outcome**.

- Evaluator source SHA-256:
  `db7d57c17eac9194ad6c38006dee6a3e888acde1a72bf3d950201e2e951f6548`
- Freeze manifest hash:
  `041b1dbc95392d7b181d9921696953093ba9e4cee69728709f5ff263daf83d42`
- Freeze JSON SHA-256:
  `b0c34a63e5b9928a074d380c6c17b21fab5e818e4a1cd4d0a24cbf418063e3fa`
- Mutable parameters after freeze: none
- Opened outcome windows: none
- Still sealed: 2021–2022 Stage 1, 2023 Stage 2, and every 2024+ window

## Sequential opening contract

1. Stage 1 may physically parse only `[2021-01-01, 2023-01-01)`.
2. Stage 2 may physically parse only `[2023-01-01, 2024-01-01)` and only after
   a Stage-1 PASS whose self-hash, evaluator source hash, evaluator-freeze
   manifest hash, configuration, gate evidence, and physical-window evidence
   all match the current freeze.
3. No evaluator path may parse 2024 or later.
4. A failed Stage 1 permanently rejects this singleton without opening 2023.
5. A failed Stage 2 rejects it without running portfolio-orthogonality tests.

The parsers stop at the first end-boundary timestamp before decoding its price
or funding values. Full-file hashes are copied from the already-frozen source
manifests; each physically opened stage records its own parsed-line hash and
exact grid diagnostics.

## Frozen schedules

Counts are `Stage 1 / Stage 2 / all pre-2024` after requiring signal, entry, and
exit to remain inside the same physical window.

| Clock | Count |
|---|---:|
| primary | 44 / 28 / 72 |
| direction flip | 44 / 28 / 72 |
| no ticket | 52 / 33 / 85 |
| no taker | 42 / 20 / 62 |
| volume only | 49 / 25 / 74 |
| flow only | 118 / 34 / 153 |
| EUR only | 36 / 28 / 64 |
| TRY only | 41 / 33 / 74 |
| BRL only | 44 / 28 / 72 |
| USDT only | 42 / 22 / 64 |
| reference suppression | 56 / 42 / 99 |
| absolute-book participation | 64 / 28 / 92 |
| one-day signal delay | 44 / 28 / 72 |
| random side | 44 / 28 / 72 |

Every schedule has an independent canonical hash in the freeze JSON. The
primary clock remains fixed long, enters at `d+1 00:05 UTC`, holds 72 hours,
uses 0.5x leverage, and is globally non-overlapping.

## Outcome-free evidence

A Python runtime audit hook around `freeze_evaluator()` observed reads of only
the preregistration/support artifacts, frozen clocks, evaluator/static source
files, and market/funding **manifests**. It observed zero opens of:

- `BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz`
- `binance_um_btcusdt_funding_marks_2020_2023.csv.gz`

The generated freeze also records:

- `opened_windows = []`
- `execution_ohlc_rows_parsed_during_freeze = 0`
- `funding_rows_parsed_during_freeze = 0`
- `simulation_run_during_freeze = false`

## Frozen accounting

- 6 bp/notional/side base cost; 10 bp/notional/side stress cost
- exact funding settlements while the position is open
- favorable-before-adverse held-bar ordering
- global high-water strict MDD including entry fee, hypothetical adverse exit
  fee, realized exit fee, and funding
- full-calendar CAGR including idle periods
- 20,000-draw, fixed-seed weekly-cluster sign-flip test
- absolute return, CAGR, strict MDD, CAGR/MDD, trade count, mean gross move,
  stress-cost, contained-subperiod, and mechanism-control gates exactly as
  preregistered

## Verification

- Focused source/preregistration/support/evaluator tests: `32 passed`
- Evaluator-only tests after the Stage-2 interlock regression: `7 passed`
- Ruff: passed
- Independent code review found one high-severity Stage-2 freeze-binding gap;
  the interlock and regression test were added before this freeze was finalized.

No Stage-1 or Stage-2 performance statistic was calculated while producing this
document.
