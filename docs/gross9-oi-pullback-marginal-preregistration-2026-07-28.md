# G9-OIDP-1 preregistration: fixed OI-pullback marginal audit

## Why this is the next practical candidate

`oi_divergence_pullback_range_rsi_h96_s6` is not a newly tuned alpha. Its four
thresholds, long side, 8-hour hold, and 30-minute clock were frozen on train
data on 2026-07-05.

It was omitted from the later Gross9 candidate universe, even though its fixed
standalone record is:

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| train through 2023 | 42.71% | 11.26% | 44.29% | 0.25 | 313 |
| 2024 | 52.54% | 52.41% | 6.34% | 8.27 | 64 |
| 2025 | 36.60% | 36.63% | 5.46% | 6.71 | 40 |
| 2026 through June | 0.62% | 1.50% | 9.65% | 0.16 | 17 |

The completed-bar July replay then produced +2.90% absolute return over four
long trades while the promoted portfolio's July trades were entirely short.
That recent result is too small and already inspected, so it can only veto. It
cannot select a weight.

## Frozen test

- Keep all Gross9 weights unchanged.
- Add only the exact fixed OI-pullback sleeve.
- Search candidate weights `{0.25, 0.50, 0.75, 1.00}`.
- Rank using train and 2024 only.
- Use the same shared 5-minute BTC upper-before-lower accounting as Gross9.
- Require both pre-2025 ratio deltas to be positive, beat a same-gross
  pro-rata Gross9 leverage control, reduce MDD in at least one selection
  window, and keep entry Jaccard at most 0.25 versus every existing sleeve.
- Open only the frozen top row on 2025/2026 and the fixed July diagnostic.

No signal threshold, feature, hold, stride, side, exit, or selector may change.

## Claim boundary

This is a missed-candidate marginal audit, not fresh alpha discovery. All
calendar outcomes are research-exposed. A passing row remains forward-shadow
until new live evidence accumulates.

The exact machine contract is
`results/gross9_oi_pullback_marginal_preregistration_2026-07-28.json`.
