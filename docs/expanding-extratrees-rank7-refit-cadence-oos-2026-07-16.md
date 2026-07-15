# Frozen rank-7 annual vs monthly refit cadence OOS

Cadence manifest: `627441e5a7a3bd070e136e771f7dcc93cea6162565c0dd2226c2140c5c836f21`

Pre-2025 selected cadence: **annual**. Future metrics below compare both frozen alternatives but do not rerank that choice.

## Results

| Cadence | Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean net | Win rate | Future/full pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| annual | eval_2025 | 16.3620% | 16.3740% | 4.9844% | 3.2850 | 21 | 73.15 bps | 71.43% | True/True |
| annual | holdout_2026h1 | 7.3132% | 18.4835% | 4.3007% | 4.2978 | 12 | 59.61 bps | 83.33% | True/True |
| annual | future_2025_2026h1 | 24.8717% | 16.9903% | 4.9844% | 3.4087 | 33 | 68.23 bps | 75.76% | True/True |
| annual | all_2023_2026h1 | 64.0433% | 15.5877% | 4.9844% | 3.1273 | 74 | 67.64 bps | 78.38% | True/True |
| monthly | eval_2025 | 8.1905% | 8.1963% | 5.2179% | 1.5708 | 19 | 42.42 bps | 63.16% | False/False |
| monthly | holdout_2026h1 | 7.3238% | 18.5117% | 4.3007% | 4.3043 | 14 | 51.22 bps | 78.57% | False/False |
| monthly | future_2025_2026h1 | 16.1141% | 11.1322% | 5.2179% | 2.1335 | 33 | 46.15 bps | 69.70% | False/False |
| monthly | all_2023_2026h1 | 43.7043% | 11.1953% | 5.2179% | 2.1456 | 79 | 46.61 bps | 69.62% | False/False |

## Future delta: monthly minus annual

- Absolute return: `-8.7576` percentage points
- CAGR: `-5.8581` percentage points
- Strict MDD: `+0.2334` percentage points
- CAGR/MDD: `-1.2752`
- Trades: `+0`
- Mean net/trade: `-22.07` bps

## Mechanism diagnostics

- Shared future trades: `28` at `55.98` mean net bps.
- Annual-only future trades: `5` at `136.80` mean net bps.
- Monthly-only future trades: `5` at `-8.89` mean net bps.
- Annual folds: minimum effective-fit fraction `0.662`, maximum sample-weight multiple `5.48`.
- Monthly folds: minimum effective-fit fraction `0.165`, maximum sample-weight multiple `38.88`.

## Integrity

- The cadence manifest and selection-result hashes are pinned in code.
- Annual and monthly pre-2025 schedules reproduce their committed prefixes.
- The annual cadence reproduces the prior frozen rank-7 OOS schedules and metrics exactly.
- Every future monthly/annual fold purges labels whose exits reach its cutoff.
