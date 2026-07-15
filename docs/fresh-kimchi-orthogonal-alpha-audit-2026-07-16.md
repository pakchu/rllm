# Fresh Kimchi/FX orthogonal alpha audit

This audit pins the historical bidirectional policy and changes only one logically required rule: signals that depend on FX/Kimchi data fail closed when the current source bar is unavailable. No threshold was re-ranked on 2025/2026.

## Candidate performance (realized funding included)

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit_2020_2023 | 11.1641% | 2.6812% | 20.1684% | 0.1329 | 152 | 45/107 |
| fit_2023 | 2.2709% | 2.2725% | 4.9030% | 0.4635 | 12 | 8/4 |
| selection_2024 | 10.6160% | 10.5931% | 3.7215% | 2.8465 | 30 | 7/23 |
| eval_2025 | 11.9212% | 11.9298% | 5.0086% | 2.3819 | 17 | 9/8 |
| holdout_2026h1 | 9.5597% | 24.5315% | 5.5692% | 4.4049 | 28 | 20/8 |
| future_2025_2026h1 | 22.6205% | 15.4963% | 5.5692% | 2.7825 | 45 | 29/16 |
| oos_2024_2026h1 | 35.6379% | 13.4381% | 5.5692% | 2.4129 | 75 | 36/39 |
| all_2023_2026h1 | 38.7181% | 10.0520% | 7.7951% | 1.2895 | 87 | 44/43 |

## Trade independence from frozen rank-7

| Window | Exact entry Jaccard | Candidate entries within 6h | Position Jaccard | Candidate position overlap | Daily PnL Pearson | Spearman | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| selection_2023_2024 | 0.0000 | 0.1667 | 0.0730 | 0.1586 | 0.2538 | 0.0809 | True |
| future_2025_2026h1 | 0.0000 | 0.2222 | 0.1300 | 0.2967 | 0.2109 | 0.1646 | True |
| all_2023_2026h1 | 0.0000 | 0.1954 | 0.1017 | 0.2263 | 0.2289 | 0.1226 | True |

## Integrity and interpretation

- Stale signal rows blocked: long `484`, short `0`; post-gate availability violations are zero.
- Entry is at the next 5-minute open; TP 4%, SL 2.5%, maximum hold 288 bars, 0.5x, 6 bp/notional/side, realized funding, and split-contained exits.
- Strict MDD includes intratrade adverse OHLC excursion and funding debit.
- Full-calendar windows are used for every CAGR, including periods with no position.
- Exact-entry Jaccard alone is not accepted as independence evidence; the audit also checks near entries, occupied bars, and zero-filled daily marked PnL.
- 2025/2026 are not pristine discovery OOS because this family was viewed previously. The freshness repair was not tuned on those windows, but live promotion still requires new forward shadow data.

## Verdict

Low-correlation regime-sleeve candidate: trade independence passes the declared audit limits, but combined 2025-2026H1 CAGR/strict-MDD is 2.7825. Keep frozen in forward shadow; do not replace rank-7 or enable live until a new untouched window confirms it.
