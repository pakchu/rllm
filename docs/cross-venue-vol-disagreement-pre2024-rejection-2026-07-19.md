# Cross-venue volatility disagreement pre-2024 rejection — 2026-07-19

## Verdict

Reject this battery before OOS. Fifteen support-passing clocks were evaluated on
the frozen 2023H2 selection window; none passed all preregistered gates. The 2024,
2025, and 2026 sources remain unopened for candidate performance.

## Best frozen result

`dvol_rich_move_follow_v80_p80_h48` was the strongest candidate by
`CAGR / strict MDD`, but it failed the required ratio of 3:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023H2 | 7.83% | 16.14% | 6.54% | 2.47 | 29 |
| 2023Q3 | 4.91% | 20.98% | 6.54% | 3.21 | 15 |
| 2023Q4 | 2.78% | 11.50% | 4.89% | 2.35 | 14 |

Its weekly cluster sign-flip p-value was 0.0867, its 10bp-per-side stress return
was +6.59%, and the frozen direction flip returned -10.86%. These are useful
diagnostics but do not override the failed primary target.

## Family diagnosis

- The **DVOL-rich move-follow** family was directionally coherent but too weak:
  its best risk-adjusted result stopped at 2.47 and only 29 trades.
- The **BVOL-rich move-fade** hypothesis was contradicted. Support-passing cells
  lost between 0.83% and 16.24%, while several direction-flip controls were
  positive. That is falsification evidence, not permission to invert the family.
- The six-month source history is too short to justify repairing quantiles,
  holding periods, or directions after seeing these outcomes.

## Audit contract

- entry is five minutes after both hourly volatility features close;
- all thresholds exclude the current hour;
- performance includes 6bp per side base cost, realized funding, fixed 0.5x
  leverage, full-calendar CAGR, and global-HWM held-path 5m high/low strict MDD;
- missing Binance BVOL archives and incomplete hours are never imputed;
- evaluator logic and source hashes were committed before candidate performance;
- pre-freeze schema validation disclosed only row counts/time bounds/missing
  counts, not price values, funding values, or candidate returns;
- no 2024+ candidate performance was opened.

The next search must use a distinct causal mechanism or source. Reversing the
BVOL fade, loosening the ratio gate, or retuning the 48h hold from this result is
selection-window repair and is prohibited.
