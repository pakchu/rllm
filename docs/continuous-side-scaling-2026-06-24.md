# Continuous side-strength scaling (2026-06-24)

## Purpose

Binary side banning was too blunt: strict thresholds fixed 2025Q3 but cut too much aggregate alpha. This pass adds optional continuous side scaling from validation completed-trade mean return.

For each side:

```text
side_scale = clip(validation_side_mean_trade_ret_pct / denominator, 0, 1)
```

If `side_min_val_trades` is set and a side has too few validation trades, its scale is 0.

Implementation:

- `training/event_candidate_ridge_ranker._write_policy(..., side_scale_by_side=...)`
- `training/event_candidate_pairwise_walkforward.py --side-scale-val-mean-ret-pct`

## Protocol

Base protocol: PA-ext, 6M/3M/3M, stats-gated, pair half-life 45d.

## Results

| Side scaling | CAGR | Strict MDD | CAGR/MDD | Trades | p approx | Mean trade |
|---|---:|---:|---:|---:|---:|---:|
| none | 13.26% | 14.10% | 0.94 | 119 | 0.087 | +0.418% |
| binary side mean >= 0 | 14.23% | 14.10% | 1.01 | 115 | 0.076 | +0.461% |
| continuous denominator 0.5 | 15.02% | 14.10% | 1.07 | 119 | 0.066 | +0.468% |
| continuous denominator 1.0 | 12.48% | 12.54% | 0.99 | 119 | 0.084 | +0.391% |
| continuous denominator 1.5 | 9.90% | 10.60% | 0.93 | 119 | 0.108 | +0.312% |

Best current report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidescale_d0p5_2026-06-24/report.json`.

## 2025Q3 note

Continuous scaling reduces 2025Q3 loss only when denominator is stricter:

- denominator 0.5: SHORT scale remains 1.0, fold unchanged at `CAGR -35.5 / MDD 14.1`
- denominator 1.0: SHORT scale 0.55, fold improves to `CAGR -24.3 / MDD 12.1`
- denominator 1.5: LONG scale 0.82, SHORT scale 0.36, fold improves to `CAGR -17.8 / MDD 9.75`

But stricter scaling reduces aggregate alpha more than it helps the bad fold.

## Conclusion

Continuous side scaling gives the current best aggregate (`CAGR 15.02 / MDD 14.10 / ratio 1.07`) but still misses the target by a wide margin. The main remaining problem is not risk sizing; it is detecting or adapting to the 2025Q3-style decay regime before trading it.
