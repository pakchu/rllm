# Side-specific validation reliability gate (2026-06-24)

## Purpose

The best time-decayed PA-ext ranker still had losing folds driven largely by SHORT trades. This pass adds an optional side-specific validation reliability filter: after q/margin selection, validation completed trades are split by LONG/SHORT; sides that fail a minimum validation sample/mean-return rule are forbidden in the next test fold.

Implementation:

- `training/event_candidate_ridge_ranker._write_policy(..., allowed_sides=...)`
- `training/event_candidate_pairwise_walkforward.py`
  - `--side-min-val-trades`
  - `--side-min-val-mean-ret-pct`

## Protocol

Base protocol: PA-ext, 6M fit / 3M validation / 3M test, stats-gated, pair half-life 45d.

## Results

| Side rule | CAGR | Strict MDD | CAGR/MDD | Trades | p approx | Mean trade |
|---|---:|---:|---:|---:|---:|---:|
| none | 13.26% | 14.10% | 0.94 | 119 | 0.087 | +0.418% |
| min side trades 3, mean >= 0.0% | 14.23% | 14.10% | 1.01 | 115 | 0.076 | +0.461% |
| min side trades 3, mean >= 0.5% | 11.38% | 14.10% | 0.81 | 104 | 0.143 | +0.421% |
| min side trades 3, mean >= 1.0% | 6.84% | 14.11% | 0.48 | 66 | 0.276 | +0.413% |

Best report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidegate_2026-06-24/report.json`.

## 2025Q3 failure note

The strict `mean >= 1.0%` side rule fixes 2025Q3 by allowing LONG only:

- Without strict side rule: `CAGR -35.5 / MDD 14.1 / ratio -2.52`
- With `mean >= 1.0%`: `CAGR 12.0 / MDD 13.0 / ratio 0.92`

But strict side filtering cuts too many profitable folds elsewhere, reducing aggregate CAGR. Binary side banning is therefore too blunt.

## Conclusion

Weak side-specific validation reliability improves the current best result slightly. Strong binary side banning can fix the known bad fold but damages aggregate alpha. Next step should use validation side strength for continuous position scaling instead of hard side inclusion/exclusion.
