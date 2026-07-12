# Alpha search: long minimal funding/premium squeeze (2026-07-12)

## Context

The previous REX-focused work improved 2024/2025-2026 behavior but still left train CAGR/MDD below target.  I therefore searched outside REX for standalone alpha candidates using the existing fixed alpha pool and state-transition scans.

## Failed branches

### State-transition alpha

Artifact: `results/state_transition_alpha_scan_2026-07-12.json`

- Candidate thresholds are train-only.
- Ranking was test2024-only; eval2025 and ytd2026 were replay-only.
- Result: 4,375 test candidates, 0 alpha-pool qualifiers, 0 live-grade qualifiers.
- The best test2024 patterns were mostly `sell_absorption_reversal`, but they collapsed in train/eval/ytd, indicating test overfit.

Train-first rerun artifact: `results/state_transition_alpha_trainfirst_2026-07-12.json`

- 13 train-positive candidates survived the train filter.
- 0 passed test/eval robustness.
- Conclusion: reject current state-transition family as standalone alpha.

### Calendar/OI/funding standalone scan

Artifact: `results/calendar_oi_funding_alpha_scan_2026-07-10.json`

- Several 2024/2025/YTD-looking candidates existed, but most had negative or absent train evidence.
- Do not use those rows directly as alpha without train extension.

## Candidate found: `long_minimal_funding_premium`

Source definition: `training/portfolio_opt_new_alpha_pool.py`

```python
ALPHAS["long_minimal_funding_premium"] = {
    "side": "long",
    "components": ["funding10_trend70", "premium20_mom90"],
    "hold": 576,
    "family": "long_squeeze",
}
```

Component rules:

- `funding10_trend70`:
  - `funding_rate <= -0.0000167`
  - `trend_96 >= 0.007485218212390219`
- `premium20_mom90`:
  - `premium_index_change <= -0.00023471`
  - `htf_1d_return_4 >= 0.0940403008961932`

Interpretation:

- Long squeeze / continuation candidate.
- Trades when funding/premium stress is depressed while trend/momentum is strong.
- This is not REX; it is a funding/premium/trend alpha.

## Individual fixed-rule validation

Artifact: `results/new_alpha_pool_train_extended_2026-07-12.json`

Unit weight 1.0, fixed thresholds, full calendar windows:

| split | abs ret | CAGR | strict MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|---:|
| train 2020-2023 | 140.66% | 24.55% | 15.32% | 1.60 | 206 | 0.58 |
| test 2024 | 31.04% | 30.97% | 5.86% | 5.28 | 29 | 0.79 |
| eval 2025 | 18.33% | 18.34% | 4.97% | 3.69 | 26 | 0.58 |
| ytd 2026 | 12.12% | 31.65% | 4.55% | 6.95 | 29 | 0.66 |

This is the best standalone candidate found in this pass because all periods are positive and test/eval/ytd all clear CAGR/MDD > 3.  The train period remains the bottleneck.

## Exit sweep

Artifact: `results/long_minimal_funding_premium_exit_sweep_fast_2026-07-12.json`

Tested hold/TP/SL/stride around the default 576-bar hold.

Best by train/test/eval minimum ratio was still the original rule:

```json
{
  "hold": 576,
  "tp": null,
  "sl": null,
  "stride": 12
}
```

| split | abs ret | CAGR | strict MDD | CAGR/MDD | trades | win |
|---|---:|---:|---:|---:|---:|---:|
| train | 140.66% | 24.55% | 15.32% | 1.60 | 206 | 0.58 |
| test2024 | 31.04% | 30.97% | 5.86% | 5.28 | 29 | 0.79 |
| eval2025 | 18.33% | 18.34% | 4.97% | 3.69 | 26 | 0.58 |
| ytd2026 | 12.12% | 31.65% | 4.55% | 6.95 | 29 | 0.66 |

TP/SL variants generally either raised train MDD, reduced eval, or hurt YTD.  A 10% TP improved eval slightly but worsened train MDD:

- hold 576, TP 10%, no SL, stride 12:
  - train: 151.33% abs, CAGR 25.91%, MDD 16.87%, R 1.54
  - test2024: 31.78% abs, CAGR 31.70%, MDD 5.86%, R 5.41
  - eval2025: 21.06% abs, CAGR 21.07%, MDD 4.91%, R 4.30
  - ytd2026: 12.12% abs, CAGR 31.65%, MDD 4.55%, R 6.95

## Decision

Promote `long_minimal_funding_premium` to the active alpha-candidate list, but not yet to full live standalone alpha.

Why it matters:

- Positive train/test/eval/ytd.
- Non-REX, funding/premium/trend mechanism.
- Test/eval/ytd ratios are strong.
- Trade count is reasonable for this event family.

Why it is not enough:

- Train strict MDD is 15.32%, just over the target MDD threshold.
- Train CAGR/MDD is 1.60, below the global objective.

Next work:

1. Decompose train drawdown months for this alpha.
2. Test regime filters that reduce train MDD without destroying 2024/2025/2026.
3. Combine with REX side-specific candidate only after standalone robustness is improved.

## Logical flaw audit before pool insertion (2026-07-12)

Verdict: no fatal future-reference flaw found in the feature construction, but promotion stays `candidate` because threshold provenance is not fresh-unseen and train-window risk is below target.

- `funding_rate`: loaded through `preprocessing/binance_aux_features.py::_merge_aux` with `pd.merge_asof(..., direction="backward")`; live usage must use only published funding history, not next scheduled funding estimates.
- `premium_index` / `premium_index_change`: premium klines use `close_time` as the availability timestamp, then backward-asof merge; `premium_index_change` is `premium.diff(96)`, so it is past-only if the source rows are causal.
- `trend_96`: computed by `_return_over(close, 95)` using `close.shift(95)`, so it uses current/past bars only.
- `htf_1d_return_4`: higher-timeframe bars are resampled, shifted by one completed HTF candle, and backward-asof aligned; the daily candle containing the current row is deliberately excluded.
- Execution caveat: the alpha is non-overlapping event logic (`hold=576`, `stride=12`). Live conversion must enforce the same no-overlap/next-allowed behavior and must not fire repeatedly every 5m while the condition remains true.
- Statistical caveat: train 2020-2023 is profitable but below target (`CAGR/strict_MDD=1.60`, strict MDD `15.32%`), while 2024/2025/2026H1 pass. Treat it as a long-regime candidate/feature, not live-grade proof.
