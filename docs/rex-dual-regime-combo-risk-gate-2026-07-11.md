# REX dual-regime combo/risk gate validation (2026-07-11)

## Protocol

- Base signal: fixed `rex_dual_regime` predictions.
- Calendar windows for CAGR include idle periods:
  - train: 2021-01-01 .. 2024-01-01
  - test: 2024-01-01 .. 2025-01-01
  - eval: 2025-01-01 .. 2026-06-01
- Strict MDD is the online strict-bar backtest MDD including intratrade adverse excursion.
- Thresholds were derived from train month-first feature quantiles or prior train-only TTE candidates.
- Selection used train/test only; eval was replayed after selection.

## Artifacts

- Same-filter TTE: `results/rex_dual_regime_month_feature_abstain_tte_same_filter_fullwindow_2026-07-11.json`
- Regime combo gates: `results/rex_dual_regime_regime_combo_gate_fullwindow_2026-07-11.json`
- Additional MDD feature gate: `results/rex_dual_regime_additional_train_mdd_gate_2026-07-11.json`
- Online risk/TP sweep: `results/rex_dual_regime_combo_online_risk_stop_sweep_2026-07-11.json`

## Findings

### Same-filter TTE

Best single same-filter candidates preserved test/eval but train remained weak:

- `usdkrw_zscore <= -1.1786`
  - train: ret 54.56%, CAGR 15.63%, strict MDD 13.25%, R 1.18, N 317
  - test: ret 17.19%, CAGR 17.15%, strict MDD 5.78%, R 2.97, N 61
  - eval: ret 24.57%, CAGR 16.82%, strict MDD 3.72%, R 4.52, N 55

### New regime combinations

Best train/test-balanced combo was an abstain OR gate:

- `block_any__usd_low__h4_drop`
- Abstain when either:
  - `usdkrw_zscore <= -1.1786`
  - `htf_4h_return_4 <= -0.01994`

Stats:

| split | abs ret | CAGR | strict MDD | CAGR/MDD | trades | p |
|---|---:|---:|---:|---:|---:|---:|
| train | 61.45% | 17.32% | 13.25% | 1.31 | 282 | 0.029 |
| test | 20.76% | 20.72% | 4.69% | 4.42 | 60 | 0.023 |
| eval | 24.57% | 16.82% | 3.72% | 4.52 | 55 | 0.00028 |

This improved train R from ~1.18 to 1.31 but did not solve the train MDD bottleneck.

### Train MDD decomposition

The largest train drawdown was concentrated around a January 2021 LONG cluster:

- max close-to-close path drawdown around 11.05% before adverse excursion adjustment.
- all cluster trades were `rex_htf_pullback_reclaim`, hold 144.
- worst sequence: 2021-01-19 .. 2021-01-25 with repeated LONG losses including -4.28% and -2.44% trade returns.

Simple extra month-feature gates did not materially solve this. Best additional feature gate (`volume_zscore >= q0.9` block) only changed:

- train: ret 52.32%, CAGR 15.07%, strict MDD 12.39%, R 1.22, N 218
- test/eval unchanged from base combo.

### Online risk / take-profit overlay

Best train/test robust overlay on top of `block_any__usd_low__h4_drop`:

```json
{
  "pause_after_losses": 2,
  "pause_bars": 144,
  "monthly_loss_stop_pct": 4,
  "trade_stop_loss_pct": 0,
  "trade_take_profit_pct": 3.0,
  "atr_trailing_stop_mult": 0
}
```

Stats:

| split | abs ret | CAGR | strict MDD | CAGR/MDD | trades | p |
|---|---:|---:|---:|---:|---:|---:|
| train | 80.39% | 21.75% | 9.05% | 2.40 | 257 | 0.0058 |
| test | 16.76% | 16.73% | 5.84% | 2.86 | 57 | 0.0409 |
| eval | 21.13% | 14.53% | 3.72% | 3.91 | 53 | 0.0013 |

Alternative with stronger test ratio:

```json
{
  "pause_after_losses": 3,
  "pause_bars": 576,
  "monthly_loss_stop_pct": 0,
  "trade_stop_loss_pct": 0,
  "trade_take_profit_pct": 3.0,
  "atr_trailing_stop_mult": 0
}
```

- train: ret 78.36%, CAGR 21.29%, strict MDD 10.38%, R 2.05, N 265
- test: ret 18.77%, CAGR 18.73%, strict MDD 4.69%, R 4.00, N 59
- eval: ret 23.90%, CAGR 16.38%, strict MDD 3.95%, R 4.15, N 53

## Decision

The raw combo gate is not enough. The first genuinely useful improvement is not another stale regime feature; it is path/risk management:

- cap winners with `take_profit_pct=3.0`, and
- pause after clustered losses.

However, the main top overlay still has test R 2.86, just below the target 3.0. The second overlay has test/eval > 4 but train R only 2.05. Neither fully clears the global objective, but both are materially better than the ungated dual REX.

## Next work

- Search path-aware exits around the January 2021 long cluster instead of more month filters.
- Focus on reducing train strict MDD from 9-10% to <=7% without degrading 2024/2025-2026 too much.
- Candidate directions: dynamic time-to-exit, intratrade adverse stop, side-specific long risk gate, and REX long/short split overlays.

## Intratrade stop sweep addendum

Artifact: `results/rex_dual_regime_intratrade_stop_sweep_2026-07-11.json`

Tested fixed intratrade stop-loss and ATR-trailing families on top of `block_any__usd_low__h4_drop`.

Best train/test minimum-ratio candidate:

```json
{
  "pause_after_losses": 3,
  "pause_bars": 288,
  "monthly_loss_stop_pct": 4,
  "trade_stop_loss_pct": 2.0,
  "trade_take_profit_pct": 3.0,
  "atr_trailing_stop_mult": 0
}
```

| split | abs ret | CAGR | strict MDD | CAGR/MDD | trades | p |
|---|---:|---:|---:|---:|---:|---:|
| train | 71.87% | 19.80% | 9.20% | 2.15 | 274 | 0.013 |
| test | 13.73% | 13.70% | 5.19% | 2.64 | 60 | 0.114 |
| eval | 22.84% | 15.67% | 4.77% | 3.29 | 54 | 0.00058 |

Best train-ratio candidate:

```json
{
  "pause_after_losses": 2,
  "pause_bars": 144,
  "monthly_loss_stop_pct": 4,
  "trade_stop_loss_pct": 3.0,
  "trade_take_profit_pct": 3.0,
  "atr_trailing_stop_mult": 0
}
```

- train: ret 76.27%, CAGR 20.81%, strict MDD 8.91%, R 2.34, N 265
- test: ret 12.62%, CAGR 12.60%, strict MDD 5.92%, R 2.13, N 57
- eval: ret 21.13%, CAGR 14.53%, strict MDD 3.72%, R 3.91, N 53

Best test/eval minimum-ratio candidate:

```json
{
  "pause_after_losses": 3,
  "pause_bars": 576,
  "monthly_loss_stop_pct": 4,
  "trade_stop_loss_pct": 2.0,
  "trade_take_profit_pct": 2.5,
  "atr_trailing_stop_mult": 0
}
```

- train: ret 47.33%, CAGR 13.80%, strict MDD 10.77%, R 1.28, N 265
- test: ret 15.94%, CAGR 15.91%, strict MDD 4.69%, R 3.39, N 60
- eval: ret 23.86%, CAGR 16.35%, strict MDD 3.96%, R 4.13, N 53

### Intratrade stop conclusion

A fixed intratrade stop does reduce train MDD in some cases, but it also clips or reshapes the return distribution enough to damage 2024 test. It did not beat the earlier no-stop online overlay on train/test balance:

- no intratrade SL overlay: train R 2.40, test R 2.86, eval R 3.91
- best fixed SL train/test minimum: train R 2.15, test R 2.64, eval R 3.29

So fixed intratrade stops are not the main unlock. The next likely path is side-specific or state-conditioned exits, especially targeting the January 2021 LONG drawdown cluster without applying the same stop bluntly to all trades.

## Side-specific exit narrow sweep

Artifact: `results/rex_dual_regime_side_specific_exit_narrow_2026-07-11.json`

The full side-specific grid was too slow, so a narrow sweep around the prior best no-stop overlay was run. The focus was whether LONG-specific stops could reduce the January 2021 LONG cluster without damaging 2024 test.

Best narrow side-specific candidate:

```json
{
  "long_sl": 0,
  "long_tp": 2.5,
  "short_sl": 0,
  "short_tp": 3.0,
  "pause_after_losses": 2,
  "pause_bars": 144,
  "monthly_loss_stop_pct": 4
}
```

| split | abs ret | CAGR | strict MDD | CAGR/MDD | trades | p |
|---|---:|---:|---:|---:|---:|---:|
| train | 74.11% | 20.32% | 9.05% | 2.25 | 258 | 0.0092 |
| test | 19.43% | 19.38% | 4.69% | 4.13 | 58 | 0.0174 |
| eval | 21.13% | 14.53% | 3.72% | 3.91 | 53 | 0.0013 |

Comparison to prior symmetric no-SL overlay:

- Prior: train R 2.40, test R 2.86, eval R 3.91.
- Side-specific: train R 2.25, test R 4.13, eval R 3.91.

The best improvement came from asymmetric take-profit (`LONG TP 2.5`, `SHORT TP 3.0`), not from LONG stop-loss. LONG stop-loss candidates generally reduced train return and did not solve the global ratio target.

### Updated interpretation

- Fixed SL remains unattractive.
- Side-specific TP is useful because it raises 2024 test robustness without hurting eval.
- Remaining bottleneck is train R < 3, caused by MDD still around 9%.
- Next likely unlock is not a stop-loss; it is side/state-conditioned entry suppression or dynamic TTE around LONG loss clusters.
