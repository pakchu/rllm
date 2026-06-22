# Bounded regime-conditioned alpha scan — 2026-06-23

## Purpose

The repaired Binance aux rolling scan showed funding/premium inputs are real but unstable.  Instead of another blind threshold sweep, this run tested simple regime-conditioned activation rules:

> if `regime_feature` is in train-fitted low/high bucket, then trade a train-fitted quantile rule on `signal_feature`.

All regime thresholds, signal thresholds, and long/short directions are fit on train only. Test ranks candidates; eval is audit only.

## Bounded search

The initial full regime scan was too broad and was stopped because it was spending time on thousands of strict simulations. The scanner now supports `--signal-features` and `--regime-features` allowlists.

Bounded run:

```bash
.venv/bin/python -m training.alpha_regime_rule_scan \
  --input-csv data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz \
  --output results/alpha_regime_rule_scan_binance_aux_bounded_2026-06-23.json \
  --train-start 2023-01-01 \
  --train-end '2024-06-30 23:59:59' \
  --test-start 2024-07-01 \
  --test-end '2025-08-31 23:59:59' \
  --eval-start 2025-09-01 \
  --eval-end '2026-02-28 23:59:59' \
  --horizons 144,288 \
  --signal-features funding_rate,funding_zscore,premium_index_zscore,premium_index_change,kimchi_premium_zscore,kimchi_premium_change,trend_96,window_drawdown \
  --regime-features funding_zscore,premium_index_zscore,kimchi_premium_zscore,kimchi_premium_change,dxy_zscore,dxy_momentum,htf_1w_return_4,window_drawdown \
  --leverage 0.5 \
  --wave-trading-root /home/pakchu/workspace/wave_trading \
  --binance-funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --binance-premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz
```

Gate:

```bash
.venv/bin/python -m training.alpha_candidate_gate \
  --input-report results/alpha_regime_rule_scan_binance_aux_bounded_2026-06-23.json \
  --output results/alpha_candidate_gate_regime_binance_aux_bounded_2026-06-23.json \
  --min-cagr-to-mdd 3.0 \
  --max-strict-mdd-pct 15.0 \
  --min-fold-trades 80 \
  --min-total-trades 200 \
  --min-positive-folds 2
```

## Result

Gate decision: `NO_GO`, passed `0 / 80`.

Best candidates by gate ordering:

1. `dxy_zscore:low -> kimchi_premium_zscore`, horizon 288
   - Test: CAGR 26.79, strict MDD 12.39, ratio 2.16, trades 278.
   - Eval: CAGR 47.26, strict MDD 11.29, ratio 4.19, trades 98.
   - Failure: test ratio below 3.
2. `dxy_zscore:low -> kimchi_premium_zscore`, horizon 144
   - Test: CAGR 21.02, strict MDD 9.74, ratio 2.16, trades 464.
   - Eval: CAGR 50.51, strict MDD 8.28, ratio 6.10, trades 171.
   - Failure: test ratio below 3.
3. `dxy_momentum:low -> premium_index_zscore`, horizon 288
   - Test: CAGR 35.81, strict MDD 17.74, ratio 2.02, trades 257.
   - Eval: CAGR 21.59, strict MDD 11.03, ratio 1.96, trades 91.
   - Failure: test ratio below 3 and test MDD above 15.

## Interpretation

This is the first direction in the reset path that is not purely dead: DXY-low regimes plus Kimchi signal produce positive test and eval with hundreds of trades.  But it still does not meet the user target because test CAGR/MDD is around 2.16, not 3+.

Next work should focus on improving this specific family rather than broad blind search:

- Use DXY-low + Kimchi state as a candidate RLLM textual regime prior.
- Ask the LLM/policy to decide activation/abstention and horizon, not raw direction from a scalar threshold.
- Add longer eval folds through 2026-05 before trusting the 2025-09..2026-02 result.
- Keep funding/premium as context features; do not treat them as standalone alpha yet.
