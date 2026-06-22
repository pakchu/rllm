# Rolling alpha with repaired Binance aux — 2026-06-23

## Setup

After restoring real BTCUSDT funding and premium-index inputs, reran rolling prior-only alpha discovery over the full 2020-2026 BTCUSDT 5m dataset with wave_trading DXY/Kimchi/USDKRW features enabled.

Command:

```bash
.venv/bin/python -m training.rolling_alpha_feature_discovery \
  --input-csv data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz \
  --output results/rolling_alpha_feature_discovery_binance_aux_2026-06-23.json \
  --wave-trading-root /home/pakchu/workspace/wave_trading \
  --external-tolerance 30min \
  --binance-funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --binance-premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz \
  --horizons 36,72,144,288 \
  --quantiles 0.10,0.20,0.30 \
  --min-eval-events 120 \
  --top-event-candidates 60 \
  --leverage 0.5 \
  --max-strict-candidates 18
```

Gate command:

```bash
.venv/bin/python -m training.alpha_candidate_gate \
  --input-report results/rolling_alpha_feature_discovery_binance_aux_2026-06-23.json \
  --output results/alpha_candidate_gate_rolling_binance_aux_2026-06-23.json \
  --min-cagr-to-mdd 3.0 \
  --max-strict-mdd-pct 15.0 \
  --min-fold-trades 30 \
  --min-total-trades 300 \
  --min-positive-folds 5
```

## Result

Gate decision: `NO_GO`.

- Passed candidates: `0 / 18`.
- Best strict-score candidate: `wave__mom_144`, horizon 288, q 0.1.
  - Positive folds: 5/7, total trades 564.
  - Failed because only 1 fold reached CAGR/MDD >= 3, some folds breached MDD 15, and worst fold CAGR was negative.
  - Broken folds: 2023H1 CAGR -29.96 / MDD 18.92, 2026H1 CAGR -23.67 / MDD 16.42.
- Best repaired derivative candidate: `mkt__funding_rate`, horizon 288, q 0.2.
  - Strong early folds: 2023H1 CAGR 33.52 / MDD 7.97; 2023H2 CAGR 35.97 / MDD 4.46.
  - Failed badly in 2024H1: CAGR -40.30 / MDD 31.83.
  - Also failed 2026H1: CAGR -12.20 / MDD 15.59.
- `premium_index_zscore` strict smoke test was also negative on 2024H2:
  - CAGR -57.98, strict MDD 36.88, trades 594.

## Interpretation

The restored Binance aux inputs fixed a real data-quality issue, but did not by itself produce a stable alpha. Funding appears regime-local: useful around 2023, destructive around 2024H1 and 2026H1 under frozen quantile rules.

Next direction should not be another blind gate/threshold optimization pass. More promising paths:

1. Regime-conditioned feature use: detect when funding premium is predictive vs contrarian before activating it.
2. Multi-feature interaction search: funding by itself is unstable; combine with macro/Kimchi/HTF trend context and require out-of-fold stability.
3. RLLM framing: use the LLM as a compact regime/rationale policy over causal textual state, but train only after a non-oracle candidate passes a weaker statistical prefilter.
4. Add real BTC open-interest history or remove OI columns everywhere; current OI remains unusable.
