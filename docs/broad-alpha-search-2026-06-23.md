# Broad alpha search checkpoint (2026-06-23)

## Context

The DXY-low / Kimchi prior path produced a target-oracle edge but did not generalize when converted into causal Gemma policy decisions. The next check broadened the search across market, wave-trading external, and Binance futures auxiliary features with rolling half-year evaluation.

## Protocol

Command:

```bash
.venv/bin/python -m training.rolling_alpha_feature_discovery \
  --input-csv data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz \
  --output results/rolling_alpha_broad_wave_aux_2026-06-23.json \
  --wave-trading-root /home/pakchu/workspace/wave_trading \
  --binance-funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --binance-premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz \
  --horizons 36,72,144,288 \
  --quantiles 0.10,0.20,0.30 \
  --min-train-rows 20000 \
  --min-eval-events 120 \
  --top-event-candidates 80 \
  --max-strict-candidates 24 \
  --leverage 0.5

.venv/bin/python -m training.alpha_candidate_gate \
  --input-report results/rolling_alpha_broad_wave_aux_2026-06-23.json \
  --output results/rolling_alpha_broad_wave_aux_gate_2026-06-23.json \
  --min-cagr-to-mdd 3.0 \
  --max-strict-mdd-pct 15.0 \
  --min-fold-trades 30 \
  --min-total-trades 300 \
  --min-positive-folds 5
```

Leakage guard: each fold is evaluated chronologically; the gate requires fold consistency and does not promote candidates from a single good period.

## Result

Gate decision: **NO_GO** (`passed_count=0 / candidate_count=24`).

Top rejected candidates:

| Candidate | Total trades | Positive folds | Passing ratio folds | Worst CAGR | Worst strict MDD | Min CAGR/MDD | Main blocker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wave__mom_144`, horizon 288, q=0.10 | 564 | 5/7 | 1/7 | -29.96% | 18.92% | -1.58 | 2023H1 and 2026H1 fail; MDD cap violated |
| `mkt__funding_rate`, horizon 144, q=0.10 | 337 | 4/7 | 0/7 | -9.84% | 11.48% | -1.21 | too few positive folds; no fold reaches ratio 3 |
| `mkt__funding_rate`, horizon 288, q=0.20 | 753 | 4/7 | 0/7 | -40.30% | 31.83% | -1.27 | large drawdown and negative worst fold |

## Interpretation

The broader univariate scan found episodic edges, especially trend/momentum and funding-related candidates, but not a deployable alpha. The failure mode is not lack of trades; it is fold instability. Candidates that work in 2024-2025 lose sharply in 2023H1 or 2026H1.

This keeps the current conclusion unchanged: single prior/gate optimization is the wrong direction. The next useful branch is a leakage-safe multi-feature model where the LLM can summarize/regularize causal state, but selection must be done on train/test only and eval must remain a holdout.
