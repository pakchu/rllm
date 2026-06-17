# Alpha edge investigation — 2026-06-17

## Protocol
- Data: `data/2023-01-01_2026-02-28_d2a88c0700504d6a5e15bc3839ad84b6.csv.gz` plus leak-safe backward-asof external joins from `/home/pakchu/workspace/wave_trading`.
- Strict execution: entry delay 1 bar, costs, non-overlapping holds, bar-by-bar intrabar adverse excursion included in strict MDD.
- Split discipline for combo scan:
  - train: 2023-01-01 .. 2024-06-30
  - test/ranking: 2024-07-01 .. 2025-08-31
  - eval/holdout: 2025-09-01 .. 2026-02-28
- No eval tuning: model weights, score thresholds, and long/short direction are fit from train only; test ranks candidates; eval is final audit.

## Univariate Kimchi premium strict backtests
Best univariate result was `kimchi_premium_change h288 q0.20`:
- eval CAGR 14.42%, strict MDD 6.88%, ratio 2.10
- 66 trades, mean trade +0.107%, p≈0.454, CI includes 0

Other h72/h144 variants were negative. Conclusion: Kimchi premium has visible IC, but as a standalone strict trading rule it is not statistically meaningful.

## Linear feature-combination scan
Candidate groups: external, kimchi-only, trend, range/reversion, candle/flow, funding/OI, and combinations. Ridge L2 values tested: 10, 100, 1000.

Most stable-but-weak candidates:
- `kimchi_plus_trend h288 q0.15 L2=100`: test 15.78/16.14=0.98, eval 24.05/13.11=1.83, 370/147 trades, p≈0.408/0.379.
- `range_reversion h288 q0.20 L2=1000`: test 14.00/22.08=0.63, eval 14.09/14.78=0.95, 406/171 trades, p≈0.447/0.601.

Important rejection:
- `trend h288 q0.10 L2=1000` had eval ratio 3.16, but test ratio only 0.23. This is not a valid success because the test split does not support selecting it.

## Current conclusion
The currently available feature families do not yet contain a robust, statistically meaningful alpha satisfying CAGR/strict-MDD ≥ 3 under train/test/eval discipline. The useful signal is weak and concentrated around 2-day horizon trend/reversion + Kimchi context, but it is insufficient as a direct policy.

## Next direction
Move from linear/global rules to regime-aware interaction discovery:
1. Detect regimes from past-only volatility/range/trend/Kimchi/DXY states.
2. Fit simple rules inside regimes, not globally.
3. Require regime candidates to pass train and test before eval is inspected.
4. Feed only robust regime descriptors into Gemma-based LLM policy; do not ask the LLM to infer raw numeric edge from weak raw features.

## Follow-up: regime-conditioned candidate audit

Candidate discovered by sensitivity scan:
- Regime: `kimchi_premium_change` in train-window low bucket.
- Signal: `trades_ratio` quantile rule.
- Horizon: 288 bars.
- Fit from 2023-01-01..2024-06-30 with rq=0.25/sq=0.25:
  - test 2024-07..2025-08: CAGR 60.70%, strict MDD 8.12%, ratio 7.47, 280 trades, p≈0.004.
  - eval 2025-09..2025-12-01 effective: CAGR 40.60%, strict MDD 11.55%, ratio 3.52, 61 trades, p≈0.062.

External data caveat:
- wave_trading Kimchi/DXY caches end in early/mid December 2025 while the market file extends to 2026-02-27.
- The apparent 2026 eval interval produced no 2026 trades for this candidate; effective OOS trading ended on 2025-12-02.

Longer split audit:
- Fit 2020..2022, test 2023..2024, eval 2025:
  - test failed: CAGR -3.22%, strict MDD 38.04%, 478 trades, p≈0.936.
  - eval 2025 strong: CAGR 52.56%, strict MDD 11.55%, 217 trades, p≈0.013.
- Fit 2020..2023, test 2024, eval 2025:
  - test weak: CAGR 23.29%, strict MDD 18.41%, ratio 1.27, 243 trades, p≈0.267.
  - eval 2025 strong: CAGR 50.03%, strict MDD 11.81%, ratio 4.24, 217 trades, p≈0.017.

Interpretation:
- The candidate is not a timeless alpha. It appears to be a strong 2025 regime-specific alpha.
- It should not be deployed as an always-on rule.
- Next LLM/RL direction: train Gemma to identify when the 2025-like Kimchi-flow regime is active and abstain otherwise, rather than directly predicting every trade from raw numeric bars.
