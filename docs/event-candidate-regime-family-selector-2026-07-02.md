# Regime-conditioned candidate-family selector (2026-07-02)

## Intent

After the zero-strength candidate bug was fixed, individual REX/price-action families still showed validation-to-eval instability.  This pass adds a fold-safe selector that treats each family as a weak expert and chooses one family per target fold from information available before that fold.

## Implementation

New script: `training/event_candidate_regime_family_selector.py`

Leakage guards:

- family thresholds are fit only on rows before each target fold;
- first fold uses pre-fold train prior only;
- later folds use previous fold outcomes plus regime similarity measured from pre-fold lookback windows;
- target fold outcomes are recorded only after selection;
- zero-strength candidate filtering from `event_candidate_pool_probe.py` is inherited.

## Smoke evaluation

Command shape:

```bash
.venv/bin/python -m training.event_candidate_regime_family_selector \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/event_candidate_regime_family_selector_rex_core_6m_2023_2026h1_2026-07-02.json \
  --train-start 2020-01-01 --eval-start 2023-01-01 --eval-end 2026-06-01 \
  --fold-months 6 --hold-bars 288 --stride-bars 24 --quantile 0.80 \
  --min-train-trades 80 --min-fold-trades 20 --memory-folds 3 \
  --family-include rex_extreme_breakout_follow,rex_compression_breakout,rex_compression_fakeout,rex_htf_pullback_resume,rex_htf_pullback_reclaim,rex_htf_deep_pullback_resume,rex_htf_context_pullback_resume,rex_htf_long_pullback_resume,rex_htf_short_pullback_resume,rex_multiscale_location_revert
```

Report: `results/event_candidate_regime_family_selector_rex_core_6m_2023_2026h1_2026-07-02.json`

Final stitched replay, 2023-01-01..2026-06-01:

| CAGR | strict MDD | CAGR/MDD | trades | p-value |
| ---: | ---: | ---: | ---: | ---: |
| -0.28% | 19.35% | -0.01 | 250 | 0.954 |

Fold highlights:

| fold | selected family | CAGR/MDD | trades | note |
| --- | --- | ---: | ---: | --- |
| 2023H1 | `rex_htf_deep_pullback_resume` | -1.85 | 22 | bad first prior |
| 2023H2 | `rex_htf_context_pullback_resume` | 0.10 | 34 | flat |
| 2024H1 | `rex_htf_context_pullback_resume` | 4.48 | 27 | strong but sparse |
| 2024H2 | `rex_htf_context_pullback_resume` | 3.70 | 31 | strong but sparse |
| 2025H1 | `rex_multiscale_location_revert` | -1.81 | 87 | major regime miss |
| 2025H2 | `rex_htf_context_pullback_resume` | -0.76 | 21 | weak |
| 2026H1 | `rex_htf_pullback_reclaim` | 2.04 | 28 | positive but sparse |

## Interpretation

This selector did not solve the target objective.  It is still useful because it localizes the failure:

1. The REX core has real pockets in 2024, but not enough stability through 2025.
2. The largest damage comes from choosing `rex_multiscale_location_revert` in 2025H1; the selector needs a drawdown/market-phase veto before using location-reversion experts.
3. Trade counts per fold are often too low for statistical confidence, so the next pool should include more independent setup families or shorter holding horizons.
4. This supports the current architectural direction: LLM should not directly choose raw trades yet.  It should first reason over compact state cards that describe family validity/veto conditions, then a transparent selector/backtester verifies them fold-by-fold.

## Next action

Add family-level diagnostics to identify when the selector should abstain or avoid a family:

- previous-fold sign flip / anti-persistence warning;
- pre-fold drawdown and high-volatility veto for reversion families;
- minimum expected trade count per fold;
- shorter horizon sweep for REX families to improve sample size.
