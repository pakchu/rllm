# Event candidate family scoreboard export (2026-07-02)

## Why

The next LLM/RLLM step should not ask Gemma to choose from raw OHLC numbers.  It needs compact state cards: which family candidates were available before the fold, what their prior evidence looked like, and whether the selector abstained.  The previous selector report only exposed the chosen family, so it was insufficient for listwise/ranking SFT.

## Change

`training/event_candidate_regime_family_selector.py` now emits `pre_fold_scoreboard` for each fold:

- family name;
- pre-fold score;
- threshold;
- evidence rows used to calculate the score;
- evidence metrics from pre-fold train or previous folds.

This remains no-leak because the scoreboard is computed before target fold metrics are used for reporting.

## Verification

Command:

```bash
.venv/bin/python -m training.event_candidate_regime_family_selector \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --output results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_6m_2023_2026h1_2026-07-02.json \
  --train-start 2020-01-01 --eval-start 2023-01-01 --eval-end 2026-06-01 \
  --fold-months 6 --hold-bars 288 --stride-bars 24 --quantile 0.80 \
  --min-train-trades 80 --min-fold-trades 20 --memory-folds 3 --min-selection-score 0.75 \
  --family-include rex_extreme_breakout_follow,rex_compression_breakout,rex_compression_fakeout,rex_htf_pullback_resume,rex_htf_pullback_reclaim,rex_htf_deep_pullback_resume,rex_htf_context_pullback_resume,rex_htf_long_pullback_resume,rex_htf_short_pullback_resume,rex_multiscale_location_revert
```

Result is unchanged from the abstention replay:

| CAGR | strict MDD | CAGR/MDD | trades | p-value |
| ---: | ---: | ---: | ---: | ---: |
| 4.75% | 14.76% | 0.32 | 217 | 0.366 |

## Next action

Build JSONL state-card records from this scoreboard:

- one prompt per target fold;
- options are top pre-fold families plus `ABSTAIN`;
- target for historical training can be best target-fold diagnostic family, but eval split must be held out chronologically;
- model should explain family validity/veto, not raw trade direction.
