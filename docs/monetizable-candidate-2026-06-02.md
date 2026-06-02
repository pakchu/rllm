# Monetizable candidate search snapshot — 2026-06-02

## Status

No candidate has passed the original production target yet:

- recent test and eval each >= 6 months
- strict OHLC bar-by-bar MDD <= 15%
- CAGR / strict MDD >= 3 on both test and eval
- statistically meaningful trade count and positive trade-return confidence interval
- no eval-period parameter selection

However, the current closest **live-watchlist** set is materially better than the failed static/bucket sweeps and is worth the next validation pass.

## Closest strict candidate

Artifact: `results/h144_trend48_candidate73_lev05_exact_strict.json`

Source candidate: `results/h144_trend48_regime_grid.json` top index `73`, rechecked with exact strict OHLC bar-by-bar execution.

Policy:

```json
{
  "hierarchical": {
    "inverse": false,
    "gate_margin_threshold": 3.0,
    "side_margin_threshold": 3.0,
    "hold_bars": 432,
    "cooldown_bars": 6
  },
  "regime_filter": {
    "name": "tf_trend_48_0p005",
    "abs_trend_min": 0.005,
    "align_mode": "trend_follow",
    "trend_col": "trend_48"
  },
  "execution": {
    "leverage": 0.5,
    "fee_rate": 0.0004,
    "slippage_rate": 0.0001,
    "entry_delay_bars": 1
  }
}
```

Strict exact results:

| Split | Period role | Trades | CAGR | Strict MDD | CAGR/MDD |
| --- | --- | ---: | ---: | ---: | ---: |
| test | selection/validation window | 102 | 25.87% | 13.29% | 1.95 |
| eval | untouched report window | 101 | 45.37% | 11.31% | 4.01 |
| all | test+eval recent year | 203 | 34.96% | 13.29% | 2.63 |

Trade significance is not yet strong enough:

- test mean trade CI95 lower bound: about `-0.138%`, p ~= `0.36`
- eval mean trade CI95 lower bound: about `-0.079%`, p ~= `0.166`
- all mean trade CI95 lower bound: about `-0.031%`, p ~= `0.102`

## Interpretation

This is **not production-qualified** under the original target, because test ratio is below 3 and the all-period confidence interval still crosses zero.  But it is the first strict recent-year candidate in this pass with:

- both 6-month windows positive,
- both windows over 100 trades,
- strict MDD under 15%,
- untouched eval CAGR/MDD above 4,
- exact OHLC bar-by-bar execution rather than forward-return accounting.

## Why this set is promising

The edge is not coming from high leverage or an eval-only lucky overlay.  It is a conservative trend-following agreement rule:

1. analyzer gate margin must be meaningfully positive (`TRADE - NO_TRADE >= 3`),
2. trader side confidence must be meaningful (`abs(LONG - SHORT) >= 3`),
3. the trade direction must align with a past-only 48-bar trend regime,
4. leverage is cut to `0.5`, which keeps strict MDD below 15%.

This supports the current thesis: the LLM should act as a **selective regime/edge filter**, while execution risk is handled conservatively.

## Next required validation

1. Rebuild this exact candidate as a first-class reproducible search artifact instead of relying on temp scripts.
2. Extend the same strict recheck to the full top-80 candidate set and rank by train/test only, with eval report-only.
3. Add a 3-year train/test/eval equivalent for this policy family; the old 3-year h144 forward-return candidate failed exact strict validation, so recent-year success is insufficient.
4. If the candidate survives, export it as a paper-trading candidate only, not live production.
