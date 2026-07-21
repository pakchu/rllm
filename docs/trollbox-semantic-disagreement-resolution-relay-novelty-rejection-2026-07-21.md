# TSDR-72 novelty result — rejected before economic evaluation

## Decision

TSDR-72 is retired. It passed source support but failed the frozen
execution-novelty gate against TBASR-24. No TSDR return, funding cash flow, PnL,
equity path, CAGR, MDD, or strict simulation was computed.

- novelty artifact:
  `results/trollbox_semantic_disagreement_resolution_novelty_2026-07-21.json`;
- artifact SHA-256:
  `202cfdc1b06a6352eb146f8f95b56dab3fa6844055a1dc6b276794214ab2fc79`;
- result hash:
  `a6525490b10e6d30fb101354451a34dfb839b90646edcc7ff9470d5078a65122`;
- pure clock:
  `results/tsdr_pure_clocks_2026-07-21.csv.gz`;
- pure-clock SHA-256:
  `cf9b875b2fa7674131e09f3b3a3fc5d9fef0c50193be45222b6955a24ff3f62e`;
- pure-clock rows: 238 TSDR plus 577 TBASR; and
- failed check: `tbasr_tolerant_coverage_at_most_0_35`.

## Full-window comparator result

The first implementation compared only the common train prefix. Independent
verification correctly rejected that narrower denominator. Before publishing
an artifact, the implementation was repaired to reconstruct the unchanged
TBASR policy over both TSDR research splits and to compare all 238 TSDR clocks.

| Metric | Result | Frozen limit |
|---|---:|---:|
| exact entry matches | 43 | diagnostic |
| exact entry Jaccard | 5.57% | <= 20% |
| maximum one-to-one matches within +/-6h | 96 | diagnostic |
| TSDR tolerant match coverage | **40.34%** | **<= 35%** |
| signed occupied-exposure correlation | -0.0736 | abs <= 0.40 |
| position-bar Jaccard | 6.68% | diagnostic |

The semantic path itself was structurally new: there were zero exact
`(onset_end, resolution_end, side)` matches against TBASR's singleton semantic
events. That is insufficient. More than two-fifths of TSDR executions still
occurred within six hours of the retired TBASR trade clock, so it is not a
sufficiently independent execution alpha under the preregistered rule.

All three live-sleeve comparisons passed their exact-entry, tolerant-match, and
exposure-correlation limits. The sole failure was the related semantic
strategy, which is the comparator most capable of detecting a renamed TBASR
repair.

## Boundary accounting

Reconstructing the frozen TBASR comparator decoded 315,648 five-minute market
rows through 2022, including 105,120 rows from 2022. They were used only to
recreate TBASR's causal one-hour displacement feature and trade clock under its
already frozen code.

```text
funding rows loaded                 = 0
performance artifacts parsed       = 0
return/PnL fields read              = 0
strict simulation calls            = 0
TBASR test economic report parsed  = 0
2023-or-later market rows loaded    = 0
post-2022 semantic rows loaded      = 0
economic outcomes computed         = false
```

The 2022 causal-price access is disclosed because it is still market-data
access even though no post-entry outcome was calculated. TSDR was already
fully frozen before that access and is now retired, so none of those rows may
be used to alter its side, deadline, hold, disagreement threshold, or overlap
limit.

## No-repair rule

The following are prohibited:

- relaxing 35% to 41% or changing the +/-6h tolerance;
- using exact Jaccard or low exposure correlation to override the failed gate;
- dropping TBASR, shortening its coverage, or returning to the train-only
  denominator;
- changing follow-resolution to fade-resolution;
- choosing 2h, 12h, or 24h after observing this result;
- reclassifying the same Trollbox messages with another model to escape the
  comparator; and
- opening a TSDR economic backtest.

## Next independent search

The next candidate must leave the Trollbox semantic source. The preserved
outcome-blind design inventory ranks **LVRT-72 — Liquidity Vacuum Replenishment
Transition** next: a rare aggTrade microstructure state transition from a
fragmented, bursty one-sided liquidity vacuum to replenishment and flow
reversal. It must be independently preregistered and must prove novelty against
MFIC/BAFR/AFCS before any market outcome is evaluated. TSDR thresholds or
results may not inform LVRT parameters.
