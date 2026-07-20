# DLPD-12 source-support preregistration — 2026-07-20

## Boundary

This freezes one **Dominance Leverage Polarity Decomposition** source clock
before production event incidence or any BTC outcome is opened.  The only
loaded predictor artifact is the checksummed hourly premium-close panel frozen
in `docs/binance-btcdom-premium-decomposition-source-audit-2026-07-20.md`.

No BTC execution price, return, funding cash flow, label, PnL, equity, CAGR,
MDD, post-2023 predictor row, or real DLPD event count is authorized by this
document.

## Singleton

- Normalize each current `BTCUSDT` and `BTCDOMUSDT` premium close against a
  strictly prior 720-hour rolling median and `IQR / 1.349` scale.
- Require at least 672 valid observations in each rolling window.
- Activate only when both absolute z-scores are at least `1.0` and their signs
  disagree.
- Side is the sign of the BTC premium z-score.
- Signal on false-to-true onset only.
- Decide after the completed UTC hour, enter at the next `+5m` open, hold
  exactly twelve hours, and enforce non-overlap and calendar containment.

There is no smoothing, threshold, direction, hold, latency, stop, take-profit,
price, regime, model, or LLM search after this freeze.  The earlier bounded
source-count probe is disclosed in the source-decision document and is the only
incidence-based choice that occurred.

## Source controls

- `btc_only_tail`: absolute BTC premium tail without dominance confirmation;
- `dom_only_mirror`: absolute dominance premium tail, opposite mapped side;
- `same_sign`: both tails point in the same direction;
- `stale_btc_1h`: previous-hour BTC z-score with current dominance z-score;
- `stale_dom_1h`: current BTC z-score with previous-hour dominance z-score.

Controls are falsification evidence only.  They cannot replace or repair the
primary policy after any outcome is opened.

## Source-only support gates

Both 2022 and 2023 must independently have:

- at least 120 non-overlapping events;
- at least 25% of events on each side;
- at least 20 events in each quarter; and
- no month containing more than 20% of annual events.

The 2023 primary entry clock must additionally remain novel against the frozen
primary clocks of PSR-30/6, PCBR-12, OPDR-24, CLD-72 and FCIR-12:

- exact Jaccard at most `0.10`; and
- maximum bidirectional event containment within one hour at most `0.35`.

All gates must pass.  Failure retires DLPD-12 before market outcomes, with no
threshold, clock, side, comparator, or support-gate repair.

## Conditional outcome sequence

Only a source-support pass permits a separately committed evaluator.  The
sequential windows are 2022 train, 2023 test, 2024-2025 eval, and 2026H1 final.
Every opened stage must have positive absolute return, full-calendar
CAGR/strict-MDD at least 3, strict MDD at most 15%, positive 10 bp stress,
positive contained subperiods, clustered sign-flip p at most 0.10, and an
inferior direction flip.  Train and test each require at least 120 trades.

Every later predictor/outcome window remains sealed until every earlier gate
passes.  Full-calendar CAGR includes idle cash.  Strict MDD must include the
global/pre-entry high-water mark, the complete held OHLC path, costs, and exact
funding boundaries.
