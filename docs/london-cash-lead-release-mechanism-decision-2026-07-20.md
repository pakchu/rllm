# London cash-lead release mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **LCLR-24 — London Cash-Lead Release**, a
two-hour Binance BTCUSDT perpetual policy triggered by a completed,
DST-aware Coinbase/Binance window from 15:00 to 16:00 `Europe/London`.

This decision opens no LCLR event incidence, post-window BTC return, PnL, or
2023+ Coinbase row. It selects only the economic mechanism, source boundary,
and frozen research sequence.

## Why this branch is admissible after the BRR rejection

LCLR does not ingest, reconstruct, name, or distribute the CME CF Bitcoin
Reference Rate. It uses only independently obtained exchange candles and asks
whether a fixed London cash-execution window leaves a temporary price-discovery
imbalance between Coinbase BTC-USD and Binance BTCUSDT perpetual.

The BRR source branch was rejected because published benchmark values require
licensed non-display use for automated research/trading and can be disseminated
at any time between 16:00 and 16:30 London. Those restrictions do not make the
underlying public exchange markets unavailable. LCLR therefore enters from the
next complete five-minute bar after its own source window, not from a BRR
publication.

BRR source decision:
[`cme-cf-brr-source-feasibility-2026-07-20.md`](cme-cf-brr-source-feasibility-2026-07-20.md)

## Economic mechanism

The hypothesis is narrower than generic venue leadership:

1. a sustained, back-loaded Coinbase move during the fixed London window is a
   proxy for cash-market execution pressure;
2. if Binance perpetual has moved less in the same direction, cash has led the
   executable derivative rather than merely co-moving with it;
3. unusually coherent cash flow that remains active in the final partitions
   can continue briefly after the fixed window ends; and
4. the policy follows the completed Coinbase direction for exactly two hours.

The source-only preregistration must freeze the exact coherence, cash-lead,
participation, back-loading, latency, and support definitions before their real
incidence is calculated.

## Why this is not the rejected Coinbase family

The rejected Coinbase–Binance search applied relative return, premium, and
activity conditions across the entire five-minute grid. It selected no policy
on 2020–2022 and never opened 2023.

LCLR is not a lower threshold or alternative hold for those policies. Its
event primitive is one fixed DST-aware 12-partition London window per eligible
day. It requires within-window path coherence and final-partition persistence,
then measures cash lead at the window boundary. Repository search found broad
UTC session/day-of-week encodings and weekend clocks, but no prior 15:00–16:00
`Europe/London` Coinbase cash-lead event policy.

The prior calendar × OI/funding scan used coarse Asia/Europe/US flags and
derivative state; it did not use Coinbase, a London-local DST clock, or a
completed 12-partition cash path.

## Frozen source boundary

The pre-2023 source is already checksum-audited and physically ends before
`2023-01-01`:

- Coinbase BTC-USD five-minute candles:
  `data/coinbase_btcusd_5m_2020_2022.csv.gz`, SHA-256
  `07f7a3bddecbbc3724994645b9ac1cd0f391378e0feed421f2c8caa145aab77b`;
- Binance BTCUSDT perpetual five-minute candles:
  `data/coinbase_leadership_binance_5m_2020_2022.csv.gz`, SHA-256
  `1a06f1f4dbbdafaf885fb03844426eed5d5bad4aa206fa72b88db2cbd98bef94`;
- exact Binance funding marks:
  `data/coinbase_leadership_funding_2020_2022.csv.gz`, SHA-256
  `0ff7952a83a38e8e1b2cbeb4eaad3e23aacf0c68920164bdfb83a13c3c2bfa36`;
- audited source manifest SHA-256:
  `3af321fdcafd0fe6680c4583341b6508124a979fefbf489f8d3376c7ec78a269`.

The manifest reports 315,648 expected five-minute rows, 315,528 complete
Coinbase rows, 120 missing Coinbase rows left unimputed, a complete Binance
grid, and 3,288 exact funding marks.

Official live/historical interfaces:

- Coinbase product candles:
  <https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles>
- Coinbase WebSocket market-data overview:
  <https://docs.cdp.coinbase.com/exchange/websocket-feed/overview>
- Binance public-data archive and checksum policy:
  <https://github.com/binance/binance-public-data>
- Binance USD-M live market streams:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams>

## Frozen research sequence

1. Commit the exact source-only feature and support preregistration plus
   synthetic DST, missing-partition, and causality tests.
2. Run real 2020–2022 incidence without loading post-window market outcomes.
   Reject without repair if train/test support or side balance fails.
3. Commit and hash-freeze one strict train/test evaluator before constructing
   any LCLR post-window return.
4. Use 2020–2021 as train and calendar 2022 as test. No 2023 row may be
   requested or parsed during selection.
5. Reject if either train or test misses the frozen profitability, strict-MDD,
   execution-delay, stress-cost, or mechanism-control gate.
6. Only a single frozen pre-2023 pass may trigger a separately committed
   2023+ source download and sequential 2023, 2024, 2025, then 2026 evaluation.
7. Stop at the first failed year; do not change side, thresholds, window,
   latency, or hold after observing a result.

The branch is globally contaminated by extensive prior BTC research. This
sequence can support a candidate-level frozen claim, not a pristine global
human holdout claim.
