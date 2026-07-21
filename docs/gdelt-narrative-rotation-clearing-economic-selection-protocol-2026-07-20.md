# GNRC pre-2024 economic-selection protocol

## Purpose and freeze boundary

This stage opens only the already frozen 2021–2023 BTCUSDT USD-M perpetual
market and funding inputs after the GNRC source-only family passed its
preregistered support gate. The evaluator, this document, tests, and a later
hash-gated launcher must be committed before any market value is parsed.
The evaluator also refuses to load either value artifact unless
`results/gdelt_gnrc_premarket_access_seal_2026-07-22.json` exists and binds the
exact committed evaluator, protocol, tests, source-support report, market data
and manifest, and funding data and manifest. A later launcher additionally
hard-codes this seal's exact SHA-256 before importing the evaluator.

Frozen ancestors:

- preregistration SHA-256:
  `ae175a242db1fa850164789e4a3e6f3f39b4ac8eae0fb877ce79e915ae3d67f3`
- source-support report SHA-256:
  `1b35c6fef694f1b352129cd3b40ae85832834561f61b731bccaf4d8b24c2a5e4`
- source-support manifest hash:
  `fa4465fa3a8f6b001d4179c692e2d0a7b11e6ce7439a474bb995541b9aa32780`
- source-support decision: `advance_to_market`, with exactly 17 of the frozen
  24 variants passing. No sign, threshold, window, hold, evidence gate, or
  support gate may be repaired after this result.

## Frozen execution inputs

- official Binance BTCUSDT USD-M 5-minute kline artifact, calendar 2020–2023;
  only `[2021-01-01, 2024-01-01)` values are parsed;
- official Binance realized funding rates with audited 8-hour mark-price-kline
  open settlement marks, calendar 2020–2023; only the same pre-2024 interval is
  parsed;
- the committed 2020–2023 GDELT daily source and exact schedules reproduced
  from the source-support report.

The evaluator verifies the exact artifact and manifest hashes, official-source
provenance, complete 5-minute and 8-hour grids, UTC timestamps, OHLC envelopes,
funding timestamp mapping, source schedule counts, and strict split containment.
It reads no 2024+ news, BTC bar, or funding value.

## Execution and risk accounting

Every frozen variant is evaluated independently in train (2021–2022) and
selection (2023). Entry is the 5-minute open exactly ten minutes after the
`+48h15m` source availability time. Exit is the 5-minute open exactly three or
seven calendar days later. Same-time re-entry is forbidden by the frozen
scheduler.

Only the 17 source-supported variants may have BTC outcomes computed. The other
seven retain their source schedules but record `market_outcome_opened=false`,
no economic metrics or daily-return hash, and adjusted p-value 1.0.

Base cost is 2 bp per entry side and 2 bp per exit side. Stress cost is 4 bp
per side. Funding cash flow is applied after the entry instant and at or before
the exit instant. The exact returned Binance `fundingTime`, its offset, and its
floor-to-8h mapping are validated; the audited `mark_open_time` is used as the
complete UTC 00:00/08:00/16:00 settlement grid required by the preregistration.
Entry and exit occur at `:25`, so this at-most-60-second source jitter cannot
change settlement inclusion.

Strict MDD visits every flat/open/close mark, both costs, funding cash changes,
and conservative intrabar order: long high→low→close, short low→high→close.
The exit bar is closed at its open before that bar's extremes. CAGR always uses
the full calendar split, including inactive intervals.

## Daily returns and familywise inference

The 2023 familywise test uses 365 daily net log equity returns. A day's endpoint
is the equity at its final UTC 5-minute close; the first return starts at split
initial equity. Overnight gaps therefore enter the following day's return.

The frozen one-sided Romano–Wolf procedure is implemented as the standard
step-down max-t test:

1. compute `sqrt(n) * mean(r) / std(r, ddof=1)`;
2. center every tested variant by its own 2023 sample mean;
3. draw 100,000 synchronized circular seven-day block samples with seed
   `20260720`;
4. order hypotheses by descending observed t;
5. at each step, compare against the bootstrap maximum over the remaining
   hypotheses;
6. use `(exceedances + 1) / (draws + 1)` and enforce monotone adjusted p-values.

Exact observed-t ties are treated as one group: every tied hypothesis is tested
against the same pre-removal remaining set and the whole group is removed
together. Lexical ID order is serialization-only and cannot change inference.

Only source-supported variants that pass all train economic qualifiers and have
positive 2023 return variance enter the step-down set. Every other frozen family
member receives adjusted p-value 1.0. This is a disambiguation of the
preregistered Romano–Wolf max-t method, not a new tunable choice.

## Qualification and selection

The executable uses the preregistered gates unchanged:

- train: positive absolute return, CAGR/strict-MDD at least 1, strict MDD at
  most 25%, at least 24 trades, and positive stress-cost absolute return;
- selection: positive absolute return, CAGR/strict-MDD at least 1, strict MDD
  at most 20%, and at least 10 trades;
- familywise adjusted p-value at most 0.10.

Among variants passing all three layers, the champion maximizes selection
CAGR/strict-MDD, then minimizes selection strict MDD, then uses lexical
`variant_id`. Failure produces `retire_without_repair`. Success produces a
write-once selection report with the champion policy hash and permits—but does
not itself open—the separately sealed 2024–2026 OOS stage.

## Outcome boundary

The output is
`results/gdelt_narrative_rotation_clearing_economic_selection_2026-07-20.json`.
It records absolute return, CAGR, strict MDD, CAGR/MDD, trade count, stress
metrics, daily-return hashes, adjusted p-values, all source hashes, and exact
pre-2024 rows read. Post-2023 market, funding, and news row counts must remain
zero until the two OOS access seals are separately committed.
