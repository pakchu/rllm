# RFXS2-576 mechanism decision — full-month source successor

## Decision and disclosure

The next candidate is **RFXS2-576**, a separately identified successor to the
source-rejected RFXS-576 policy. It preserves the exact economic feature,
direction, threshold, event onset, 48-hour execution, support floors, novelty
gates, controls, staged outcome gates, costs, and no-repair rules frozen in:

- `docs/regional-fiat-cross-rate-stress-mechanism-decision-2026-07-20.md`
- commit `1d5805397ed72c98bc83597544b949b07d425f32`
- SHA-256
  `c3f7bcfd12c4412be0ad8696b2fa339c709fa94f1a5e61a22cf33c45e4d3ae89`

This document incorporates that contract by reference and replaces only the
candidate identity and source-start clauses listed below. If a clause is not
explicitly replaced here, the original frozen clause remains normative.

RFXS-576 failed before source-support because `BTCBRL` began part-way through
October 2020. The failure and non-observation audit are frozen in:

- `docs/regional-fiat-cross-rate-stress-rfxs576-source-rejection-2026-07-20.md`
- commit `8ff99fec6f100537c260df8b1d484c32ebf56d8d`

RFXS2-576 therefore does **not** claim pristine source-feasibility selection.
Its 2020-11-01 start was chosen after learning only that the preceding archive
was partial and that the November archive passed structural validation. The
failed run displayed or persisted no close value, residual, z-score, event
incidence, comparator statistic, execution value, funding value, or outcome.
This successor is frozen before any such statistic is computed.

## Exact replacements

### Candidate identity

Every normative occurrence of `RFXS-576` in the incorporated contract becomes
`RFXS2-576`. In particular, the deterministic random-side control is:

```text
SHA256("RFXS2-576-random-side-20260720|" + entry_time)
```

Its first byte below 128 is long and otherwise short. The weekly cluster seed
remains `20260720`; no other threshold, side map, or randomization changes.

### Complete source boundary

The exact source horizon is now:

```text
[2020-11-01 00:00:00, 2024-01-01 00:00:00) UTC
```

This expressly voids and replaces the incorporated document's descriptive
claim that 2020-10 was a complete common monthly boundary; the October ZIP
existence observation remains only part of the rejected-candidate audit.

The builder must require all four symbols on every UTC day of this complete
horizon. It must reject a partial horizon, a partial month, a missing day, a
duplicate row, or a source start before or after 2020-11-01. It may not retain
October's partial BTCBRL history, backfill it, splice another venue, or shorten
the baseline.

The first regional residual is on 2020-11-02. The exactly 180 strictly prior
residuals for the first eligible z-score are 2020-11-02 through 2021-04-30, so
the first eligible source day is 2021-05-01. This is deterministic calendar
arithmetic, not observed incidence.

The production source artifact must be named:

```text
data/binance_regional_fiat_cross_rate_btc_2020-11_2023/
  BTC_regional_fiat_cross_rate_1d_2020-11-01_2023-12-31.csv.gz
  build_manifest.json
```

The manifest must bind this document and the incorporated original document,
their commits and SHA-256 values, the source-builder commit and SHA-256, every
companion-checksum response hash, every published archive hash, every locally
computed archive hash, and the exact four-symbol complete grid.

### Warmup wording

Where the incorporated strict evaluator says “with 2020-10 history used only
for causal warmup,” RFXS2-576 replaces it with “with 2020-11-01 through
2020-12-31 history used only for causal warmup.” No 2020 event or outcome is
eligible.

The original 2021 support floors remain unchanged: at least 18 accepted events
in 2021 and at least four accepted events in each of 2021Q2, Q3, and Q4. An
event cannot exist before 2021-05-01 under the frozen 180-prior-residual rule.
Failure of the shortened eligible Q2 interval is terminal, not grounds to relax
the floor.

## Outcome and source seals

The source builder may open only the four completed Binance Spot daily kline
archives through 2023. It may not import or open USD-M five-minute execution
OHLC, funding, future return, PnL, equity, CAGR, MDD, or any 2024+ source ZIP or
row. The source-support evaluator remains a separate, tested, committed,
hash-frozen work unit and may run only after the full source artifact passes
byte, date-grid, column, and manifest validation.

The original sequential outcome order remains unchanged:

1. 2021-2022 train;
2. 2023 test;
3. 2024 evaluation only after both preceding stages pass;
4. 2025 forward only after 2024 passes; and
5. 2026-H1 final only after 2025 passes.

No stage may use this source-boundary failure to alter the 180-day window,
`+/-1` threshold, median aggregation, contrarian direction, 00:05 entry,
576-bar hold, 0.5x exposure, support thresholds, costs, controls, significance
test, or CAGR/strict-MDD gates.

## Successor no-repair rule

Any source-integrity, source-support, novelty, train, test, evaluation, forward,
or final failure retires RFXS2-576. Another boundary, symbol, normalization,
direction, threshold, or support floor is a third candidate and requires a new
pre-result preregistration. RFXS-576 results may never be pooled with this
successor to improve support or performance.
