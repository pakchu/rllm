# GDELT Bitcoin narrative source protocol

## Objective

Promote an outcome-blind daily news source that is independent of the existing
BTC price, funding, basis, liquidation, FX, and on-chain feature families. This
stage acquires source counts only; it does not define or test a trading rule.

Official references:

- [GDELT data access](https://www.gdeltproject.org/data.html)
- [DOC 2.0 API and timeline parameters](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- [DOC 2.0 fixed historical search horizon](https://blog.gdeltproject.org/doc-2-0-updates-1-5-year-searching-and-updated-mobile-interface/)

## Frozen API contract

- endpoint: `https://api.gdeltproject.org/api/v2/doc/doc`
- mode: `timelinevolraw`
- format: `json`
- source interval: `[2020-01-01, 2024-01-01)`
- logical request windows: calendar quarters, half-open
- API `enddatetime`: one second before each logical exclusive end; this avoids
  GDELT's observed inclusion of the boundary calendar-day bin
- required response resolution: one UTC day
- availability assigned to source date + 48 hours + 15 minutes
- every query must return a complete daily grid
- the global article normalization count must agree across all queries

Queries:

1. `broad`: `(bitcoin OR cryptocurrency)`
2. `failure`: `(bitcoin OR cryptocurrency) AND (hack OR hacked OR exploit OR scam OR fraud OR theft OR bankruptcy OR bankrupt OR collapse OR liquidation OR liquidated)`
3. `constraint`: `(bitcoin OR cryptocurrency) AND (ban OR banned OR regulation OR regulator OR crackdown OR lawsuit OR investigation)`
4. `adoption`: `(bitcoin OR cryptocurrency) AND (ETF OR institutional OR adoption OR approval OR approved OR investment)`

The 48-hour-plus-15-minute availability lag is deliberately conservative. It
does not assume that all articles assigned to a source day were indexed by that
day's close.

## Outputs

- daily count source: `data/gdelt_bitcoin_narrative_daily_2020_2023.csv.gz`
- raw response bundle: `data/gdelt_bitcoin_narrative_timeline_raw_2020_2023.jsonl.gz`
- source manifest: `results/gdelt_bitcoin_narrative_source_manifest_2026-07-20.json`

The raw response bundle records each request URL, request interval, response
hash, and parsed payload. Cache files exist only for interruption-safe resume.

## Outcome boundary

This stage may not open BTC market data, funding, labels, future returns, PnL,
CAGR, or MDD. It also does not request any news row dated 2024 or later.

Before this protocol was frozen, one connectivity/schema probe opened the
`broad` query for January 2021 and confirmed a daily `value` plus global `norm`
schema. The probe did not open category queries or any market outcome. No count,
threshold, or trading rule is selected from that probe.

After the downloader was first committed but before any source artifact was
written, one 2020Q1 request showed that `enddatetime=20200401000000` returns an
April 1 daily bin. The request builder was therefore corrected to send
`enddatetime=20200331235959` for logical `[2020-01-01, 2020-04-01)`. The failed
response was not cached and no feature or outcome was evaluated.

After source promotion, source-only feature definitions and minimum incidence
gates must be frozen before any BTC outcome is opened.
