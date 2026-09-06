# High-volatility independent-source resumption audit — 2026-08-16

## Decision

No additional historical economics candidate is authorized by this audit.
The existing local/Postgres observables remain exhausted, and the genuinely
independent sources still lack an admissible point-in-time archive or the
required data entitlement.  No Gross9 rows, execution prices, funding values,
or post-entry outcomes were opened during this audit.

This audit follows the terminal `HVCELVPQA-8` train result.  That exact joint
confirmation passed source support and Gross9 novelty, but failed the frozen
train CAGR/MDD, stress CAGR/MDD, and calendar-half gates.  It must not be
repaired or used to choose another subset of the same events.

## Current Postgres inventory

The updated database configuration was queried through
`preprocessing.live_db_features.postgres_url_from_env` using credential names
only.  The non-operational research tables are:

| Table | Historical span observed | Observable family |
| --- | --- | --- |
| `bars_binance` | 2019-09-08 to 2026-08-16 | futures OHLCV, trade count, taker flow |
| `bars_binance_premium` | 2020-01-01 to 2026-08-16 | premium-index OHLCV |
| `bars_binance_spot` | 2020-01-01 to 2026-08-16 | spot OHLCV, trade count, taker flow |
| `bars_polygon` | 2018-01-01 to 2026-08-14 | already-audited Polygon OHLCV symbols |
| `bars_upbit` | 2018-01-01 to 2026-08-16 | KRW spot OHLCV |
| `funding_rates_binance` | 2020-01-01 to 2026-08-16 | exact funding settlements |
| `open_interest_binance` | 2020-09-01 to 2026-08-03 | historical open interest |
| `open_interest_binance_live` | forward-only | live open interest |

The schema contains no option surface, option order book, ETF primary-market
inventory, licensed benchmark-resolution feed, node-topology archive, or
historical mempool-state archive.  The environment exposes Binance API key
names only; it contains no Deribit, CME, ETF-provider, or Bitnodes entitlement.

## Official-source recheck

### CME CF BRR

CME now displays a page saying that a free historical BRR workbook is
available:

- <https://www.cmegroup.com/trading/cf-bitcoin-reference-rate/historical-data.html>

The linked workbook path is
`/trading/files/cme-cf-brr-historical-data.xlsx`, but direct retrieval from the
research host still returns Akamai HTTP 403.  More importantly, CME's current
FAQ states that customers require an MDLA for CME CF reference-rate products
and that derived works require a separate agreement:

- <https://www.cmegroup.com/articles/faqs/cme-cf-cryptocurrency-benchmarks-faq.html>

Therefore the visible workbook link does not establish an automated-use or
derived-signal entitlement.  No workbook values were opened or committed.

### US spot-BTC ETF primary inventory

The current IBIT product page exposes a current shares-outstanding snapshot and
an Excel download, but it still does not expose a replayable daily
shares-outstanding archive with publication timestamps:

- <https://www.ishares.com/us/products/333011/i>

Current snapshots cannot reconstruct what was available at every historical
decision, so this axis remains forward-only.

### Deribit option surface

Deribit's public API supports current active instrument metadata and current
book summaries, including mark IV and open interest:

- <https://docs.deribit.com/api-reference/market-data/public-get_instruments>
- <https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency>

Those endpoints support a forward collector but not a complete replay of the
historical surface required for the frozen train/test/eval/final windows.

## Research-program contamination boundary

The 2024 test window has been opened repeatedly across the high-volatility
search and has produced no passing candidate.  It is now adaptive validation
at the program level, even when a new candidate is preregistered immediately
before its own run.  Further combinations of frozen OHLCV/flow/OI/funding/
premium/cross-alt clocks would not supply fresh confirmatory evidence.

Accordingly:

1. do not create another subset, veto, router, threshold, or component swap
   from the exhausted local clocks;
2. treat 2024 as contaminated for any future design informed by these results;
3. admit a new singleton only after an independent point-in-time source and an
   untouched forward confirmation window are frozen; and
4. retain the existing 0.5 gross, exact funding, 6/10 bp costs, full-calendar
   CAGR, strict MDD, Gross9 novelty, and stop-on-first-failure contracts.

## Exact unblock conditions

One of the following is required before another honest economics candidate can
be opened:

1. documented automated-use rights plus a complete point-in-time historical
   option-surface or benchmark-resolution archive;
2. a forward collector that accumulates sufficient independent source and
   untouched outcome history for a newly frozen split design; or
3. another genuinely independent observable with complete causal history that
   is absent from the repository and Postgres inventory above.

