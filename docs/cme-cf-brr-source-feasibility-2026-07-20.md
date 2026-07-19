# CME CF BRR source feasibility — 2026-07-20

## Verdict

**Rejected as a production research input under the repository's current
unlicensed data contract. No BRR values or BTC outcomes were opened.**

The CME CF Bitcoin Reference Rate (BRR) is economically interesting and has
enough daily observations for a multi-year experiment. The public methodology
also defines a reproducible benchmark window. The blocker is not event support;
it is lawful, automatable access for research and live signal generation.

The repository must not ingest the free CME website workbook, scrape it, or use
the published BRR in an automated trading system without the applicable data
license. A separately calculated London-window proxy made solely from
independently licensed exchange data may still be researched, but it must not be
called or represented as BRR and needs its own frozen protocol.

## Causal clock

The current CF Benchmarks methodology guide is version 17.3 dated 2026-06-08.
For BRR it specifies:

- a 15:00–16:00 `Europe/London` observation window;
- 12 five-minute partitions;
- one volume-weighted median across all relevant constituent-exchange trades
  per partition;
- an equally weighted average of the 12 partition medians;
- retrieval one minute after the window ends; and
- dissemination once per day, including weekends and holidays, at an
  unspecified point between 16:00 and 16:30 London time.

Therefore a historical file containing only the effective date and final value
does **not** justify a fixed 16:05 decision clock. Without an authenticated
publication timestamp, the earliest conservative fixed decision time is after
16:30 London, followed by the next complete execution bar. Restatements and
market-failure repeats also need explicit handling.

Official methodology:
<https://docs.cfbenchmarks.com/CME%20CF%20Reference%20Rates%20Methodology.pdf>

Current and historical constituent membership is versioned separately. The
2023 roster differs from the current roster: Bullish joined Bitcoin-Dollar
indices on 2024-12-30 and Crypto.com on 2025-03-31. Any exact reconstruction
must use the constituent set effective on each calculation day.

Official constituent list:
<https://docs.cfbenchmarks.com/CME%20CF%20Constituent%20Exchanges.pdf>

## Access and use gate

1. CME publishes a page advertising a free historical BRR workbook:
   <https://www.cmegroup.com/trading/cf-bitcoin-reference-rate/historical-data.html>.
   That page is a manual website distribution surface, not an automation or
   trading-use grant.
2. CME's 2023-12-07 data-terms advisory says website content is for personal,
   non-commercial use and prohibits scripts, bulk retrieval, data mining, and
   automated analysis:
   <https://www.cmegroup.com/content/dam/cmegroup/notices/clearing/2023/12/Chadv23-364.pdf>.
3. A direct automated request to the workbook was rejected with HTTP 403 and a
   response explicitly directing automated or commercial users to CME data
   delivery channels. No circumvention was attempted.
4. CF Benchmarks' REST API requires an API key obtained by contacting CF
   Benchmarks for a license:
   <https://docs.cfbenchmarks.com/api/>.
5. The historical-value endpoint additionally requires authorization for the
   requested index and `STREAM_HISTORICAL_VALUES`; recent values can also be
   delayed or amended:
   <https://docs.cfbenchmarks.com/api/rest/historical-values/> and
   <https://docs.cfbenchmarks.com/api/rest/values/>.
6. CME Schedule 7 defines both automated trading and research/analysis using
   benchmark information as non-display benchmark use governed by an index and
   benchmark license:
   <https://www.cmegroup.com/market-data/files/schedule-7-to-the-ila-september-2024.pdf>.

This is a source-contract decision, not legal advice and not a negative alpha
result. Under the current repository environment, the BRR values cannot be
used for model training, strategy selection, backtesting, or live order
generation.

## Reopening conditions

Reopen this branch only after all of the following exist:

1. a BRR historical and live license that explicitly permits automated
   research and trading-signal use;
2. authenticated historical values covering the full train/test/eval period;
3. publication or amendment timestamps sufficient to enforce a causal clock;
4. effective-dated constituent and methodology versions; and
5. a committed source manifest and outcome-blind event-support protocol before
   any BTC return following a BRR event is inspected.

## Next admissible branch

A London benchmark-window hypothesis can be tested without CME benchmark
information by using only independently obtained public exchange trades or
candles. That experiment would measure a market-session flow proxy rather than
BRR itself. It must:

- use an `Europe/London` DST-aware clock;
- finish all source features before the next-bar decision;
- avoid reconstructing, naming, or distributing CME's benchmark value;
- freeze its train/selection/eval sequence separately; and
- prove novelty against the rejected generic Coinbase leadership and
  spot-perpetual families.
