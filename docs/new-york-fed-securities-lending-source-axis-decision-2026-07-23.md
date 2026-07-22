# New York Fed SOMA securities-lending source-axis decision — 2026-07-23

## Decision

Open one new **source-only** research axis: daily New York Fed System Open
Market Account (SOMA) securities-lending auction results. This source family is
not present in the repository's existing alpha code or documents.

This decision authorizes only a hash-reproducible 2019–2023 source build and
schema audit. It does **not** authorize a candidate clock, source incidence,
BTC market data, funding, forward returns, PnL, CAGR, MDD, or model training.
A separate committed mechanism decision must precede every candidate-derived
count.

## Official economic object

The New York Fed states that it lends Treasury and agency debt securities from
SOMA to primary dealers temporarily, using a competitive multiple-price auction
each business day at noon. The stated purpose is smooth clearing of Treasury
and agency debt securities in support of monetary-policy implementation.
Summary results are released publicly after each auction.

Official references:

- program and publication description:
  <https://www.newyorkfed.org/markets/domestic-market-operations/monetary-policy-implementation/securities-lending>;
- operation-result definitions and historical search:
  <https://www.newyorkfed.org/markets/desk-operations/securities-lending>;
- program FAQ and auction timing:
  <https://www.newyorkfed.org/markets/sec_faq.html>;
- official Markets Data API documentation:
  <https://markets.newyorkfed.org/static/docs/markets-api.html>;
- frozen OpenAPI document used by the builder:
  <https://markets.newyorkfed.org/static/docs/markets-api.yml>.

The result page defines:

- `Par Amount Submitted`: propositions received for a security;
- `Par Amount Accepted`: securities lent;
- `Weighted Average Rate`: the accepted lending fee, not a repo rate;
- `Pre-Auction Par Available to Borrow`: SOMA inventory available for lending;
- `Pre-Auction Par Outstanding Loans`: loans not returned before the auction;
- `Par Amount Extended`: loans not returned by maturity before Fedwire close.

These fields are direct auction/settlement observables. They do not identify
dealer direction in BTC or guarantee that high borrowing demand is a broad
funding shock.

## Frozen retrieval contract

The source builder will request only the official production endpoint:

```text
GET https://markets.newyorkfed.org/api/seclending/seclending/results/details/search.json
    ?startDate=YYYY-MM-DD
    &endDate=YYYY-MM-DD
```

The OpenAPI document was last updated on 2026-06-12 and defines the search
range as inclusive. Retrieval is split into one request per calendar year for
2019 through 2023. The builder must persist deterministic gzip copies of:

1. the exact OpenAPI bytes;
2. each exact annual JSON response; and
3. a request/response ledger containing URL, retrieval UTC time, HTTP status,
   content type, byte count, and SHA-256.

No beta endpoint, page-scraped HTML, current-only shortcut, CUSIP filter, or
post-2023 operation is allowed. Redirects away from `markets.newyorkfed.org`,
non-JSON responses, duplicate annual operations, or schema drift fail closed.

## Allowed source schema

Operation-level fields:

- `operationId`, `auctionStatus`, `operationType`;
- `operationDate`, `settlementDate`, `maturityDate`;
- `releaseTime`, `closeTime`, `lastUpdated`, `note`;
- `totalParAmtSubmitted`, `totalParAmtAccepted`, `totalParAmtExtended`.

Security-detail fields:

- `cusip`, `securityDescription`;
- `parAmtSubmitted`, `parAmtAccepted`, `weightedAverageRate`;
- `somaHoldings`, `theoAvailToBorrow`, `actualAvailToBorrow`;
- `outstandingLoans`.

Unknown fields are retained in the raw bytes but reject normalized-panel
construction until this decision is amended before any candidate use. Nulls
remain null; missing demand is never converted to zero. Numeric values must be
finite, nonnegative exact decimals. Operation totals must reconcile to detail
sums under explicitly documented null handling.

## Conservative availability and revision boundary

The official program says auctions occur at noon Eastern and results are posted
after the operation, but it does not promise a historical result-publication
minute in the API documentation. The normalized source therefore uses:

```text
base_available = operationDate + 1 calendar day at 00:00 UTC
revision_available = lastUpdated interpreted in America/New_York
available_at_utc = max(base_available, revision_available)
```

If `lastUpdated` is missing, malformed, earlier than the operation, or
ambiguous at a daylight-saving transition, the row is quarantined rather than
backfilled. Candidate execution must later wait at least one complete five-
minute bar after `available_at_utc`.

This is deliberately later than the normal same-day result and prevents using
the noon auction before its public result. It also moves a recorded revision
forward to its update time. It cannot prove that the current API preserves
every historical revision timestamp. The source manifest must record that
residual current-vintage limitation, and live promotion will require forward
collection of raw response hashes and retrieval timestamps.

## Normalized outputs

The builder will produce two frozen panels:

1. one row per securities-lending operation; and
2. one row per operation/CUSIP detail.

Both panels are limited physically to operation dates from 2019-01-01 through
2023-12-31. The source audit may report row counts, missingness, date coverage,
schema membership, reconciliation, update-lag distributions, and duplicate
identities. It may not compute submitted/accepted ratios, cross-sectional
breadth, ranks, z-scores, state transitions, event clocks, sides, holds, or any
other candidate-derived feature before a later mechanism freeze.

## Candidate ideas deliberately not yet selected

Plausible later mechanisms include collateral-scarcity breadth, auction-demand
concentration, unreturned-loan persistence, or a disagreement between demand
and accepted supply. Listing them here does not select one and authorizes no
incidence calculation. Exactly one mechanism, direction, clock, hold, controls,
support floor, and no-repair rule must be committed before source values are
combined into candidate features.

## Stop conditions

Retire the source axis before candidate work if any of the following occurs:

- the official endpoint cannot reproduce complete annual responses;
- required operation or detail identity fields are absent;
- operation dates or identifiers conflict across annual responses;
- `lastUpdated` cannot support a deterministic conservative clock;
- operation/detail totals materially fail reconciliation without an official
  explanation; or
- the builder accesses any market, funding, return, portfolio, or model value.

The immediate next work unit is a tested source builder. It must be committed
before the first full 2019–2023 retrieval is run.
