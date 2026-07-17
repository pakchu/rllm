# U.S. Treasury auction-demand source audit (2026-07-17)

## Decision

The official TreasuryDirect auction-query archive is suitable for a new,
crypto-exogenous alpha family, subject to a conservative availability clock.

No BTC price, funding, OI, premium, FX, Kimchi, order-flow, label, or portfolio
outcome was opened during this source audit.

## Official source

- Auction query and historical export:
  <https://www.treasurydirect.gov/auctions/auction-query/>
- Auction query help:
  <https://www.treasurydirect.gov/auctions/auction-query/auction-query-help/>
- Announcements and result press-release archive:
  <https://www.treasurydirect.gov/auctions/announcements-data-results/announcement-results-press-releases/>
- Official definition of bid-to-cover and auction-result disclosures:
  <https://www.treasurydirect.gov/laws-and-regulations/auction-regulations-uoc/>
- TreasuryDirect auction FAQ; results are available in accounts after 5 p.m. ET:
  <https://www.treasurydirect.gov/help-center/auction-faqs/>

The machine-readable endpoint used by TreasuryDirect's own auction-query page
is:

`https://www.treasurydirect.gov/TA_WS/securities/jqsearch`

## Frozen universe

Only original-issue (`reopening=No`) nominal fixed-rate coupon auctions are
retained:

- 2-year, 3-year, 5-year, 7-year, and 10-year notes;
- 20-year and 30-year bonds;
- TIPS, FRNs, bills, CMBs, and reopenings are excluded.

| Coverage | Value |
|---|---:|
| Rows | 445 |
| Same-day complete rows | 440 |
| Quarantined later-updated rows | 5 |
| First auction | 2016-02-24 |
| Last auction | 2023-12-28 |
| 2021 rows | 60 |
| 2022 rows | 59 |
| 2023 rows | 60 |

The panel retains official bid-to-cover, competitive accepted amounts, and
primary/direct/indirect bidder accepted amounts. Every row is checked so the
three bidder buckets exactly equal `competitiveAccepted`.

Five February/March 2023 rows carry an API `updatedTimestamp` of 2023-04-28.
Their current values cannot be proven identical to the auction-day values, so
their demand fields are blanked and `source_complete=false`. Downstream clocks
must not bridge across those quarantined observations.

## Causal availability contract

The archive exposes an Eastern-time `updatedTimestamp`, normally a few minutes
after the competitive close, but it does not attach an explicit time-zone
offset. The evaluator must not treat that field as a directly tradable clock.

Every result is instead exposed at **22:00 UTC on the auction date**:

- 17:00 EST in winter;
- 18:00 EDT in summer;
- therefore never earlier than TreasuryDirect's documented after-17:00-ET
  account-availability guarantee;
- live and historical code must use the same conservative clock.

The official result PDF and XML URLs are retained for every row. The current
API is a historical archive, not a formal revision-vintage service; the frozen
raw responses and their SHA-256 hashes therefore define this research snapshot.

## Artifacts

- Builder: `training/build_treasury_auction_demand_panel.py`
- Tests: `tests/test_build_treasury_auction_demand_panel.py`
- Panel:
  `data/us_treasury_auction_demand_2016_2023/us_treasury_nominal_original_auctions_2016_2023.csv.gz`
- Manifest:
  `data/us_treasury_auction_demand_2016_2023/build_manifest.json`
- Raw pages:
  `data/us_treasury_auction_demand_2016_2023/raw/auction_query_page_{0,1}.json.gz`

## Alpha boundary

This audit authorizes a single source-only-screened family based on **changes
in auction demand**, not Treasury yield levels and not any crypto input. The
candidate must be preregistered before BTC outcomes are opened. Its 2023 market
outcomes remain sealed until an unchanged 2021-2022 Stage1 passes.
