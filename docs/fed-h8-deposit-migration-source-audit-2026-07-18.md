# Federal Reserve H.8 deposit-migration source audit — 2026-07-18

## Decision

The H.8 source axis is suitable for an outcome-blind preregistration **only when
dated archive releases are used**. Current-vintage FRED/DDP history is not used
for historical features because H.8 is benchmarked and revised.

No BTC OHLC, funding, return, portfolio, or existing-alpha outcome was opened
while this source panel or its transformations were built.

## Official source contract

- release calendar and timing rule:
  <https://www.federalreserve.gov/releases/h8/>
- H.8 definitions and construction:
  <https://www.federalreserve.gov/releases/h8/about.htm>
- technical questions, benchmarking, and methodology changes:
  <https://www.federalreserve.gov/releases/h8/h8_technical_qa.html>
- notes on historical data changes:
  <https://www.federalreserve.gov/releases/h8/h8notes.htm>
- dated archive template:
  `https://www.federalreserve.gov/releases/h8/YYYYMMDD/default.htm`

The Federal Reserve states that H.8 is generally released Friday at 4:15 p.m.
Eastern, or Thursday when Friday is a federal holiday. Weekly levels are
Wednesday observations. “Large” means the top 25 domestically chartered
commercial banks by domestic assets at the previous benchmark Call Report;
“small” is the residual set. The Fed adjusts the groups for mergers and panel
shifts, so this is not a clean G-SIB-versus-regional-bank partition.

## Frozen coverage

- 365 official release-calendar records, 2017-01-06 through 2023-12-29;
- 365 dated HTML archive snapshots, each with an individual response SHA-256;
- 190 legacy split-table pages and 175 modern single-table pages;
- retained groups: all domestic, large domestic, and small domestic banks;
- retained vintages: seasonally adjusted and not seasonally adjusted;
- retained levels: cash assets, total assets, deposits, large time deposits,
  other deposits, and borrowings for the prior and latest printed week;
- accounting checks on every release:
  `deposits = large time deposits + other deposits` and
  `domestic = large + small`, within printed rounding tolerance.

The normalized panel has 365 rows and 85 columns. It uses the release date
printed inside each page rather than assuming that the release-calendar key or
archive path is the actual publication date.

## Official archive irregularities preserved

- calendar key `2017-01-12` / archive path `2017-01-12` prints release date
  `2017-01-13`;
- calendar key / archive path `2020-12-21` prints release date `2020-12-18`;
- calendar key `2022-11-11` has no page, while the holiday release exists at
  archive path and printed release date `2022-11-10`.

The panel stores the calendar key, archive-path date, and printed release date
separately. It contains 355 Friday, 8 Thursday, and 2 Monday printed release
dates. The latest observation lag is 9 days for 355 releases, 8 days for 8
releases, and 12 days for 2 releases.

## Methodology boundaries

The eventual evaluator must not silently delete or repair these known breaks:

- 2020-10-02: revised large/small panel-shift adjustment method;
- 2023-03-31 release: revised treatment of FDIC bridge-bank data for the week
  ending 2023-03-15;
- 2023-06-30: revised weekly seasonal-adjustment treatment of outliers;
- 2023-12-15: transition toward annual seasonal-factor updates.

They must be reported as frozen source controls. The primary source transform
uses the seasonally adjusted values printed in each dated release; the exact
not-seasonally-adjusted replay is frozen as a mechanism control.

## Candidate transform boundary

The outcome-blind candidate combines three weak balance-sheet signals:

1. large-bank minus small-bank one-week log growth in other deposits;
2. small-bank one-week log growth in borrowings;
3. the negative of small-bank one-week log growth in cash assets.

Each component is robustly standardized against exactly the prior 104 H.8
releases. Their equal mean is a stress score. Positive stress is provisionally
risk-off/SHORT BTC; negative relief is provisionally risk-on/LONG BTC. A
strictly prior rolling quantile of the previous 52 absolute stress scores
supplies the event threshold. The two-stage history consumes 156 source
releases before the 2020 Stage1 start.
The exact quantile and support density will be selected from source counts only
before any BTC/funding row is opened.

This interpretation is an economic hypothesis, not a Federal Reserve claim.
“Other deposits” can also reflect deposit repricing or liability substitution,
and cash/borrowings are not unique measures of banking stress.

## Frozen identities

- release-date snapshot SHA-256:
  `20a7d218ffbe2c4a47508ff4c547fdee7047663f585b31ceba62c6f66b771629`
- archive snapshot SHA-256:
  `65edb50eb2b2a01785518fe30a92acf1a35f0f8b78cd332a96557ff9bad8601a`
- normalized panel SHA-256:
  `c8d1bfb0bbd13ef6d35f09ad7367ef8d2d5bb28981376223b735746ade68a572`
- build manifest SHA-256:
  `1f0a194e628ab9c44c23fc4a923145dcf89a62bface745cc36872eeee919eda9`
