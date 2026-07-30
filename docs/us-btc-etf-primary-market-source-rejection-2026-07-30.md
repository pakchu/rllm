# US spot-BTC ETF primary-market source rejection — 2026-07-30

## Decision

Reject the US spot-BTC ETF daily shares/holdings axis before mechanism
selection, source incidence, or any BTC outcome.

The public issuer surfaces inspected on 2026-07-30 did not provide an
immutable, point-in-time, machine-replayable daily history from fund launch
through 2026H1. They exposed current snapshots, factsheets, quarterly
documents, or publication obligations. A current snapshot is not evidence of
the value that was publicly available on each historical date.

No ETF source row before `2026-06-01`, flow, candidate clock, BTC market row,
funding row, return, PnL, CAGR, or MDD was opened.

## Official surfaces checked

- [iShares IBIT](https://www.ishares.com/us/products/333011/ishares-bitcoin-trust-etf)
  and its
  [latest holdings CSV](https://www.ishares.com/us/products/333011/ishares-bitcoin-trust-etf/latest-holdings.csv)
  expose a current fund snapshot. No public immutable date-indexed archive was
  found.
- [ARKB](https://www.ark-funds.com/funds/arkb) exposes a current holdings CSV.
  ARK documents daily holdings publication timing, but no complete immutable
  launch-to-2026H1 archive was found. The asset host also disallows generic
  crawling.
- [BITB](https://bitbetf.com/) exposes a current reserve snapshot and a
  time-limited signed proof-of-reserves report URL. The page describes a
  `T+1` reporting delay but not a replayable historical archive.
- The
  [BRRR SEC registration statement](https://www.sec.gov/Archives/edgar/data/1841175/000183988224000518/bitcoin-s1a_010824.htm)
  describes a business-day publication obligation. That establishes a
  publication rule, not a surviving point-in-time dataset.
- WisdomTree, VanEck, Franklin Templeton, Invesco, Hashdex, and Grayscale
  public product/document surfaces similarly yielded current pages,
  factsheets, or periodic documents rather than a complete daily archive.

## Bounded post-cutoff feasibility probe

Only excluded dates after `2026-06-01` were used to test endpoint behavior.
The iShares historical-looking query forms returned the latest snapshot rather
than the requested date. No historical ETF row was retained or used to choose
a rule.

## Rejection reason

The axis fails all three minimum provenance requirements:

1. immutable point-in-time historical daily values;
2. replay through a documented official historical endpoint; and
3. an independent official replay or archive against which completeness can
   be checked.

Issuer publication timing could support a forward collector started now, but
such a collector cannot backfill historical evidence. This exact historical
ETF axis is therefore closed for the current alpha search.
