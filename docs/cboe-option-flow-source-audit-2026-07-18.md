# Cboe option-flow source audit — 2026-07-18

## Decision

Freeze the Cboe daily U.S. options statistics panel as a **source-only research
input**.  No BTC price, funding, return, trade, portfolio, or alpha-overlap row
was opened while building or validating this panel.

This is a distinct data axis from the repository's crypto price, taker flow,
funding, open interest, FX, Kimchi, REX, and Cboe implied-index experiments.
Whether it has a causal next-session BTC edge remains deliberately unknown.

## Official source

- [Cboe daily market statistics](https://www.cboe.com/us/options/market_statistics/daily/)
- [Cboe historical options data information](https://www.cboe.com/us/options/market_statistics/historical_data/)
- Date-addressable page:
  `https://www.cboe.com/us/options/market_statistics/daily/?dt=YYYY-MM-DD`

The daily page publishes put/call ratios and call, put, and total volumes for
all products, index options, equity options, VIX options, and SPX/SPXW options.
The historical-information page warns that the displayed volume and put/call
data are informational and points users who need detailed historical products
to Cboe DataShop.

## Frozen artifact

| Item | Value |
|---|---:|
| Research horizon | 2020-01-01 through 2023-12-31 |
| Valid Cboe option-statistics dates | 1,006 |
| First / last date | 2020-01-02 / 2023-12-29 |
| Panel | `data/cboe_option_flow_2020_2023/cboe_option_flow_2020-01-01_2023-12-31.csv.gz` |
| Panel SHA-256 | `35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78` |
| Response-ledger hash | `33d0ba076914c338c3e5f6c795c20eccfee97ab3e205f9890756fd20a467de8f` |
| Manifest | `data/cboe_option_flow_2020_2023/build_manifest.json` |

Each normalized row retains the SHA-256 of the corresponding Cboe HTML
response.  Raw HTML was not retained to avoid hundreds of megabytes of
duplicated presentation markup.  The parser and schema validation are frozen
in `training/build_cboe_option_flow_panel.py`.

## Validation performed

1. Exactly one Next.js `optionsData` object must be present.
2. An official `optionsData: null` page is treated as a no-data date, never as
   a zero-volume session and never forward-filled.
3. Required ratio and volume groups must all exist.
4. Every call, put, and total volume must be a finite nonnegative integer.
5. `call + put == total` for every retained group.
6. VIX and SPX/SPXW volume cannot exceed index-option volume; index and equity
   volume cannot exceed all-product volume.
7. Cboe's rounded put/call ratios must agree with exact retained volumes within
   0.011.
8. Dates are unique, sorted, and bounded to the frozen horizon.
9. Deterministic gzip replay reproduces the panel byte-for-byte.

## Interpretation boundary

The later alpha may use index/equity put-call divergence, VIX call pressure,
and index share of total option volume as **weak flow proxies**.  Cboe does not
state that these aggregates identify institutional investors, opening trades,
buyers, sellers, or directional intent.  Multi-leg spreads and closing trades
are mixed into volume.  Any “institutional hedge migration” interpretation is
therefore a research inference, not an official Cboe claim.

## Vintage and live boundary

The date query currently renders historical daily values, but it is not being
claimed as a point-in-time archive.  A next-Cboe-session decision clock removes
same-close publication ambiguity, but **does not prove vintage immutability**.
Before live promotion, forward collection must record retrieval time and
response hash, and the live parser must demonstrate parity with this frozen
schema.  No stale carry across a no-data date is permitted.
