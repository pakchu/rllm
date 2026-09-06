# OI/supply leverage source-axis rejection — 2026-08-13

## Decision

The proposed high-volatility BTC leverage-occupation object
`sum_open_interest / cmc_circulating_supply` is rejected before
preregistration. No candidate incidence, execution prices, returns, PnL,
funding values, or Gross9 rows were opened for this proposal.

The terminal decision is
`source_axis_reject_missing_causal_historical_provenance_no_repair`.

## Intended source object

The source-blind proposal would have used exact BTCUSDT 5-minute rows from
`public.open_interest_binance` and ranked the completed eight-hour arithmetic
mean of `sum_open_interest / cmc_circulating_supply`. The physical
interpretation was perpetual open contracts per circulating BTC, rather than
an OI change or OI-to-turnover measure.

## Provenance findings

1. Repository search found readers of `cmc_circulating_supply`, but no schema,
   collector, migration, immutable first-seen ledger, raw response archive,
   checksum, or upsert contract proving how historical values were acquired or
   whether old values can be revised.
2. Binance's `openInterestHist` response identifies
   `CMCCirculatingSupply` as circulating supply supplied by CMC, but the
   current public endpoint exposes only a short recent history. It therefore
   cannot independently reproduce the multi-year database values.
3. The Binance Vision historical metrics archive audited in
   `docs/binance-cross-collateral-positioning-metrics-source-audit-2026-07-17.md`
   does not contain the CMC circulating-supply field.
4. The sealed HVOTSC-8 source execution audit established that all 376,570
   matching 2023-01-01 through 2026-08-01 `open_interest_binance` rows had
   `observed_at IS NULL`. Consequently this database cannot prove that each
   historical supply value was observed by the proposed decision boundary.
5. `training/build_oi_enriched_cache.py` forward-fills the supply column after
   an as-of join. That helper is unsuitable evidence for an exact-row,
   no-imputation candidate and does not repair the missing source provenance.

## No-repair boundary

Deriving availability as `ts+5m`, substituting a deterministic BTC issuance
curve, changing supply providers, using a present-day historical series, or
forward-filling the field would alter the proposed source and causal-clock
contract after the provenance defect became known. Those substitutions are
forbidden repairs. The OI/supply leverage axis is not preregistered and will
not be reopened in this research round.
