# BFC-3: Blockspace Fee Confirmation preregistration

Status: **frozen before any BFC-3 post-entry return was inspected**.

## Hypothesis

BFC-3 is a price-independent Bitcoin blockspace-demand clock. It combines:

- transactor-paid fees relative to protocol issuance, and
- transactions per main-chain block.

Coin Metrics' official catalog defines `FeeTotNtv` as transactor-paid fees
excluding issuance, `IssTotNtv` as newly issued native units, `BlkCnt` as
main-chain blocks, and `TxCnt` as ledger transactions:
<https://community-api.coinmetrics.io/v4/catalog/metrics?metrics=FeeTotNtv%2CIssTotNtv%2CBlkCnt%2CTxCnt%2CAssetEODCompletionTime>.
API behavior is documented at <https://docs.coinmetrics.io/api/v4/>.

The economic claim is narrow: unusually high fee-funded security revenue is
more credible as a demand signal when block transaction density is not weak.
That joint state should precede a short multi-day BTC demand continuation.

## Frozen singleton

Policy `BFC-3`:

1. `fee_share = log(FeeTotNtv / IssTotNtv)`.
2. `transaction_density = log(TxCnt / BlkCnt)`.
3. Standardize each current daily value against the last 180 strictly earlier,
   already-published observations; require 120.
4. `composite = fee_share_z + 0.5 * transaction_density_z`.
5. Eligible state:
   - `fee_share_z >= 1.0`,
   - `transaction_density_z >= 0.0`,
   - `composite >= 1.5`, and
   - publication lag no more than three days.
6. Emit only the first eligible day after an ineligible day.
7. Long at one complete five-minute latency bar after the first five-minute
   open at or after `AssetEODCompletionTime`.
8. Hold 864 five-minute bars (three days), with no overlapping trade.
9. Use 0.5x exposure, 6 bp/notional/side base cost, and 10 bp/notional/side
   stress cost.

There is one candidate and no threshold, side, latency, or hold repair.

## Leakage boundary

- Signal construction loads no market, price, funding, premium, OI, exchange
  flow, or other derivative feature.
- The source file is physically limited to observations before 2024.
- `AssetEODCompletionTime` is the earliest semantic availability timestamp.
- Old backfilled rows may seed a reference after publication but cannot signal
  when their lag exceeds three days.
- Exchange inflow/outflow/supply metrics were deliberately excluded because a
  historical point-in-time address-tag vintage archive is unavailable.
- The downloaded network file is a frozen current vintage, not a complete
  revision archive. Live promotion requires forward-vintage parity.

## Outcome-blind support gate

The non-overlapping clock must have at least:

- 35 train events across 2021-2022,
- 14 in each train year,
- 14 in 2023,
- 5 in each 2023 half,
- no month above 20% of all events.

Failure rejects BFC-3 without loading post-entry outcomes.

## Frozen performance gate

Both 2021-2022 train and 2023 selection must have:

- positive absolute return,
- CAGR / strict MDD at least 3.0,
- strict MDD at most 15%,
- weekly-cluster sign-flip `p <= 0.10`,
- mean gross underlying move at least 30 bp,
- positive result at 10 bp/notional/side stress cost.

Both 2023 halves and a one-bar-delayed execution must be positive. CAGR uses
the entire wall-clock split, including idle cash. Strict MDD uses the global
pre-entry high-water mark, favorable-before-adverse held OHLC, funding, and
entry/exit/hypothetical-liquidation costs.

## Controls and promotion

Controls are exact direction flip, fee-only, density-only, low-fee mirror,
seven-day stale state, one-bar delayed entry, and year-stratified random clocks.
A component-only, mirror, or stale control passing every primary gate rejects
the claimed joint mechanism.

Only a performance pass opens orthogonality testing against all previously
frozen promoted/live/shadow sleeves. Promotion then requires low clock/PnL
overlap and positive marginal portfolio contribution. The 2024, 2025, and 2026
YTD outcomes remain sealed until those pre-2024 gates pass.
