# IFDA-72 source-axis decision — 2026-07-20

## Decision

The next BTC candidate will use Binance USD-M **individual `trades` fill-size
distributions**.  The provisional identifier is **IFDA-72** (Individual-Fill
Dispersion Absorption, six-hour hold).

This document freezes only the source axis and the research boundary.  It does
not freeze a trading rule, inspect any post-entry BTC return, or claim that the
observable is profitable.  Exact features, action, scheduler, controls, support
floors, and rejection gates must be committed separately before source
incidence is opened.

## Why TCDA was stopped before preregistration

The alternative TCDA idea would have treated confirmed Ethereum
`usdt_eth:destroyed_black_funds` events as compliance shocks.  A source-only
audit of the immutable stablecoin panel found:

- 642 event rows but only **66 distinct UTC event days**;
- distinct event days by year: **18 / 16 / 16 / 16** for 2020–2023;
- **106 rows** on 2021-03-17 and 72 rows on 2022-10-06;
- 40 active months, 642 unique transactions, and 567 decoded addresses; and
- no zero-amount rows.

The contract rows are therefore heavily batched.  Counting the 642 rows as
independent observations would be pseudo-replication, while collapsing each
batch to an executable 48-hour event leaves at most 66 independent clocks over
four years.  A tail-shock threshold would reduce that count further.  This is
too weak for the project's long-window statistical contract, so TCDA is
rejected at source feasibility without opening BTC market data or outcomes.

`DestroyedBlackFunds` also must not be relabelled as customer redemption or
generic stablecoin contraction.  The event records destruction of a blocked
address's balance; the proposed short direction would have remained a
speculative compliance-risk hypothesis rather than a contract semantic.

Source panel:

- `data/ethereum_stablecoin_issuance_redemption_2020_2023/ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz`
- `results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json`
- <https://tether.to/en/legal/>
- <https://api-docs.hadron.tether.to/reference/patchtokenwipewalletaddress>

## New observable

Binance's official public-data specification exposes USD-M Futures `trades`
archives with:

```text
trade_id, price, qty, quoteQty, time, isBuyerMaker
```

Each archive has an adjacent published checksum.  The historical source is:

- <https://github.com/binance/binance-public-data>
- `https://data.binance.vision/data/futures/um/daily/trades/BTCUSDT/`

The raw rows permit five-minute, side-conditional distributions of individual
match notional `price * qty`: HHI, effective count, normalized effective count,
tail share, and cross-side dispersion asymmetry.  `isBuyerMaker=true` means the
aggressive side is sell; `false` means aggressive buy.

These are reported match records only.  IFDA may not infer participant
identity, parent-order identity, maker ownership, or resting-book depth from
them.

## Why this is not a repair of prior fragmentation candidates

The repository's existing Binance microstructure source is `aggTrades`, where
one row is an aggregate trade event.  Prior families used:

- MFIC: HHI across **aggregate-event** notional plus price-impact curvature;
- AFCS: aggregate-event compression and underlying trade-ID span;
- VTMS: quote notional divided by trade-ID count as an **average** ticket proxy;
- minute-packet topology: five one-minute aggregate summaries; and
- QLCD: exact quantity lattices on aggregate events.

None reconstructs the distribution of the individual trade records inside the
aggregate-ID spans.  IFDA's mandatory coarse control will recompute the same
dispersion statistic on `aggTrades`; individual-fill granularity must add
economic value over that control or the candidate is rejected.

The mechanism is motivated only at a broad level: price response depends on
available liquidity and on how persistent order flow is absorbed.  The cited
microstructure literature supports testing that mechanism, not IFDA
profitability:

- <https://arxiv.org/abs/cond-mat/0312703>
- <https://arxiv.org/abs/0804.3818>
- <https://arxiv.org/abs/1102.5457>

## Historical build contract

1. Cover `[2020-01-01, 2024-01-01)` only.
2. Download one daily ZIP at a time, verify its published `.CHECKSUM`, parse it
   in memory, and discard raw bytes immediately.
3. Never persist raw ZIP archives.  Write only deterministic monthly or final
   five-minute aggregates and a checksum-bound manifest.
4. Fail closed on missing days, duplicate or non-monotone trade IDs, unknown
   maker flags, malformed rows, checksum mismatch, or a source revision.
5. Record explicitly that no future OHLC, funding PnL, return, label, strategy
   result, or existing alpha outcome was read.
6. Check filesystem usage before every archive.  Abort before download when
   used space is at least **300 GiB**.  The decision-time usage was 287 GiB.
7. The source-only support stage must finish before any market outcome evaluator
   is implemented or opened.

## Live-parity boundary

Binance USD-M documents recent and historical individual-trade REST endpoints
(`/fapi/v1/trades` and `/fapi/v1/historicalTrades`) but its futures market
WebSocket catalog exposes aggregate trades rather than a raw-trade stream.
Production use is therefore conditional on a later gap-free replay test that
polls recent trades, catches up by trade ID, and reconciles every completed
five-minute bar.  A historical alpha pass cannot waive this operational gate.

If exact individual-fill continuity cannot be maintained within documented
rate limits, IFDA is research-only and may not enter the live portfolio.

## Next immutable sequence

1. Commit an exact single-policy preregistration and unit tests without reading
   source incidence.
2. Build and audit only the 2020–2023 raw-trade feature source under the disk
   guard above.
3. Run source-only support and clock novelty against the frozen microstructure
   comparator bundle.
4. Reject without repair on source, support, control, novelty, or live-parity
   infeasibility.
5. Only a frozen pass may authorize a separately committed strict economic
   evaluator, opening 2020–2022 first and keeping 2023 outcomes sealed.

