# Stablecoin denominator-dislocation source decision — 2026-07-20

## Decision

The next BTC candidate will test a previously unused observable: the relative
quote-denominator value implied by simultaneous Binance Spot hourly closes in
`BTCUSDT`, `BTCUSDC`, and `BTCFDUSD`.

The provisional candidate name is **SDDR** (Stablecoin Denominator Dislocation
Reversion). This document freezes only the source axis and evidence boundary.
It does not freeze a trading rule and opens no BTCUSDT perpetual outcome,
funding cash flow, return, PnL, CAGR, MDD, or post-entry excursion.

## Source observable

For a completed common UTC hour `h`, the source builder may calculate only:

```text
usdc_vs_usdt(h)  = log(BTCUSDC_close(h) / BTCUSDT_close(h))
fdusd_vs_usdt(h) = log(BTCFDUSD_close(h) / BTCUSDT_close(h))
alt_consensus(h) = median(usdc_vs_usdt(h), fdusd_vs_usdt(h))
alt_disagreement(h) = abs(usdc_vs_usdt(h) - fdusd_vs_usdt(h))
```

The persisted source panel must discard every raw open, high, low, close,
volume, quote-notional, trade-count, and taker-flow field after validating the
official archive schema. It may retain only the two cross-price log ratios,
their contemporaneous consensus/disagreement, exact source timestamps,
availability, and completeness flags.

The ratios mostly cancel the common BTC/USD price component. They are treated
as an inferred relative stablecoin-denominator observable, not as an official
exchange peg or FX rate. The source may still contain residual BTC-market
microstructure because the three books are not sampled trade-for-trade; that
claim must be tested by controls rather than assumed away.

## Why this is a new axis

The existing stablecoin-quote experiment, SQFD-6, retained only hourly BTC base
volume, trade count, taker-buy volume, taker-sell volume, and signed taker flow.
Its source contract explicitly discarded all price fields. SDDR uses none of
those flow observables. It studies cross-quote price-denominator dislocation,
not alternative-book flow diffusion.

Repository search on 2026-07-20 found no prior alpha, beta feature, or support
clock based on `BTCUSDC / BTCUSDT` or `BTCFDUSD / BTCUSDT` price ratios. SDDR
must nevertheless include SQFD clock overlap and flow-only controls because the
same books can react to the same demand shock.

## Official history and live parity

The source reuses the checksum-verified archive family already audited in
[`binance-stablecoin-quote-flow-source-audit-2026-07-19.md`](binance-stablecoin-quote-flow-source-audit-2026-07-19.md):

- Binance public-data schema and timestamp policy:
  <https://github.com/binance/binance-public-data>
- official monthly Spot kline archive root:
  <https://data.binance.vision/?prefix=data/spot/monthly/klines/>
- archive pattern:
  `https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1h/{SYMBOL}-1h-{YYYY-MM}.zip`
- published checksum pattern: archive URL plus `.CHECKSUM`
- live exchange metadata:
  <https://data-api.binance.vision/api/v3/exchangeInfo>

Frozen common-history boundary:

- `BTCUSDT` and `BTCUSDC`: `2023-07-01 00:00 UTC` onward;
- `BTCFDUSD`: first official row `2023-08-04 08:00 UTC` onward;
- initial source-only prefix: `[2023-08-04 08:00, 2024-01-01)`;
- later sequential replay may extend through `[2024-01-01, 2026-07-01)` only
  after the preceding frozen stage passes.

The complete common source therefore supports nearly three calendar years and
includes all of 2024, 2025, and 2026H1. It does not create a globally pristine
human holdout because this research programme has inspected those market years
in other alpha families.

At research time, an hourly row becomes available only after its exact close
timestamp. A production implementation must reconstruct the same completed
hour from live Binance Spot klines for all three active symbols and fail closed
if any row is missing, late, duplicated, or not final. Historical monthly ZIP
publication on a later date is not the live decision timestamp.

## Frozen research sequence

1. Build and checksum-audit only the initial common 2023 prefix. Do not load
   BTCUSDT perpetual OHLC, funding, future return, labels, or PnL.
2. Validate exact timestamp alignment, ratio algebra, stablecoin-book breadth,
   missing-row behavior, and source-only event incidence.
3. Commit one low-multiplicity SDDR policy with exact normalization, direction,
   trigger, hold, scheduler, controls, and stopping gates before reading any
   post-entry BTC price.
4. Commit and hash-freeze a strict outcome evaluator before opening the first
   permitted outcome stage.
5. Open stages sequentially and stop on the first failure. No threshold, side,
   holding-period, or feature repair is allowed under the same SDDR name.

## Model boundary

No LLM or RL model is authorized to generate an entry from this source before a
deterministic, source-only mechanism demonstrates adequate incidence and a
frozen base clock demonstrates positive gross edge above costs. If that base
survives, a compact Gemma/RLLM may later consume symbolic SDDR state only to
abstain, route risk, or explain the decision; numeric accounting and execution
remain deterministic. This prevents model capacity from hiding a source with
no economic edge.
