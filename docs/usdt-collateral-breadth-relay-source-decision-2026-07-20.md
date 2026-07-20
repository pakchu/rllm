# USDT Collateral Breadth Relay source decision — 2026-07-20

## Decision

The next BTC alpha candidate will test a previously unused observable:
**direct stablecoin prices against USDT**, without using BTC prices or BTC-book
flow in the source clock. The provisional candidate is **UCBR-12** (USDT
Collateral Breadth Relay, 12-hour hold).

This document freezes only the source axis and economic direction. It does not
freeze a threshold or calculate a real event clock. No BTCUSDT perpetual OHLC,
funding, future return, label, PnL, absolute return, CAGR, or MDD was opened.

## Observable and economic hypothesis

The source basket is:

```text
USDCUSDT, TUSDUSDT, USDPUSDT, FDUSDUSDT
```

For each completed common UTC hour, the retained observable is the logarithm
of the direct stablecoin/USDT close. A positive move means one unit of the
alternative stablecoin buys more USDT, so USDT is relatively weak in that
pair. A negative move means relative USDT strength.

The falsifiable BTC mechanism is a **collateral-demand relay**:

- broad relative USDT strength across independent issuers indicates demand for
  the dominant Binance collateral and provisionally maps `LONG BTCUSDT`;
- broad relative USDT weakness indicates USDT-specific collateral outflow or
  stress and provisionally maps `SHORT BTCUSDT`.

Issuer-specific depegs, exchange promotions, tick rounding, and thin stablecoin
books can produce the same prices without a BTC implication. A later singleton
rule must therefore require cross-issuer breadth, reject stale/zero-volume
members, and include leave-one-issuer-out, latency, direction, SDDR, and SQFD
clock controls. The economic language is a hypothesis, not an exchange claim.

## Official source and live parity

The source is Binance's checksum-published monthly Spot hourly-kline archive:

- public-data schema and checksum policy:
  <https://github.com/binance/binance-public-data>
- archive root:
  <https://data.binance.vision/?prefix=data/spot/monthly/klines/>
- exact archive pattern:
  `https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1h/{SYMBOL}-1h-{YYYY-MM}.zip`
- checksum pattern: archive URL plus `.CHECKSUM`;
- live symbol metadata:
  <https://data-api.binance.vision/api/v3/exchangeInfo>
- live Spot kline endpoint:
  <https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data>

On 2026-07-20, `exchangeInfo` reported all four symbols as `TRADING` with Spot
trading enabled. A production source can therefore reconstruct each completed
hour from live Spot klines and must fail closed when fewer than the frozen
breadth of final rows are available. Historical ZIP publication time is not a
live decision timestamp; source hour `[h,h+1h)` becomes usable only at `h+1h`.

## Outcome-blind 2023 coverage probe

Twenty monthly ZIPs covering August–December 2023 were downloaded to memory,
verified against their official SHA-256 companions, inspected, and discarded.
The probe calculated only source integrity and coarse range statistics.

| Symbol | Rows | Missing / duplicate hours | Positive quote-volume hours | Close range | ZIP bytes |
|---|---:|---:|---:|---:|---:|
| `USDCUSDT` | 3,672 | 0 / 0 | 100.00% | 0.9989–1.0021 | 147,795 |
| `TUSDUSDT` | 3,672 | 0 / 0 | 100.00% | 0.9955–1.0043 | 146,095 |
| `USDPUSDT` | 3,672 | 0 / 0 | 98.75% | 0.9969–1.0123 | 110,682 |
| `FDUSDUSDT` | 3,672 | 0 / 0 | 100.00% | 0.9976–1.0100 | 146,292 |

The common initial source prefix is `[2023-08-01, 2024-01-01)`. Official
checksum files were also confirmed for January 2024, January 2025, and June
2026 for all four symbols, providing the required later-history path. Complete
later rows remain unopened until a prior frozen stage authorizes them.

The 46 zero-quote-volume `USDPUSDT` hours are a real source-quality warning.
They cannot be imputed as active evidence. A source builder may retain only a
per-symbol current-hour validity flag alongside the log close and must discard
raw OHLC, volume, trade count, and taker-flow values after validation.

## Repository novelty

An exact repository search found no prior reference to any of the four direct
stablecoin-pair symbols. Existing stablecoin candidates are materially
different:

- SQFD uses BTCUSDT/BTCUSDC/BTCFDUSD hourly BTC volumes and signed taker flow;
- SDDR uses simultaneous BTC cross-quote price ratios and was rejected before
  outcomes because of six-hour clock proximity to an SQFD control;
- existing premium, basis, FX, and kimchi features do not observe direct
  stablecoin/USDT prices.

UCBR is admissible only if its frozen source clock proves low overlap with both
SDDR and SQFD. A different formula alone is not sufficient independence.

## Frozen research sequence

1. Implement and test a checksum-bound 2023 source builder that persists only
   direct log closes, current-hour validity, timestamps, and breadth.
2. Build the source twice and commit byte-identical source/manifest artifacts.
   Do not calculate a real UCBR event clock in that work unit.
3. Commit one low-multiplicity breadth rule, direction, hold, scheduler,
   support gates, and novelty controls before opening event incidence.
4. Reject without repair if support, side balance, calendar dispersion,
   source-quality, SDDR overlap, or SQFD overlap fails.
5. Only a source-support pass may authorize a separately committed strict
   evaluator, followed by sequential train 2023, test 2024, evaluation 2025,
   and final 2026H1 outcome stages.

An LLM/RL component remains downstream. It may later consume symbolic
issuer-breadth state for abstention or risk routing only after a deterministic
base clock proves gross edge above costs; it may not invent missing prices,
repair timing, or select a threshold after outcomes.
