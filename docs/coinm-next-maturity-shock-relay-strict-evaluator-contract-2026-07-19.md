# CMSR-36 strict evaluator contract — 2026-07-19

## Purpose

Freeze the complete outcome-evaluation path before reading any BTCUSDT return or
funding cash flow. The 2023 test remains physically sealed until the unchanged
train report passes every preregistered gate.

## Frozen execution

- Instrument: Binance USD-M BTCUSDT perpetual.
- Signal source: completed COIN-M front/next quarterly paths only.
- Feature availability: signal open plus 5 minutes.
- Entry: signal open plus 10 minutes, leaving one complete 5-minute bucket empty.
- Exit: entry plus 36 completed 5-minute bars (3 hours).
- Sizing: fixed 0.5x notional.
- Costs: 6 bp/notional/side base and 10 bp/notional/side stress.
- Funding: exact recorded funding rate and settlement mark.
- Calendar: every idle second remains in CAGR's denominator.

## Strict drawdown

The path starts from the global high-water mark and retains it across idle
periods and trades. For every trade it records, in order:

1. actual entry cost;
2. each funding settlement mark;
3. each held 5-minute bar's favorable extreme and then adverse extreme;
4. a virtual exit cost at every adverse mark;
5. actual exit cost.

Interior funding is symmetric. At exact entry or exit timestamps, a funding
credit is discarded and a debit is retained. The settlement mark is still
visited even when a boundary credit is discarded, so funding timestamp
ambiguity cannot hide drawdown.

## Sequential opening

1. `--freeze` reads only preregistration, source-support artifacts, and the
   frozen clock ledger. It records zero parsed OHLC rows, zero funding rows, and
   zero simulations.
2. `--train` opens only the hash-bound monthly OHLC files from 2020-08 through
   2022-12. Funding uses a calendar-derived exact row slice whose iterator ends
   at the final 2022 settlement without requesting the first 2023 row.
3. `--test` requires an exact replay of a hash-bound passing train report before
   opening the separately listed 2023 monthly files and funding row slice.
4. 2024 and later remain sealed.

## Train gates

- positive absolute return;
- CAGR / strict MDD at least 3.0;
- strict MDD at most 15%;
- at least 90 trades;
- two-sided weekly-cluster sign-flip p-value at most 0.10;
- positive absolute return in each frozen half-year;
- positive absolute return and CAGR / strict MDD at least 2.5 at 10 bp/side;
- primary CAGR / strict MDD exceeds every mechanism control by at least 0.25.

No failed gate may be repaired with a different direction, threshold, hold,
latency, sizing, cost, or subperiod after train outcomes are opened.
