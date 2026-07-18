# Binance Spot/USD-M minute-dispersion source design — 2026-07-19

## Purpose

This source adds a causal axis that is absent from the stored five-minute
cross-venue frame: how activity, average ticket, and signed flow are distributed
across the **five completed one-minute aggregates** inside a bar.

It does not claim trade-level reconstruction. In particular, the HHI fields are
HHI across five minute totals, not HHI across individual fills.

## Source and timing contract

- official Binance Spot and USD-M monthly `BTCUSDT` one-minute kline archives;
- archive `.CHECKSUM` verification before parsing;
- exact UTC `open_time` join;
- exactly five source rows per output bar;
- no interpolation of a missing or invalid minute;
- feature availability and earliest execution time are `bar_open + 5 minutes`;
- raw ZIP bytes are discarded after the monthly artifact is built;
- 2024+ source construction fails closed unless `--open-oos` is explicitly used
  after a pre-2024 candidate has been frozen.

## Descriptor groups

For both Spot and USD-M the builder emits:

- quote-notional, trade-count, and absolute signed-flow time HHI;
- quote-HHI minus trade-count-HHI, a minute-level large-ticket concentration
  proxy;
- aggregate mean ticket, log-ticket standard deviation/range, and ticket timing
  centroid;
- net taker-flow fraction, sign persistence, and adjacent-minute sign-switch
  rate;
- flow-signed price impact and impact per absolute net-flow fraction.

Cross-venue differences and net-flow sign agreement are emitted explicitly.
All accepted feature rows are finite and source-complete.

## Research boundary

The source artifact is outcome-blind. Candidate direction, entry, hold, cost,
funding, strict-MDD, and train/test/eval rules must be frozen separately before
opening return outcomes. Existing cross-venue leadership artifacts are not
overwritten or reinterpreted.
