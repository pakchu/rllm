# CCBS-12 preregistration — 2026-07-17

## Hypothesis and orthogonality

CCBS-12 tests a **hypothesized relative-value mechanism** in the same-maturity
USD-M versus COIN-M BTC current-quarter futures wedge. A rich USD-M leg is sold
against a long COIN-M leg; a rich COIN-M leg is sold against a long USD-M leg.
It has no explicit BTC-direction, perpetual funding/premium, OI, Kimchi/FX,
REX, Markov, tree, LLM, or regime gate. The physical panel comes from Binance's official
[continuous-contract kline endpoint](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#continuous-contract-kline-candlestick-data).

No entry-to-exit spread return or 2023 spread PnL was opened before this
protocol. Feature support counts, including 2023 counts, were inspected.
Therefore 2023 is honestly labeled **outcome-blind development**, not pristine
OOS. Threshold selection is restricted to 2021-2022; 2024 is the first code-
frozen source-and-outcome-unopened OOS year.

## Frozen feature and support-only selection

- `w = log(USD-M close / COIN-M close)`;
- all state resets on every delivery-contract segment;
- strictly prior 14-day/4,032-bar rolling median and recursive MAD, with 3,226
  prior observations required;
- state onset requires `abs(z) >= threshold` and at least a 20 bp deviation
  from the rolling center;
- grid: `1.5, 2.0, 2.5, 3.0`; using 2021-2022 only, choose the **largest**
  value with at least 130 events, 50 per year, 25 per half, 6 per quarter, and
  30% per sign;
- PnL, future OHLC paths, and return columns are forbidden in this selection.

The completed signal bar at `t` is available at `t+5m`; entry is delayed to
the `t+10m` open. Each accepted onset reserves a full 12 hours. The spread
exits after a completed `abs(z)<=0.5` trigger with the same one-full-bar delay,
or at 12 hours. No possible path may touch or cross delivery.

## Frozen accounting and strict risk

Before fees, each leg freezes face `F=0.5*pre-entry equity`. The research
ledger uses fractional USD-M BTC quantity `F/entry` and fractional COIN-M
contracts `F/100`; neither is rounded. USD-M uses linear PnL. COIN-M uses the
inverse coin formula and converts PnL with the **same** contemporaneous price
used in that mark. Entry costs do not resize either quantity. Both legs pay 6
bp per transaction side (10 bp stress); delivery futures pay no funding.
Strict MDD uses the global/pre-entry HWM, combines
cross-venue favorable extrema before adverse extrema, and includes hypothetical
two-leg liquidation cost at every adverse mark. CAGR covers the full calendar,
including warm-up and idle time, and absolute return is always reported.

## Hard collateral boundary

The research ledger deliberately measures **derivative spread PnL only**. It
does not model BTC collateral posted for COIN-M, transfers, margin interest, or
the liquidation engine. Therefore a historical pass is not proof of account-
level neutrality and is not live-promotable. It also omits executable integer
contract rounding. An exact BTC-collateral ledger, live contract constraints,
and either a causal collateral hedge or documented unified-margin treatment
are hard prerequisites.

## Sequential gate and no repair

The single pre-2023-support-selected policy first opens 2023 development PnL.
It must produce
positive absolute return, CAGR/strict-MDD >= 3, strict MDD <= 15%, at least 50
trades, positive H1/H2 and both spread branches, positive 10 bp stress, and
direct wedge-convergence attribution. Only then may the first untouched OOS
year, 2024, open, followed by 2025 and 2026 under the identical policy. Every
failure writes a terminal rejection artifact and seals later years. No sign,
lookback, threshold, floor, exit, cost, branch, sizing, or directional-gate
repair is allowed after outcomes open.

Entry-clock overlap is measured against the hash-bound gross-3.85 live anchor.
PnL orthogonality uses full UTC calendars with no-trade days zero-filled, the
hash-bound BTCUSDT daily close return, and a fixed 0.25 CCBS incremental-weight
portfolio comparison. These are gates to test the relative-value hypothesis,
not claims made in advance.

Protocol hash: `33b6d8e4dec120b9cf1177e4fa37695f9a0e485d8dd29872e9d403fd72eacc25`
