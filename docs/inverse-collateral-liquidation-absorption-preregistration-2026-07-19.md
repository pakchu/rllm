# ICLA-60: inverse-collateral liquidation absorption preregistration — 2026-07-19

## Hypothesis

`ICLA-60` tests a cross-collateral forced-flow mechanism that has not been used
by the repository's previous liquidation policies. A one-hour COIN-M
liquidation wave is mechanical flow from inverse-collateral accounts. When
USD-M aggressor flow over the same completed hour points against that forced
flow, stablecoin-margined traders are absorbing rather than confirming the
cascade. The policy fades the forced side for one hour.

This is not `CLBR-24`. CLBR requires a single five-minute COIN-M burst followed
by a low-liquidation release bar, then holds two hours with a structural stop.
ICLA uses a 60-minute cumulative liquidation state, requires opposing USD-M
taker flow in the same fully completed window, enters without waiting for a
release bar, holds one hour, and has no stop.

## Frozen single policy

For each completed five-minute bar:

1. Sum COIN-M total and signed liquidation-snapshot USD notional over the last
   12 bars. All 12 source bars must be valid.
2. Compute the 90th percentile of positive 12-bar totals over the strictly
   prior 14 calendar days, requiring 200 positive observations.
3. Require current total at or above that threshold and absolute one-hour
   liquidation imbalance of at least `0.60`.
4. Aggregate USD-M taker quote flow over the same 12 bars:
   `imbalance = sum(2*taker_buy_quote - quote_volume) / sum(quote_volume)`.
5. Fade the forced side: long after dominant long-liquidation forced sells and
   short after dominant short-liquidation forced buys. Require the fade
   direction times USD-M taker imbalance to be strictly positive.
6. Both inputs become available at the current bar end plus one second. Enter
   at the following five-minute open, normally current bar start plus ten
   minutes.
7. Hold exactly 12 five-minute bars. Positions cannot overlap. There is no
   price stop, take profit, price feature, funding gate, OI gate, regime gate,
   tree, Markov model, REX rule, or LLM decision.

Thresholds were fixed from the economic object and source-only support shape;
no post-entry price, return, PnL, CAGR, drawdown, or funding cash was inspected.
There is one candidate and no repair after a stage outcome is opened.

## Chronological stages

| Stage | Start inclusive | End exclusive | Minimum clocks | Minimum per side | Maximum month share |
| --- | --- | --- | ---: | ---: | ---: |
| train | 2023-06-25 | 2023-10-15 | 25 | 8 | 35% |
| test | 2023-10-15 | 2024-04-15 | 90 | 20 | 25% |
| eval | 2024-04-15 | 2024-10-15 | 90 | 20 | 30% |

The train interval is only 112 days because the official COIN-M archive starts
on 2023-06-25. Test and eval are each 183 days. Even a historical pass remains
a forward-shadow candidate, not production proof.

## Strict performance gates

- Discovery exposure: `1.0x`.
- Base cost: `6 bp/notional/side`; stress: `10 bp/notional/side`.
- Exact realized funding is applied at its recorded timestamp.
- Strict MDD uses the global/pre-entry high-water mark, entry cost, every held
  favorable then adverse five-minute extreme, conservative funding-boundary
  handling, and virtual liquidation cost at each adverse mark.
- CAGR uses every second in the declared calendar, including idle periods.
- Every report includes absolute return, CAGR, strict MDD, CAGR/MDD, trades,
  side counts, mean gross/net trade return, win rate, and exposure.

Train must be positive, have CAGR/strict-MDD at least `2.0`, strict MDD at most
`15%`, at least 25 executed trades, both sides represented, positive 10bp-side
stress return, and one-sided stationary trade-block bootstrap `p <= 0.10`.
Failure keeps test and eval sealed.

Test must independently be positive with ratio at least `2.0`, MDD at most
`15%`, at least 90 trades, at least 20 trades per side, positive stress return,
and bootstrap `p <= 0.10`. Failure keeps eval sealed.

Eval requires positive return, ratio at least `3.0`, MDD at most `15%`, at
least 90 trades, at least 20 per side, positive stress return, and bootstrap
`p <= 0.10`.

## Frozen mechanism controls

- exact direction flip;
- same liquidation-wave fade without the USD-M absorption requirement;
- one additional five-minute execution delay;
- within-stage random clocks preserving trade count and side count;
- exact entry-clock overlap against CLBR-24, with Jaccard at most `0.10`.

The primary cannot be promoted if a component-only or timing control satisfies
the complete primary gate, or if the result is merely an alias of CLBR's clock.

## Leakage and source boundary

- The COIN-M source is Binance's censored force-order snapshot stream, not a
  complete liquidation-fill tape.
- USD-M activity is retained from checksum-verified official five-minute
  kline archives; price columns are not present in the activity artifact.
- Rolling wave thresholds use `shift(1)` and strictly earlier completed source
  windows.
- Signal construction reads no execution OHLC, funding, return label, strategy
  PnL, or later stage result.
- Missing COIN-M archive bars invalidate every 12-bar window touching them;
  they are never imputed as zero liquidation.
- All train/test/eval **source-only** clocks and support counts are intentionally
  frozen up front, because they contain no price, return, funding, or PnL.
- Evaluation market/funding files are physically split by stage. The evaluator
  may load only train first; test/eval outcome paths remain inaccessible until
  the preceding frozen result passes.

Historical 2024 BTC outcomes have been viewed elsewhere in the repository, so
test/eval prevent local parameter repair but are not pristine human holdouts.
Fresh versioned live snapshots and execution parity remain mandatory.
