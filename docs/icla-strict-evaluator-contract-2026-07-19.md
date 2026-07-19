# ICLA-60 strict staged evaluator contract — 2026-07-19

This contract freezes outcome accounting before any ICLA-60 train, test, or
eval market/funding frame is parsed. The evaluator is write-once. It physically
splits the already-frozen source clocks and may open only one chronological
stage at a time.

## Inputs and stage isolation

- Primary clocks are the source-only ICLA-60 artifact with byte hash
  `a55c23a7a0c296b98bb7a8958f713548c4313c0c682f1693c8f8be80b70dd053`.
- Execution OHLC and exact funding events reuse the checksum-verified official
  Binance USD-M files in `clbr_execution_sources_2023_2024`. Reuse changes no
  strategy logic; those files were built without loading either CLBR or ICLA
  clocks and contain a complete five-minute grid for the same three dates.
- Freeze verifies file hashes but does not parse any execution or funding
  frame. `train` is opened first. A failed train keeps test/eval sealed; a
  failed test keeps eval sealed.
- Stage files and results are immutable: creation uses exclusive writes and
  every later stage recomputes and verifies every prior result from frozen
  bytes.

## Execution and full-calendar metrics

- Exposure is `1.0x`; quantity is fixed at
  `pre_entry_equity / entry_open` for the whole trade.
- Enter at the frozen next-five-minute open and exit exactly 12 held bars later
  at the planned-exit open. There is no stop or take profit.
- Base cost is `6 bp/notional/side`; stress cost is
  `10 bp/notional/side`.
- Exact interior funding events are symmetric. At an exact entry or exit
  boundary, funding credits are dropped and funding debits are retained. This
  is deliberately conservative under settlement/execution ordering ambiguity.
- Absolute return compounds all net trade cash flows. CAGR uses every second
  from the declared stage start to exclusive stage end, including idle time.
- Reports always include absolute return, CAGR, strict MDD, CAGR/strict-MDD,
  trades and side counts, gross and net mean trade return, win rate, exposure,
  costs, and funding cash.

## Hardened strict MDD

The strict equity path starts from the global/pre-entry high-water mark and
applies, in order:

1. entry fee;
2. conservative funding cash and its exact settlement mark;
3. for every held five-minute bar, the favorable OHLC extreme first;
4. then the adverse OHLC extreme, including a virtual exit fee at that mark;
5. planned-exit open and the actual exit fee.

No post-exit high/low is included. The favorable-before-adverse convention is
the worst admissible intrabar ordering for drawdown. Virtual adverse-mark exit
cost prevents MDD from omitting the cost required to liquidate the position at
the observed adverse price.

## Statistical contract

- One-sided circular stationary trade-block bootstrap under a centered null.
- Mean block length: four trades.
- Resamples: 50,000; seed: 20,260,719.
- Reported p-value uses the finite-sample correction
  `(1 + exceedances) / (50,000 + 1)`.

All stages require positive base return, positive 10bp/side stress return,
strict MDD at most 15%, and bootstrap `p <= 0.10`.

| Stage | Minimum CAGR/MDD | Minimum trades | Minimum long | Minimum short |
| --- | ---: | ---: | ---: | ---: |
| train | 2.0 | 25 | 8 | 8 |
| test | 2.0 | 90 | 20 | 20 |
| eval | 3.0 | 90 | 20 | 20 |

## Frozen mechanism controls

The evaluator freezes four outcome-free controls alongside the primary clocks:

1. exact direction flip at every primary entry;
2. liquidation-wave fade with the USD-M absorption requirement removed;
3. primary entry and exit delayed by one additional five-minute bar;
4. deterministic random non-overlapping hourly clocks preserving each stage's
   primary trade count and long/short count.

Each control uses the identical execution, funding, cost, MDD, CAGR, bootstrap,
and stage gate. Primary promotion additionally requires every control to fail
the complete gate for that stage. No control parameters or random seed may be
changed after any stage outcome is opened.

CLBR alias rejection remains a separate source-only precondition rather than a
fifth market-outcome control. The evaluator freeze reproduces the support
artifact's exact-entry Jaccard, requires `support_passes=true`, and requires
`0.007662835249042145 <= 0.10`. Both booleans are copied into every stage's
promotion checks, so a forged or changed support/overlap artifact cannot be
promoted even if its performance gates pass.
