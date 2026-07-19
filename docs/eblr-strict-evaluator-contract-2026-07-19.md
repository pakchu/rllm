# EBLR-60/30 strict staged evaluator contract — 2026-07-19

This contract freezes all outcome accounting and falsification controls before
any EBLR train, test, or eval market/funding frame is parsed. The evaluator is
write-once and may open only one chronological stage at a time.

## Inputs and stage isolation

- Primary source-only clocks SHA-256:
  `b4b35a0e9ae0cf26bf08df67b5c2fc832393c638c97f5b91a86894ee693b430e`.
- Source-only support artifact SHA-256:
  `58726a37bd632cccdf7a5320ec81b2f190f6448e1fd8392295d251683ab8ca17`.
- Execution OHLC and exact funding events reuse the checksum-verified official
  Binance USD-M stage files from `clbr_execution_sources_2023_2024`. Those
  files were built without loading EBLR clocks or computing strategy returns.
- Freeze verifies all file hashes without parsing an execution or funding
  frame. Train opens first. Failed train keeps test/eval sealed; failed test
  keeps eval sealed.
- Stage clock, control clock, freeze, and result artifacts are immutable and
  written with exclusive creation. Each later stage reproduces every earlier
  result from frozen bytes before it may open.

## Execution and calendar metrics

- Exposure: `1.0x` notional. Quantity is fixed at
  `pre_entry_equity / entry_open` for the entire trade.
- Enter at the frozen next-five-minute open and exit at the planned open after
  exactly **6 held bars / 30 minutes**.
- No stop or take profit.
- Base cost: **6bp per notional side**.
- Stress cost: **10bp per notional side**.
- Exact interior funding is symmetric. At an exact entry or exit boundary,
  credits are dropped and debits retained under settlement-order ambiguity.
  A dropped boundary credit still contributes its settlement mark to the
  strict-MDD path; only the favorable cash flow is omitted.
- Absolute return compounds every net cash flow. CAGR uses all seconds from the
  declared stage start to exclusive stage end, including idle periods.
- Every report includes absolute return, CAGR, strict MDD, CAGR/strict-MDD,
  trade and side counts, win rate, mean gross/net trade return, exposure,
  costs, and funding cash.

## Hardened strict MDD

The path begins from the global/pre-entry high-water mark and applies:

1. entry fee;
2. conservative funding cash and its settlement mark;
3. for every held five-minute bar, the favorable OHLC extreme first;
4. then the adverse OHLC extreme including a virtual exit fee at that mark;
5. planned exit open and actual exit fee.

No post-exit high or low is included. Favorable-before-adverse is the worst
admissible intrabar ordering for drawdown, and the virtual fee represents the
cost needed to liquidate at the observed adverse mark.

## Statistical contract

- One-sided circular stationary trade-block bootstrap under a centered null.
- Mean block length: four trades.
- Resamples: 50,000.
- Seed: 20,260,719.
- Finite-sample p-value: `(1 + exceedances) / (50,000 + 1)`.

Every stage requires:

- positive base absolute return;
- positive 10bp/side stress absolute return;
- CAGR/strict-MDD **at least 3.0**;
- strict MDD **at most 15%**;
- bootstrap `p <= 0.10`.

| stage | minimum trades | minimum long | minimum short |
|---|---:|---:|---:|
| train | 20 | 6 | 6 |
| test | 50 | 12 | 12 |
| eval | 50 | 12 | 12 |

## Frozen source-only preconditions

The evaluator reproduces and binds:

- `support_passes=true`;
- exact CLBR entry Jaccard `0.000 <= 0.10`;
- exact ICLA entry Jaccard `0.000 <= 0.10`;
- primary clock counts train/test/eval = 21/132/113.

These checks are copied into every promotion decision. A changed or forged
support artifact cannot promote even if the market metrics appear favorable.

## Frozen mechanism controls

All controls are generated and written before any market frame is opened and
use the same execution, funding, costs, MDD, CAGR, bootstrap, and complete stage
gate as the primary:

1. **direction_flip** — identical primary clocks with every side reversed;
2. **btc_only_direct_shock** — BTC's own q95 severity and imbalance trigger,
   no ETH trigger;
3. **no_btc_quiet_gate** — identical ETH trigger without the BTC quietness
   condition;
4. **delayed_5m** — primary entry and exit delayed by one additional bar;
5. **future_eth_placebo** — primary clock shifted 30 minutes earlier, so the
   ETH trigger is deliberately in the future; noncausal and never promotable;
6. **random_clocks** — deterministic non-overlapping clocks preserving each
   stage's primary month, side, and count distributions.

Primary promotion requires every control to fail its complete stage gate. The
future placebo can only veto the primary; it can never be selected or promoted.
No control threshold, seed, or clock may change after freeze.

## Interpretation stop condition

The archive is retrospective and ends in 2024. A complete train/test/eval pass
would establish a hardened historical candidate only. It still requires
forward live-shadow evidence before production allocation. Any failed stage
rejects this exact rule; test/eval thresholds may not be repaired.
