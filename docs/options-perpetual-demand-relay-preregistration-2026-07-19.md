# OPDR-24 preregistration — Options–Perpetual Demand Relay

## Decision boundary

Freeze one candidate before opening any exact-candidate post-entry outcome.
`OPDR-24` combines two weak, independently observed derivative states:

1. Deribit DVOL is unusually rich relative to Binance BTCBVOL;
2. the completed BTCUSDT premium-index path displaces efficiently in one
   direction.

The action is the sign of the premium move. The clock contains no BTC price,
BTC return, volume, funding, OI, macro/FX, or existing-alpha state.

This is not represented as a pristine global clean room. Other repository
research already observed 2023H2 outcomes. In particular, a separate
`dvol_rich_move_follow_v80_p80_h48` price-follow candidate reached 2.47
CAGR/strict-MDD on 29 trades and was rejected. `OPDR-24` does not repair that
clock: it removes BTC price/return from both trigger and direction, introduces
premium-path demand as a required independent source, fixes a 24-hour hold, and
reserves the exact OPDR 2023H2 path as train and calendar 2024 onward for
sequential OOS. Exact OPDR outcomes have not been opened.

## Frozen causal rule

At completed UTC-hour boundary `T`:

- `R[T] = log(BVOL_close[T] / DVOL_close[T])`;
- aggregate exactly 60 source-valid premium-index one-minute bars over
  `[T-1h,T)`;
- `M[T] = last_close - first_open`, in basis points;
- `E[T] = abs(M[T]) / sum(high-low)`, over those 60 minutes.

Each threshold uses only the preceding 720 completed hourly anchors, excludes
the current hour, and requires 672 valid joint observations:

- `R[T] <= prior q20`;
- `abs(M[T]) >= prior q80`;
- `E[T] >= prior q70`.

Only a false-to-true transition opens a clock. Side is `sign(M[T])`. BVOL and
DVOL are available at `T`; the last premium minute is conservatively available
at `T+1s`. Entry is the BTCUSDT open at `T+5m`. The position is held for exactly
24 hours, globally non-overlapping, at fixed 0.5x notional.

There is one direction, one threshold tuple, one hold, and no candidate grid.

## Source opening sequence

The preregistration binds the already frozen pre-2024 BVOL, existing DVOL, and
premium-only source hashes. Future BVOL and a refreshed DVOL cut may be
downloaded only after this preregistration commit. BVOL uses the
checksum-validating official Binance Vision builder over
`[2023-06-20, 2026-07-01)`; DVOL uses the frozen official Deribit downloader and
joins only on candle `close_time`. Missing archives or incomplete hours remain
invalid and are never filled.

The source-support builder may read BVOL, DVOL, and premium paths only. It may
not read BTC execution OHLC, future returns, funding cash flow, PnL, CAGR, or
drawdown. An outcome evaluator is written and hash-frozen only after support
passes.

## Support gates

| Window | Minimum non-overlapping events |
|---|---:|
| 2023H2 train support | 20 |
| 2023 Q3 / Q4 | 6 / 8 |
| 2024 test support | 40 |
| 2025 eval support | 40 |
| 2026H1 final support | 20 |

Every parent window must have at least 25% on each side. Maximum one-month
share is 35% in train, 20% in each full OOS year, and 30% in 2026H1.
Clock novelty is checked against the old price-follow family and the frozen
PSR, PCBR, and CMSR clocks using explicit common coverage windows.

Any support failure retires OPDR without opening candidate outcomes. No support
minimum, threshold, direction, latency, or holding period can be changed from
the observed counts.

## Sequential strict evaluation

If source support passes, open only 2023H2 train. Open 2024 only if unchanged
train passes every gate, open 2025 only if unchanged 2024 passes, and open
2026H1 only if unchanged 2025 passes. The declared train-through-final calendar
is exactly three years, `[2023-07-01, 2026-07-01)`.

Every opened OOS window must show absolute return, full-calendar CAGR, strict
MDD, CAGR/strict-MDD, and trades, and must satisfy:

- positive absolute return;
- CAGR/strict-MDD at least 3;
- strict MDD at most 15%;
- weekly-cluster two-sided sign-flip `p <= 0.10`;
- mean gross underlying move at least 20 bp;
- positive contained half-year returns;
- positive 10 bp/notional/side stress return and stress ratio at least 2.5;
- primary ratio at least 0.25 above every frozen mechanism control.

Strict MDD includes the global/pre-entry HWM, entry cost, conservative funding
boundary marks, favorable-then-adverse movement inside every held five-minute
bar, virtual adverse-mark exit cost, and actual exit cost. CAGR includes all
warm-up and idle wall-clock time.

## Controls and RLLM boundary

Frozen controls remove the options-volatility state, remove premium efficiency,
replace DVOL-rich with BVOL-rich, flip direction, delay one hour, or assign a
deterministic random side. Controls may falsify the mechanism; none may replace
the singleton candidate.

Only after deterministic passage may a compact LLM reason over the frozen
symbolic vol-disagreement and premium-path state to abstain or size. It may not
create a new signal, reverse the side, or repair a failed deterministic alpha.
