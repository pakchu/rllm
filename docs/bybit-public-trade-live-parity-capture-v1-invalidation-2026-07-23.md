# Bybit public-trade live-parity capture v1 invalidation — 2026-07-23

## Decision

The first prospective Bybit REST/WebSocket capture is **invalid and
non-authoritative**.  Its emitted
`REST_WS_PARITY_PASS_PENDING_ARCHIVE` decision may not authorize archive
verification, a BSEA mechanism, a feature clock, or any outcome evaluation.

This is an implementation failure, not a Bybit source rejection.  The frozen
capture contract required rejection on local clock reversal, but the v1 code
only checked whether timestamps remained on the same UTC date.  It therefore
reported PASS despite clock evidence that violated the contract.

## Preserved v1 evidence

- capture manifest file SHA-256:
  `2c554b15479cf2053f2bbcac5f64cd378fcaa4070653af36805f4c1ce0e5ddb1`
- capture manifest self-hash:
  `aad20f8478d3567bf65cdafe7575589718c0a97aaa22e2e1ff253c370765b560`
- v1 capture script SHA-256:
  `efbce8e773679304396b091c3c63e4352f0cc8144f3eacab12d91fb36305fc1a`
- WebSocket raw artifact SHA-256:
  `5dc400d4601dfe0cf123861818194d640c12d507e37e11faf9f4210b52f43983`
- REST raw artifact SHA-256:
  `8c6cadc7add1c7424d8027dbbdff7d211e06ce5d64a5dff7a6a71f286069ce55`
- immutable invalidation artifact SHA-256:
  `d7eb515b4571a767d5efebc18ea179ebdcdd0f345c425bf2840755807f32dcbf`
- invalidation manifest hash:
  `78c0b14d552b090f923e2ca445e31184fed60f46a203f6fca4873871c741c4cb`

The raw public-trade files remain local and ignored.  They are retained only
to reproduce this transport audit and are forbidden as parity input for a
later run.

## Independent clock audit

The audit sorted every recorded WebSocket receipt and REST request/response
clock pair by monotonic time:

| Metric | Value |
|---|---:|
| clock events | 3,031 |
| monotonic elapsed | 604.125409051 s |
| UTC elapsed | 591.297083281 s |
| adjacent UTC reversals | 11 |
| maximum adjacent UTC/monotonic disagreement | 1.473088890 s |

Thus v1 failed an explicit frozen gate.  The fact that 4,753 exact REST/WS IDs
otherwise had zero field mismatches and zero interior omissions is retained
only as debugging evidence.  It cannot offset or waive clock failure.

## Outcome boundary

No BSEA event clock, candidate incidence, Binance comparator, post-entry market
value, return, PnL, CAGR, or strict MDD was opened.  No archive row for the
capture day was opened.

## Required correction

Before another network run, the implementation must:

1. bracket each UTC read with monotonic reads and retain the midpoint plus
   sampling uncertainty;
2. combine WS and REST clock samples in monotonic order;
3. fail the manifest on any UTC reversal rather than merely a date change;
4. expose clock-integrity counts and elapsed-time disagreement in the manifest;
5. cover the failure with synthetic cross-task ordering tests; and
6. commit and independently review the correction before a wholly fresh
   600-second capture.

The v1 rows cannot be relabeled or repaired in place.
