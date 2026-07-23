# Bybit public-trade live parity v3 rejection — 2026-07-23

## Decision

Provisional **BSEA-24 is rejected as source-infeasible under its frozen
transport**.  The corrected prospective capture passed its host-UTC/raw-clock
contract, but Bybit REST recent-trade windows could not cover two observed
WebSocket trade bursts.  Archive verification, a BSEA event clock, a feature,
direction selection, and outcome evaluation remain unauthorized.

This result may not be repaired by dropping missing rows, changing the poll
interval, accepting sampled parity, switching endpoints, or retrying in a
quieter period.  Any such design would be a separately preregistered source
axis rather than BSEA-24.

Immutable rejection evidence:

- rejection result SHA-256:
  `493cde97193c3c837cda4ca2101c7d2068cab8972836fdb511a68b2b7b9fc5d5`
- rejection result manifest hash:
  `cb0cade10439c4ba083db528a65b4ffc04a55b46a6ffe7e29928f4a9642c8661`
- raw capture manifest SHA-256:
  `38027f767122c9f7d5a57ae5a5a0f1445525e63bb05fcd75981de42677f68e16`
- raw capture manifest self-hash:
  `34bf6d6eb26033ffc556c41cad5260c208942f8b3dd6f17e59bc56d5d4fe44d1`
- WebSocket raw SHA-256:
  `b63ec8ee4f8f2c260631e09ce02d8dec71b8f17827a8e3592d792e4b2f5b6937`
- REST raw SHA-256:
  `43b75634964d84bbb55eb0affd83ba3a87a5de7e157c340a0182b9c95490f483`

Raw public-trade payloads remain local and ignored.  They are retained only to
reproduce this source rejection and are not committed.

## Clock correction passed

The v3 run used one fixed non-shell Windows host-clock process and
`CLOCK_MONOTONIC_RAW`; no WSL process-clock fallback occurred.

| Clock check | Result |
|---|---:|
| 60-second preflight samples | 622 |
| preflight UTC reversals | 0 |
| capture clock samples | 14,175 |
| capture UTC reversals | 0 |
| nonincreasing raw-monotonic samples | 0 |
| missing/excess ledger entries | 0 / 0 |
| capture elapsed disagreement | 1.589823 ms |
| one UTC day | pass |
| provider clean close | pass |

Therefore the v1 clock failure did not recur and cannot explain the v3 source
rejection.

## Frozen REST cadence passed

The capture produced 601 REST snapshots.  Request starts stayed on the frozen
one-second raw-monotonic cadence:

| Request-start delta | Value |
|---|---:|
| minimum | 0.980141143 s |
| median | 0.999968823 s |
| p95 | 1.001284574 s |
| p99 | 1.001834973 s |
| maximum | 1.069814020 s |

REST response latency remained below 0.504 seconds.  Thus a scheduler stall or
missed one-second poll does not classify the failure.

## Source parity failed

| Metric | Value |
|---|---:|
| unique WebSocket IDs | 38,761 |
| unique REST IDs | 33,851 |
| exact common IDs | 32,945 |
| common-field mismatches | 0 |
| eligible WebSocket IDs missing from REST | 5,816 |
| interior REST IDs missing from WebSocket | 0 |
| adjacent REST pairs with zero overlap | 2 |
| conflicting duplicate IDs | 0 |

All 5,816 missing IDs clustered in exactly two Bybit server match-time
seconds: 4,659 and 1,157 missing IDs respectively.  The WebSocket source
recorded a maximum of 5,451 trades in one server match-time second and 2,449
in one 100-millisecond bucket.  Both affected seconds exceeded the frozen REST
`limit=1000` window.

The only zero-overlap transitions were REST ordinals 50→51 and 51→52.  Their
request-start deltas were 0.999905808 and 1.000050182 seconds, while their REST
ID windows jumped across 735 ms / 33,436 sequence units and 463 ms / 15,497
sequence units.  This is consistent with the documented 1,000-row recent
window being overwritten by source bursts between correctly timed polls.

The asymmetry is decisive: every eligible interior REST ID appeared in the
WebSocket set, and all 32,945 common IDs matched exactly on frozen fields, but
the bounded REST surface could not observe every eligible WebSocket ID.

## Outcome boundary and next axis

No archive row, BSEA event clock, candidate incidence, Binance comparator,
direction, market outcome, return, PnL, CAGR, or strict MDD was opened.  BSEA-24
is closed.  The alpha search must move to a new orthogonal source axis whose
historical and live surfaces can be proven complete before feature or outcome
inspection.
