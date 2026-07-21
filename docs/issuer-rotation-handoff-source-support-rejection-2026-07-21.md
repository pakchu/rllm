# IRH-36 source-support rejection — 2026-07-21

## Verdict

**Retire IRH-36 before pair incidence, comparator novelty, or economic
outcomes.**

The frozen SHORT template requires a large `usdt_eth:redeem` event paired with
a large `usdc_eth:mint`. A redeem can first qualify for the frozen 90th-percentile
tail only after 32 strictly prior same-type rows. Therefore at least 33 USDT
redeem rows are necessary for even one SHORT tail event.

The independently replayed 2020–2023 source contains **3** USDT redeem rows.
The maximum possible qualifying SHORT tail count is exactly zero.

## Earliest failed gates

| Gate input | Frozen requirement | Observed | Result |
|---|---:|---:|:---:|
| Strictly prior USDT redeem rows | 32 | at most 2 before the last event | fail |
| Total rows needed for first SHORT tail | 33 | 3 | fail |
| Maximum possible SHORT tail events | at least 1 | 0 | fail |
| LONG/SHORT support balance | each at least 30% | SHORT impossible | fail |

Because this logical impossibility precedes pair construction, the source gate
did not read any source CSV row, derive any accepted pair, open any SQFD/SDDR/
UCBR or supply-breadth comparator timestamp, or read BTC price, funding, future
return, PnL, absolute return, CAGR, or strict MDD.

## No-repair rule

The failure cannot be repaired under the IRH-36 name:

- reducing the 32-row warmup would replace the frozen tail estimator;
- removing the SHORT template would violate the frozen side-balance contract;
- relabelling `DestroyedBlackFunds` as ordinary redemption would be a semantic
  error because confiscation is not customer redemption;
- using USDT transfers or another chain would introduce a new observable; and
- selecting a different pair window, quantile, ratio, or direction after seeing
  the counts would be post-source optimization.

The Ethereum event source remains valid and promotion-eligible as a source.
Only IRH-36 is rejected. A later candidate must be independently frozen and
must use a mechanism that is structurally supported by the observed event
types without reopening this rule.

## Integrity anchors

- mechanism freeze commit: `a418281`
- source promotion commit: `d367303`
- source gate evaluator commit: `323935f`
- source CSV SHA-256:
  `70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901`
- source manifest hash:
  `a0c7740db64f7779fade68d76985c629cabe81983bf594e8258cef16a5725a1b`
- source-gate report:
  `results/issuer_rotation_handoff_source_gate_2026-07-21.json`
- source-gate report SHA-256:
  `277793990f9e8935d1b7fbd9bccbe7a7addbd4ffb4af24e8b21016f42a40cc57`
- source-gate manifest hash:
  `5c018b45fabddf6c645ead58a84b914840e136b4d39da1f8849e9d092ae8306e`

