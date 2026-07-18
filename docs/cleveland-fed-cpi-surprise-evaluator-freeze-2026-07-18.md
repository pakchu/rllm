# CFCS-1 strict evaluator freeze — 2026-07-18

## Decision

The CFCS-1 evaluator is frozen and may now open only the physical
`[2020-01-01, 2023-01-01)` Stage1 execution window.  No execution OHLC,
funding row, return, or simulation was opened during this freeze.

## Immutable evaluation contract

- singleton policy: concordant headline/core actual-minus-nowcast surprise;
- threshold: absolute equal-mean surprise at least 0.05 percentage points;
- direction: negative surprise LONG, positive surprise SHORT;
- execution: release-day 08:35–16:00 America/New_York;
- leverage: 0.5x;
- base/stress costs: 6/10 bp per notional side;
- exact funding on `[entry, exit)`;
- full-calendar CAGR including idle cash;
- strict MDD with global/pre-entry HWM and held-bar adverse OHLC;
- deterministic weekly clustered two-sided sign-flip test;
- all Stage1/Stage2, subperiod, trade-count, mechanism-control, and stress
  gates fixed in the preregistration;
- no mutable parameter.

## Sequential seal

1. Stage1 may physically parse only 2020–2022 market and funding rows.
2. Calendar 2023 remains physically inaccessible unless the stored Stage1
   artifact passes every gate and exactly replays under this evaluator hash.
3. Calendar 2024+ remains sealed regardless of Stage1/Stage2 outcome.
4. Portfolio orthogonality is inspected only after standalone Stage2 passes.

## Frozen identity

| Item | SHA-256 / value |
|---|---|
| Evaluator | `92aba5e648ee4a0ac7119d37a271edd86df99f4177b8a533d4338d0e88bb5ff2` |
| Freeze artifact | `4e53c9b5d890ee9f8f15b0f993340401b26c3b29b0f35ccdd74c701505b2b381` |
| Freeze manifest | `76f91543c284535dcf46f01c38e2bbb47f7192a57422c2d247198822f442feae` |
| OHLC rows parsed | `0` |
| Funding rows parsed | `0` |
| Simulation run | `false` |

Any evaluator-source change invalidates the freeze rather than creating a
repair path.
