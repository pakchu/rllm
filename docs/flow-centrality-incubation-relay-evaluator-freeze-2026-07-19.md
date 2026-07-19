# FCIR-12 strict evaluator freeze — 2026-07-19

## Status

The `FCIR-12` evaluator and its six falsification controls are frozen. No BTC
execution OHLC or funding row was parsed, no execution-outcome bytes were
hashed, and no simulation ran during this freeze. All four outcome windows
remain sealed.

## Immutable evaluation contract

- Sequential opening: `train 2023 -> test 2024 -> eval 2025 -> final 2026H1`.
- Stop permanently on the first failed frozen gate; no control-guided repair.
- Position: BTCUSDT USD-M perpetual, `0.5x`, fixed `12h` hold.
- Base cost: `6bp` per notional side; stress cost: `10bp` per notional side.
- CAGR: full declared calendar, including warm-up and idle cash.
- Strict MDD: global/pre-entry HWM, entry cost, every exact funding settlement
  mark, every held 5-minute favorable-then-adverse path, virtual adverse-mark
  exit cost, and actual exit cost.
- Funding: interior exact-time events are symmetric; exact entry/exit credits
  are dropped while debits are retained. Entry `+47ms` is interior; exit
  `+47ms` is outside.
- Statistical test: deterministic weekly-cluster sign flip, 20,000 Monte Carlo
  draws after 20 clusters, seed `20260719`.

## Frozen clocks

Each clock family has 247 events: train/test/eval/final counts are
`62 / 90 / 61 / 34`.

1. Primary FCIR-12.
2. Direction flip on identical clocks.
3. Equal-weight-flow side on identical clocks.
4. Network weights stale by 24 hours.
5. Deterministic one-symbol permutation of network weights.
6. Deterministic random side on identical clocks.
7. Primary side with one additional hour of latency.

## Integrity identities

- Support commit: `a57fc55cf1507bc5d5cab33f3e5c936e7ac39e05`
- Evaluator SHA-256: `036b22442a2080e7ea5ffe914c605a9b1b1a55b128a315a2f2f05be7b37a736d`
- Control clock SHA-256: `3ffd7cd3b55aa81d3dbe4f46ec611c0209c947fcfac1c02cc9afc3274cc04711`
- Freeze file SHA-256: `05a08704b99086148752fee19d291b36ff412e9c53af72b3b6f0cf4c178ae204`
- Freeze manifest: `db449785021045afda156a1e1772c11b2b8bcdaf19db9714c3424f0e4e2e88d9`

The next permitted operation is one opening of the physical 2023 BTC execution
window. The 2024, 2025, and 2026H1 outcome sources remain unopened until every
prior stage passes all gates.
