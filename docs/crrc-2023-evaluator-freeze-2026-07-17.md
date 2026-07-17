# CRRC-72 strict 2023 evaluator freeze — 2026-07-17

The strict evaluator and its tests are frozen at commit
`1ba0e8ca50e870958969d5e3f7d129e0b28ce0fb` before any 2023 return is
calculated.

Frozen properties:

- primary event clock: 156 events, immutable hash
- five mechanism-control clocks: diagnostic only, no reranking
- execution: BTCUSDT USD-M only, 0.5x gross
- costs: 6bp per notional side; 10bp stress
- funding: exact settlement timestamp/rate, interval `(entry, exit]`
- strict MDD: global/pre-entry HWM, entry and hypothetical liquidation costs,
  favorable-before-adverse held OHLC, exact funding cash, and exit costs
- CAGR: full declared calendar; calendar 2023 is exactly one year
- output paths: immutable and created exclusively once
- sealed windows: 2023, 2024, 2025, 2026

The freeze itself loaded zero market rows, zero funding rows, and ran no
simulation. An independent verifier must pass before the one-shot 2023 open.
