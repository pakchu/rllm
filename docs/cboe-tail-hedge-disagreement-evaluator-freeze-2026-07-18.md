# CTHD-1 strict evaluator freeze — 2026-07-18

Status: **sealed before parsing any BTC OHLC or funding outcome**.

The evaluator is bound to:

- support commit `95a3fcd`
- preregistration manifest
  `4daf843d48b7fcc259c3f5a6bc533e74a3ae94ffd8a172fef49c0bb8ad8ddb91`
- support manifest
  `57b7a756851c6cf86e7948d4a8a2f66f2a528b263587b362d5625bf859651339`
- strict execution engine SHA-256
  `e309f5217f033d57d2eadfec936843e736ce287f5c47f957c0ac6f0c71879c23`
- evaluator source SHA-256
  `7bdb67fc82b46cfbcca8bdd076b196cf84a9bca9662dd12223b8508939ec6fd5`

Frozen clocks are the primary hidden-pressure schedule, four source-component
controls, exact direction flip, one-release delay, and seven-release placebo.
The primary Stage-1 physically contained schedule has 156 short trades; the
one source event crossing the `2023-01-01` boundary is excluded rather than
partially evaluated.

Freeze diagnostics:

- execution OHLC rows parsed: `0`
- funding rows parsed: `0`
- simulations run: `false`
- mutable parameters: none
- opened windows: none
- sealed windows: Stage 1 (2021–2022), Stage 2 (2023), and 2024+

Integrity anchors:

- freeze manifest hash:
  `7d8e08053bfebe85dfb973818f810427fde80e1025c10eb6c6e464b126866018`
- freeze JSON SHA-256:
  `fb4661314b634d76250df00f63082dbfb612a53ef7a1b694888087c27b1f7018`

Only `--stage1` may now parse the physical interval
`[2021-01-01, 2023-01-01)`. The `--stage2` path fails before loading market data
unless the byte-identical Stage-1 report passes every frozen gate.
