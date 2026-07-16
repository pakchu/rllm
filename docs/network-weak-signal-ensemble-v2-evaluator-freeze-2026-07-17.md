# NWE-8 evaluator pre-outcome freeze

The exact online model, controls, strict accounting, source hashes, and random
seeds were committed and frozen before any NWE-8 return label was constructed.

## Anchors

- Evaluator source commit:
  `8048870381da60f4a3fca2f540d506905d454421`
- Evaluator source SHA-256:
  `ccc04175f853caf4642d6a9d5b8670dc097553d883e02038bd4d32e51191bf64`
- Evaluator-freeze JSON SHA-256:
  `02557349b71a3f2216816d70c08c61e6e3869c0ca269107691e73d7f4c7bb124`
- Frozen feature-clock SHA-256:
  `3cc7eaa3b80944580651bf36541f0fde8edf4c66fd881d659f32396d1dda1c36`
- Pre-2024 evaluation-clock rows: `207` (`133` prediction-eligible).

At freeze time:

- labels constructed: `false`
- market rows parsed: `0`
- funding rows loaded: `0`
- simulations run: `false`
- mutable parameters: none

The immutable evaluator physically stops parsing market values before the first
timestamp on or after `2024-01-01`. It also drops the final support-only row
whose label would exit in 2024 before mapping any price.

Strict MDD uses the global/pre-entry HWM and a conservative intratrade envelope:
all favorable held OHLC and funding credits raise the peak before all adverse
held OHLC, funding debits, and hypothetical liquidation costs lower the trough.
Net realized funding is used at scheduled exit. Full split wall-clock time,
including abstained cash weeks, determines CAGR.
