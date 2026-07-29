# PSIM-D8 CDP1 Untouched 2023 Future-Veto Pass

Date: 2026-07-29  
Frozen candidate: `CDP_S50_G05`  
Decision: `pass`

## Untouched result

| Cost | Return | Strict MDD | CAGR/MDD | Trades | Long / short |
|---|---:|---:|---:|---:|---:|
| 6 bp/side | +20.13% | 20.31% | 0.992 | 59 | 43 / 16 |
| 10 bp/side | +14.60% | 21.59% | 0.676 | 59 | 43 / 16 |

Every preregistered veto check passed:

- base net return was strictly positive;
- stress net return was nonnegative;
- base CAGR/MDD was at least `0.50`;
- strict MDD was no more than twice the 2022 selection MDD;
- at least 15 trades closed.

No threshold, direction, holding period, cost, source feature, or candidate was
changed after the 2022 top1 freeze.

## Authorization boundary

This pass authorizes only a separate preregistration for same-gross marginal
portfolio evaluation. It does **not** authorize:

- live trading;
- addition to Gross9 or any current live configuration;
- substitution for a currently deployed sleeve;
- portfolio performance claims before exact frozen Gross9 inputs return.

The original PPOSM future-veto branch remains paused because its exact frozen
cache files are still unavailable. Reconstructing those files is not an
acceptable substitute.

Result hash:
`8c720c0000f8743e75f415f2f09f0867e92d61caef26ac1b63525385e6010f03`.
