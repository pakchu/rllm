# HVLSRA-24 source support

Date: 2026-08-12

## Decision

The unchanged preregistered **HVLSRA-24** Powerball sales risk-appetite clock
passes every source-support gate and may advance only to Gross9 novelty.
No entry/exit price, funding settlement, post-entry return, PnL, Gross9 row,
CAGR, or drawdown was opened in this stage.

## Causal source

- Authority: official Texas Lottery Powerball Winner Summary text reports.
- Frequency: scheduled Monday, Wednesday, and Saturday draws.
- Observable: finite positive `NET SALES` printed in each report.
- Frozen decision: 12:00 UTC on the calendar day after the draw.
- Integrity: each report's embedded draw date, draw number, and local
  report-generation timestamp are parsed; late reports are ineligible.
- Signal sign: strict sign of the completed draw-to-draw log net-sales change.
- Volatility condition: strict-prior 180-event BTC variation midrank, minimum
  60 observations, current excluded, rank at least 0.65.

The deterministic snapshot contains 716 official reports from 2022-01-01
through 2026-07-29. The source report, source-derived panels, controls, and
primary clock are hash-bound by the support artifact.

## Gate result

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| 2023H2 train | 17 | 10 | 7 | 0.4118 | 0.4118 |
| 2024 test | 77 | 39 | 38 | 0.4935 | 0.1299 |
| 2025 eval | 39 | 21 | 18 | 0.4615 | 0.2051 |
| 2026 through July final | 36 | 21 | 15 | 0.4167 | 0.3056 |

Required minima are 8/12/12/8, minority-side share is at least 0.20, and
maximum month share is at most 0.45. All checks pass.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -B \
  training/build_high_volatility_lottery_sales_risk_appetite_relay_support.py
```

Two consecutive runs produced the same support artifact SHA-256:
`6c30ce499fe99d3442bdfa0c7a1aa9b89f44d5623af9c5bb5e7d2294e02cc371`.

The next authorized action is the frozen Gross9 novelty comparison. Economic
outcomes remain sealed unless that comparison passes.
