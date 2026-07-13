# Leveraged-inventory conservation residual alpha — 2026-07-13

## Idea

Open interest is a stock of leveraged claims.  Its change should be partly
explainable by completed price movement and the contemporaneous cost of holding
perpetual exposure.  The experiment fits, using data through 2022 only,

`Δlog(OI) = β0 + β1|return| + β2 return + β3 funding + β4 premium + β5 Δpremium + β6 |carry| + residual`.

A large positive residual means more leveraged inventory was created than price
and carry conditions normally explain.  The sign of a standardized
funding/premium carry composite estimates who owns that excess inventory:

- positive carry + positive inventory residual: crowded longs, trade short;
- negative carry + positive inventory residual: crowded shorts, trade long.

This is an accounting-residual hypothesis, not an OI threshold renamed as an
alpha.  Raw OI-change and carry-only versions are explicit controls.

## Frozen protocol

- Phase 1 physically truncated market, funding, premium and metrics before
  `2024-01-01`.
- Binance OI was delayed by one complete 5-minute source bar.
- Funding was backward-as-of joined; premium was available only at completed
  hourly `close_time`.
- Linear coefficients and every threshold used fit rows through 2022 only.
- Search/selection used 2023 and its two half-years; 2023 quarters were stability
  checks. No 2024+ row was loaded before the manifest was written.
- The manifest froze coefficients, the selected policy, control threshold,
  source audit, market/feature prefix hashes and signal hash.
- Full replay verified all prefixes before scoring OOS.
- Execution: following 5-minute open, 0.5x, 6bp/side, hourly candidate stride,
  non-overlapping fixed holds, split-contained exits, conservative strict OHLC
  high-water MDD.

Search: inventory window `{4h,12h,24h}` × positive-residual fit quantile
`{80%,90%,95%}` × absolute carry `{0.5,1.0,1.5}` × hold
`{12h,24h,48h}`. Controls were searched separately but could not replace the
primary residual policy.

## Frozen candidate

Selected primary: 24h inventory residual above fit q95, `|carry_z| >= 1`, fade
carry direction, hold 48h.

Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.

| Pre-2024 window | Result |
|---|---:|
| Fit through 2022 | `+23.93 / +10.18 / 21.59 / 0.47 / 68` |
| 2023 | `+28.50 / +28.52 / 7.39 / 3.86 / 38` |
| 2023 H1 | `+20.37 / +45.38 / 7.26 / 6.25 / 21` |
| 2023 H2 | `+7.05 / +14.48 / 7.39 / 1.96 / 16` |

All four 2023 quarters were positive, with minimum quarter ratio `2.22`; this
was enough for one exploratory frozen OOS opening despite the weak fit ratio.

## Untouched OOS

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +11.16% | +11.13% | 10.86% | 1.03 | 51 |
| Eval 2025 | -13.44% | -13.45% | 25.93% | -0.52 | 58 |
| 2026 through Jun 01 | -9.30% | -20.90% | 17.64% | -1.18 | 24 |
| Combined 2024–2026 | **-11.72%** | **-5.03%** | **29.61%** | **-0.17** | 134 |

Combined mean-trade evidence is null (`p≈0.600`, effect size `d≈-0.045`).
The apparent 2023 accounting edge was a transient regime fit.

## Matched controls, combined OOS

| Variant | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Primary, 10bp/side | -16.33% | -7.11% | 31.61% | -0.22 | 134 |
| Direction flip | -7.30% | -3.09% | 28.56% | -0.11 | 134 |
| Carry only | -31.47% | -14.47% | 44.64% | -0.32 | 420 |
| Raw OI-change + carry | -5.44% | -2.29% | 25.21% | -0.09 | 126 |

Residualization changes behavior and improves on carry-only, but neither the
selected direction nor its flip earns a positive OOS return. It therefore does
not identify a persistent ownership asymmetry.

## Decision

**Reject OOS; do not trade or retune this mapping.** Preserve the continuous
inventory-conservation residual and carry state as weak beta context only. The
exact q95/carry-1/48h static policy is gamma failure provenance. Any retry must
add new information about inventory ownership or debt transfer rather than tune
these inspected OOS periods.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_inventory_conservation_residual_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_inventory_conservation_residual_alpha.py
```
