# Funding-age rollover transfer alpha preflight — 2026-07-13

## Hypothesis

Immediate funding-settlement curl failed, so this experiment instead models
**debt cohorts that survive multiple settlements**. Positive delayed OI
additions are attributed to long/short cohorts from causal global-account and
aggressive-flow ownership state. Each cohort tracks:

- weighted entry price;
- number of funding settlements survived;
- cumulative side-specific funding burden;
- proportional OI-contraction survival and an exponential memory decay.

For old cohorts, burden is funding paid minus side-adjusted price return.
Ownership moving away from burdened longs produces a short transfer signal;
ownership moving away from burdened shorts produces a long transfer signal.
This differs from immediate curl and prior cost basis: settlement age and
cumulative carry burden are explicit state variables.

## Causal protocol

- Market, funding and metrics rows were physically truncated before 2024.
- Binance metrics were delayed by one complete 5-minute source bar.
- Exact millisecond funding times were rounded up to the first 5-minute bar that
  cannot precede settlement.
- Current-bar new OI is appended only after that bar's cohort pressure is
  emitted; entry is at the next open.
- Global-account ratio plus market taker flow avoids sparse 2022 top-trader
  fields.
- Every z-score uses history shifted through `t-1`.
- Thresholds use data through 2022; 2023/H1/H2 select. OOS was not opened.
- 0.5x, 6bp/side, non-overlapping holds and conservative strict OHLC MDD.

Grid: minimum age `{1,3,6}` settlements × cohort half-life `{24h,72h}` × q90
absolute transfer threshold × hold `{6h,12h}` = 12 candidates.

## Result

Best adequately populated candidate: age at least six settlements, 24h
half-life, 12h hold. Metric format:
`absolute return / CAGR / strict MDD / ratio / trades`.

| Window | Result |
|---|---:|
| Fit through 2022 | `-24.30 / -11.82 / 48.21 / -0.25 / 849` |
| 2023 | `+7.11 / +7.11 / 21.61 / 0.33 / 372` |
| 2023 H1 | `+12.86 / +27.65 / 14.67 / 1.89 / 190` |
| 2023 H2 | `-5.10 / -9.86 / 13.45 / -0.73 / 182` |

Only two of six half-year robustness segments were positive. All twelve primary
policies had negative full-fit returns.

## Matched controls

| Variant | Fit result | 2023 result |
|---|---:|---:|
| Direction flip | `-58.89 / -33.09 / 61.76 / -0.54 / 849` | `-41.89 / -41.92 / 44.32 / -0.95 / 372` |
| Fake settlement +4h | `-36.25 / -18.41 / 59.06 / -0.31 / 857` | `+5.75 / +5.76 / 20.70 / 0.28 / 378` |
| Ignore settlement age | `-36.89 / -18.78 / 62.89 / -0.30 / 910` | `+0.64 / +0.64 / 16.01 / 0.04 / 371` |
| Remove funding burden | `-23.97 / -11.65 / 48.51 / -0.24 / 845` | `+6.05 / +6.06 / 22.67 / 0.27 / 368` |

The exact flip is worse, so age/carry state affects direction, but the selected
mechanism is still unprofitable and nearly matched by a carry-blind control.

## Decision

**Reject in preflight; do not open OOS.** Do not tune more ages, tails or holds
for this cohort formula. A future cohort model needs directly observed
ownership/position lifecycle data, not another proxy combination.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_funding_age_rollover_transfer_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_funding_age_rollover_transfer_alpha.py
```
