# SFRD-1 source-only support replay (2026-07-17)

## Decision

**PASS exact source-clock replay and advance only to the frozen 2021-2022
Stage-1 outcome evaluation.** This is not evidence of profitability or source
generalization.

SFRD-1 was explicitly screened on SOFR event density through 2023 before
freeze. This work unit proves only that the committed exact-decimal algorithm,
frozen ledger, causal clock, counts, sides, and concentration reproduce without
opening a crypto market field. Absolute return, CAGR, strict MDD, and
CAGR/strict-MDD remain **N/A**, not zero.

## Isolation

Files opened by the support builder:

1. committed SFRD-1 preregistration;
2. frozen SOFR source panel;
3. frozen SOFR source manifest;
4. frozen source-only event ledger.

Opened execution OHLC, funding, return, or portfolio sources: `[]`.

The tests instrument support-side hash reads and fail if the BTCUSDT kline or
funding paths are touched.

## Exact arithmetic and clock replay

- parse each SOFR rate as base-10 Decimal;
- require an exact integer basis point;
- difference integer rates;
- compute the prior-120 mid-rank as integer numerator
  `2*count(<) + count(==)` over denominator 240;
- tightening when numerator `>=204`, easing when `<=36`;
- signal only on a nonzero state change;
- enter exactly five minutes after `sofr_available_at_utc`;
- exit exactly five calendar days later;
- ignore rather than queue overlapping signals.

The rebuilt 158-event full-source clock equals every row of the committed gzip
ledger. Matching only the count is insufficient.

## Replayed support

| Window | Events | Long | Short | Max month share | Frozen floor |
|---|---:|---:|---:|---:|---:|
| 2021-2022 | 48 | 31 | 17 | 10.42% | >=45; each side >=15; month <=15% |
| 2021 | 12 | 8 | 4 | 33.33% | >=10 |
| 2022 | 36 | 23 | 13 | 13.89% | >=35 |
| 2023 | 40 | 20 | 20 | 12.50% | >=35; each side >=18; month <=15% |
| 2023 H1 | 18 | 9 | 9 | 22.22% | >=15 |
| 2023 H2 | 21 | 11 | 10 | 23.81% | >=18 |

The H1 and H2 counts sum to 39 rather than 40 because one five-day trade is
contained in full-year 2023 but crosses the H1/H2 boundary and is therefore
excluded from both half-year slices.

Every exact-count, side, concentration, delay, hold, rank-denominator, state,
side-mapping, non-overlap, and pre-2024 check passed.

## Artifacts

- preregistration commit:
  `3a5e4659db98dc02422671410c7d5adce9931b3a`
- preregistration SHA-256:
  `cbb80c25e4b4c627b95d1992ce4ad00043acfff586e0fe3fcd086af5b4e80b06`
- frozen clock SHA-256:
  `391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69`
- builder:
  `training/build_sofr_rate_dislocation_support.py`
- tests:
  `tests/test_build_sofr_rate_dislocation_support.py`
- result:
  `results/sofr_rate_dislocation_support_2026-07-17.json`
- result SHA-256:
  `477286e74987c2371d92bc91eeef6546ba0d3796e6f03239b75e880ec05a29e8`

Two builds must produce byte-identical support JSON.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m training.build_sofr_rate_dislocation_support

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_build_sofr_rate_dislocation_support.py
```

## Next gate

Only now may the evaluator be frozen. Stage 1 may then read BTCUSDT execution
OHLC and exact funding through 2022-12-31. It must evaluate the unchanged
SFRD-1 primary and frozen controls. 2023 remains outcome-sealed unless Stage 1
passes every preregistered performance gate; 2024+ remains sealed.
