# UGCI-288 source-support result

## Verdict

**Retire UGCI-288 without repair.** The frozen candidate failed the
outcome-blind source-support gate, so BTC market data, funding, returns, PnL,
CAGR, strict MDD, and novelty comparator clocks were not opened.

## Frozen incidence result

| Window | Events | Long | Short | Maximum month share |
| --- | ---: | ---: | ---: | ---: |
| Train 2021–2022 | 87 | 71 | 16 | 19.54% |
| Train 2021 | 80 | 65 | 15 | 21.25% |
| Train 2022 | 7 | 6 | 1 | 42.86% |
| Selection 2023 | 11 | 4 | 7 | 27.27% |
| Selection 2023 H1 | 2 | 0 | 2 | 100.00% |
| Selection 2023 H2 | 9 | 4 | 5 | 33.33% |

The preregistered minimums were 120 train events, 50 selection events, 45 per
train year, and 20 per selection half. The primary clock also failed train side
balance and month-concentration gates. The failure is structural: the fixed
USDC gross-tail plus imbalance conjunction nearly disappears in 2022 and early
2023.

## Integrity boundary

- `available_at` formed the six-hour packet clock.
- q95 used only the strictly prior 180-day packet grid, including zero packets.
- entries waited ten minutes; exits were fixed at 24 hours and contained inside
  the pre-2024 interval.
- `stale_6h` was derived only from accepted primary signals.
- no original or sealed comparator clock was opened because incidence failed.
- no threshold, direction, latency, hold, or support gate was changed after the
  result.

Artifacts:

- `data/usdc_gross_clearing_imbalance_clocks_2021_2023.csv.gz`
- `results/usdc_gross_clearing_imbalance_support_2026-07-22.json`
