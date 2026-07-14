# CATCH-12 frozen selection result — 2026-07-14

## Decision

**REJECT CATCH-12 v1. Keep 2024+ sealed.** The unchanged preregistered primary
failed train, full 2023, both 2023 halves, the strict-MDD limit, clustered
significance, the gross-move hurdle, and multiple score-bearing controls. No
threshold, side, hold, cost, or gate repair is permitted after this result.

- evaluator source commit:
  `19aa5245cfd60c814b3f08a9c6212eaf1b707c0d`
- evaluator source SHA-256:
  `d1fe5f04ec4e9ca7302291f1518ded827a7aca986ab9a6c9b9249c779f9ec1fd`
- pre-outcome freeze commit: `52c42e0`
- pre-outcome freeze SHA-256:
  `48bd81409c243d8cc4d0f9d9fbce523f0d09d1018abf7316081bd3de618cbae7`
- result:
  `results/cash_auction_transfer_catchup_handoff_selection_2026-07-14.json`
- result SHA-256:
  `8b3822f2ca748b219ccb9ad88129f71fc905cc1d2071a2d98daa034a717aec2e`

## Frozen primary statistics

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Mean gross underlying move | Trades |
|---|---:|---:|---:|---:|---:|---:|
| 2020–2022 train | -89.09% | -52.21% | 89.16% | -0.59 | -2.17 bp | 3,102 |
| 2023 selection | -38.30% | -38.32% | 38.33% | -1.00 | +0.74 bp | 855 |
| 2023 H1 | -23.18% | -41.27% | 23.21% | -1.78 | -0.66 bp | 415 |
| 2023 H2 | -19.67% | -35.27% | 19.76% | -1.78 | +2.07 bp | 440 |

The weekly entry-cluster one-sided p-value is `1.0` in train and full 2023
because the cost-inclusive observed mean return is negative. The frozen half
count requirement passed, but profitability did not.

## Why the support success did not become alpha

The support stage established that the event was frequent, balanced, causal in
availability, and distinct from its placebos. It did **not** establish that the
next one-hour move was large or persistent. The required underlying
round-trip break-even is about 12 bp under the frozen `0.5x` leverage and
`6 bp` account round-trip cost. CATCH produced only `-2.17 bp` gross in train
and `+0.74 bp` in 2023, so repeated execution compounded costs rather than an
edge.

The exact direction flip reinforces the non-stationarity diagnosis: it had
`+2.17 bp` mean gross in train but `-0.74 bp` in 2023. Neither orientation is
large enough to pay costs, and the preferred sign reverses out of sample. The
other score-bearing controls also remained within only a few basis points of
zero gross move. The observed Spot-to-USD-M handoff is therefore a measurable
microstructure state, but not a tradable one-hour directional alpha under this
contract.

## Verification

- A second complete evaluator run reproduced the result file byte-for-byte.
- An independent calculation from the frozen 3,957-row clock and exact USD-M
  open/high/low arrays reproduced absolute return, full-clock CAGR, strict MDD,
  mean gross move, and trade count for all four primary windows to numerical
  precision.
- All primary and control schedules were reserved before execution OHLC was
  loaded.
- The execution source ends at `2023-12-31 23:55:00`; the final frozen CATCH
  exit is `2023-12-31 16:30:00`.
- Calendar 2024 test, calendar 2025 eval, and 2026 YTD were not read.

## Consequence for the next search

CATCH-12 v1 is closed. Its failure says the next preregistration must target an
economically larger and lower-turnover payoff object before any return is
opened; simply retuning this percentile, side, or one-hour hold would be
post-outcome repair rather than a new alpha.
