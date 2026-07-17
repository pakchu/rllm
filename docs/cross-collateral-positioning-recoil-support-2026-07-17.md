# CCPR-1 source-only support freeze — 2026-07-17

## Decision

The preregistered cross-collateral positioning recoil family has enough
source-only support to open the sealed Stage1 execution window. The selected
rotation rank is **Q=0.85**. No OHLC, funding, future return, label, portfolio
PnL, CAGR, or drawdown was parsed in this work unit, so performance metrics are
**N/A**.

## Frozen source result

| Item | Result |
|---|---:|
| Five-minute source rows | 261,216 |
| Joint source-complete rows | 220,370 |
| Hourly anchors | 21,768 |
| Complete causal feature anchors | 6,675 |
| Selected Q | 0.85 |
| Outcome sources opened | 0 |
| Simulations run | 0 |

The causal completeness requirement is intentionally severe: each current
anchor needs its full 73-row six-hour endpoint path and all 168 prior hourly
rank anchors. Missing exchange metrics are never filled or carried forward.

## Support selection

| Q | Train | 2021 partial | 2022 | 2023 | 2023H1 | 2023H2 | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.80 | 147 | 47 | 100 | 66 | 47 | 19 | yes |
| 0.85 | 113 | 35 | 78 | 49 | 37 | 12 | **yes** |
| 0.90 | 75 | 24 | 51 | 34 | 25 | 9 | no |

Q=0.90 fails the frozen train-power, 2021, 2022, 2023, 2023H2, and 2023
single-month concentration support floors. The protocol therefore selects the
highest complete passer, Q=0.85. No outcome was used to make this choice and
there is no fallback after Stage1 opens.

At Q=0.85, the primary train clocks are balanced across long and short sides,
no month dominates the event set, and exact timestamp Jaccard is far below the
0.85 duplicate ceiling for each OI-only, taker-only, USD-M-only, and COIN-M-only
control. These are source-clock diagnostics, not evidence of profitability or
portfolio orthogonality.

## Sealed next step

The strict evaluator may now open only `[2021-07-08, 2023-01-01)`. It must test
the frozen 4h and 8h holds, realized funding, 6bp base cost, 10bp stress, strict
intratrade MDD, clustered sign-flip significance, all subperiods, and the full
falsification battery. The 2023 execution window remains sealed unless Stage1
passes every preregistered gate.

## Artifacts

- Builder: `training/build_cross_collateral_positioning_recoil_support.py`
- Support result:
  `results/cross_collateral_positioning_recoil_support_2026-07-17.json`
- Frozen clocks:
  `results/cross_collateral_positioning_recoil_clocks_2026-07-17.csv`
- Tests: `tests/test_build_cross_collateral_positioning_recoil_support.py`
