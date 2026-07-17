# CCPR-1 Stage1 rejection postmortem — 2026-07-17

## Verdict

Reject CCPR-1 without opening 2023. The cross-collateral source is useful and
its event clocks are distinct, but fading USD-M-versus-COIN-M positioning
concordance is not a standalone alpha under the frozen execution contract.

| Candidate | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| CCPR-H4 | -0.65% | -0.44% | 20.64% | -0.02 | 96 | 0.9390 |
| CCPR-H8 | -2.97% | -2.01% | 21.39% | -0.09 | 83 | 0.7256 |

At 10bp per side the 4h candidate loses 4.39% with a -2.98% CAGR, 20.80%
strict MDD, -0.14 CAGR/MDD, and 96 trades. The 8h candidate loses 6.15% with a
-4.19% CAGR, 21.55% strict MDD, -0.19 CAGR/MDD, and 83 trades. The failure is
not a marginal transaction-cost miss.

## Temporal failure

### CCPR-H4

| Subperiod | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021 partial | -3.77% | -7.63% | 20.64% | -0.37 | 32 |
| 2022H1 | +0.44% | +0.90% | 1.82% | 0.49 | 11 |
| 2022H2 | +2.79% | +5.62% | 2.72% | 2.06 | 53 |

### CCPR-H8

| Subperiod | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2021 partial | -1.16% | -2.37% | 21.39% | -0.11 | 29 |
| 2022H1 | +1.59% | +3.24% | 2.06% | 1.57 | 10 |
| 2022H2 | -3.38% | -6.59% | 8.09% | -0.81 | 44 |

The sign is not stable across market eras or horizons. The 4h rule behaves
better late in 2022, while the 8h rule reverses and loses in the same half.
This is inconsistent with one durable collateral-crowding recoil mechanism.

## Mechanism controls

- H4 primary CAGR/MDD: -0.02; best component control: OI-only -0.05.
  Margin is only +0.03 versus the frozen +0.25 requirement.
- H8 primary CAGR/MDD: -0.09; OI-only is +0.13. The conjunction underperforms
  the OI component by 0.22.
- Every component and falsification control fails the complete gate battery.

Thus the negative result is not caused by accidentally choosing the wrong
component from a clearly profitable mechanism. The source contains distinct
state information, but the fixed symmetric fade mapping does not convert it
into reliable PnL.

## Integrity

- Physical OHLC window: `[2021-07-08, 2023-01-01)` only.
- Parsed market rows: 156,096; final row 2022-12-31 23:55 UTC.
- Parsed funding rows: 1,626; final row 2022-12-31 16:00 UTC.
- Stage2 command exits before loading execution data with
  `CCPR-1 Stage1 failed; 2023 remains sealed`.
- Report manifest:
  `95f7d74048b8ed8e5199ad3cd6456e58ddf569dee6530673e0584ba9bd9504c0`.

## Research consequence

Do not reverse direction, change Q, add a price regime, or tune the hold inside
CCPR-1. A future family may retain cross-collateral positioning as a weak
feature only under a new clean preregistration. The next standalone alpha
search should move to a different causal axis rather than repair this result.
