# Pullback weak-state exit router — pre-OOS selection

## Verdict

**PRE_OOS_CANDIDATE_FROZEN.** The exact state router passed the strict pre-2024 contract.
2024+ rows were not opened.

## Frozen state/action rule

- Base event: corrected live-parity confirmed pullback squeeze.
- Support: 48-hour time exit.
- Neutral: 24-hour time exit.
- Adverse absorption: 48-hour cap with 4% take, no stop.
- Weak states: completed-bar Wasserstein flow-response strain, causal-cone rupture, and dual price/flow clock.
- Entry: next 5-minute open; 6bp/notional/side plus realized funding.
- Strict MDD: global/pre-entry HWM plus favorable-before-adverse position envelope.

## Frozen performance

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| train | +88.41% | +28.81% | 7.89% | 3.65 | 86 |
| train_2020h2 | +11.12% | +23.28% | 5.70% | 4.09 | 17 |
| train_2021 | +31.60% | +31.62% | 7.89% | 4.01 | 27 |
| train_2022 | +26.12% | +26.14% | 7.81% | 3.35 | 41 |
| select_2023 | +13.94% | +13.95% | 4.59% | 3.04 | 18 |
| select_2023_h1 | +10.15% | +21.53% | 4.59% | 4.70 | 9 |
| select_2023_h2 | +3.44% | +6.95% | 3.08% | 2.26 | 9 |
| pre_2024 | +114.67% | +24.38% | 7.89% | 3.09 | 104 |

## Search accounting

- Structured cells: 1,296
- Strict qualifiers: 2
- Two top cells had identical executed schedules; deterministic tie-breaking chose the higher 2.0 clock-dominance threshold.
- Frozen manifest hash: `9045f3fc1f8a92ea5e933e222817114633f58c54710e25e0fe396c8d47f6689c`

The programme has inspected related pullback families on later years. The exact router is frozen before its own OOS replay, but that replay is contamination-aware rather than a pristine programme-level holdout.
