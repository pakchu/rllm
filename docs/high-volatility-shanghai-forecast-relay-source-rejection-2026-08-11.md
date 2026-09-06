# HVSFR-24 terminal source-support rejection

HVSFR-24 was preregistered before opening Shanghai source values or candidate
incidence. The frozen source builder was then committed and pushed before the
source-support stage ran. The source stage opened only completed pre-entry
Shanghai and BTC inputs; it did not open post-entry prices, PnL, Gross9 rows, or
economic outcomes.

## Frozen source result

| Split | Events | Long | Short | Minority share | Maximum month share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train 2023H2 | 35 | 32 | 3 | 0.0857 | 0.4000 |
| Test 2024 | 119 | 111 | 8 | 0.0672 | 0.1681 |
| Eval 2025 | 60 | 40 | 20 | 0.3333 | 0.2000 |
| Final 2026 through July | 51 | 20 | 31 | 0.3922 | 0.2941 |

All four splits passed the minimum-event and maximum-month-concentration gates.
Train and test failed the preregistered 0.20 minority-side-share gate. The
candidate is therefore rejected at the first source/scientific gate.

The unchanged build was run twice after removing dynamic Yahoo response metadata
from the persisted source snapshot. The source snapshot, source manifest, clock,
and terminal support result hashes reproduced exactly. This normalization changed
neither source values nor the frozen signal.

No Gross9 novelty comparison, execution price, funding, return, drawdown,
sign-flip test, stress result, or RV20-q90 audit was opened. No predictor, VAR
lag/window, volatility threshold, side, hold, clock, subset, model, or diagnostic
control repair is authorized.
