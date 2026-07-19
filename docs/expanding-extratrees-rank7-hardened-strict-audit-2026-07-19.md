# ExtraTrees rank-7 hardened strict audit — 2026-07-19

Verdict: **SURVIVES_HARDENED_STRICT_AUDIT**

This is a retrospective accounting/parity audit, not pristine discovery OOS. The frozen model, thresholds, annual refits, selected positions, and trade clocks were required to match before hardened metrics were accepted.

## Metrics

| Period | Abs return | CAGR | Hardened strict MDD | CAGR/MDD | Trades | 10bp/side stress abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 12.8641% | 12.8735% | 3.1464% | 4.0915 | 19 | 12.0093% |
| 2024 | 16.3961% | 16.3599% | 3.4920% | 4.6849 | 22 | 15.3760% |
| 2025 | 16.3620% | 16.3740% | 5.0129% | 3.2664 | 21 | 15.3882% |
| 2026h1 | 7.3132% | 18.4835% | 4.3294% | 4.2693 | 12 | 6.7991% |
| future | 24.8717% | 16.9903% | 5.0129% | 3.3893 | 33 | 23.2336% |
| all | 64.0433% | 15.5877% | 5.0129% | 3.1095 | 74 | 59.2569% |

## Statistical checks

- `future`: weekly-cluster sign-flip p `0.007400`; stationary trade-bootstrap p `0.000100`.
- `all`: weekly-cluster sign-flip p `0.000020`; stationary trade-bootstrap p `0.000020`.

## Integrity

- selected-position hash: `8ffbd55f07ceda0e82c270fe4b370fffba44bb3fcfc807368c4385d2ba97f531`
- every frozen per-window trade-clock hash matched: `True`
- model/feature/policy changes: none
- exact funding boundary: interior symmetric; entry/exit credits dropped, debits retained
- adverse mark: actual entry cost plus virtual liquidation cost
