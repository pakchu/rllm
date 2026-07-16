# Wikimedia Attention-Divergence — Selection Result

- Status: **rejected_before_holdout**
- Policies opened: 14 (preregistered only)
- Data opened: Wikimedia + BTC/funding through 2022-12-31 only; 2023 and 2024+ remain sealed.
- Diagnostic policy: `{"attention_threshold": 3.0, "family": "broad_attention_reversal", "hold_days": 3, "price_horizon_days": 1, "price_threshold": 0.08}`

## Diagnostic policy statistics

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Long/Short |
|---|---:|---:|---:|---:|---:|---:|
| fit_2020 | 3.2439% | 3.2372% | 7.8546% | 0.4121 | 3 | 1/2 |
| fit_2021 | 0.0000% | 0.0000% | 0.0000% | 0.0000 | 0 | 0/0 |
| selection_2022 | -1.9666% | -1.9679% | 5.0358% | -0.3908 | 1 | 1/0 |
| combined_2020_2022 | 1.2136% | 0.4028% | 7.8546% | 0.0513 | 4 | 2/2 |

## Decision

No preregistered policy passed every 2020-2022 selection gate. The 2023 holdout and all 2024+ data remain unopened.

Same-count/same-side random control: 5000 samples, ratio p=0.4103, random-positive fraction=0.4858.
The random control is a selection diagnostic; the preregistered Bonferroni weekly block-bootstrap gate belongs to the still-sealed 2023 holdout phase.

Historical Wikimedia snapshots do not prove point-in-time publication; even a future passing result would require retrieval-timestamped forward shadow evidence.

A passing selection policy is not an alpha yet. It may open exactly one frozen 2023 holdout; 2024+ remains sealed until the holdout gates pass.
