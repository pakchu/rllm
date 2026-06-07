# Side preference data export (2026-06-07)

## Purpose
The side specialist SFT underperformed eval majority baseline, suggesting hard side labels are too noisy or weak. This dataset builds explicit side preferences by comparing realized LONG and SHORT net returns for the same prompt/time.

## Method
For each row from `data/stable_trader_policy_h144_t1p8_s1p5_all.jsonl`:
1. Simulate LONG net return with the fixed `h144 / target 1.8% / stop 1.5%` economics.
2. Simulate SHORT net return with the same entry, cost, and horizon.
3. Keep the row only if absolute return difference is at least 0.05%.
4. Set `chosen` to the better side JSON and `rejected` to the worse side JSON.

## Outputs
- `data/side_preference_h144_t1p8_s1p5_diff0p05_train.jsonl`
- `data/side_preference_h144_t1p8_s1p5_diff0p05_val.jsonl`
- `data/side_preference_h144_t1p8_s1p5_diff0p05_eval.jsonl`
- `data/side_preference_h144_t1p8_s1p5_diff0p05_all.jsonl`
- `data/side_preference_h144_t1p8_s1p5_diff0p05.summary.json`

## Counts
- Total preference pairs: 2260
- Train: 1221
- Val: 532
- Eval: 507
- Chosen LONG: 1155
- Chosen SHORT: 1105

Reward-difference stats:
- Minimum: 0.0503%p
- Mean: 1.0956%p
- Maximum: 1.65%p

## Interpretation
This is a much stronger side-learning signal than the previous stable-policy hard labels:
- it is balanced between LONG and SHORT,
- it directly teaches comparative side quality,
- it avoids NO_TRADE dilution,
- it can be used for DPO or candidate scoring.

## Next step
Run DPO starting from the side SFT adapter, then evaluate whether side accuracy beats eval majority baseline and whether chosen-side candidate scoring improves before combining with the gate.
