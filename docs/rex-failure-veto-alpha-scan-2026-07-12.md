# REX failure-mode veto alpha scan (2026-07-12)

Base action is deterministic REX event side; scan searches past-only keep/veto gates over fixed REX events.

## Protocol
- Base event: `rex_htf_pullback_reclaim` rows from `data/rex_event_reasoning_policy_sft_20260712.jsonl`.
- Action: deterministic `base_event.base_side` only; no future label/LLM action is used for entries.
- Features: signal-time/past-only numeric REX/macro/flow features plus symbolic tokens.
- Primary ranking: train-only.
- Secondary diagnostic ranking: train+test only; eval remains holdout.
- Entry: next bar; hold: 144 x 5m bars; leverage 0.5; fee+slippage 5bp round-turn assumption from existing simulator.

## Baseline raw REX
- train: abs +25.50%, CAGR +4.76%, strict MDD 21.17%, R 0.22, trades 505, p 0.360
- test: abs +8.74%, CAGR +10.27%, strict MDD 5.31%, R 1.93, trades 49, p 0.196
- eval: abs +0.01%, CAGR +0.05%, strict MDD 5.57%, R 0.01, trades 28, p 0.979

## Top train-only selected gates
### #1: `dxy_zscore >= -1.2189522865171263 AND htf_1w_return_4 >= -0.26588062806734514`
- train: abs +72.38%, CAGR +11.81%, strict MDD 13.94%, R 0.85, trades 431, p 0.030
- test: abs +5.94%, CAGR +6.96%, strict MDD 5.45%, R 1.28, trades 44, p 0.352
- eval: abs +0.91%, CAGR +3.43%, strict MDD 4.47%, R 0.77, trades 22, p 0.797

### #2: `htf_1w_return_4 >= -0.26588062806734514 AND rex_144_max_to_cur_pct >= 0.007868509927537293`
- train: abs +66.93%, CAGR +11.14%, strict MDD 12.66%, R 0.88, trades 406, p 0.036
- test: abs +5.96%, CAGR +6.99%, strict MDD 4.34%, R 1.61, trades 34, p 0.231
- eval: abs +0.15%, CAGR +0.57%, strict MDD 2.39%, R 0.24, trades 16, p 0.949

### #3: `htf_1w_return_4 >= -0.26588062806734514 AND htf_1w_return_4 <= 0.37578432203548623`
- train: abs +78.32%, CAGR +12.56%, strict MDD 15.35%, R 0.82, trades 410, p 0.009
- test: abs +8.74%, CAGR +10.27%, strict MDD 5.31%, R 1.93, trades 49, p 0.196
- eval: abs +1.28%, CAGR +4.85%, strict MDD 4.63%, R 1.05, trades 25, p 0.751

### #4: `htf_1w_return_4 >= -0.26588062806734514 AND return_zscore_48 <= 0.8245814517313907`
- train: abs +72.66%, CAGR +11.82%, strict MDD 14.81%, R 0.80, trades 422, p 0.024
- test: abs +2.64%, CAGR +3.09%, strict MDD 6.08%, R 0.51, trades 42, p 0.591
- eval: abs +0.77%, CAGR +2.90%, strict MDD 4.63%, R 0.63, trades 25, p 0.832

### #5: `bb_z >= -1.5961253627040404 AND htf_1w_return_4 >= -0.26588062806734514`
- train: abs +66.59%, CAGR +11.01%, strict MDD 14.27%, R 0.77, trades 405, p 0.041
- test: abs +7.77%, CAGR +9.12%, strict MDD 6.68%, R 1.37, trades 48, p 0.268
- eval: abs +1.41%, CAGR +5.35%, strict MDD 4.51%, R 1.19, trades 23, p 0.728

### #6: `htf_1w_return_4 >= -0.26588062806734514`
- train: abs +63.75%, CAGR +10.62%, strict MDD 15.35%, R 0.69, trades 462, p 0.051
- test: abs +8.74%, CAGR +10.27%, strict MDD 5.31%, R 1.93, trades 49, p 0.196
- eval: abs +1.28%, CAGR +4.85%, strict MDD 4.63%, R 1.05, trades 25, p 0.751

### #7: `htf_1w_return_4 >= -0.26588062806734514 AND kimchi_premium_zscore >= -1.0163830394297868`
- train: abs +63.99%, CAGR +10.65%, strict MDD 14.75%, R 0.72, trades 429, p 0.046
- test: abs +4.20%, CAGR +4.92%, strict MDD 4.77%, R 1.03, trades 45, p 0.419
- eval: abs +2.70%, CAGR +10.45%, strict MDD 4.19%, R 2.49, trades 23, p 0.463

### #8: `htf_1w_return_4 >= -0.26588062806734514 AND oi_zscore <= 1.5910475818293068`
- train: abs +69.16%, CAGR +11.36%, strict MDD 16.16%, R 0.70, trades 428, p 0.033
- test: abs +3.68%, CAGR +4.30%, strict MDD 6.34%, R 0.68, trades 45, p 0.534
- eval: abs +2.08%, CAGR +8.08%, strict MDD 3.32%, R 2.43, trades 22, p 0.587

### #9: `bb_z >= 1.3606616015142925 AND rex_144_cur_to_min_pct <= 0.04456585444179015`
- train: abs +43.45%, CAGR +7.79%, strict MDD 7.43%, R 1.05, trades 122, p 0.005
- test: abs -2.54%, CAGR -3.04%, strict MDD 4.61%, R -0.66, trades 13, p 0.469
- eval: abs -2.34%, CAGR -9.90%, strict MDD 5.18%, R -1.91, trades 11, p 0.395

### #10: `htf_1d_return_1 <= 0.03522957407649755 AND rex_144_max_to_cur_pct >= 0.007868509927537293`
- train: abs +52.37%, CAGR +9.07%, strict MDD 12.42%, R 0.73, trades 408, p 0.087
- test: abs +9.43%, CAGR +11.08%, strict MDD 2.49%, R 4.45, trades 30, p 0.032
- eval: abs -3.46%, CAGR -12.91%, strict MDD 4.99%, R -2.59, trades 16, p 0.342


## Top train+test diagnostic gates (eval holdout)
These are not live promotion; this ranking uses test as validation and opens eval only after selection.
### TTE #1: `htf_1w_return_4 >= -0.26588062806734514 AND rex_144_max_to_cur_pct >= 0.007868509927537293`
- train: abs +66.93%, CAGR +11.14%, strict MDD 12.66%, R 0.88, trades 406, p 0.036
- test: abs +5.96%, CAGR +6.99%, strict MDD 4.34%, R 1.61, trades 34, p 0.231
- eval: abs +0.15%, CAGR +0.57%, strict MDD 2.39%, R 0.24, trades 16, p 0.949

### TTE #2: `dxy_zscore >= -1.2189522865171263 AND htf_1w_return_4 >= -0.26588062806734514`
- train: abs +72.38%, CAGR +11.81%, strict MDD 13.94%, R 0.85, trades 431, p 0.030
- test: abs +5.94%, CAGR +6.96%, strict MDD 5.45%, R 1.28, trades 44, p 0.352
- eval: abs +0.91%, CAGR +3.43%, strict MDD 4.47%, R 0.77, trades 22, p 0.797

### TTE #3: `htf_1w_return_4 >= -0.26588062806734514 AND htf_1w_return_4 <= 0.37578432203548623`
- train: abs +78.32%, CAGR +12.56%, strict MDD 15.35%, R 0.82, trades 410, p 0.009
- test: abs +8.74%, CAGR +10.27%, strict MDD 5.31%, R 1.93, trades 49, p 0.196
- eval: abs +1.28%, CAGR +4.85%, strict MDD 4.63%, R 1.05, trades 25, p 0.751

### TTE #4: `bb_z >= -1.5961253627040404 AND htf_1w_return_4 >= -0.26588062806734514`
- train: abs +66.59%, CAGR +11.01%, strict MDD 14.27%, R 0.77, trades 405, p 0.041
- test: abs +7.77%, CAGR +9.12%, strict MDD 6.68%, R 1.37, trades 48, p 0.268
- eval: abs +1.41%, CAGR +5.35%, strict MDD 4.51%, R 1.19, trades 23, p 0.728

### TTE #5: `htf_1w_return_4 >= -0.26588062806734514 AND kimchi_premium_zscore >= -1.0163830394297868`
- train: abs +63.99%, CAGR +10.65%, strict MDD 14.75%, R 0.72, trades 429, p 0.046
- test: abs +4.20%, CAGR +4.92%, strict MDD 4.77%, R 1.03, trades 45, p 0.419
- eval: abs +2.70%, CAGR +10.45%, strict MDD 4.19%, R 2.49, trades 23, p 0.463

### TTE #6: `htf_1w_return_4 >= -0.26588062806734514`
- train: abs +63.75%, CAGR +10.62%, strict MDD 15.35%, R 0.69, trades 462, p 0.051
- test: abs +8.74%, CAGR +10.27%, strict MDD 5.31%, R 1.93, trades 49, p 0.196
- eval: abs +1.28%, CAGR +4.85%, strict MDD 4.63%, R 1.05, trades 25, p 0.751

### TTE #7: `htf_1w_return_4 >= -0.26588062806734514 AND range_vol >= 0.02098580437041066`
- train: abs +55.46%, CAGR +9.45%, strict MDD 13.54%, R 0.70, trades 394, p 0.070
- test: abs +6.08%, CAGR +7.20%, strict MDD 5.07%, R 1.42, trades 37, p 0.276
- eval: abs +2.02%, CAGR +7.82%, strict MDD 4.47%, R 1.75, trades 19, p 0.547

### TTE #8: `htf_1w_return_4 >= -0.26588062806734514 AND oi_zscore <= 1.5910475818293068`
- train: abs +69.16%, CAGR +11.36%, strict MDD 16.16%, R 0.70, trades 428, p 0.033
- test: abs +3.68%, CAGR +4.30%, strict MDD 6.34%, R 0.68, trades 45, p 0.534
- eval: abs +2.08%, CAGR +8.08%, strict MDD 3.32%, R 2.43, trades 22, p 0.587

### TTE #9: `htf_1w_return_4 >= -0.26588062806734514 AND htf_4h_return_4 >= -0.030833617703247055`
- train: abs +63.86%, CAGR +10.65%, strict MDD 15.83%, R 0.67, trades 423, p 0.037
- test: abs +7.78%, CAGR +9.20%, strict MDD 5.86%, R 1.57, trades 47, p 0.245
- eval: abs +1.28%, CAGR +4.85%, strict MDD 4.63%, R 1.05, trades 25, p 0.751

### TTE #10: `bb_z <= 1.3606616015142925 AND htf_1w_return_4 >= -0.26588062806734514`
- train: abs +51.32%, CAGR +8.85%, strict MDD 14.04%, R 0.63, trades 424, p 0.084
- test: abs +5.49%, CAGR +6.44%, strict MDD 5.55%, R 1.16, trades 44, p 0.334
- eval: abs +0.11%, CAGR +0.41%, strict MDD 2.54%, R 0.16, trades 22, p 0.959

### TTE #11: `htf_1d_return_1 <= 0.03522957407649755 AND return_zscore_48 <= 0.8245814517313907`
- train: abs +51.89%, CAGR +8.93%, strict MDD 13.81%, R 0.65, trades 423, p 0.081
- test: abs +4.84%, CAGR +5.66%, strict MDD 4.02%, R 1.41, trades 37, p 0.280
- eval: abs +0.21%, CAGR +0.78%, strict MDD 4.93%, R 0.16, trades 26, p 0.946

### TTE #12: `htf_1w_return_4 >= -0.26588062806734514 AND rex_144_range_width_pct >= 0.021008280967639882`
- train: abs +53.96%, CAGR +9.23%, strict MDD 14.37%, R 0.64, trades 393, p 0.076
- test: abs +6.08%, CAGR +7.20%, strict MDD 5.07%, R 1.42, trades 37, p 0.276
- eval: abs +2.02%, CAGR +7.82%, strict MDD 4.47%, R 1.75, trades 19, p 0.547

### TTE #13: `htf_1w_return_4 >= -0.26588062806734514 AND return_zscore_48 >= -0.8089380722762467`
- train: abs +50.17%, CAGR +8.77%, strict MDD 14.76%, R 0.59, trades 417, p 0.101
- test: abs +7.85%, CAGR +9.22%, strict MDD 6.29%, R 1.47, trades 47, p 0.262
- eval: abs +0.78%, CAGR +2.94%, strict MDD 3.71%, R 0.79, trades 23, p 0.828

### TTE #14: `htf_1w_return_4 >= -0.26588062806734514 AND kimchi_premium_change >= -0.00324675352454635`
- train: abs +42.56%, CAGR +7.53%, strict MDD 13.05%, R 0.58, trades 425, p 0.121
- test: abs +5.82%, CAGR +6.82%, strict MDD 5.16%, R 1.32, trades 42, p 0.260
- eval: abs +1.28%, CAGR +4.85%, strict MDD 4.63%, R 1.05, trades 25, p 0.751

### TTE #15: `htf_1w_return_4 >= -0.26588062806734514 AND range_pos >= -0.5996501056443556`
- train: abs +50.63%, CAGR +8.74%, strict MDD 15.69%, R 0.56, trades 407, p 0.088
- test: abs +6.53%, CAGR +7.66%, strict MDD 5.80%, R 1.32, trades 46, p 0.292
- eval: abs +1.24%, CAGR +4.68%, strict MDD 4.81%, R 0.97, trades 25, p 0.759

### TTE #16: `htf_1w_return_4 >= -0.26588062806734514 AND rex_144_range_pos >= -0.5996501056443556`
- train: abs +50.63%, CAGR +8.74%, strict MDD 15.69%, R 0.56, trades 407, p 0.088
- test: abs +6.53%, CAGR +7.66%, strict MDD 5.80%, R 1.32, trades 46, p 0.292
- eval: abs +1.24%, CAGR +4.68%, strict MDD 4.81%, R 0.97, trades 25, p 0.759

### TTE #17: `htf_1w_return_4 >= -0.26588062806734514 AND range_pos <= 0.5519438499929247`
- train: abs +45.33%, CAGR +8.01%, strict MDD 13.94%, R 0.57, trades 424, p 0.119
- test: abs +9.15%, CAGR +10.76%, strict MDD 5.43%, R 1.98, trades 44, p 0.111
- eval: abs +1.26%, CAGR +4.78%, strict MDD 2.54%, R 1.88, trades 22, p 0.671

### TTE #18: `htf_1w_return_4 >= -0.26588062806734514 AND rex_144_range_pos <= 0.5519438499929247`
- train: abs +45.33%, CAGR +8.01%, strict MDD 13.94%, R 0.57, trades 424, p 0.119
- test: abs +9.15%, CAGR +10.76%, strict MDD 5.43%, R 1.98, trades 44, p 0.111
- eval: abs +1.26%, CAGR +4.78%, strict MDD 2.54%, R 1.88, trades 22, p 0.671

### TTE #19: `bb_z >= -1.5961253627040404 AND htf_1d_return_1 <= 0.03522957407649755`
- train: abs +46.48%, CAGR +8.12%, strict MDD 13.34%, R 0.61, trades 413, p 0.114
- test: abs +10.09%, CAGR +11.87%, strict MDD 4.86%, R 2.44, trades 43, p 0.130
- eval: abs +0.66%, CAGR +2.48%, strict MDD 4.80%, R 0.52, trades 24, p 0.870

### TTE #20: `dxy_zscore <= 1.236722665970137 AND htf_1w_return_4 >= -0.26588062806734514`
- train: abs +50.73%, CAGR +8.76%, strict MDD 14.20%, R 0.62, trades 421, p 0.085
- test: abs +11.31%, CAGR +13.32%, strict MDD 5.07%, R 2.63, trades 43, p 0.085
- eval: abs +1.22%, CAGR +4.64%, strict MDD 4.54%, R 1.02, trades 24, p 0.758


## Current best candidate interpretation
The most useful pattern is a broad REX veto: avoid REX entries when weekly HTF return is deeply negative. Additional range/kimchi/OI filters can improve one split but do not produce statistically strong eval evidence.

Promising but not live-grade examples:
- `htf_1w_return_4 >= -0.2659 AND range_pos <= 0.5519`: train/test/eval all positive, but train R 0.57 and eval only 22 trades.
- `htf_1w_return_4 >= -0.2659 AND kimchi_premium_zscore >= -1.0164`: eval R 2.49, but test R 1.03 and p-values weak.
- `dxy_momentum <= 0.00153 AND oi_zscore <= 1.591`: eval R 5.34, but train/test are too weak and this was not primary-selected.

## Verdict
REX failure-mode veto is useful as a context/risk filter, not a standalone alpha. It improves raw train MDD and creates several positive holdout candidates, but no candidate reaches the target of CAGR/strict MDD >= 3 with statistically meaningful trades across train/test/eval.

## Leakage guard
- Fixed REX candidate rows are generated before this scan.
- Gate thresholds are quantiles from train rows.
- Primary top list ranks on train only.
- TTE diagnostic list ranks on train+test only; eval is not used for ranking.
- All features are computed from rows at or before signal time; entry occurs after signal.