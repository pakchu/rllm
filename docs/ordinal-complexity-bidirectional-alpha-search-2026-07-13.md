# Ordinal Complexity Bidirectional Alpha Search — 2026-07-13

## Verdict

**No alpha was promoted.**  The frozen low-permutation-entropy continuation
policy was strong in 2024 but reversed in 2025.  The exact static standalone
mapping is rejected; the raw ordinal-complexity features are retained as a
regime-sensitive beta family because they are genuinely distinct from existing
sign entropy, path efficiency and jump/volatility features.

## New feature family

For each completed hour, the last three and four hourly closes are converted to
an ordinal permutation.  Only rank order is retained; price magnitude is not.

- `oc_pattern_{3,4}`: permutation ID of completed hourly closes.
- `oc_direction_{3,4}`: endpoint rank displacement in `[-1,1]`.
- `oc_o{3,4}_w{168,720}_entropy`: normalized permutation entropy of patterns
  strictly before the current pattern.
- `*_pattern_surprise`: smoothed negative log probability of the current
  pattern under its trailing historical distribution.
- `*_transition_surprise`: smoothed negative log conditional probability of the
  current transition, again using only earlier transitions.

Each source hour requires all 12 five-minute rows. Hour `H` is timestamped at
`H+1h`, so no partial-hour close enters a signal.

## Frozen protocol

- Feature thresholds: 2020-06-01 through 2022-12-31 only.
- Policy selection: full 2023 plus positive H1/H2 stability while all 2024+
  market rows were physically absent.
- Fixed search: 64 signal masks × holds `{144,288,576}` = 192 policies.
- Signal families: low/high entropy, high pattern surprise and high transition
  surprise, each with continuation and reversal direction mapping.
- Execution: next 5m open, stride 12, one position, 0.5x, 6bp/side, no TP/SL,
  strict intraposition high-water MDD.
- Frozen manifest contained two eligible policies. OOS did not alter any
  threshold, mapping or hold.

## Results

All percentages use the complete calendar window, including periods without
trades.

### Rank 1 — low entropy continuation

Order 3, trailing 168 hours, lower 30% entropy threshold
`0.9642215792721767`, fixed 576-bar hold.

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades (L/S) |
|---|---:|---:|---:|---:|---:|
| Fit 2020-06–2022 | +49.99% | 16.98% | 21.86% | 0.78 | 195 (97/98) |
| Select 2023 | +22.40% | 22.41% | 12.10% | 1.85 | 71 (32/39) |
| Test 2024 | +30.16% | 30.09% | 9.55% | **3.15** | 68 (35/33) |
| Eval 2025 | **-10.49%** | -10.50% | 22.10% | -0.47 | 106 (54/52) |
| 2026 to Jun 02 | +4.62% | 11.45% | 9.28% | 1.23 | 43 (18/25) |
| 2024–2026 combined | +28.29% | 10.86% | 24.09% | 0.45 | 217 (108/109) |

The 2024 result was nontrivial (`p≈0.038`), but 2025 sign reversal and five
negative quarters out of ten reject stable alpha. At 10bp/side the combined
ratio fell to 0.26.

### Rank 2 — high entropy reversal

Order 4, trailing 720 hours, upper 20%, fixed 288-bar hold. It lost money in
2024 and 2025 and generated no 2026 trade; combined return was -5.13%, ratio
-0.23, 43 trades.

## Independence audit

Every ordinal feature passed the predeclared `|Spearman rho| < 0.40` audit
against existing close cousins:

- binary sign entropy;
- sign autocorrelation;
- variance ratio;
- 72-bar path efficiency;
- short/long realized-volatility ratio.

The largest observed absolute correlation was `0.1599` for four-point,
720-hour transition surprise. Low correlation proves distinct information, not
profitable information.

## Leakage and replay checks

- 2024+ rows were physically absent during feature thresholding and selection.
- Entropy and surprise counts are shifted one pattern, so the current pattern
  never estimates its own probability.
- Completed hourly features are exposed from the next hour only.
- Market-prefix, feature-prefix, reference-feature and activation hashes are
  frozen in the manifest.
- Exact replay reproduced the manifest, correlations, selected policies and
  source hash.
- Manifest SHA-256:
  `b3a0e9f537eee13b6bbc44fc433b9a9896072fb048234d2221c7caf8791d59d6`.

## Decision

1. Do not trade or retune the static low-entropy continuation/high-entropy
   reversal mappings on 2024–2026.
2. Keep continuous entropy/surprise values as beta inputs for a separately
   frozen regime model or RLLM tokenization experiment.
3. Promotion requires a new mechanism and fresh forward evidence; the 2024
   pocket alone is not an alpha.

## Artifacts

- `training/search_ordinal_complexity_bidirectional_alpha.py`
- `tests/test_search_ordinal_complexity_bidirectional_alpha.py`
- `results/ordinal_complexity_bidirectional_top10_manifest_2026-07-13.json`
- `results/ordinal_complexity_bidirectional_alpha_scan_2026-07-13.json`
- `results/ordinal_complexity_bidirectional_replay_verification_2026-07-13.json`
