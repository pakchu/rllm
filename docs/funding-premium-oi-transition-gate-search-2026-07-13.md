# Funding/Premium + OI Transition Gate Search — 2026-07-13

## Verdict

**Rejected as an alpha and recorded as a gamma usage.**  A causal Binance BTC
open-interest transition veto reduced both the trade count and the risk-adjusted
return of the already-fixed funding/premium + `lr_impact_72` long.  No frozen
policy passed the alpha-pool gate, and neither is suitable for live shadow use.

This rejects the exact **hourly OI transition veto on this fixed base**, not raw
open interest as a feature family.

## Frozen protocol

- Base rule was not retuned:
  - funding leg: `funding_rate <= -0.0000167`, `trend_96 >= 0.007485218212390219`,
    and `-0.20030301257467914 <= lr_impact_72 <= 0.24664964484849766`;
  - premium leg: `premium_index_change <= -0.00023471` and
    `htf_1d_return_4 >= 0.0940403008961932`.
- OI source: Binance USD-M `sum_open_interest`, backward-asof joined with a
  mandatory one-complete-5m-bar delay.
- An hourly OI observation is usable only after all 12 source rows are complete;
  the last delayed OI from hour `H` becomes visible at `H+1h`.
- State thresholds were fitted only on 2021-04-15 through 2022-12-31:
  - 24h log-OI-change q30: `-0.015744523516136515`;
  - q70: `0.022633799389715922`;
  - trailing-seven-day OI-z median: `0.36480192149050933`.
- Three-state transition: previous/current 24h OI-change bucket.
- Six-state transition: `2 * change_bucket + OI_level_bucket`, then the
  previous/current transition.
- Search was capped at 36 predeclared policies; the data supported only two
  stable negative-transition vetoes.
- Fit/ranking used no row on or after 2024-01-01.  The source-prefix, derived
  feature, base-feature and activation hashes were frozen in the manifest.
- Execution: next 5m open, fixed 576 bars, stride 12, one position, long only,
  0.5x, 6bp/side, no TP/SL, strict intraposition high-water MDD.

### Transparent preflight correction

The first pre-2024-only pass selected zero policies because it simultaneously
required at least eight trades per transition and 60 trades for a policy while
the fixed base had only 71 fit trades.  Before any candidate OOS result existed,
the policy-level floors were corrected to 36 fit and 12 selection trades.  The
zero-candidate manifest is retained as
`results/funding_premium_oi_transition_gate_zero_candidate_preflight_2026-07-13.json`.
No threshold, state definition, execution parameter or OOS gate was changed.

## Frozen replay results

All percentages are full-window values. CAGR counts periods with no trades.

| Policy | Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Rank 1: 3-state veto `[7,8,4]` | 2024 | +16.48% | 16.44% | 6.78% | 2.42 | 14 |
|  | 2025 | +8.72% | 8.72% | 6.56% | 1.33 | 9 |
|  | 2026 to Jun 02 | +0.57% | 1.37% | 7.18% | 0.19 | 13 |
|  | 2024–2026 combined | +27.35% | 10.52% | 7.53% | 1.40 | 36 |
| Rank 2: 6-state veto `[21,35,33]` | 2024 | +17.59% | 17.55% | 6.78% | 2.59 | 17 |
|  | 2025 | +9.97% | 9.98% | 6.56% | 1.52 | 15 |
|  | 2026 to Jun 02 | +5.75% | 14.37% | 7.18% | 2.00 | 18 |
|  | 2024–2026 combined | +36.74% | 13.82% | 9.03% | 1.53 | 50 |

The unchanged base was materially better:

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2024 | +24.48% | 24.43% | 5.67% | 4.31 | 26 |
| 2025 | +22.44% | 22.46% | 6.56% | 3.42 | 21 |
| 2026 to Jun 02 | +10.80% | 27.95% | 6.82% | 4.10 | 22 |
| 2024–2026 combined | +68.88% | 24.21% | 7.99% | 3.03 | 69 |

The better OI policy remained statistically positive in aggregate
(`p≈0.00250`, two-policy Bonferroni `≈0.00500`) but failed the actual economic
criterion in 2025, 2026 and combined, and removed 19 of 69 base trades.  A low
p-value does not repair the negative incremental value versus the fixed base.
At 10bp/side its combined ratio fell further to 1.38.

## Independence and leakage audit

- Fit Spearman maximum absolute correlation:
  - `oi_logchg24`: `0.0565` versus base admission features;
  - `oi_z168`: `0.1420`.
- Thus the OI state is statistically distinct, but distinction alone did not
  create incremental alpha.
- Every source was physically truncated before manifest selection.
- Full replay reproduced the manifest hash, thresholds, baseline, selected
  policies and source hashes exactly.
- Manifest SHA-256:
  `4e38e3be76329174c661ee80bae4330255553ff6e2f2b42c845d50f81213e082`.

## Decision

Do not retry static first-order OI transition vetoes on this base, do not relax
the veto states using 2024–2026, and do not replace the stronger ungated base.
A future OI experiment must use a materially different mechanism and fresh
forward evidence.

## Artifacts

- `training/search_funding_premium_oi_transition_gate_alpha.py`
- `tests/test_search_funding_premium_oi_transition_gate_alpha.py`
- `results/funding_premium_oi_transition_gate_top10_manifest_2026-07-13.json`
- `results/funding_premium_oi_transition_gate_alpha_scan_2026-07-13.json`
- `results/funding_premium_oi_transition_gate_replay_verification_2026-07-13.json`
- `results/funding_premium_oi_transition_gate_zero_candidate_preflight_2026-07-13.json`
