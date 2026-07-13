# Funding/premium × alt premium-change gate alpha — 2026-07-13

## Decision

Register frozen manifest rank 9 as an **alpha-pool candidate**, but explicitly
not as live-grade. It passes the predeclared 2024/2025 alpha-pool gate and has
adequate trade counts, while its 2026 and combined CAGR/MDD are too weak for
deployment.

## Rule

The BTC premium component is unchanged. The BTC funding squeeze component is
allowed only when six-alt premium crowding is outside its central fit range:

```text
funding component:
  funding_available > 0.5
  and funding_rate <= -0.0000167
  and trend_96 >= 0.007485218212390219
  and alt_premium_available > 0.5
  and (
    alt_premium_median_change288_z2016 <= -0.41777540743350977
    or alt_premium_median_change288_z2016 >= 0.4309560030698776
  )

premium component:
  premium_available > 0.5
  and premium_index_change <= -0.00023471
  and htf_1d_return_4 >= 0.0940403008961932

entry = funding component OR premium component
```

The external feature uses ETH/SOL/BNB/XRP/ADA/DOGE Binance USD-M hourly
premium-index closes. Each value becomes visible only at `close_time`, may be
at most 65 minutes old, and all six symbols must be available. The cross-symbol
median's 24-hour change is standardized over 2016 five-minute bars.

Execution is fixed: next 5-minute open, 576 bars, stride 12, one
non-overlapping long, no TP/SL, 0.5x, and 6 bp per side.

## Clean-room protocol

- Feature threshold fit: 2023-02-15 through 2023-06-30.
- Policy selection: 2023-07-01 through 2023-12-31 only.
- Market, BTC funding/premium, and all 12 six-alt source files were physically
  truncated before 2024 before the manifest was written.
- 16 external features were screened against BTC funding, BTC premium change,
  8-hour trend, completed daily momentum, and `lr_impact_72`; 10 passed
  `max |rho| < 0.30`.
- The selected feature's maximum fit `|rho|` is 0.1743; correlation with
  `lr_impact_72` is 0.0609.
- 360 raw policies produced 358 unique masks and 24 eligible pre-2024 variants.
- Frozen Top-10 manifest rank: 9.
- Manifest internal SHA-256:
  `f2c92510f5a234c00577b9f2b006915e27926fa3433150688d821e92af9fa001`.
- Manifest file SHA-256:
  `5296906b0e454653919ebc293d459cebf033002918608c8b81a19563334128a1`.
- Replay reused the manifest without mutation and matched the complete selected
  output exactly.

## Strict statistics

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Approx. p |
|---|---:|---:|---:|---:|---:|---:|
| Fit 2023H1 | +25.50% | 84.05% | 9.81% | 8.57 | 13 | 0.0262 |
| Select 2023H2 | +0.43% | 0.85% | 4.24% | 0.20 | 15 | 0.8678 |
| Test 2024 | +23.21% | 23.16% | 6.33% | 3.66 | 28 | 0.00694 |
| Eval 2025 | +17.84% | 17.85% | 6.56% | 2.72 | 25 | 0.0693 |
| 2026 to Jun02 | +6.83% | 17.20% | 11.01% | 1.56 | 26 | 0.3329 |
| 2024–2026 to Jun02 | +55.10% | 19.91% | 11.41% | 1.75 | 79 | 0.00135 |

The combined approximate p-value is about 0.0135 after Top-10 Bonferroni
correction. Eight OOS quarters are positive, one has no split-contained trade,
and one is negative (2025Q4, -2.64%). These are useful alpha-pool statistics,
but the 2026 and combined risk-adjusted returns remain below live requirements.

## Distinctness

Combined OOS executed-date Jaccard is 0.2174 versus the OI-gate watchlist and
0.3333 versus the `lr_impact_72` candidate. Annual overlap varies, but 2026 is
low at 0.122 versus OI and 0.231 versus `lr_impact_72`. The timing is therefore
not a duplicate, although all three share the fixed funding/premium base.

## Limits

- `passes_alpha_pool=true`; `passes_live_grade=false`.
- The fit and selection windows are short because six-alt auxiliary history
  starts in 2023.
- The broader program has inspected 2024–2026, so fresh forward proof is still
  required even though this manifest itself was frozen before 2024 replay.
- No threshold, hold, stride, or gate target may be retuned from later results.

## Artifacts

- `training/search_funding_premium_alt_crowding_gate_alpha.py`
- `tests/test_search_funding_premium_alt_crowding_gate_alpha.py`
- `results/funding_premium_alt_crowding_gate_top10_manifest_2026-07-13.json`
- `results/funding_premium_alt_crowding_gate_alpha_scan_2026-07-13.json`
- `results/funding_premium_alt_crowding_gate_replay_verification_2026-07-13.json`
- `configs/policies/funding_premium_alt_premium_change_outer_alpha_candidate.json`
