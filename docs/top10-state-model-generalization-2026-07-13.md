# Top-10 state-model family generalization

Date: 2026-07-13

## Policy update

The promotion unit is now the **pre-evaluation ranked Top-10 family**, rather
than only rank 1. A member may demonstrate family-level generalization if it was
already inside the frozen Top-10 before later-period metrics were inspected.

This does not prove that its exact parameterization is uniquely optimal. Within
each family, the representative is therefore the highest pre-evaluation rank
among members that pass the fixed OOS gates—not the member with the best OOS
return.

## Surviving Top-10 members

| candidate | pre-eval rank | Test 2024 | Eval 2025 | 2026 YTD | trades 2024/25/26 |
|---|---:|---:|---:|---:|---:|
| Kalman | 4 | 4.74 | 3.16 | 10.25 | 19 / 15 / 18 |
| Kalman | 5 | 7.26 | 3.07 | 6.51 | 18 / 10 / 11 |
| BOCPD | 8 | 4.79 | 6.48 | 8.18 | 25 / 21 / 22 |
| BOCPD | 10 | 4.26 | 5.46 | 12.02 | 22 / 21 / 24 |
| Semi-Markov | 7 | 10.49 | 4.80 | 6.20 | 22 / 17 / 25 |

Table values are full-window CAGR/strict-MDD. Every candidate has positive
absolute return in all three later windows.

## Canonical family representatives

- Kalman: `kalman_rank4`
- BOCPD: `bocpd_rank8`
- Semi-Markov: `semimarkov_rank7`

These are chosen by the highest frozen pre-evaluation rank among passing family
members.

## Headline statistics

### Kalman rank 4

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +17.01% | 16.97% | 3.58% | 4.74 | 19 |
| Eval 2025 | +12.43% | 12.44% | 3.94% | 3.16 | 15 |
| 2026 YTD | +13.54% | 35.69% | 3.48% | 10.25 | 18 |

At 10 bp/side, Eval 2025 falls to ratio `2.86`. This is the most
execution-sensitive representative.

### BOCPD rank 8

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +22.94% | 22.89% | 4.77% | 4.79 | 25 |
| Eval 2025 | +19.83% | 19.85% | 3.06% | 6.48 | 21 |
| 2026 YTD | +14.18% | 37.52% | 4.59% | 8.18 | 22 |

At 10 bp/side, ratios remain `4.45 / 5.96 / 7.33`; at 15 bp/side they
remain `4.04 / 5.37 / 6.37`. This is the strongest cost-robust family.

### Semi-Markov rank 7

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +37.10% | 37.01% | 3.53% | 10.49 | 22 |
| Eval 2025 | +16.44% | 16.45% | 3.43% | 4.80 | 17 |
| 2026 YTD | +8.66% | 22.08% | 3.56% | 6.20 | 25 |

At 10 bp/side, ratios remain `9.99 / 4.49 / 5.19`; 2026 falls to `4.00`
at 15 bp/side.

## Statistical diagnostics

Trade-level bootstrap intervals are clearly positive in Test 2024 for all five
survivors. BOCPD rank 8 also has positive lower bounds in Eval 2025 and 2026.
Semi-Markov rank 7 has a positive Eval-2025 interval but its partial-2026
interval crosses zero. Kalman Eval-2025 intervals narrowly cross zero.

Accordingly:

1. BOCPD is the strongest statistically supported representative.
2. Semi-Markov remains a valid family-level candidate but is less certain in
   2026.
3. Kalman is a lower-confidence, execution-sensitive candidate.

## Signal diversity

OOS entry Jaccard overlap is low across model families:

- BOCPD rank 8 vs Semi-Markov rank 7: `0.128`
- BOCPD rank 8 vs Kalman rank 4: `0.062`
- Kalman rank 4 vs Semi-Markov rank 7: `0.064`

The two BOCPD variants overlap more strongly (`0.484`), so they should remain
one family for portfolio caps rather than count as independent alphas.

## Caveats

- The base funding/premium setup has prior research-history exposure.
- Top-10 survival establishes family-level evidence, not exact-parameter
  optimality.
- Most annual blocks contain only 10-25 trades; live promotion should begin as
  shadow or tightly capped size.
- 2022 remains weak for most candidates; Semi-Markov rank 7 is negative in that
  year.

## Artifacts

- Shared validator: `training/validate_top10_state_model_candidates.py`
- Result: `results/top10_state_model_validation_2026-07-13.json`
- Kalman candidate: `research/pools/alphas/kalman_top10_funding_premium_long_20260713.json`
- BOCPD candidate: `research/pools/alphas/bocpd_top10_funding_premium_long_20260713.json`
- Semi-Markov candidate: `research/pools/alphas/semimarkov_top10_funding_premium_long_20260713.json`
