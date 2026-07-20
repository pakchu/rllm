# BATE-288 2021–2022 train rejection

Status: **rejected at the frozen train gate; 2023 selection remains unopened**.

The evaluator source was committed at
`d837c58e78f93e6ef492610d45ffb263648f59c3` and then sealed by the
zero-outcome freeze artifact before any market or funding value was parsed.
The train run used 1.0x leverage, next-open execution, exact frozen funding,
6 bp/notional/side base costs, 10 bp/notional/side stress costs, full-calendar
CAGR, and global strict path MDD.

## Frozen train result

| Policy/window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Primary, 2021–2022, 6 bp | +29.99% | +14.02% | 81.97% | 0.17 | 644 |
| Primary, 2021–2022, 10 bp stress | -22.38% | -11.90% | 86.34% | -0.14 | 644 |
| Primary, 2021 | +295.25% | +295.62% | 52.81% | 5.60 | 321 |
| Primary, 2022 | -66.59% | -66.61% | 74.46% | -0.89 | 322 |
| Primary HIGH-long contribution | -6.42% | -3.27% | 76.30% | -0.04 | 386 |
| Primary LOW-short contribution | +38.91% | +17.87% | 52.07% | 0.34 | 258 |
| Direction flip | -93.79% | -75.10% | 94.64% | -0.79 | 644 |
| Weight-only | -25.88% | -13.92% | 79.77% | -0.17 | 671 |
| Transaction-only | -11.70% | -6.04% | 70.79% | -0.09 | 665 |
| Denominator-free | -85.55% | -62.02% | 87.02% | -0.71 | 529 |
| 24-hour stale state | -89.09% | -66.99% | 92.35% | -0.73 | 643 |
| Year/side-matched random clock | -92.95% | -73.47% | 95.15% | -0.77 | 645 |
| One-bar delayed primary | +31.15% | +14.53% | 82.52% | 0.18 | 644 |

Primary mean gross side-adjusted move was `25.15 bp`, below the frozen
`30 bp` floor. The weekly-cluster sign-flip result was `p=0.21963` over 105
weekly clusters, above the frozen `0.10` ceiling. Funding was present in all
644 trades (1,932 settlements); aggregate funding cash was `-0.0130%` of the
sum of trade entry equities.

## Frozen gate failures

- CAGR / strict MDD below 3.0;
- strict MDD above 15%;
- 10 bp/notional/side stress return non-positive;
- mean gross side-adjusted move below 30 bp;
- weekly-cluster sign-flip `p > 0.10`;
- HIGH-long net contribution non-positive; and
- calendar 2022 absolute return non-positive.

The extra five-minute latency control stayed positive, but it inherited the
same extreme drawdown and does not rescue the singleton. None of the three
mechanism-null controls passed the primary gates, so the rejection is caused
by unstable primary economics rather than a stronger component-only null.

## Stopping decision

The frozen singleton does not permit a side, threshold, hold, cost, or regime
repair. BATE-288 is therefore rejected before 2023. The untouched 2023 values,
orthogonality checks, portfolio contribution checks, and all 2024+ windows
remain sealed for this candidate.

Artifacts:

- train result:
  `results/block_arrival_throughput_elasticity_train_2021_2022_2026-07-20.json`;
- train file SHA-256:
  `04150f70f838d092100a01b3fe6a07a6efb7e9f99b2ca05e164bb16ce14d15f4`;
- train result hash:
  `486c8fd02ad016734aa6f6ab482b5ea87a0c07cc0633f6cf0fd17c41d190a093`;
- evaluator freeze SHA-256:
  `9473f5ba1b821e1692b608dfa54c47fa8d666bb188e58f72d03dd08355e03a3c`.
