# Cross-collateral near-pressure pre-2024 selection

The candidate uses no funding, premium, REX, price, or return input.  It measures the signed near-book (1%-2%) net depth flow in USD-M and COIN-M, standardizes each venue against a strictly lagged robust 30-day baseline, and combines their pressure.

## Frozen policy

- Feature: `near_plain`
- H1-only absolute-score quantile: `0.985` = `4.434387570833191`
- Entry: threshold-onset side at next 5-minute open; fixed hold `288` bars
- 0.5x, 6 bp/notional/side, realized funding, no TP/SL, non-overlap, split-contained exits
- Multiplicity disclosed: `104` unique cells; 2024+ remained unopened

## Results

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit_2023h1 | 40.8407% | 99.5845% | 6.3180% | 15.7621 | 107 | 52/55 |
| selection_2023h2 | 14.3118% | 30.4108% | 9.2815% | 3.2765 | 131 | 65/66 |
| q1 | 24.4003% | 142.5584% | 6.3180% | 22.5640 | 47 | 21/26 |
| q2 | 13.2158% | 64.5766% | 5.7069% | 11.3155 | 60 | 31/29 |
| q3 | 1.9696% | 8.0514% | 5.9545% | 1.3522 | 67 | 31/36 |
| q4 | 12.1037% | 57.3972% | 9.2815% | 6.1841 | 64 | 34/30 |
| full_2023 | 60.9975% | 61.0500% | 9.2815% | 6.5776 | 238 | 117/121 |

## Interpretation

- Both H1 fit and H2 selection clear CAGR/strict-MDD 3, and every 2023 quarter is positive.
- Q3 is the weak block (ratio about 1.35), so this is not live-grade evidence by itself.
- Selection used 104 cells and one year of book data; overfit risk remains high until 2024+ OOS.
- The policy is now immutable. Future data may reject it but cannot change its formula, threshold, or hold.
