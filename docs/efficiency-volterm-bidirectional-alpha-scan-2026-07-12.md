# Path-efficiency / volatility-term alpha scan

Date: 2026-07-12

## Question

Can directional path efficiency and the short/long realised-volatility term structure provide a causal BTC alpha that is structurally different from the existing REX, OI, funding/premium and calendar families?

## Protocol

- Input: BTCUSDT 5-minute data, 2020-01-01 through 2026-06-01.
- Features use only completed current/past bars.
- Entry is delayed by one bar.
- Feature thresholds are frozen on the pre-2024 Train split.
- Candidate ranking uses Test 2024 only.
- Eval 2025 and 2026 YTD are report-only diagnostics.
- Cost: 6 bp per side.
- strict MDD includes intraposition adverse excursion.
- Tested parameter combinations: 6,897.

## Result

No candidate passed the alpha-pool or live-grade validators.

The best Test-2024-ranked candidate was `efficient_expansion_144_0.8_0.9_0.8` with hold 144 bars, stride 24 bars, TP 2.5% and SL 1.5%.

| split | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long/short |
|---|---:|---:|---:|---:|---:|---:|
| Train | +38.86% | 8.55% | 11.47% | 0.75 | 655 | 301 / 354 |
| Test 2024 | +21.62% | 21.57% | 4.72% | 4.57 | 186 | 88 / 98 |
| Eval 2025 | -3.72% | -3.73% | 10.63% | -0.35 | 170 | 67 / 103 |
| 2026 YTD | -6.67% | -15.29% | 9.87% | -1.55 | 84 | 33 / 51 |

## Interpretation

The family found a strong local 2024 continuation effect but reversed in both later windows despite ample trade counts. This is not a sample-size failure: it is regime instability. The most visible deterioration is on the short side in 2025 and on both sides in 2026.

Path efficiency and volatility-term structure may still be useful as regime context or a veto feature, but this scan provides no evidence for promoting it as a standalone alpha. It must not be added to the live portfolio based on the 2024 result.

## Next structurally different searches

1. Completed-session handoff events using only the previous Asia/Europe/US session.
2. Funding-settlement events with funding cashflow separated from price PnL.
3. Causal interaction/hazard rules over the three sleeves of the Train-MDD-40 portfolio.

## Artifacts

- Script: `training/search_efficiency_volterm_bidirectional_alpha.py`
- Result: `results/efficiency_volterm_bidirectional_alpha_scan_2026-07-12.json`
