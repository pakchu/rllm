# OI divergence pullback alpha — 2026-09-06

## Decision

**SHADOW/PAPER CANDIDATE; live disabled.** This is a distinct second alpha candidate, but the fresh replay contains only eight trades and does not justify live promotion.

## Frozen formula

Enter long at the next 5-minute open when all four pre-existing gates pass:

- 4-hour OI growth minus price return z-score >= 0.895401863
- 48-bar return z-score <= -0.738957066
- range volatility >= 0.040084155
- normalized RSI <= -0.045076568

Evaluate every 30 minutes, prohibit overlapping positions, and hold for 8 hours. The recent replay additionally requires current OI availability.

## Evidence

| Window | Return | strict MDD | CAGR/MDD | Trades | Win rate |
|---|---:|---:|---:|---:|---:|
| Historical 2024 | 52.54% | 6.34% | 8.27 | 64 | — |
| Historical 2025 | 36.60% | 5.46% | 6.71 | 40 | — |
| Original 2026 YTD | 0.62% | 9.65% | 0.16 | 17 | — |
| Fresh Jun 1–Aug 3, 6 bp/side | 6.96% | 5.25% | 8.95 | 8 | 75% |
| Fresh Jun 1–Aug 3, 10 bp/side | 6.28% | 5.25% | 7.94 | 8 | 75% |

The fresh report reproduced byte-for-byte with SHA-256 `ec108c6a7ebf6634431fbae42dbf34a26fadfb1792d418be026a2f54b253c725`. Annualized recent ratios are descriptive only because the window and trade count are small.

## Interpretation

The signal buys price pullbacks when open interest remains unusually strong, only in elevated range volatility and weak RSI conditions. It is structurally different from the hourly dollar-flow/regime-switch sleeve, but both can acquire long BTC exposure during stress regimes.

## Risks and next gate

- Eight fresh trades are insufficient for a robust standalone claim.
- Pre-2024 training CAGR/MDD was only 0.25; the strong 2024/2025 results may reflect selection luck.
- The authoritative OI sample stops on 2026-08-03, so there is no September confirmation.
- Keep disabled and collect forward paper trades without changing thresholds. Reassess only after materially more independent events.
