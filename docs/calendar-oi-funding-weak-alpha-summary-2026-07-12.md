# Calendar/OI/Funding weak alpha candidate (2026-07-12)

## Candidate
`calendar_oi_funding_friday_asia_long_20260712`

Long-only BTC setup:
- `cf_funding_flip_long >= 1.4055940540679024`
- `cal_post_funding_1h >= 0.5`
- `cal_asia >= 0.5`
- `cf_basis_z <= -0.5452289044223437`
- hold 144 x 5m bars, stride 12

## Stats
| split | abs return | CAGR | strict MDD | CAGR/MDD | trades | win rate |
|---|---:|---:|---:|---:|---:|---:|
| train pre-2024 | +30.66% | +8.36% | 15.85% | 0.53 | 203 | 0.48 |
| test 2024 | +11.96% | +11.94% | 16.16% | 0.74 | 72 | 0.56 |
| eval 2025 | +5.65% | +5.65% | 10.62% | 0.53 | 63 | 0.52 |
| ytd 2026 | +1.73% | +4.19% | 9.64% | 0.43 | 33 | 0.52 |

## Verdict
This is not live-grade. CAGR/MDD is much lower than the target 3.0.

But it is worth keeping as a weak alpha because:
1. It is positive in train/test/eval/ytd.
2. Trade count is more meaningful than many sparse candidates.
3. Logic is orthogonal to REX price-action: derivatives/calendar/funding basis.
4. It can be used as a portfolio sleeve or LLM context feature.

## Failed neighboring scans
- Trade-size/impact: test2024 strong, eval/ytd negative.
- Extrema-arrival: train/eval/ytd negative.
- Volume-clock: test2024 strong, eval negative.
- OI/liquidation: test/eval unstable, ytd negative.
- Jump-variation: one qualifier exists but train is negative; keep only as historical candidate.

## Leakage/contamination note
This candidate is selected after inspecting an existing scan, so it is **candidate_weak**, not promotion-grade. It requires fresh OOS before live use.
