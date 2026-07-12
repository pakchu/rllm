# Regime-routed sparse REX alpha (2026-07-13)

## Verdict

A slow-state router repaired the side-balanced critics' sign instability, but
the resulting alpha is still too weak for promotion.  The best selected pair
was profitable in every future window and cleared the ratio target in 2026,
but missed `CAGR / strict MDD >= 3` in both 2024 and 2025.

## Protocol

- Specialists: the frozen five-long/five-short pre-2024 manifest.
- Routers: six predeclared zero-sign rules over signal-time daily, weekly,
  medium-trend, or 30-day range state.
- Pair selection: 2023 full/H1/H2 only; 150 pairs tested.
- Pre-future manifest hash:
  `714a6c75f7b89fbd9d4812f3354afd756d5eb65c3f133bfc23e485cef294f372`
- Pre-2024 OHLC is physically bounded before pair selection; the full market
  and future candidate files are opened only after the routed manifest write.
- Costs: 0.5x and 6 bp per side.
- MDD: corrected favorable-to-adverse intraposition high-water path.

## Best selected pair

- Long: ExtraTrees TAKE critic, q80, all REX families.
- Short: ExtraTrees path-utility critic, q70, all REX families.
- Router: long when the signal-time four-day daily-context return is nonnegative;
  short when it is negative.

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | p approx |
|---|---:|---:|---:|---:|---:|---:|
| 2023 Select | +11.76% | 11.77% | 2.32% | 5.06 | 37 | 0.0038 |
| 2024 Test | +6.87% | 6.85% | 4.69% | 1.46 | 33 | 0.2683 |
| 2025 Eval | +3.66% | 3.67% | 1.64% | 2.23 | 16 | 0.1519 |
| 2026 YTD | +2.78% | 6.82% | 2.01% | 3.40 | 11 | 0.2097 |

## Interpretation

The router is directionally useful: unlike the specialists alone, its return
did not flip negative across 2024-2026.  The effect size and trade count are too
small to call it a standalone alpha, and leverage cannot fix the ratio.

Its next legitimate use is as a complementary sparse sleeve.  In particular,
the routed short trades may offset the 2025 drawdown of the independently
selected continual positioning + DVOL long critic.  That combination must be
simulated as one chronological equity path; standalone MDD values cannot be
added or averaged.

## Artifacts

- Search: `training/search_rex_regime_routed_alpha.py`
- Tests: `tests/test_search_rex_regime_routed_alpha.py`
- Manifest: `results/rex_regime_routed_top10_manifest_2026-07-13.json`
- Result: `results/rex_regime_routed_alpha_scan_2026-07-13.json`
