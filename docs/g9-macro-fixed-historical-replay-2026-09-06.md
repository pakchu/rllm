# Fixed G9/macro combinations — historical replay, 2026-09-06

## Scope

The same four combinations previously discussed were replayed without weight optimization: G9, G9 half + macro1, G9 + macro0.5, G9 + macro1.
2024 and 2025 are complete calendar years. **2026H1 runs January1 through June30**, with end-exclusive July1 00:00UTC. Earlier cache coverage through May31 was extended using the frozen recent DB snapshot.

## Results: return / strict MDD, 6bp per side and realized funding

| Combination | 2024 | 2025 | 2026H1 |
|---|---:|---:|---:|
| g9 | 215.26% / 16.78% | 181.51% / 15.24% | 99.50% / 18.67% |
| g9_half_macro1 | 108.38% / 11.33% | 87.67% / 8.57% | 43.68% / 9.75% |
| g9_macro0.5 | 238.68% / 17.73% | 195.20% / 15.24% | 99.93% / 18.75% |
| g9_macro1 | 262.92% / 19.03% | 208.93% / 15.24% | 100.20% / 18.83% |

## Cost stress: return at 10bp per side

| Combination | 2024 | 2025 | 2026H1 |
|---|---:|---:|---:|
| g9 | 180.51% | 158.32% | 82.65% |
| g9_half_macro1 | 91.57% | 74.80% | 35.63% |
| g9_macro0.5 | 197.55% | 167.14% | 81.83% |
| g9_macro1 | 214.84% | 175.72% | 80.86% |

## CAGR/MDD at base cost

| Combination | 2024 | 2025 | 2026H1 |
|---|---:|---:|---:|
| g9 | 12.78 | 11.92 | 16.23 |
| g9_half_macro1 | 9.54 | 10.24 | 11.06 |
| g9_macro0.5 | 13.41 | 12.82 | 16.25 |
| g9_macro1 | 13.77 | 13.72 | 16.24 |

## Interpretation

The prior recent-window impression does not generalize: halving G9 reduces drawdown but also reduces return and CAGR/MDD in every historical period here. It is lower exposure, not demonstrated long-run dominance.
Keeping G9 and adding macro improves 2024/2025 returns, with increased 2024 drawdown. 2026H1 incremental base-cost return is very small and turns negative relative to G9 under 10bp cost stress. There is no uniform all-regime winner.

## Source and accounting contract

2024/2025 Fresh and Rank7 schedules are reconstructed with the published historical contexts. Rank7 uses maturity-purged causal annual refits; the2026 model is not applied to2024/2025. Frozen annual-reference hashes and historical trade counts pass.
REX clocks use hash-verified original JSONL sources; Markov uses its frozen historical rule. 2024 constituent counts exactly match the authoritative203 trades. 2025 totals133.
Full2026H1 uses fixed native runtime adapters with only the2026 Rank7 bundle, genuine cached spot/premium metadata, and the recent database/5m-delayed OI archive extension. This is not the same source contract as legacy H1 summary metrics; the G9 control is recalculated under the same ledger for each comparison.
All strategies use the shared signed-unit BTC ledger. Costs and funding act on aggregate net units; native event holds, post-fee4.5x event cap and conservative barrier/ruin treatment are unchanged. Coefficients are equity-notional multipliers, not percentages.
Historical source schedulers retain their split-contained trade policy: entries whose eventual exits cross the source/window boundary may be omitted. The calendar grid is complete, but these schedules are not a continuously running live account across year boundaries. Macro inventory is liquidated at each report end.
All periods were exposed in prior research; none is pristine OOS. No period-specific weight fitting occurred. No live config was changed.

## Evidence

`research/g9_macro_historical_v2/report.json` — full figures and counts.
`research/g9_historical_barriers/report.json` — annual-model proof and exact barrier clocks.
`training/evaluate_g9_macro_historical.py` — fixed-weight historical evaluator.
Initial registration tuple/list serialization failure was corrected before economics; `research/g9_macro_historical/failure.json` records it.
