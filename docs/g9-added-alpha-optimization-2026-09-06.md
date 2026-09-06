# G9 plus additional-alpha portfolio optimization — 2026-09-06

## Result

All five G9 constituents were included alongside macro-flow, OI pullback and regional trend. **546 unique allocations** were ranked on June stress Calmar. Frozen top five all lost in July–September4. No optimizer winner is promoted.
An additional descriptive audit reports all 27 G9-plus-new local cells already present in the original frozen grid. They do not replace the failed June-selected winner.

## Common accounting

Window June1–September5 00:00 UTC. G9 source clocks: Fresh 7, Rank7 4, REX taker13, REX veto19, Markov4 trades. OI pullback9 trades.
Repository mainnet declaration is G8 (Rank7 weight2). Frozen G9 raises Rank7 to3. Actual deployed process not verified. Both are separate controls.
Legacy .5x base-sleeve convention becomes G9 notional coefficients [1,1.5,.2,.8,1], sum4.5. New targets use native1x coefficients. Net signed units determine cost/funding and event risk; no overlap/frequency/fee-ratio rejection. Random coefficient budgets4.5/6 allow opposite exposures to offset under netcap4.5.
Fixed-unit holds and source-specific stop/take prices are retained. Net-cap sizing includes fees. Conservative intrabar ruin is absorbing; all hedge-break barrier-exit subsets contribute to MDD. No exchange liquidation model.

## Results, 6bp per side + funding

| Configuration | June–Sep4 return | CAGR | MDD | CAGR/MDD | July–Sep4 return | Sep1–4 return | Entries / orders | 10bp full return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| repository_live_g8 | 9.89% | 43.16% | 14.35% | 3.01 | 3.68% | -0.28% | 32 / 86 | 7.09% |
| frozen_g9 | 10.19% | 44.63% | 17.31% | 2.58 | 3.89% | -0.28% | 32 / 86 | 7.21% |
| g9x0.5_macro_flow1.0 | 11.01% | 48.78% | 7.67% | 6.36 | 5.14% | -2.55% | 50 / 875 | 8.50% |
| g9x1.0_macro_flow0.5 | 13.24% | 60.47% | 16.77% | 3.61 | 5.52% | -1.49% | 54 / 875 | 9.67% |
| g9x1.0_macro_flow1.0 | 16.31% | 77.71% | 16.22% | 4.79 | 7.13% | -2.69% | 50 / 875 | 12.14% |

## Interpretation

Lower-risk discussion candidate: G9 x0.5 plus macro coefficient1.0. Full return11.01%, MDD7.67%, versus G9 return10.19% and MDD17.31%. This is not merely a fully invested percentage allocation; actual average/peak exposures differ.
Retain-G9 discussion candidate: G9 plus macro coefficient0.5; return13.24%, MDD16.77%. Larger macro addition1.0 raises full return16.31% but worsens September-only loss to-2.69%.
All shortlisted macro additions perform worse than G9 during Sep1–4. That tradeoff must not be hidden by the aggregate improvement.
June-selected top1 was Fresh1.75 + Markov0.25 + OI4.0 (notional coefficients): full+40.96%, but July onward-5.02% versus G9+3.89%. This demonstrates unstable selection and OI concentration, not a deployable optimum.

## Data and boundaries

OI uses official daily archives delayed5m. Spot/premium sources and the immutable annual Rank7 bundle are required; absent sleeves are not silently replaced by zero. Runtime source-readiness filters may reject individual signals as in the frozen adapters.
An expensive retrospective live-OI snapshot sort was replaced in the loader by archived DB OI; evaluated May onward OI is overwritten by the publication-delayed official archive. Raw cached frames are local and ignored. Predictor frames are recomputed after OI overlay, funding near-aliases canonicalized.
Every period was previously exposed. June is only one selection month. These diagnostics do not establish pristine OOS or an exhaustive/global optimum. Independent windows restart from cash and initialize known target states; interval returns are not chained contributions.
Live files were not modified. Disabled discussion candidates: `configs/shadow/g9_added_alpha_sensitivity_2026-09-06.json`.

## Evidence

`research/g9_september_inputs/report.json` — G9 constituent source receipts/trades.
`research/g9_added_alpha_optimization/report.json` — 546-allocation inventory and frozen-finalist reports.
`research/g9_added_alpha_optimization/fixed_additions/report.json` — all 27 predeclared local cells.
Ledger received independent code review after post-fee cap and intrabar ruin fixes; regression tests cover net fees, barriers, hedge break, carry quantities, cap sizing and insolvency.
