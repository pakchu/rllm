# Online RLS price-impact alpha preflight (2026-07-13)

Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.

| policy | fit | 2023 | 2023H1 | 2023H2 | flipped 2023 |
|---|---:|---:|---:|---:|---:|
| `H576 high continuation` | -47.84/-22.26/49.66/-0.45/553 | -0.89/-0.89/1.90/-0.47/6 | -0.89/-1.79/1.90/-0.94/6 | 0.00/0.00/0.00/0.00/0 | 0.14/0.14/1.48/0.09/6 |
| `H576 high fade` | 9.83/3.69/18.02/0.21/279 | -0.06/-0.06/2.67/-0.02/4 | -0.06/-0.11/2.67/-0.04/4 | 0.00/0.00/0.00/0.00/0 | -0.47/-0.47/3.26/-0.15/4 |
| `H576 low continuation` | -26.42/-11.19/27.73/-0.40/483 | -40.27/-40.29/41.69/-0.97/483 | -17.57/-32.29/19.29/-1.67/180 | -27.54/-47.24/28.80/-1.64/303 | -7.04/-7.04/8.01/-0.88/483 |
| `H576 low fade` | 3.84/1.47/12.17/0.12/233 | -10.65/-10.65/15.11/-0.70/216 | -1.26/-2.53/5.30/-0.48/84 | -9.51/-17.99/15.11/-1.19/132 | -14.51/-14.52/17.85/-0.81/216 |
| `H2016 high continuation` | -26.91/-11.42/34.30/-0.33/235 | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 |
| `H2016 high fade` | 1.32/0.51/29.64/0.02/174 | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 | 0.00/0.00/0.00/0.00/0 |
| `H2016 low continuation` | -20.66/-8.57/21.91/-0.39/216 | -11.99/-12.00/16.32/-0.74/214 | -7.70/-14.93/8.77/-1.70/88 | -4.65/-9.03/9.50/-0.95/126 | -13.18/-13.18/19.11/-0.69/214 |
| `H2016 low fade` | -5.82/-2.29/18.85/-0.12/165 | -23.24/-23.26/28.25/-0.82/179 | -13.02/-24.53/20.89/-1.17/75 | -11.75/-21.98/17.58/-1.25/104 | 3.15/3.15/12.22/0.26/179 |

## Verdict

- Eligible fixed policies: **0 / 8**; OOS metrics calculated: **no**.
- Fast/high/fade was the only policy with a meaningful positive fit result, but it had only four 2023 trades and lost slightly; no 2023H2 trade existed.
- Low-impact regimes generated enough trades but lost materially in 2023 in continuation and fade mappings.
- Direction flips did not create fit plus two-half stability. Reject these exact static mappings without spending 2024-2026 evidence.
- The continuous pre-update slope and prior-scaled residual remain beta context only, not an executable alpha.

## Leakage controls

- The selection frame is verified to contain no row at or after 2024 before feature construction. The CSV reader may buffer a cutoff-crossing chunk; buffered later rows are filtered before entering the market frame.
- The current target never updates the slope used at that bar; the current residual never updates its own z-score denominator.
- All trades enter at the next 5m open and strict MDD includes intratrade adverse OHLC.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_online_rls_price_impact_alpha --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz --manifest-output results/online_rls_price_impact_top8_manifest_2026-07-13.json --output results/online_rls_price_impact_alpha_scan_2026-07-13.json --docs-output docs/online-rls-price-impact-alpha-preflight-2026-07-13.md
```
