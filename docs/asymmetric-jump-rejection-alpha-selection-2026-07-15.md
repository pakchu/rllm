# Asymmetric jump/rejection candidate

- Selection data are physically truncated before `2024-01-01`.
- Long: signed jump variation + OI build + clean path.
- Short: upper rejection + rising DXY + fast causal volume clock.
- Entry: next 5-minute open; 24-hour max hold; 3% stop; 4% take.
- Same-bar ambiguity is conservative: stop is evaluated before take.
- Cost: 6bp/notional/side plus realized funding; leverage 0.5x.
- Strict MDD: global high-water plus conservative favorable-then-adverse intratrade path.
- Multiplicity: 448 source rules, 625 asymmetric pairs (460 unique stat signatures), then bounded execution refinements.
- Status: frozen pre-2024 candidate, not yet an OOS-confirmed alpha.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | L/S |
|---|---:|---:|---:|---:|---:|---:|
| fit | 109.28% | 39.63% | 7.90% | 5.01 | 224 | 136/88 |
| fit_2020q4 | 11.03% | 63.23% | 5.30% | 11.94 | 22 | 19/3 |
| fit_2021 | 52.01% | 52.05% | 7.90% | 6.59 | 101 | 65/36 |
| fit_2021_h1 | 39.03% | 94.44% | 7.90% | 11.95 | 52 | 37/15 |
| fit_2021_h2 | 9.34% | 19.38% | 7.20% | 2.69 | 49 | 28/21 |
| fit_2022 | 24.00% | 24.02% | 6.60% | 3.64 | 101 | 52/49 |
| fit_2022_h1 | 13.99% | 30.24% | 6.09% | 4.97 | 59 | 29/30 |
| fit_2022_h2 | 8.78% | 18.18% | 6.60% | 2.76 | 42 | 23/19 |
| select_2023 | 32.30% | 32.33% | 6.73% | 4.80 | 80 | 51/29 |
| select_2023_h1 | 15.30% | 33.29% | 4.02% | 8.27 | 42 | 27/15 |
| select_2023_h2 | 14.74% | 31.39% | 6.73% | 4.67 | 38 | 24/14 |
