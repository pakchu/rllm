# Expanding ExtraTrees top-10 selection before 2025

2025+ was not available to this selection run. The feature graph was physically truncated at `2025-01-01`; ranking used only 2023 test and 2024 validation.

- Grid: `1152` cells
- Selection-pass cells: `313`
- Models: five deterministic `300`-tree ExtraTrees ensembles
- Manifest hash: `0bd68c3fda187c9e399b533442f9e3a9ecb4c7efaab756c374f18d1702d56394`

| Rank | Learner | Policy | 2023 abs/CAGR/MDD/ratio/trades | 2024 abs/CAGR/MDD/ratio/trades | Combined ratio/trades |
| ---: | --- | --- | --- | --- | --- |
| 1 | `d3/leaf32/mf0.5` | `λ0.5/fq0.35/pq0.5/rq0.8` | 13.07%/13.08%/3.12%/4.20/20 | 16.32%/16.29%/3.46%/4.70/23 | 4.24/43 |
| 2 | `d3/leaf32/mf0.5` | `λ0.5/fq0.35/pq0.5/rq0.85` | 13.07%/13.08%/3.12%/4.20/20 | 16.32%/16.29%/3.46%/4.70/23 | 4.24/43 |
| 3 | `d2/leaf32/mf0.8` | `λ0.5/fq0.45/pq0.55/rq0.85` | 12.86%/12.87%/3.12%/4.13/19 | 18.66%/18.62%/3.46%/5.38/23 | 4.54/42 |
| 4 | `d2/leaf32/mf0.8` | `λ0.5/fq0.45/pq0.55/rq0.75` | 12.86%/12.87%/3.12%/4.13/19 | 17.40%/17.36%/3.46%/5.01/22 | 4.36/41 |
| 5 | `d2/leaf32/mf0.8` | `λ0.5/fq0.45/pq0.55/rq0.8` | 12.86%/12.87%/3.12%/4.13/19 | 17.40%/17.36%/3.46%/5.01/22 | 4.36/41 |
| 6 | `d2/leaf32/mf0.8` | `λ0.5/fq0.45/pq0.55/rq0.7` | 12.86%/12.87%/3.12%/4.13/19 | 17.33%/17.29%/3.46%/4.99/21 | 4.35/40 |
| 7 | `d2/leaf32/mf0.8` | `λ0.25/fq0.4/pq0.55/rq0.7` | 12.86%/12.87%/3.12%/4.13/19 | 17.02%/16.98%/3.46%/4.90/20 | 4.31/39 |
| 8 | `d2/leaf32/mf0.8` | `λ0.5/fq0.4/pq0.55/rq0.85` | 12.86%/12.87%/3.12%/4.13/19 | 16.67%/16.64%/3.46%/4.80/23 | 4.26/42 |
| 9 | `d2/leaf32/mf0.8` | `λ0.25/fq0.4/pq0.55/rq0.75` | 12.86%/12.87%/3.12%/4.13/19 | 16.40%/16.36%/3.46%/4.72/22 | 4.22/41 |
| 10 | `d2/leaf32/mf0.8` | `λ0.5/fq0.4/pq0.55/rq0.75` | 12.86%/12.87%/3.12%/4.13/19 | 15.43%/15.40%/3.46%/4.45/22 | 4.08/41 |

## Frozen execution contract

- Completed signal at t, next-open entry at t+1.
- All predictive features delayed 12×5m; current source identity only is retained.
- Exact source-owned exits, 6bp/notional/side, realized funding.
- Stop-before-take ambiguity, non-overlap, split-contained exits.
- Wall-clock CAGR and favorable-before-adverse strict global HWM MDD.
- Annual expanding refits purge labels whose exits reach the cutoff.

## Limitation

This is a retrospective clean-room reconstruction: the program does not use 2025+, but earlier human research in this repository had already viewed those periods. The separate evaluator must therefore report this as algorithmically isolated, not human-pristine.
