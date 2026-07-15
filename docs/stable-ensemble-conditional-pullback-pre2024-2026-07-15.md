# Stable ensemble conditional-pullback selection — 2026-07-15

**Selection candidate found; 2024+ remains sealed. This is not yet an OOS claim.**

Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.

## Weak-signal interaction

- Average five independently seeded 2,000-tree shallow forests; this removes the earlier single-seed lottery.
- Forest fitting remains parallel, but tree predictions are reduced in a fixed single-thread order. Parallel accumulation differed by only a few ULPs (`~5e-18`) yet changed quantile-tie membership; two fresh processes now produce identical prediction, activation, and result hashes.
- Calibrate funding and premium score thresholds independently from fit-window examples only.
- Premium events use only their source score threshold.
- Funding events use the score threshold, then require either a non-compressed 28-day range or a deep completed-daily-bar pullback.
- Economic interpretation: avoid ordinary funding entries inside quiet ranges; permit them only at a range extreme, while retaining high-dispersion opportunities.
- Source-owned exits are unchanged: funding 48h/4% take/no stop; premium 12h/no take/3% stop.
- All thresholds are computed on 2020-07-01..2022-12-31; selection is 2023; physical source cutoff is 2024-01-01.

## Search evidence

- Complete cells: **240**.
- Formal-gate passes: **6**.
- Passing cells with an adjacent pass: **6**.
- All six passes form one contiguous 3×2 neighborhood; no isolated winner was selected.

## Selected pre-OOS specification

- Funding score quantile: `0.3`; premium score quantile: `0.5`.
- Low-width quantile: `0.2` (`0.09761263`).
- Pullback quantile: `0.4` (`-0.24066869`).

| Window | Result |
|---|---:|
| train | 104.16% / 33.01% / 8.15% / 4.05 / 128 |
| train_2020h2 | 4.22% / 8.54% / 4.35% / 1.96 / 25 |
| train_2021 | 53.74% / 53.79% / 7.84% / 6.86 / 62 |
| train_2022 | 26.32% / 26.34% / 8.15% / 3.23 / 40 |
| select_2023 | 11.09% / 11.10% / 3.12% / 3.56 / 19 |
| select_2023_h1 | 8.90% / 18.78% / 3.12% / 6.02 / 13 |
| select_2023_h2 | 2.01% / 4.03% / 2.30% / 1.75 / 6 |
| pre_2024 | 126.80% / 26.35% / 8.15% / 3.23 / 147 |

## Stop condition before OOS

The exact selected rule must survive per-seed, ensemble-size, one-hour-delay, and interaction-ablation audits. Only then may a freeze commit be created and 2024+ opened once.

Result hash: `2e0c80328bf33c16a2784190964e6b5ca923cf122c4f52eff67a1a0d3900026f`
