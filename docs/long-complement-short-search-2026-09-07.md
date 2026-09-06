# Long-complement short search — 2026-09-07

## Objective and decision

The target is protection/complement for existing longs, **not a standalone profitable short strategy**. Tested148 rule/control settings (144 dynamic rules) and17 ML/control settings (16 models/horizons/quantiles/weights).

**No new robust production short was established in this batch.** Both selection rank-one candidates deteriorate in2025. Lower-ranked frozen alternatives have partial protection but inconsistent payoff; they are not substituted as validated winners.

## Execution

Parent is G9 + macro1. The hedge is capped at observable positive parent net units, cannot create an extra net short when the parent is flat/short, and closes/reduces with parent barrier exits. Same-BTC exposures net before costs/funding. Thus this overlay is economically a dynamic reduction of net long exposure, not an independent alpha while flat.

Tests cover net-cost equivalence, parent absence, oversized hedges, compulsory barrier closes, and phantom post-close ruin. For multiple intrabar barriers, possible orderings are evaluated and the lower terminal-equity path retained; risk scenarios likewise honor compulsory hedge closure. Tick-level fill ordering remains unobserved.

Controls include no hedge and always-on reduction at each searched coefficient. No frequency, overlap, correlation or standalone-profit rejection gate is used.

## Rule search

Eight families: failed rebound, sell ignition, quiet breakdown, crowded-long breakdown, distribution, currency risk-off, regional unwind, trend acceleration. Fixed holds3/6/12h and hedge coefficients0.25/0.5/1.0. Completed hourly features execute at the next5m open. 2024 selects five distinct paths;2025 and full2026H1 are fixed reports.

Rank-one `distribution_t0.5_h12_w1.0` improved2024 but reduced2025 return from208.93% to175.67% with almost unchanged MDD. It also worsened2026H1 MDD. Reject as an established complement.

A preselected alternative, `failed_rebound_t0.5_h12_w0.25`, uses standardized168h momentum < -0.5, six-hour price z-score >0.5, and six-hour aggressive flow <-0.01. It holds up to12h, capped by the parent's actual long exposure.

| Period | Parent return / MDD | With failed-rebound overlay return / MDD |
|---|---:|---:|
| 2024 | 262.92% /19.03% | 256.69% /16.75% |
| 2025 | 208.93% /15.24% | 195.37% /14.81% |
| 2026H1 | 100.20% /18.83% | 102.03% /17.97% |

These are6bp/side figures including funding.2025 sacrifices13.56pp return for only0.43pp less MDD. Long-only parent tests also do not establish uniform improvement.

## ML search

Ridge and shallow histogram gradient boosting predict6/12h future BTC return, training only on observed positive-parent exposure states. Labels must mature strictly before the fit cutoff.2024H1 fits2024H2;2024 fits2025;2024–2025 fit2026. Quantile thresholds are fitted on training predictions and capped at -12bp. No LLM training was resumed.

2024H2 selects three fixed alternatives. The Ridge rank-one hedge worsens2025 return and MDD, although2026H1 return improves. It is not promoted.

The frozen HGB6h,20th-percentile,0.25x alternative has smaller upside drag but limited protection:

| Period | Return delta / MDD delta, actual parent,6bp | Return delta / MDD delta, long-only parent,6bp |
|---|---:|---:|
| 2024H2 | -2.70pp / -1.44pp | -2.58pp / -0.82pp |
| 2025 | -3.03pp /0.00pp | -0.94pp /0.00pp |
| 2026H1 | +0.65pp /-0.16pp | +1.56pp /-0.48pp |

At10bp/side the actual parent's2026H1 incremental return becomes -0.62pp; the long-only parent retains only+0.25pp. That is not strong, cost-robust evidence of a new hedge alpha.

## Boundaries

- Every period was already exposed historically; none is pristine OOS.
- Rule2024 results cover the full year; ML2024 results cover **H2 only**.
- Long-only controls remove existing negative target states but preserve the frozen entry/exit clock; they are diagnostic portfolios.
- Parent source reconstruction retains its original split-contained clock conventions.
- No September extension, live activation, or production deployment is claimed for these new short candidates.
- No new dependencies or live configuration changes were made.

## Artifacts

- `research/short_complement_context/report.json` and three NPZ files: exact parent baseline parity, feature timing and source hashes.
- `research/long_complement_shorts/report.json`: all rule selection inventory and fixed reports, including long-only controls.
- `research/ml_long_complement_shorts/report.json`: mature-label fit receipts, selected alternatives and fixed reports.
- `training/short_complement_ledger.py`: portfolio-aware short execution and risk logic.
