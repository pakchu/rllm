# Conformal SR leverage-pressure alpha search (2026-07-14)

## Decision

**Reject both static mappings; preserve the sequential pressure-evidence state
as weak beta; keep 2024+ excluded.**

This experiment introduced a new event layer rather than another static tail.
At each completed hour it formed a leverage-pressure residual:

`flow_z * (1 + clip(oi_z, 0, 3)) - hourly_price_return_z`.

All z-scores use the preceding 30 days. The current residual is then ranked
against only the preceding 180 days using conservative one-sided conformal
ranks. Two power-betting Shiryaev–Roberts restart statistics accumulate
repeated upper- or lower-tail departures. An event occurs only when evidence
crosses the fixed boundary 200, after which both statistics reset.

The sequential detector is the novel object. Primitive flow/price/OI residuals
overlap prior impact and inventory work, but the evidence-event timestamps do
not materially overlap those prior static families.

## Causal and statistical protocol

- Returned analysis frame is hard-filtered before `2024-01-01`. The shared CSV
  parser can read and immediately discard later rows in its cutoff-crossing
  chunk; no such row enters features, thresholds, signals, or outcomes.
- Hourly signal uses the 12 completed 5-minute rows ending at minute 55 and
  enters the next minute-00 open.
- OI is delayed one complete 5-minute source row before hourly sampling.
- Normalization: 30 days, minimum 15 days, always shifted one hour.
- Conformal reference: preceding 180 days, minimum 90 days; current residual is
  excluded. Exact formulas are
  `(1 + count(previous >= current)) / (n + 1)` and
  `(1 + count(previous <= current)) / (n + 1)`, with conservative ties.
- Power bet is `0.5 * p^-0.5`; SR recursion is `(1 + R_previous) * bet`.
- Boundary was raised from nominal 100 to 200 before outcomes as a conservative
  two-tail union adjustment. No anytime-valid guarantee is claimed under
  dependent market data; this is a sequential change detector.
- Two co-primary economic maps only: fade absorbed pressure or follow stored
  pressure release. Both are always reported. One fixed 12-hour hold.
- 0.5x, 6 bp/side, split-contained non-overlapping execution, and conservative
  favorable-first/adverse-second OHLC strict MDD.
- 2023 is inspected internal selection; 2024+ stayed excluded from computation.

## Support-only and novelty preflight

Before opening returns, both exact-opposite maps had identical support:

| Split | Raw events | Strict executable |
|---|---:|---:|
| Fit | 150 | 142 |
| 2023 | 49 | 48 |

There were 199 total events, balanced 111 positive-pressure / 88
negative-pressure. The frozen event-overlap gate required every baseline
Jaccard below 0.50:

| Baseline event family | Jaccard |
|---|---:|
| Online-RLS `abs(residual_z)>=1.5` | 0.021 |
| Nonequilibrium probability-current q80 | 0.005 |
| Single conformal tail `p<=0.01` | 0.137 |
| Same SR detector without OI multiplier | 0.098 |

Maximum Jaccard was 0.137, so the event layer passed the preregistered novelty
gate. Feature-level Spearman was high versus the no-OI residual (`0.922`), but
OI materially changed when evidence crossed and later proved economically
necessary.

## Primary results

| Mapping / split | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades (L/S) |
|---|---:|---:|---:|---:|---:|
| Release / fit | +38.71% | 15.94% | 15.63% | 1.02 | 142 (83/59) |
| Release / 2023 | +6.67% | 6.67% | 7.33% | 0.91 | 48 (22/26) |
| Fade / fit | -41.17% | -21.32% | 42.71% | -0.50 | 142 (59/83) |
| Fade / 2023 | -11.88% | -11.89% | 17.70% | -0.67 | 48 (26/22) |

Release was positive in four full fit half-years plus the short 2020Q4 segment,
and strongly rejected the fade interpretation. However, 2021H2 was weak
(`ratio 0.51`) and 2023H2 lost `-1.86%` (`ratio -0.53`). Zero of two mappings
passed the required fit-and-2023 CAGR/MDD 3 gate.

## Structural controls on release

| Control | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| Remove OI multiplier | -3.91% / -0.10 | -10.86% / -0.71 |
| Single tail; no sequential evidence | +15.59% / 0.25 | +3.57% / 0.25 |
| Reverse flow sign before residual | -40.73% / -0.50 | -19.57% / -0.85 |
| Delay signal 1h | +26.10% / 0.88 | +2.02% / 0.25 |
| Delay signal 24h | -7.19% / -0.15 | -13.80% / -0.96 |
| Delay signal 7d | -2.96% / -0.08 | -2.09% / -0.52 |

OI amplification, flow orientation, sequential accumulation, and local timing
all contribute. This is stronger representation-level evidence than a simple
residual tail, but not an executable static alpha.

## Cost stress on release

| Cost/side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0 bp | +51.05% / 1.31 | +9.78% / 1.56 |
| 1 bp | +48.92% / 1.26 | +9.26% / 1.45 |
| 3 bp | +44.75% / 1.16 | +8.21% / 1.21 |
| 6 bp | +38.71% / 1.02 | +6.67% / 0.91 |
| 10 bp | +31.05% / 0.82 | +4.64% / 0.58 |
| 15 bp | +22.06% / 0.58 | +2.15% / 0.24 |

The effect survives high costs, so turnover is not the root failure. The
problem is temporal/path risk, especially 2023H2. Freeze boundary, windows,
multiplier, mapping and hold. Any reuse should expose continuous residual,
conformal ranks and SR evidence to a materially different preregistered learner
rather than retune this sample.

## Artifacts

- `training/search_conformal_sr_pressure_alpha.py`
- `tests/test_search_conformal_sr_pressure_alpha.py`
- `results/conformal_sr_pressure_alpha_scan_2026-07-14.json`
