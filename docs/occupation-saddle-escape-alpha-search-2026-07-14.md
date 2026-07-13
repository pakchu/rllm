# Causal occupation-saddle escape alpha search (2026-07-14)

## Decision

**Reject the static policy; preserve only the frozen occupation-landscape state
as weak beta; keep 2024+ sealed.**

The experiment treated the prior 30 days of log-price occupation as a causal
potential landscape. At each UTC day boundary it froze a 64-bin profile jointly
weighted by elapsed time and quote volume. A low-density saddle between two
adjacent high-occupation modes represented a candidate basin boundary. A
completed hourly close crossing that saddle entered at the next 5-minute open
and continued in the crossing direction.

This is a label-free geometric construction. It does not use future returns,
outcome-fitted thresholds, or 2024+ data.

## Protocol

- Source physically truncated before `2024-01-01`; 2024+ was never opened.
- Every daily profile contains only the 8,640 bars strictly before that UTC day.
- Fixed design: 30 days, 64 bins, five-bin smoothing, saddle depth ratio <=0.50,
  mode separation >=3 bins, and 6h/12h holds: two policies total.
- Completed minute-55 crossing enters next minute-00 open.
- 0.5x exposure, 6 bp per side, non-overlapping holds, split-contained exits,
  and conservative favorable-first/adverse-second OHLC strict MDD.
- 2023 is inspected internal selection; all pre-2024 evidence is exploratory.
- A support-only preflight used an index-relative day boundary and counted
  277 fit plus 157 selection signals. Before any return evaluation, that bug was
  corrected to true UTC timestamp boundaries; the final state has 467 raw
  pre-2024 crossings. A regression test covers non-midnight source starts.

## Primary results

| Hold / split | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades (L/S) |
|---|---:|---:|---:|---:|---:|
| 12h / fit | +5.14% | 2.29% | 12.75% | 0.18 | 104 (46/58) |
| 12h / 2023 | +10.04% | 10.05% | 6.62% | 1.52 | 72 (46/26) |
| 6h / fit | -17.42% | -8.29% | 22.20% | -0.37 | 138 (71/67) |
| 6h / 2023 | -3.88% | -3.88% | 6.91% | -0.56 | 90 (51/39) |

The 12-hour version was the only plausible row, but zero of two policies passed
admission. Its fit approximate mean-return p-value was 0.697 and its 2023 value
was 0.207. It lost in 2020H2 and 2021H1, was nearly flat in 2021H2, and its
2023H2 ratio fell to 0.50. This is not statistically or temporally robust.

## Falsification controls on the 12-hour policy

| Control | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| Exact direction flip | -18.22% / -0.35 | -17.18% / -0.96 |
| Delay crossing 1h | -1.19% / -0.04 | +1.56% / 0.20 |
| Delay crossing 24h | -9.64% / -0.22 | -2.47% / -0.29 |
| Delay crossing 7d | -21.11% / -0.30 | -7.62% / -0.73 |
| Time-only occupation | +8.64% / 0.29 | +5.18% / 0.65 |
| Volume-only occupation | +19.31% / 0.58 | +6.23% / 0.98 |
| Reversed price-volume association | +5.08% / 0.13 | +5.67% / 0.93 |

Direction flip and timing delays support a weak local crossing effect. However,
the joint time-volume profile is not uniquely identified: time-only,
volume-only, and even reversed price-volume association remain similarly weak,
with volume-only stronger in fit. The defensible beta object is therefore the
generic frozen occupation/saddle geometry, not the claimed joint-liquidity
mechanism or this static continuation rule.

## Cost stress

| Cost/side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0 bp | +11.91% / 0.45 | +14.90% / 2.39 |
| 1 bp | +10.75% / 0.40 | +14.08% / 2.24 |
| 3 bp | +8.47% / 0.31 | +12.45% / 1.94 |
| 6 bp | +5.14% / 0.18 | +10.04% / 1.52 |
| 10 bp | +0.85% / 0.03 | +6.92% / 1.01 |
| 15 bp | -4.26% / -0.12 | +3.14% / 0.44 |

The 2023 row approaches but still misses the target even at zero cost; fit is
far below it. Costs weaken the result but do not explain the regime instability.

## Artifacts

- `training/search_occupation_saddle_escape_alpha.py`
- `tests/test_search_occupation_saddle_escape_alpha.py`
- `results/occupation_saddle_escape_alpha_scan_2026-07-14.json`
