# Order-flow campaign terminal-absorption alpha search (2026-07-14)

## Decision

**Reject the entire static transition; add no new beta; keep 2024+ sealed.**

The parent trophic campaign follows repeated sponsor-to-crowd continuation
events. This experiment tested a different economic transition: after a frozen
campaign confirmation, wait for the first same-side aggressive-flow phase that
no longer produces price progress, then fade the campaign side. The hypothesis
was coherent and causally executable, but its primary direction lost in nearly
every pre-2024 segment. The opposite direction was not stable in 2023.

## Frozen construction

- Parent source fixed before this experiment: q95 profile `(12,24,6)`, trailing
  144 bars, at least two same-side events, at most one opposite event, and a
  144-bar campaign cooldown.
- For campaign side `s`, terminal score is
  `s*a_imbalance_z - s*a_return_z - a_impact_z - s*a_clv`.
- The score uses the parent's already-frozen q95 numeric absorption threshold;
  this is not a literal later parent-role event because `s` remains the earlier
  campaign side.
- Campaign confirmed at completed bar `i`; scan only `i+1..i+h` inclusive.
  First threshold hit emits a fade at completed bar `j`, entering `j+1` open.
- One pending campaign at a time. New confirmations while pending, including
  at the endpoint, are ignored. Canonical execution skips overlapping trades.
- Two co-primary policies only: wait/hold tied at 6h or 12h. Their deterministic
  rank rule and all controls were fixed before opening returns.
- The returned analysis frame is hard-filtered before `2024-01-01`; 0.5x,
  6 bp/side, split-contained exits, and favorable-first/adverse-second OHLC
  strict MDD. No 2024+ row enters features, thresholds, signals, or outcomes.
- I/O disclosure: the shared 100,000-row chunk parser reads the cutoff-crossing
  chunk before immediately discarding its 2024+ rows. “Sealed” here means
  excluded from the returned analysis frame and every computation, not that
  the compressed file parser stopped reading bytes exactly at midnight.

This is contaminated exploratory recombination: the parent settings came from
an outcome-ranked rejected pre-2024 campaign scan. It is not represented as a
fresh independent discovery.

## Support-only preflight

The following counts were computed before any return simulation:

| Wait/hold | Fit raw / executable | 2023 raw / executable |
|---|---:|---:|
| 6h / 6h | 74 / 74 | 26 / 26 |
| 12h / 12h | 90 / 86 | 35 / 35 |

## Primary results

| Policy / split | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades (L/S) |
|---|---:|---:|---:|---:|---:|
| 6h / fit | -18.80% | -7.74% | 19.96% | -0.39 | 74 (39/35) |
| 6h / 2023 | -1.21% | -1.21% | 4.22% | -0.29 | 26 (16/10) |
| 12h / fit | -23.43% | -9.81% | 26.56% | -0.37 | 86 (47/39) |
| 12h / 2023 | -7.12% | -7.13% | 12.48% | -0.57 | 35 (19/16) |

The 6-hour policy lost in all five fit half-years and 2023H1; only 2023H2 was
positive (`+0.80%`, ratio `0.85`). Its fit mean-return p-value was `0.031`, but
for a negative effect: evidence against the proposed fade, not for an alpha.
The 12-hour policy showed the same failure and zero of two policies passed.

## Controls on the 6-hour policy

| Control | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| Exact direction flip (terminal continuation) | +11.65% / 0.54 | -1.96% / -0.62 |
| Parent campaign continuation | +9.62% / 0.43 | +8.92% / 1.56 |
| Absorption without campaign | -73.20% / -0.54 | -31.16% / -0.99 |
| Sponsor/crowd phase-order swap | -9.85% / -0.32 | -6.12% / -0.74 |
| Delay terminal signal 1h | -17.61% / -0.39 | +1.02% / 0.35 |
| Delay terminal signal 6h | -7.15% / -0.26 | -0.89% / -0.21 |
| Delay terminal signal 7d | +2.68% / 0.08 | -1.96% / -0.37 |

The parent continuation dominates the new transition. Exact inversion helps
fit but loses in 2023, so the terminal event does not identify a stable new
direction. Absorption without a campaign is extremely high-turnover noise.

## Cost stress on the 6-hour fade

| Cost/side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0 bp | -15.11% / -0.37 | +0.34% / 0.09 |
| 1 bp | -15.74% / -0.37 | +0.08% / 0.02 |
| 3 bp | -16.97% / -0.38 | -0.44% / -0.11 |
| 6 bp | -18.80% / -0.39 | -1.21% / -0.29 |
| 10 bp | -21.17% / -0.40 | -2.24% / -0.48 |
| 15 bp | -24.03% / -0.41 | -3.50% / -0.60 |

Fit loses materially at zero cost, so execution friction is not the root cause.
Campaign density and the parent role scores remain their previously recorded
weak-beta objects; their terminal-fade recombination creates no additional
representation worth preserving.

## Artifacts

- `training/search_orderflow_campaign_terminal_absorption_alpha.py`
- `tests/test_search_orderflow_campaign_terminal_absorption_alpha.py`
- `results/orderflow_campaign_terminal_absorption_alpha_scan_2026-07-14.json`
