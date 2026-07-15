# Causal BOCPD pullback overlay audit — 2026-07-15

## Verdict

**Rejected.** Four of eight overlays satisfy the absolute pre-2024 gate, but none beats the frozen pullback premium-overheat comparator on the same lexicographic selection score. The best overlay therefore is not incremental alpha, and 2024+ was not opened for this family.

## Causal contract

- Hour `H` contains exactly `[H-1h,H)`; the unfinished `HH:00` 5-minute bar is excluded.
- BOCPD output is mapped only to the exact hour boundary; no stale two-hour carry-forward.
- Standardization, state thresholds, and state trade quality use 2020-07 through 2022-12 only.
- Entry, realized funding, 6 bp/notional/side cost, split-contained exits, and strict MDD are inherited unchanged from the frozen state machine.

Metric format: absolute return / CAGR / strict MDD / CAGR-MDD / trades.

## Comparator versus best overlay

| Policy | Train | 2023 selection | Pre-2024 | Score |
|---|---:|---:|---:|---:|
| Frozen comparator | 116.08% / 36.06% / 7.96% / 4.53 / 75 | 15.04% / 15.05% / 3.92% / 3.84 / 15 | 148.58% / 29.70% / 7.96% / 3.73 / 90 | `[3.7287757935873365, 3.8402707506691334, 90.0]` |
| Best BOCPD overlay | 109.68% / 34.43% / 7.96% / 4.32 / 75 | 15.04% / 15.05% / 3.92% / 3.84 / 15 | 141.22% / 28.59% / 7.96% / 3.59 / 90 | `[3.5896560063912846, 3.8402707506691334, 90.0]` |

## Best overlay specification

```json
{
  "hazard_hours": 336,
  "primary_quantiles": [
    0.25,
    0.75
  ],
  "short_mass_quantile": 0.75,
  "secondary_quantile": 0.5,
  "min_state_trades": 3,
  "min_state_edge": 0.0
}
```

Passing overlay cells: `4/8`; comparator-beating cells: `0/8`.
