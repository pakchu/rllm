# Causal weak tensor exit router selection

Metric: absolute return / CAGR / strict MDD / CAGR-MDD / trades.

## Verdict

**Frozen for OOS.** The base remains the sole entry trigger. The tensor combines three price-structure weak features with five completed-hour BOCPD diagnostics to route TP4/TP8/TP12; earlier exits may admit a later base trigger under the unchanged non-overlap rule.

Multiplicity: 1008 cells; 426 beat the frozen base after the absolute gate.

| Policy | Train | 2023 selection | Pre-2024 | Score |
|---|---:|---:|---:|---:|
| Frozen base | 116.08% / 36.06% / 7.96% / 4.53 / 75 | 15.04% / 15.05% / 3.92% / 3.84 / 15 | 148.58% / 29.70% / 7.96% / 3.73 / 90 | `[3.7287757935873365, 3.8402707506691334, 90.0]` |
| Tensor router | 122.52% / 37.66% / 7.96% / 4.73 / 76 | 16.31% / 16.33% / 3.92% / 4.16 / 15 | 158.82% / 31.20% / 7.96% / 3.92 / 91 | `[3.917555353824354, 4.164847814208693, 91.0]` |

## Leakage controls

- Every source is physically truncated before 2024 for selection.
- Market features are prior-bar live features; BOCPD uses `[H-1h,H)` and exact H mapping.
- Scaling, BOCPD standardization, and counterfactual action labels stop before 2023.
- Entry is next-open; costs, realized funding, split-contained exits, and strict MDD are unchanged.
