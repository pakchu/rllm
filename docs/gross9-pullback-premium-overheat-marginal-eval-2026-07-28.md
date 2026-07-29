# Gross9 + frozen PPOSM future veto

Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

- decision: **terminal_veto_fixed_pposm_marginal**
- frozen weight: `0.75`
- reranked/repaired: `false/false`

| window | combined | same-gross comparator | ratio Δ | MDD reduction |
|---|---:|---:|---:|---:|
| eval2025 | 197.81% / 198.03% / 16.48% / 12.01 / 144 | 209.82% / 210.06% / 15.87% / 13.24 / 133 | -1.224 | -1.755%p |
| ytd2026 | 95.65% / 396.40% / 20.08% / 19.75 / 127 | 87.66% / 349.38% / 19.30% / 18.10 / 108 | +1.643 | -2.150%p |
| combined 2025–2026H1 | 482.67% / 246.51% / 20.08% / 12.28 / 271 | 481.41% / 245.98% / 19.30% / 12.74 / 241 | -0.466 | -2.150%p |

## Paired weekly evidence

- active weeks: `55`
- sign-flip p: `0.4846`
- bootstrap 90% lower mean log-effect: `-0.00106503`

## Boundary

- Standalone 2024–2026 results were known before this audit; this is a contamination-aware portfolio marginal audit, not pristine discovery OOS.
- The PPOSM signal, state thresholds, skip rule, exits, leverage, costs, and schedules were replayed from the frozen artifacts without tuning.
