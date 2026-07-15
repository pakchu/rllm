# Causal BOCPD pullback exit router selection

Metric format: absolute return / CAGR / strict MDD / CAGR-MDD / trades.

## Verdict

**Frozen for OOS.** BOCPD does not gate entries. It routes only normal state-5 trades to TP4 while preserving every base trade and all overheat/capitulation behavior.

| Policy | Train | 2023 selection | Pre-2024 | Score |
|---|---:|---:|---:|---:|
| Frozen base | 116.08% / 36.06% / 7.96% / 4.53 / 75 | 15.04% / 15.05% / 3.92% / 3.84 / 15 | 148.58% / 29.70% / 7.96% / 3.73 / 90 | `[3.7287757935873365, 3.8402707506691334, 90.0]` |
| Exit router | 122.32% / 37.61% / 7.96% / 4.72 / 75 | 15.04% / 15.05% / 3.92% / 3.84 / 15 | 155.76% / 30.76% / 7.96% / 3.86 / 90 | `[3.8402707506691334, 3.861743984294992, 90.0]` |

## Leakage controls

- The market source is physically truncated before 2024.
- Hour H is built from `[H-1h,H)` and mapped only at exact H.
- Standardization, state thresholds, and TP action quality use only 2020-07 through 2022-12.
- The one-shot manifest must be committed before `--open-oos` can read later rows.
- Next-open entry, realized funding, 6 bp/notional/side, non-overlap, split-contained exits, and strict MDD remain unchanged.
