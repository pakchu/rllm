# Preisach flow-scar Barkhausen alpha search (2026-07-14)

## Decision

**Reject the trading policy. Preserve the persistent relay/scar state as weak
beta only. Do not open 2024+.**

This experiment was designed to avoid relabeling ordinary breakout, dual-clock,
or rolling signed-area features:

- Twelve fixed Preisach relays retain `-1/+1` state while the price field stays
  inside each relay deadband. The same current field can therefore imply a
  different state after a different path.
- Every relay accumulates normalized aggressive taker flow since its last
  switch. This per-domain flow scar is not a rolling event count.
- When at least three coherent relays switch together, the policy measures flow
  that had opposed the new avalanche direction, divides it by current same-side
  taker chase, and follows the avalanche only in the fit-only q80 score tail.

The price field is current log close relative to a prior-only 12-hour rolling
median, scaled by prior-only 12-hour realized volatility. Flow is
`(2*taker_buy_quote - quote_volume) / previous_completed_hour_quote_volume`.

## Honest design history

Before reading any return, a state-support-only probe found that a seven-day
field with q90 produced only four fit signals. Windows 12h/24h/48h were checked
for event density only; the 12-hour field and q80 tail were frozen because they
supported the repository minimum trade-count target. No outcome was read during
that support check. There is one primary policy: 6-hour fixed hold.

All pre-2024 rows nevertheless remain globally exploratory/contaminated. 2023
is inspected internal selection, not pristine OOS.

## Causal and execution protocol

- Physical source cutoff strictly before `2024-01-01`; 2024+ stayed sealed.
- Relay thresholds are fixed dimensionless constants, not fitted.
- Prior rolling center, volatility, and quote denominator exclude the signal
  row; current completed close/flow may define the signal.
- Avalanche scars use flow only through `t-1`; current `t` flow is added only
  after switch scoring/reset for future bars.
- Signal at completed 5-minute close enters next 5-minute open.
- 0.5x exposure, 6 bp per side, non-overlapping 6-hour hold.
- Strict MDD uses conservative favorable-first/adverse-second OHLC ordering and
  split-contained exits.

## Primary result

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +1.68% | +0.75% | 10.74% | 0.07 | 101 | 61/40 |
| 2023 | -10.49% | -10.50% | 14.46% | -0.73 | 57 | 29/28 |
| 2023 H1 | -8.87% | -17.09% | 11.61% | -1.47 | 36 | 18/18 |
| 2023 H2 | -1.79% | -3.51% | 6.24% | -0.56 | 21 | 11/10 |

The policy had 1,377 coherent multi-relay avalanches, 1,067 positive opposing
scar events, and 265 raw q80 signals. Non-overlap left adequate support, so the
failure is not a small-sample gate artifact. Fit effect size was only `d=0.022`
with approximate p-value 0.821. In 2023 the mean effect was negative
(`d=-0.259`, approximate p-value 0.051).

## Structural falsification

| Control | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -13.62% / -0.36 / 101 | +4.01% / 0.65 / 57 |
| Remove scar; avalanche only | -17.51% / -0.28 / 606 | -26.10% / -0.92 / 291 |
| Erase deadband memory | -64.30% / -0.58 / 1,117 | -44.99% / -0.96 / 607 |
| Delay flow scar 7 days | -14.57% / -0.35 / 74 | -6.85% / -0.72 / 43 |
| Same-direction scar instead | +1.03% / 0.04 / 57 | +1.21% / 0.24 / 24 |
| Current flow chase only | +7.86% / 0.35 / 140 | -6.37% / -0.60 / 113 |
| Delay signal one bar | +2.12% / 0.09 / 101 | -8.91% / -0.68 / 57 |
| Delay signal one hour | -2.58% / -0.08 / 101 | -10.83% / -0.76 / 57 |
| Delay signal seven days | -8.14% / -0.29 / 104 | -1.71% / -0.33 / 56 |

The deadband-memory, scarless, and delayed-flow controls demonstrate that this
is not merely a renamed breakout or event clock. The representation contains
path-specific information. It still does not yield a stable side rule: the
primary mapping changes sign in 2023, while the exact flip is also too weak and
fails 2023 H2. The same-direction-scar variant is positive in both full windows
but has ratios 0.04/0.24 and loses in 2023 H2, so it is not a rescue.

## Cost stress

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0 bp | +8.03% / 0.37 | -7.38% / -0.61 |
| 1 bp | +6.94% / 0.32 | -7.91% / -0.63 |
| 3 bp | +4.80% / 0.22 | -8.95% / -0.68 |
| 6 bp | +1.68% / 0.07 | -10.49% / -0.73 |
| 10 bp | -2.35% / -0.09 | -12.51% / -0.78 |
| 15 bp | -7.16% / -0.21 | -14.97% / -0.84 |

Even zero-cost 2023 is negative. Costs are not the root cause.

## Conclusion

Freeze the 12-hour field, relay lattice, scar ordering, q80 tail, side mapping,
and 6-hour hold. The exact static policy belongs in gamma. Persistent relay
magnetization, relay ages, opposing/same-direction scars, avalanche count and
coherence may remain weak beta tokens for a materially different causal learner
or genuinely fresh-forward evidence.

## Artifacts

- `training/search_preisach_flow_scar_alpha.py`
- `tests/test_search_preisach_flow_scar_alpha.py`
- `results/preisach_flow_scar_alpha_scan_2026-07-14.json`
