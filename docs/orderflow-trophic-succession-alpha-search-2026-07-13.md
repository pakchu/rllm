# Order-Flow Trophic Succession Alpha — Preflight

Date: 2026-07-13

## Hypothesis

Rather than treating volume and trade count as static bar features, this experiment
models a causal market-participant succession across three completed,
non-overlapping 5-minute phases:

1. a **sponsor** phase with large tickets, directional flow and price progress;
2. a **crowd** phase with smaller tickets, higher trade intensity and the same flow;
3. an **absorption** phase where flow no longer produces proportional price progress.

The economic claim is that large-ticket sponsorship followed by fragmented crowd
participation can either continue while absorption is absent or reverse after
absorption appears. Only OHLCV, quote volume, trade count and taker-buy quote volume
are used; no open interest, funding, transfer entropy or liquidation-scar inputs are
shared with the immediately preceding research families.

An initial hard-AND prototype produced only zero to two crowd events and was
discarded as an underpopulated representation, not reported as a tradable scan.
The final transparent experiment uses continuous equal-weight role scores.

## Protocol

- Physical source rows strictly before `2024-01-01`; 2024+ OOS stayed unopened.
- Every role score uses only completed non-overlapping phases and prior-only
  2,016-bar standardization; entry is at the next 5-minute open.
- 96 fixed policies: six phase profiles, role tails `{q70,q80,q90,q95}`, mappings
  `{continuation, absorption reversal}`, and holds `{24,72}` bars.
- 0.5x exposure, 6 bp/side implementation cost, split-contained exits, and
  favorable-first/adverse-second OHLC high-water strict MDD.
- Admission required positive fit and 2023, fit CAGR/MDD at least 2, 2023 at least
  1.25, adequate long/short support, and non-negative 2023 halves.

## Best adequately populated policy

Profile `(6,6,3)`, q95 role tails, continuation mapping, 72-bar hold:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Fit (2020-06..2022) | -9.33% | -3.72% | 23.70% | -0.16 | 327 |
| Selection 2023 | +5.80% | +5.80% | 9.97% | +0.58 | 117 |
| 2023 H1 | +4.69% | +9.69% | 9.97% | +0.97 | 66 |
| 2023 H2 | +1.06% | +2.11% | 6.82% | +0.31 | 51 |

The fit half-years were unstable: 2021 H1 returned -15.48% and 2022 H2 -9.64%,
despite positive 2021 H2, 2022 H1 and both 2023 halves. No policy passed admission.

## Cost decomposition

| Cost per side | Fit return / CAGR / MDD / ratio | 2023 return / CAGR / MDD / ratio |
|---|---:|---:|
| 0 bp | +10.33% / +3.88% / 21.15% / +0.18 | +13.50% / +13.50% / 8.83% / +1.53 |
| 1 bp | +6.78% / +2.57% / 21.56% / +0.12 | +12.17% / +12.18% / 9.02% / +1.35 |
| 3 bp | +0.02% / +0.01% / 22.39% / +0.00 | +9.58% / +9.59% / 9.40% / +1.02 |
| 6 bp | -9.33% / -3.72% / 23.70% / -0.16 | +5.80% / +5.80% / 9.97% / +0.58 |

The phase sequence contains a weak gross directional edge, but it does not pay the
repository-standard round-trip cost in fit.

## Structural controls at 6 bp/side

| Variant | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -28.50% / -0.35 / 327 | -18.67% / -0.98 / 117 |
| Sponsor/crowd phase-order swap | -41.46% / -0.42 / 361 | -12.13% / -0.64 / 168 |
| Ticket-role definition swap | -28.19% / -0.37 / 223 | -9.89% / -0.71 / 69 |
| Sponsor-only collapse | -71.30% / -0.53 / 2,112 | -36.78% / -0.93 / 820 |
| Full-sequence delayed by 15 bars | -14.67% / -0.27 / 327 | +0.91% / +0.08 / 117 |

The controls show that role order and timing matter; the result is not a disguised
single-bar flow threshold. They do not rescue its economic weakness.

## Decision

Reject the exact 96 static tail/onset/hold policies before OOS. Retain the continuous
sponsor/crowd/absorption role scores only as a weak beta for a structurally different
campaign-aggregation test. That follow-up must reduce turnover by confirming repeated
same-direction succession episodes; it must not retune the role tails on 2024+ data.

Artifacts:

- `training/search_orderflow_trophic_succession_alpha.py`
- `results/orderflow_trophic_succession_alpha_scan_2026-07-13.json`
- `tests/test_search_orderflow_trophic_succession_alpha.py`
