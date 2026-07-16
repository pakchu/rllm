# Cross-Venue Temporal Torsion preregistration (2026-07-16)

## Hypothesis

CVTT does not ask only whether Spot or USD-M was more active or moved later.
It asks whether the **flow-to-price temporal order inside each venue is
crossed**:

1. Spot flow precedes Spot price response while USD-M price response precedes
   USD-M local flow: follow the Spot flow direction.
2. USD-M flow precedes USD-M price response while Spot price response precedes
   Spot local flow: follow the USD-M flow direction.

This is consistent with a source venue preloading information and the other
venue echoing it before its own local aggressive flow arrives. It is not proof
of causality or participant identity.

## Why it is a new axis

The repo already tested venue lateness, basis stretch, lagged flow response,
activity/ticket surprise, and aggregate flow. Before this preregistration,
repo search found no alpha implementation using either within-venue
`return_time_centroid - flow_time_centroid` delay or their crossed Spot/USD-M
torsion. CVTT also excludes OI, funding, premium, FX/kimchi, REX,
HHI/effective-count, TAAR tail/arrival, and Coinbase inputs.

## Frozen family

| Policy | Route | Hold after entry |
|---|---|---:|
| V01 | Spot preload → USD-M echo | 6 bars (30m) |
| V02 | Spot preload → USD-M echo | 18 bars (90m) |
| V03 | USD-M preload → Spot echo | 6 bars (30m) |
| V04 | USD-M preload → Spot echo | 18 bars (90m) |

For each route, the score is the geometric mean of positive source preload
and positive destination echo. It must exceed the strictly-prior rolling
30-day 95th percentile among clean, directionally confirmed crossed-clock
bars. The threshold is shifted one bar and needs seven days of eligible prior
history. An episode starts only after 12 completed bars without the same route.

## Data and execution

- Source: official-checksummed Binance Spot and USD-M one-minute archives,
  aggregated into completed five-minute timing centroids.
- Source feature SHA-256:
  `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc`.
- Selection source must physically stop before parsing 2023 non-date values.
- Invalid source bucket plus the next 24 bars is quarantined; no imputation.
- Feature for bucket `t` becomes available at `t+5m`; assumed fill is the
  USD-M open at `t+10m`, leaving one complete latency bucket.
- Leverage 0.5x; base cost 6bp and stress cost 8bp per notional side; exact
  realized funding.
- Strict MDD includes global/pre-entry high-water, held favorable-before-
  adverse OHLC, entry/hypothetical liquidation costs, slippage, and funding.
- CAGR uses the full calendar, including idle time.

## Selection and holdout

- Fit/report years: 2020 and 2021; selection year: 2022.
- A policy must be positive in every year and at least five of six half-years,
  have combined CAGR/strict-MDD at least 2, survive 8bp cost, and pass a
  Bonferroni-adjusted weekly sign-flip test across all four policies.
- Only one exact policy may open 2023. It must produce positive absolute
  return, CAGR/strict-MDD at least 3, strict MDD at most 10%, at least 150
  trades, nonnegative H1/H2, and positive 8bp and delayed-entry controls.
- 2023 has been viewed in unrelated repo research, so this is a mechanically
  sealed exact-policy test, not pristine market history. Untuned forward
  shadow/live evidence remains mandatory.

## Orthogonality gate

Only after standalone and holdout survival, compare executed behavior against
the full frozen portfolio: exact-entry Jaccard ≤2%, entries near an existing
entry within ±6h ≤25% (target ≤10%), occupied-position Jaccard ≤15%, and
absolute daily-PnL Pearson ≤30% with at least ten nonzero days. Undefined
metrics fail closed, and synchronized portfolio MDD must improve.
