# Execution Metronome Absorption Alpha Search — 2026-07-14

## Decision

**Rejected as alpha and rejected as a novel static composite.** The policy loses
fit and 2023 before costs, its exact flip does not generalize, and the score is
dominated by ordinary flow pressure. The spectral-regularity component is not
necessary under the preregistered ablation gate. No 2024+ outcome was opened.

## Thesis

Algorithmic execution can split a parent order into regularly sized child
orders. If average ticket size follows a concentrated within-hour frequency,
taker flow is persistent and large tickets fail to move price in the same
direction, passive liquidity may be absorbing the execution. The declared
policy fades that flow.

For each decision hour `T`:

1. use exactly the 12 completed BTCUSDT five-minute bars in `[T-60m,T)`;
2. calculate average ticket `quote volume / trade count` per bar;
3. linearly detrend the 12 log-ticket observations and calculate normalized
   rFFT power entropy over the six non-zero frequencies;
4. set regularity to `1 - entropy`, while invalidating near-zero residual power;
5. combine regularity with positive prior-only average-ticket z-score,
   persistent signed-flow pressure and failed signed price acceptance;
6. fade the fit-only q90 score tail.

## Pre-outcome review and repairs

The independent critic returned `REVISE`. Before any future return was opened,
the implementation added:

- invalidation of constant/linear near-zero-power ticket paths;
- finite/positive 12-bar accounting checks;
- fit-only q01 floors for hour quote volume, trade count, absolute-flow
  fraction and price-path length;
- an explicit minute-00 signal and minute-05 entry contract;
- split-contained non-overlap support with both sides in both 2023 halves;
- direct overlap checks against frozen trophic continuation, terminal
  absorption, campaign and chirp events;
- a required outcome ablation: primary must beat the no-regularity control in
  return and CAGR/MDD in both fit and 2023.

Ten targeted tests passed before outcome opening.

## Causal protocol

- The returned market frame is strictly before `2024-01-01`.
- A cutoff-crossing parser chunk may be read and discarded, but no discarded
  row enters a returned frame or computation.
- Bars through minute-55 complete at minute-00. The signal is assigned to the
  minute-00 row and enters only at minute-05.
- Fit is `2020-06-01..2022-12-31`; 2023 is inspected internal selection with
  H1/H2 robustness. 2024+ remains sealed.
- Hold is fixed 12 hours, leverage `0.5x`, implementation cost `6bp/side`.
- Strict MDD uses favorable-first/adverse-second OHLC high-water ordering.

## Support-only gate

Support was measured before future extremes, simulation, legacy event audit or
result writing.

| Split | Executable trades | Long | Short |
|---|---:|---:|---:|
| Fit | 532 | 324 | 208 |
| 2023 | 260 | 148 | 112 |
| 2023 H1 | 110 | 63 | 47 |
| 2023 H2 | 150 | 85 | 65 |

All support requirements passed.

## Primary result

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| Fit | -38.07% | -16.92% | 42.46% | -0.40 | 532 | 324/208 |
| 2023 | -28.89% | -28.91% | 30.70% | -0.94 | 260 | 148/112 |
| 2023 H1 | -11.91% | -22.58% | 17.45% | -1.29 | 110 | 63/47 |
| 2023 H2 | -19.28% | -34.63% | 20.48% | -1.69 | 150 | 85/65 |

Only `2022 H2` was marginally positive among the seven reported fit/selection
half-years.

## Structural controls

| Control | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Remove regularity | -32.28% / -0.34 / 529 | -34.36% / -0.97 / 257 |
| Remove ticket pressure | -38.62% / -0.41 / 593 | -24.46% / -0.90 / 269 |
| Remove nonacceptance | -46.00% / -0.38 / 589 | -30.05% / -0.95 / 273 |
| Plain flow pressure | -55.20% / -0.44 / 617 | -22.62% / -0.94 / 281 |
| Regularity only | -62.57% / -0.49 / 645 | -24.44% / -0.87 / 277 |
| Exact direction flip | -21.32% / -0.28 / 532 | +1.12% / +0.10 / 260 |
| Signal delay 1h | -29.50% / -0.36 / 532 | -28.14% / -0.90 / 260 |
| Signal delay 24h | -14.70% / -0.23 / 532 | -15.04% / -0.77 / 259 |
| Signal delay 7d | -25.00% / -0.24 / 532 | -27.67% / -0.94 / 262 |

The flip loses fit, so the declared failure is not repaired by reversing the
economic sign. Removing regularity is less negative in fit and more negative in
2023; primary therefore fails the fixed regularity-necessity gate.

## Cost stress

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0bp | -14.78% / -0.19 | -16.89% / -0.80 |
| 1bp | -19.20% / -0.24 | -19.02% / -0.84 |
| 3bp | -27.35% / -0.31 | -23.12% / -0.91 |
| 6bp | -38.07% / -0.40 | -28.89% / -0.94 |
| 10bp | -49.95% / -0.45 | -35.92% / -0.97 |
| 15bp | -61.64% / -0.49 | -43.74% / -0.98 |

Zero-cost failure rejects turnover as the root cause.

## Novelty audit

Overlap with the older trophic families is near zero:

- q95 trophic continuation: `0.0025`;
- q95 terminal absorption: `0.0000`;
- campaign: `0.0020`;
- chirp: `0.0000`.

However, the primary is not novel relative to its own simpler components:

- no-nonacceptance event Jaccard: `0.621`;
- no-regularity event Jaccard: `0.545`;
- score Spearman with flow pressure: `0.782`.

Maximum fixed-control event Jaccard exceeds the preregistered `0.60` gate, and
regularity is not outcome-necessary. The exact static composite is therefore a
flow-pressure variant despite its low overlap with older named families.

## Freeze decision

- Keep within-hour ticket spectral entropy/regularity only as a weak beta
  representation for a materially different learner.
- Record the exact q90 large-ticket/coherent-flow/nonacceptance fade as gamma
  failure provenance.
- Do not tune frequency bins, detrending, q tail, denominator floors, component
  weights, direction, delay or hold on this inspected sample.

## Artifacts

- `training/search_execution_metronome_absorption_alpha.py`
- `tests/test_search_execution_metronome_absorption_alpha.py`
- `results/execution_metronome_absorption_alpha_scan_2026-07-14.json`
