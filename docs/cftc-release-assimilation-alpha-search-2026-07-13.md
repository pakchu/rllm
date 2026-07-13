# CFTC Release-Assimilation Alpha Search — 2026-07-13

## Thesis

CFTC positions describe Tuesday inventory but become public later. At the
conservative release time, both the position surprise and the intervening BTC
price path are known. This experiment asks whether that surprise has already
been assimilated:

- **unpriced**: price moved opposite the position surprise or not with it;
  follow the newly disclosed side;
- **over-assimilated**: same-direction price movement exceeded the standardized
  surprise magnitude; fade the stale inventory disclosure.

The feature is not a static CFTC level. It is a causal relationship between a
weekly participant-inventory surprise and the market's price digestion during
the publication lag.

## Causal protocol

- Market rows are physically cut before `2024-01-01`.
- Every CFTC report is unavailable until `report_date + 8d`, deliberately later
  than normal publication.
- Participant net share and its weekly change are normalized by open interest.
- Position surprise and report-to-release BTC return use prior-only 104-report
  z-scores with 52-report minimum history.
- The release decision uses the first completed 5-minute bar at/after the
  conservative release; execution is the next open.
- Reporting at `report_date` was documented as an invalid lookahead placebo and
  was not simulated.
- Position size is `0.5x`, cost is `6bp/side`, and strict MDD is
  favorable-first/adverse-second OHLC high-water.
- `2024+` OOS was not opened.

## Frozen grid

Sixteen policies:

- participant: leveraged money, asset manager;
- absolute fit surprise tail: q80, q90;
- state: unpriced follow, over-assimilated fade;
- hold: `48h`, `96h`.

## Best ranked policy

Asset-manager q90, unpriced follow, `48h` hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +5.24% | +2.00% | 9.28% | 0.22 | 7 | 5/2 |
| 2023 | +3.75% | +3.76% | 3.72% | 1.01 | 3 | 1/2 |
| 2023H1 | +0.71% | +1.43% | 2.49% | 0.58 | 1 | 0/1 |
| 2023H2 | +3.02% | +6.09% | 3.72% | 1.64 | 2 | 1/1 |

The aggregate sign is interesting but statistically unusable: seven fit trades
and three 2023 trades are far below the `25/8/3-per-half` admission floor.
`2022H1` was also negative.

## Falsification controls

At the selected policy and standard costs:

- exact direction flip: fit `-5.99%`, 2023 `-4.07%`;
- ignore assimilation and use the static surprise tail: fit `-0.23%`, 2023
  `+3.75%`;
- execute the same release four weeks late: fit `-3.88%`, 2023 `-2.36%`;
- swap to leveraged-money surprises: fit `-2.36%`, 2023 `-0.61%`.

At zero cost, the selected policy reaches only fit ratio `0.24` and 2023 ratio
`1.06`. Cost is not the limiting factor; weekly support and fit instability are.

## Decision

**Rejected as alpha.** Zero of 16 policies passed preflight, and OOS remained
sealed. The positive top sign is not promoted to beta because its sample is too
small to distinguish an effect from chance. Retain only exact gamma failure
provenance; do not tune assimilation cutoffs, tails or holds on this sample.

Artifacts:

- `training/search_cftc_release_assimilation_alpha.py`
- `tests/test_search_cftc_release_assimilation_alpha.py`
- `results/cftc_release_assimilation_alpha_scan_2026-07-13.json`
