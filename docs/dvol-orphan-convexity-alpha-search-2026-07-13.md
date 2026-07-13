# DVOL Orphan-Convexity Alpha Search — 2026-07-13

## Thesis

An implied-volatility rise is most interesting when it has no visible parent.
The experiment models each completed hourly Deribit BTC DVOL change from the
preceding realized variance, high-low range, absolute price jump and Binance
premium-index change. A large **positive residual** is an "orphan convexity"
event: options repriced before contemporaneous spot/perpetual stress explained
the move.

Direction is not taken from the DVOL number. It is assigned independently by
either completed prior-hour taker flow or a spot-perpetual basis impulse. The
economic hypothesis is that unexplained convexity demand plus signed price
discovery identifies latent directional information.

## Causal protocol

- Market, spot/premium and DVOL inputs are physically cut before `2024-01-01`.
- DVOL OHLC becomes visible only at the hourly candle `close_time` and joins by
  backward as-of with `65min` tolerance.
- Only consecutive hourly DVOL changes are accepted; gaps are not collapsed
  into one-hour shocks.
- Rolling `30d`/`90d` OLS predicts each update using only earlier hourly
  observations. The current target is added only after its residual is emitted.
- Predictor scaling and residual q90/q95 thresholds use `2021-04-15..2022-12-31`
  fit rows only.
- Five completed spot/premium one-minute rows are required in each decision bar.
- Signals enter at the next 5-minute open, use `0.5x`, pay `6bp/side`, and use
  favorable-first/adverse-second strict MDD.
- Source-file SHA-256 hashes are stored in the result artifact.
- `2024+` OOS was not opened.

## Frozen grid

Sixteen policies:

- residual history: `30d`, `90d`;
- positive residual tail: q90, q95;
- direction proxy: prior-hour taker flow, spot-perp basis impulse;
- hold: `12h`, `24h`.

## Best ranked policy

`90d`, q95 residual, taker-flow direction, `24h` hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | +11.86% | +4.43% | 31.50% | 0.14 | 279 | 105/174 |
| 2023 | -23.06% | -23.07% | 32.17% | -0.72 | 190 | 91/99 |
| 2023H1 | +1.07% | +2.16% | 12.86% | 0.17 | 88 | 35/53 |
| 2023H2 | -23.55% | -41.32% | 24.46% | -1.69 | 101 | 55/46 |

The fit result itself was unstable: `2021H1` ratio `3.01`, but `2022H2`
returned `-14.89%` with ratio `-1.30`.

## Falsification controls

At the selected hold and standard costs:

- exact direction flip: 2023 `-0.26%`, ratio `-0.01`;
- causal 12-hour signal lag: 2023 `-25.26%`, ratio `-0.73`;
- raw DVOL-change tail: 2023 `-25.19%`, ratio `-0.72`;
- raw realized-variance tail: fit `-5.54%`, ratio `-0.06`; 2023
  `+13.21%`, ratio `1.65`, with a negative 2023H2;
- direction-proxy swap: 2023 `+0.19%`, ratio `0.01`.

At zero cost the selected policy still lost `13.76%` in 2023. The failure is
therefore not explained by turnover alone. Residualization did not isolate a
stable direction; in 2023 the exact flip was materially less bad than the
declared mapping.

## Decision

**Rejected as alpha.** Zero of 16 policies passed preflight and OOS remained
sealed. Do not tune nearby residual windows, tails, proxy mappings or holds on
this sample. The exact orphan-convexity execution family is retained only as
gamma failure provenance.

Artifacts:

- `training/search_dvol_orphan_convexity_alpha.py`
- `tests/test_search_dvol_orphan_convexity_alpha.py`
- `results/dvol_orphan_convexity_alpha_scan_2026-07-13.json`
