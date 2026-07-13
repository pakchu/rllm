# CME–offshore debt handoff alpha preflight — 2026-07-13

## Hypothesis

A venue-level leverage handoff may occur when weekly CME leveraged-money net
positioning changes opposite to a Binance offshore state that combines OI growth
and ownership change. The handoff score is

`-z(CME leveraged-money weekly change) × z(Binance ΔOI × Δownership)`.

A positive score means CME and offshore leverage are moving in opposite
directions; the policy fades the offshore receiver. This is cross-venue debt
migration, distinct from funding, price-only momentum and single-venue OI
levels.

## Causal protocol

- Market and Binance metrics were physically truncated before 2024.
- Binance metrics were delayed by one complete 5-minute source bar.
- CFTC report rows were exposed only at `report_date + 8 days`, a deliberately
  conservative availability rule; later releases were physically excluded.
- Offshore ownership uses causal global-account and market taker-flow state,
  avoiding sparse 2022 top-trader fields.
- Weekly and bar z-scores use histories shifted through `t-1`.
- Signals occur once at the conservative CFTC release anchor and execute at the
  next 5-minute open.
- Fit/thresholds use data through 2022; 2023/H1/H2 select. OOS was not opened.
- 0.5x, 6bp/side, fixed non-overlapping holds and strict OHLC MDD.

Grid: Binance handoff window `{7d,14d}` × CFTC standardization `{52w,104w}` ×
fit tail `{q80,q90}` × hold `{24h,48h}` = 16 candidates.

## Strongest populated candidate

7d Binance handoff, 104w CFTC standardization, q80, 24h hold.
Metric format: `absolute return / CAGR / strict MDD / ratio / trades`.

| Window | Result |
|---|---:|
| Fit through 2022 | `+13.09 / +5.72 / 15.08 / 0.38 / 22` |
| 2023 | `+11.92 / +11.93 / 4.41 / 2.70 / 19` |
| 2023 H1 | `+6.13 / +12.76 / 3.28 / 3.88 / 10` |
| 2023 H2 | `+5.45 / +11.11 / 3.55 / 3.13 / 9` |

The appealing 2023 split is not enough: 2022 H1 and H2 were both negative
(`-2.44%`, `-3.32%`), full-fit ratio was only `0.38`, and only 22 fit trades
exist. This does not meet the user's ratio-3 target or statistical sample size.

## Matched controls

| Variant | Fit result | 2023 result |
|---|---:|---:|
| Direction flip | `-15.04 / -7.10 / 23.10 / -0.31 / 22` | `-12.88 / -12.89 / 13.83 / -0.93 / 19` |
| Asset-manager net instead | `-0.12 / -0.05 / 13.50 / -0.00 / 22` | `+7.21 / +7.21 / 3.49 / 2.07 / 7` |
| CFTC delayed four extra weeks | `+7.12 / +3.16 / 7.42 / 0.43 / 22` | `+11.17 / +11.18 / 4.38 / 2.56 / 10` |
| Offshore state only | `+3.62 / +1.62 / 15.69 / 0.10 / 22` | `+5.84 / +5.85 / 4.38 / 1.34 / 11` |

The exact direction matters, but a four-week-extra-stale CFTC control nearly
matches 2023. Therefore the result cannot be attributed confidently to the
release-time handoff mechanism.

## Decision

**Reject the static policy in preflight; do not open OOS.** Preserve the
continuous CME/offshore handoff state as weak beta because it is genuinely
cross-venue and directionally active, but never call it standalone alpha until
a fresh, much longer event sample clears fit and forward gates.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_cme_offshore_debt_handoff_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_cme_offshore_debt_handoff_alpha.py
```
