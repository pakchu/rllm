# Confirmed pullback-squeeze alpha — 2026-07-15

> **Superseded / rejected under the current live-strict contract.** The original
> report used a `:55` positional clock with a two-hour premium tolerance, omitted
> realized funding, and used a less conservative MDD path. The live-parity audit
> reduces the full-history 0.5x ratio from `3.56` to `1.97`; at 0.9x, strict MDD
> rises to `21.14%`. See
> `docs/confirmed-pullback-squeeze-live-parity-audit-2026-07-15.md`.

## Decision

Promote this rule to the **research/shadow alpha pool**. This is the first weak-
signal interaction in this search that clears the long-period target without
requiring another alpha sleeve:

- full 2020H2-2026-05-31 at 0.5x: CAGR `27.33%`, strict MDD `7.68%`, ratio
  **`3.56`**, 139 trades;
- target operating point at 0.9x: CAGR **`52.75%`**, strict MDD **`13.52%`**,
  ratio **`3.90`**, 139 trades;
- at 0.9x and 10 bp per side: CAGR **`50.18%`**, strict MDD **`13.83%`**,
  ratio **`3.63`**.

This supersedes the unconfirmed candidate in
`specific-pullback-squeeze-alpha-2026-07-15.md`. It remains a shadow candidate,
not a live-capital recommendation, because the broader programme has already
inspected 2024-2026 and the effective research multiplicity is large.

## Specific interaction

The result is not a score average. Every layer has a different causal role.

### 1. Squeeze opportunity

Funding/trend branch:

```text
funding_available
and funding_rate <= train q10 (-0.00002222)
and trend_96 >= train q70 (0.009017208457522975)
```

Premium/momentum branch:

```text
premium_available
and premium_index_change <= train q20 (-0.00026817)
and completed htf_1d_return_4 >= train q90 (0.09605902316678483)
```

### 2. Pullback without immediate overextension

```text
rex_576_range_pos <= train-event q60 (0.4633067898243715)

funding branch: completed htf_1d_return_1 <= q70 (0.028038610394397256)
premium branch: completed htf_3d_return_1 <= q70 (0.045815363295544476)
```

### 3. Broad confirmation vetoes

```text
bb_z <= train-event q80 (1.4007434227732993)
quote_vol_z_1d <= train-event q90 (1.1338897334483824)
```

The final mechanism is therefore:

> depressed futures positioning + established medium-term trend + lower-half
> 48-hour pullback + no completed higher-timeframe overextension + no local
> Bollinger/turnover frenzy.

The q80/q90 confirmation gates are intentionally broad. They remove the most
overheated entries rather than trying to identify a rare perfect signal.

## Leakage-safe protocol

- Base and all context thresholds: fitted on 2020-07-01 through 2022-12-31.
- Structure selection: 2023 only, including separate H1/H2 reporting.
- Fourth-stage family: Top-20 frozen before replaying 2024 onward; this rule was
  pre-2024 rank 9.
- Signal: completed 5-minute bar evaluated hourly.
- Entry: t+1 5-minute open.
- Exit: fixed 576 bars / 48 hours.
- One non-overlapping long position.
- Base audit leverage: 0.5x; cost: 6 bp per side.
- CAGR includes every idle day in the named calendar window.
- Strict MDD includes pre-entry equity high water and favorable-then-adverse
  marking inside every held OHLC bar.
- Trades whose exit crosses a split boundary are purged.
- Funding/premium availability is required; stale or absent auxiliary values do
  not create a signal.

## 0.5x strict results

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Train 2020H2-2022 | +122.02% | 37.54% | 7.68% | **4.89** | 76 |
| Select 2023 | +17.18% | 17.19% | 3.47% | **4.95** | 16 |
| Test 2024 | +23.75% | 23.69% | 5.24% | **4.52** | 15 |
| Eval 2025-2026-05-31 | +29.68% | 20.22% | 4.87% | **4.15** | 32 |
| OOS 2024-2026-05-31 | +60.48% | 21.65% | 5.24% | **4.13** | 47 |
| Full 2020H2-2026-05-31 | +317.50% | 27.33% | 7.68% | **3.56** | 139 |

Annual ratios are `8.01` (2020H2), `4.97` (2021), `4.33` (2022), `4.52`
(2024), `4.00` (2025), and `5.73` (2026 through May 31). 2023H2 remains the
weak subwindow at +2.44% absolute return and ratio `2.19`, but the full 2023
selection window passes.

## Target operating point: 0.9x

No signal or threshold is changed; only leverage is scaled from 0.5x to 0.9x.

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Train | +300.28% | 74.07% | 13.52% | **5.48** | 76 |
| Select 2023 | +32.47% | 32.50% | 6.19% | **5.25** | 16 |
| Test 2024 | +45.94% | 45.83% | 9.25% | **4.95** | 15 |
| Eval 2025-2026 | +58.35% | 38.48% | 8.69% | **4.43** | 32 |
| OOS 2024-2026 | +131.09% | 41.49% | 9.25% | **4.48** | 47 |
| Full | +1125.41% | **52.75%** | **13.52%** | **3.90** | 139 |

At 10 bp per side, the full-period result remains +1008.63% absolute return,
CAGR `50.18%`, strict MDD `13.83%`, and ratio `3.63`.

Three-bar entry delay alone also passes: CAGR `52.17%`, strict MDD `14.78%`,
ratio `3.53`. The combined 10 bp + three-bar delay stress narrowly misses the
absolute target: CAGR `49.61%`, strict MDD `15.02%`, ratio `3.30`. This boundary
case is a reason to shadow the rule before capital deployment.

## Why both confirmation signals matter

| Variant | Pre-2024 ratio / trades | OOS ratio / trades | Full ratio / trades |
|---|---:|---:|---:|
| Unconfirmed pullback squeeze | 2.98 / 101 | 3.40 / 51 | 2.65 / 152 |
| Bollinger veto only | 2.90 / 94 | 4.05 / 48 | 2.58 / 142 |
| Quote-volume veto only | 3.79 / 98 | 4.25 / 49 | 3.31 / 147 |
| Both vetoes | **4.09 / 92** | **4.13 / 47** | **3.56 / 139** |

Quote-volume overheat removal supplies most of the edge. The Bollinger veto is
not useful alone, but conditioned on the quote-volume veto it lowers full-period
strict MDD from 8.44% to 7.68%, raising the ratio from 3.31 to 3.56. That is the
specific interaction sought here.

The funding and premium branches are also inadequate alone under the final
context. Their full-period ratios are 1.85 and 1.62; their conditional union is
3.56 because the branches cover different episodes and alter subsequent
non-overlapping execution opportunities.

## Robustness

- 10 bp per side at 0.5x: full ratio `3.32`.
- Two-bar delay: full ratio `3.44`.
- Three-bar delay: full ratio `3.24`.
- Nearby q75-q85 Bollinger and q85-q90 quote-volume recipes generally retain
  full-period ratios from `3.33` to `3.57`.
- Changing the fixed 48-hour hold to 36 or 60 hours fails; the holding horizon
  is part of the mechanism and must not be retuned from later data.
- Four-trade moving-block bootstrap 95% mean-return intervals remain positive:
  train `+0.628%` to `+1.648%`, 2024 test `+0.733%` to `+1.944%`, 2025-2026
  eval `+0.347%` to `+1.296%`, and full `+0.754%` to `+1.428%` per trade.
- Only 2025Q4 is negative in the later quarterly replay, at -0.38%; 2025Q3 has
  no split-contained trade.

## Remaining risks

1. This is retrospective research. The programme has repeatedly inspected
   2024-2026, so later periods are not pristine global OOS.
2. The first interaction stage tested 29,133 rules and the confirmation stage
   1,280. Descriptive p-values cannot eliminate that selection multiplicity.
3. OOS 2024-2026 contains 47 trades; the full sample has 139. This is meaningful
   but still modest for a 48-hour event strategy.
4. Actual funding cash flows during positions are not yet included.
5. The 0.9x operating point is near the MDD boundary under simultaneous high
   cost and delayed entry. Begin with shadow execution, not full sizing.

## Artifacts

- `training/search_specific_pullback_squeeze_alpha.py`
- `training/search_confirmed_pullback_squeeze_alpha.py`
- `tests/test_search_specific_pullback_squeeze_alpha.py`
- `results/confirmed_pullback_squeeze_alpha_audit_2026-07-15.json`

Reproduce:

```bash
.venv/bin/python training/search_confirmed_pullback_squeeze_alpha.py
.venv/bin/python -m pytest -q \
  tests/test_search_specific_pullback_squeeze_alpha.py \
  tests/test_strict_bar_backtest.py \
  tests/test_market_features.py
```
