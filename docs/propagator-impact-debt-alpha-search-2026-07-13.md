# Propagator Impact-Debt Alpha Search — 2026-07-13

## Thesis

Aggressive flow is persistent because orders are split, while its price impact
can be distributed over time. A completed bar may therefore leave a causal
**impact debt**: the part of prior unexpected taker flow that a learned response
kernel says has not yet reached price.

This experiment is different from a flow tail, path-area gate or contemporaneous
price-impact regression:

1. normalize signed taker notional by prior 30-day median activity;
2. fit an AR(12) model on fit data and retain only the flow innovation;
3. fit a finite impulse response from current/past innovations to 5-minute
   returns;
4. at completed bar `t`, sum only future kernel mass still owed by innovations
   observed through `t`;
5. trade the sign of an extreme prior-standardized debt at the next open.

The rule is economically falsifiable: if BTC absorbs flow essentially
immediately, the kernel tail and cost-surviving debt edge should disappear.

## Causal protocol

- The source is physically truncated before `2024-01-01`.
- AR and response coefficients use only `2020-06-01..2022-12-31`; 2023 is
  internal selection and 2024+ stays sealed.
- Current signed flow and return are from a completed bar; execution is the next
  5-minute open.
- Activity scaling uses a prior-only 30-day median. Debt z-score uses history
  through `t-1` only.
- Fit returns are in-sample model diagnostics; 2023 is the relevant frozen-model
  preflight.
- Leverage is `0.5x`, cost is `6bp/side`, and strict MDD is conservative
  favorable-first/adverse-second OHLC high-water.

## Bounded exploratory grid

Forty policies were inspected entirely before 2024:

- response horizon: 1 hour or 3 hours (`12`, `36` bars);
- debt threshold: `|z| >= 1.5, 2.0, 2.5, 3.0, 3.5`;
- event mode: remain in state or first onset;
- hold: one or three response horizons.

No OOS row was opened after this internal search.

## Kernel evidence

The fitted response is overwhelmingly immediate:

| Kernel | Current response | Total response | Tail after current |
|---|---:|---:|---:|
| 1h | +0.006119 | +0.005806 | -0.000313 |
| 3h | +0.006133 | +0.004626 | -0.001507 |

The negative tail implies partial reversal rather than delayed continuation.
That is a coherent microstructure estimate, but it still needs to survive
turnover and regime splits.

## Best ranked policy

Three-hour kernel, first `|debt z|>=3.5` onset, 3-hour hold:

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| fit | -12.40% | -4.99% | 22.69% | -0.22 | 199 | 112/87 |
| 2023 | -2.85% | -2.85% | 5.89% | -0.48 | 89 | 46/43 |
| 2023H1 | -3.72% | -7.37% | 5.89% | -1.25 | 50 | 27/23 |
| 2023H2 | +0.91% | +1.82% | 4.98% | 0.37 | 39 | 19/20 |

Zero of 40 policies passed admission.

## Controls and cost

- exact direction flip: fit `-11.83%`, ratio `-0.23`; 2023 `-7.71%`, ratio
  `-0.60`;
- current innovation direction without the tail: identical to the flip for the
  top policy because the estimated tail has the opposite sign;
- one-hour signal lag: fit `-17.58%`, ratio `-0.35`; 2023 `-5.67%`, ratio
  `-0.65`;
- zero-cost top policy: fit `-1.29%`, CAGR `-0.50%`, strict MDD `22.22%`, ratio
  `-0.02`; 2023 `+2.48%`, CAGR `2.49%`, strict MDD `4.98%`, ratio `0.50`.

Costs are material, but they do not explain the fit failure or the negative
2023H1 result.

## Decision

**Rejected.** The creative mechanism is logically valid and causal, but the
data says BTC's measurable response is mostly immediate. The remaining tail is
too small, temporally unstable and cost-fragile to trade. Record the exact
static propagator-debt thresholds/holds as gamma failure provenance; do not
tune more AR orders, ridge penalties, thresholds or holds on this sample.

Artifacts:

- `training/search_propagator_impact_debt_alpha.py`
- `tests/test_search_propagator_impact_debt_alpha.py`
- `results/propagator_impact_debt_alpha_scan_2026-07-13.json`
