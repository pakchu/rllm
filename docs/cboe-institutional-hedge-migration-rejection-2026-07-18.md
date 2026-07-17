# CIHM-1 rejection — 2026-07-18

## Decision

**Reject CIHM-1. Keep 2023 sealed. Do not repair this candidate.**

CIHM-1 tested a source-only Cboe option-flow migration hypothesis under the
unchanged preregistration and strict evaluator.  It failed Stage 1 before any
2023 BTC outcome was opened.

## Stage-1 result, 2021–2022

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross edge | Weekly sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | **-2.31%** | **-1.16%** | **35.31%** | **-0.03** | 151 | 8.34 bp | 0.9622 |
| 10 bp/notional/side | -8.04% | -4.10% | 36.70% | -0.11 | 151 | 8.34 bp | 0.8487 |

The full-calendar period was used, including all idle days.

### Contained years

| Year | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross edge |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | -24.27% | -24.29% | 30.64% | -0.79 | 74 | -64.48 bp |
| 2022 | +29.01% | +29.03% | 9.82% | 2.96 | 77 | +78.32 bp |

The sign reversal is too large to describe as a stable next-session edge.  The
2022 success does not rescue the negative two-year result and cannot justify a
post-hoc calendar or regime gate.

## Controls

| Clock | Absolute return | CAGR/MDD | Trades |
|---|---:|---:|---:|
| primary | -2.31% | -0.033 | 151 |
| institutional-gap only | -15.16% | -0.286 | 156 |
| VIX-call-pressure only | +1.48% | +0.024 | 151 |
| index-share only | +8.09% | +0.170 | 141 |
| level composite | +0.39% | +0.007 | 211 |
| direction flip | -20.34% | -0.301 | 151 |
| one-release delay | -19.32% | -0.432 | 151 |
| seven-release placebo | +6.49% | +0.150 | 150 |

Both the primary and its direction flip lose after costs, so this is not a
simple sign mistake.  The seven-release placebo and index-share control beat
the primary, while neither is independently investable.  That is evidence
against the proposed timing mechanism, not permission to substitute a control.

## Root cause assessment

1. **Aggregate volume is ambiguous.** It mixes opening and closing trades,
   buyers and sellers, and multi-leg spreads.
2. **The three changes are not a stable joint state.** Index/equity put-call
   change and VIX call-pressure change are mechanically anti-correlated in the
   source panel, weakening the equal-weight conjunction.
3. **The clock is regime-dependent.** The candidate loses severely in 2021 and
   works only in 2022; no preregistered, source-only rule distinguishes them.
4. **Timing specificity is falsified.** A seven-release shift performs better
   than the intended next-session clock.

## Leakage boundary

- Parsed BTC market rows: 210,240, exactly `[2021-01-01, 2023-01-01)`.
- Parsed funding rows: 2,190, same physical boundary.
- Both parsers stopped before reading their end boundary.
- Stage-1 result manifest:
  `e39d43f7d485a1f55fa45699c28a99137a99bac7657abfce7e92fceb4e6a66cf`.
- Invoking Stage 2 raises `CIHM-1 Stage1 failed; 2023 remains sealed` before the
  Stage-2 market/funding loader is called.

No threshold, sign, weight, holding period, source component, BTC gate, or
2022-specific regime will be fitted to this candidate.  The next search must
use a new independent mechanism and candidate ID.
