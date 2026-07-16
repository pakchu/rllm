# CRRC-72 outcome-blind qualification — 2026-07-17

## Verdict

CRRC-72 passed its frozen **support and causal-clock overlap gates without
opening any price, funding, return, PnL, CAGR, MDD, or equity value**. This is
permission to build and freeze the strict 2023 evaluator, not evidence that
the strategy is profitable.

## Selected support clock

- Raw candidates: 117 long, 91 short, 0 two-sided conflicts
- Quarter-contained non-overlap events: **156**
- Executed sides: 91 long, 65 short
- Halves: H1 57, H2 99
- Quarters: Q1 32, Q2 25, Q3 47, Q4 52
- Maximum month share: **15.38%** (limit 20%)
- Maximum quarter share: **33.33%** (limit 40%)

The 14-cell outcome-blind incidence grid replayed from the frozen 2023 panels.
The deterministic selection rule again chose add q85, outer-withdraw q75,
inner-net q55, and flicker q85. The selected clock hash is stored in
`results/cross_venue_radial_refill_compression_event_clock_2026-07-17.json`.

## Causal-clock overlap

| Prior clock | Events | Exact entry matches | Exact Jaccard | +/-12-bar matches / CRRC | Position-time Jaccard | Gate |
|---|---:|---:|---:|---:|---:|---|
| PDF-10 | 591 | 3 | 0.40% | 46 / 156 = 29.49% | 1.81% | pass |
| CCLH | 167 | 0 | 0.00% | 7 / 156 = 4.49% | 9.31% | pass |
| RLWC-144 | 0 | 0 | 0.00% | 0 / 156 | 0.00% | pass, non-evidentiary |
| near-pressure | 238 | 1 | 0.25% | 24 / 156 = 15.38% | 13.27% | pass |

RLWC-144 had no supported events, so zero overlap with it is reported but is
not treated as affirmative independence evidence. Clock overlap is also not a
claim of PnL orthogonality; daily/weekly PnL correlation and synchronized
portfolio marginal value remain sealed until CRRC passes standalone returns.

## Next irreversible boundary

The next step is still outcome-blind: physically isolate 2023 BTCUSDT 5m OHLC
and exact funding, implement the strict simulator, hash-freeze both source and
evaluator, and independently verify the freeze. Only then may 2023 returns be
opened once. Any failed preregistered gate retires CRRC without repair and
keeps 2024+ sealed.
