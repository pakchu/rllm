# ExtraTrees rank-7 leverage battery preregistration — 2026-07-27

Status: **PREREGISTERED — levered grid metrics not computed**

## Fixed question

Can the already-frozen annual ExtraTrees rank-7 alpha be scaled to the user target without changing its features, learner, thresholds, direction, exits, refit cadence, or trade clocks?

- fixed leverage grid: `0.50x, 0.75x, 1.00x, 1.25x, 1.50x`;
- selection period: `2023-01-01`–`2025-01-01` only;
- selection rule: highest cell passing every 2023, 2024, combined, and 10 bp stress gate;
- 2025–2026H1: report-only; no repair or reselection;
- full-calendar CAGR includes idle time; absolute return is always shown;
- hardened strict MDD includes entry cost, favorable-before-adverse 5m path, conservative funding, virtual adverse liquidation cost, and exit cost.

## Fixed target

Both `future` (2025–2026H1) and `all` (2023–2026H1) must independently reach `CAGR >= 50%`, `strict MDD <= 15%`, and `CAGR/MDD >= 3`. A policy can survive the robustness gates without satisfying this stronger target.

## Research boundary

This is not globally pristine OOS: rank-7's 2025+ outcomes were already viewed. The useful protection is narrower: the new sizing choice is fixed from pre-2025 windows and later outcomes cannot change it.

Manifest hash: `01fa88ba5e1398c06ea192749c81a15e516982688761c570f160d6e416a16659`

Next authorized action: `IMPLEMENT_REVIEW_COMMIT_AND_PUSH_FIXED_BATTERY_THEN_EXECUTE_ONCE`.
