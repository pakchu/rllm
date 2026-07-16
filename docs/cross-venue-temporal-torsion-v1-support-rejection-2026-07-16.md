# CVTT v1 support rejection (2026-07-16)

## Decision

**Reject CVTT v1 before computing any threshold, event clock, or trade return.**
The crossed-clock mechanism occurred often, but the preregistration confused
seven calendar days with 2,016 eligible route events inside a 30-day window.
That condition was nowhere achievable.

## Outcome-blind evidence

| Route | Confirmed crossed-clock rows | Maximum eligible rows in any prior 30d | Required | Estimable rows |
|---|---:|---:|---:|---:|
| Spot preload → USD-M echo | 29,176 | 1,028 | 2,016 | 0 |
| USD-M preload → Spot echo | 29,701 | 979 | 2,016 | 0 |

Because the rolling q95 threshold could not exist, all four policy clocks have
zero eligible events by construction. Market OHLC, future return, and realized
post-signal funding were not read.

## Source support

- Raw unavailable feature rows: 522 of 315,648.
- Current plus following-24-bar quarantine: 1,386 rows (0.4391%); global 1%
  gate passed.
- Worst month: 2022-12 at 3.3602%; preregistered monthly 3% gate failed.

The official source is globally dense and invalid periods are explicitly
quarantined, but v1 remains rejected. Neither gate is silently relaxed.

## Permitted repair boundary

One new version may repair only support feasibility because no return was
opened:

1. distinguish minimum clean **calendar bars** from minimum eligible route
   events;
2. use a declared eligible-event minimum below the observed route-window
   maxima;
3. use the already established 5% monthly quarantine ceiling while retaining
   the 1% global ceiling and identical no-imputation rule.

Economic mechanism, source columns, direction rules, holds, execution delay,
costs, strict MDD, selection gates, and 2023 holdout gates may not change.
Any later adjustment after return inspection is forbidden.

Artifact:
`results/cross_venue_temporal_torsion_support_v1_2026-07-16.json`.
