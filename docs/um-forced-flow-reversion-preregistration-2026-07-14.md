# UMFR-36 preregistration — 2026-07-14

## Status

**Support-only; outcomes unopened.** UMFR-36 uses only completed five-minute
cross-venue descriptors before `2024-01-01`. Entry is the next Binance USD-M
five-minute open and exit is fixed 36 bars later.

## Economic hypothesis

A late USD-M impulse that moves with high path excess, larger relative ticket
size, and basis stretch while Spot under-responds is more likely a forced
derivatives-flow shock than informed cash discovery. UMFR therefore fades the
USD-M flow direction instead of following it.

## Frozen formula

Let `u = sign(um_flow_fraction)` and trade `side = -u`. Require:

- `u * um_log_return_5m * 10000 > 0`;
- `u * spot_log_return_5m * 10000 < u * um_log_return_5m * 10000`;
- all USD-M activity/flow/return centroids are later than Spot centroids;
- `u * basis_change_bp > 0`;
- USD-M path-excess exceeds Spot path-excess;
- `log(um_avg_ticket / spot_avg_ticket)` exceeds its strictly prior 30-day
  median.

Score is the geometric mean of late pressure, `log1p(basis stretch)`,
`log1p(path excess advantage)`, `log1p(ticket surprise)`, and absolute USD-M
flow fraction. Thresholds are rolling quantiles of prior eligible events only
over 17,280 bars, minimum 64 prior events. The frozen grid is
`0.50, 0.65, 0.75, 0.80, 0.85`; select the highest support-passing quantile.

## Frozen support floors

- total non-overlapping events at least 900;
- each year 2020–2023 at least 130;
- 2023 H1/H2 at least 100 each and each 2023 quarter at least 40;
- each side at least 40% overall and at least 60 events per year;
- at least 44 months with at least five scheduled events;
- novelty vs temporal/stale controls and prior clocks must pass the frozen
  overlap floors in the support artifact.

Failure rejects before returns. Passing only permits a clock freeze and then a
pre-2024 evaluation; it is not a profitability claim.
