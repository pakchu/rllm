# Treasury Auction Demand Impulse (TADI-1) preregistration — 2026-07-17

## Hypothesis

An abrupt, concordant improvement in both Treasury auction bid-to-cover and
indirect-bidder participation reveals stronger global duration absorption and
easier risk-bearing capacity; BTC is traded long for the following 24 hours.
The exact concordant deterioration is traded short.

This is deliberately external to the existing crypto price-action, REX, OI,
taker-flow, funding/premium, Kimchi/FX, cross-venue, options, and on-chain
families.

## Frozen causal rule

For each original nominal coupon tenor independently:

1. expose official auction results at 22:00 UTC on the auction date;
2. compute change in bid-to-cover from the previous complete same-tenor auction;
3. compute change in indirect competitive share, where the denominator is
   primary + direct + indirect competitive accepted amounts;
4. mid-rank each change against exactly 12 prior valid same-tenor changes;
5. current values enter history only after their ranks are emitted;
6. do not compute a change across any `source_complete=false` auction;
7. LONG when both ranks are at least 0.75; SHORT when both are at most 0.25;
8. enter at 22:05 UTC and exit exactly 24 hours later;
9. reserve globally, with shortest-tenor-first priority for a timestamp tie.

There is one candidate. No direction, rank length, threshold, hold, gate, or
crypto regime repair is allowed after outcomes are opened.

## Outcome-blind support seen before freeze

| Window | Events | Long | Short | Max events in one month |
|---|---:|---:|---:|---:|
| 2021 | 13 | 8 | 5 | 2 |
| 2022 | 15 | 10 | 5 | 3 |
| Stage1 2021-2022 | 28 | 18 | 10 | 3 |
| 2023 H1 | 12 | 8 | 4 | 5 |
| 2023 H2 | 11 | 1 | 10 | 4 |
| Sealed 2023 | 23 | 9 | 14 | 5 |

These are source timestamps and sides only. No BTC OHLC, return, funding cash
flow, CAGR, MDD, win rate, or portfolio overlap was joined.

## Execution and accounting

- 0.5x fixed gross;
- 6 bp/notional/side base cost and 10 bp stress cost;
- exact realized BTCUSDT funding on `[entry, exit)`;
- strict MDD includes global and pre-entry high-water, entry cost, adverse held
  OHLC, hypothetical liquidation, funding, and exit cost;
- CAGR uses the full split wall clock, including idle cash.

## Validation sequence

1. Stage1: 2021-01-01 through 2022-12-31, with 2021 and 2022 contained checks.
2. Only an unchanged Stage1 pass may load 2023 market outcomes.
3. Stage2: 2023 full year plus H1/H2.
4. Orthogonality against the frozen portfolio is measured only after a
   standalone pass; it cannot rescue a failed candidate.

Primary gates include positive absolute return, CAGR/strict-MDD at least 3,
strict MDD at most 15%, two-sided weekly-cluster sign-flip p-value at most 0.10,
minimum trade and gross-edge floors, positive 10 bp stress return, positive
contained subperiods, and at least a 0.25 ratio margin over each mechanism
control.

## Falsification controls

- bid-to-cover-change tails only;
- indirect-share-change tails only;
- exact direction flip;
- one complete same-tenor auction delay;
- deterministic random side on the primary entries.

## Artifacts

- Source audit: `docs/us-treasury-auction-demand-source-audit-2026-07-17.md`
- Clock: `training/treasury_auction_demand_impulse_clock.py`
- Preregistration: `training/preregister_treasury_auction_demand_impulse.py`
- Manifest: `results/treasury_auction_demand_impulse_preregistration_2026-07-17.json`
- Frozen clock: `results/treasury_auction_demand_impulse_preregistered_clock_2026-07-17.csv.gz`
