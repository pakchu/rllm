# DEWH-144 — Deribit Expiry Wall Handoff mechanism decision

## Decision and unopened boundary

The next standalone BTC candidate is **DEWH-144**, a twelve-hour post-expiry
test of whether price continues away from a concentrated strike wall after that
wall is removed by delivery.

This document freezes the source fields, feature math, side, timing, hold,
support floors, novelty limits, and later economic gates before the new
strike-wall aggregate or any DEWH candidate incidence is computed. It opens no
Binance OHLC, funding, future return, PnL, CAGR, MDD, or 2024+ outcome.

Historical BTC outcomes have been seen elsewhere in the repository. This can
therefore establish only a candidate-level frozen sequence, not a globally
pristine holdout.

## Why this is not a DEHR repair

Retired DEHR-72 used terminal option type and moneyness:

```text
net_release = ITM put position - ITM call position
side        = sign(net_release)
```

DEWH intentionally discards call/put type, ITM/OTM state, and
`net_release_position`. It combines call and put position at each strike and
asks a different question: whether a uniquely dominant strike acted as a
local expiry wall, and whether the delivery index finished just above or below
that wall. Its side follows the index away from the removed wall.

DEHR's support rejection is not loosened or reopened. DEWH receives a new
identifier, source panel, feature, hold, support contract, controls, and a
mandatory related-source novelty comparison against the unchanged DEHR clock.

## Official source and source expressiveness

Use Deribit's public BTC delivery history only:

```text
GET https://www.deribit.com/api/v2/public/get_last_settlements_by_currency
    ?currency=BTC&type=delivery&count=1000
```

Official documentation:
<https://docs.deribit.com/api-reference/market-data/public-get_last_settlements_by_currency>

The existing validated loader proves that each option row exposes the expiry
code and strike in `instrument_name`, positive reported `position`, a common
delivery `index_price`, and the actual delivery event timestamp. Its exact
source code SHA-256 at this freeze is
`1e698db869ef263b692a950a3ecc4f4fafb834dd99db8476fd4da11bc1852cda`.

The retained DEHR aggregate is **not sufficient** for DEWH. It stores only the
largest individual instrument share, not the position combined by strike, the
dominant strike itself, its distance from the delivery index, or local strike
spacing. DEWH must therefore create a separate source-only aggregate from raw
responses. Raw responses may be held only in memory and must not be persisted
or committed.

Frozen source interval: `[2019-01-01, 2024-01-01)` UTC. The builder must:

- prove continuation pagination crossed the lower boundary;
- reject duplicate instruments and unsupported instrument names;
- exclude futures;
- require every option position and index price to be finite and positive;
- require all rows of an expiry to share one index price and a delivery clock
  span no greater than five seconds;
- retain the scheduled expiry and actual delivery event as separate clocks;
- bind every canonical response page and the aggregate by SHA-256; and
- record zero market, funding, return, and post-entry outcome access.

## Frozen strike-wall aggregate

For expiry `e`, strike `K`, reported position `p`, and delivery index `S`:

```text
P_e(K)       = sum(position across the call and put at K)
P_e          = sum_K P_e(K)
K*_e         = unique argmax_K P_e(K)
wall_share_e = P_e(K*_e) / P_e
spacing_e    = min_{K != K*_e} abs(log(K / K*_e))
u_e          = log(S / K*_e) / spacing_e
```

An expiry is wall-valid only when it has at least three distinct positive
strikes, a unique exact maximum `P_e(K)`, positive `spacing_e`, and finite
values. Tied maxima are unavailable; they are not resolved using index
distance, option type, future price, or arbitrary strike order.

The committed aggregate may retain only:

- `expiry_time`, `delivery_event_time`, and historical earliest observation;
- `index_price`, distinct strike count, total position;
- dominant strike and position, wall share, strike-position HHI, and the
  largest individual instrument share used only by the frozen ablation;
- local log spacing and signed normalized wall distance `u_e`;
- source-quality flags and timing diagnostics.

It must not retain mark price, settlement PnL, option type, terminal state,
instrument-level rows, a return, or a trading label.

## Causal availability and execution

Deribit does not document a first-publication SLA for the complete delivery
set. Preserve the conservative live rule established by the source audit:

1. wait at least 60 minutes after the actual reported delivery event;
2. observe two identical canonical delivery sets five minutes apart;
3. historical earliest observation is `delivery_event_time + 65 minutes`;
4. let `B` be that timestamp rounded up to the next five-minute boundary;
5. wait one complete bucket and enter at `B + 5 minutes`; and
6. a later live observation delays or cancels; it is never backdated.

For the normal 08:00 event this yields 09:10 UTC. A non-grid event is rounded
forward, never backward. Exit is exactly 144 five-minute bars (twelve hours)
after entry. Entries are globally chronological and non-overlapping. A split
keeps a row only when expiry, delivery, observation, entry, and exit are all
contained in that split.

## Frozen strictly-prior features and singleton signal

For each wall-valid expiry, build empirical midranks from wall-valid expiries
in the preceding 365 calendar days, excluding the current expiry. Require at
least 180 prior expiries. Ties use:

```text
(count(prior < current) + 0.5 * count(prior == current)) / prior_count
```

There is no expanding fallback, imputation, stale carry, threshold grid, side
grid, hold grid, or post-support repair.

Exactly one setup is selected when all conditions hold:

```text
rank(total_position) >= 0.50
rank(wall_share)     >= 0.70
0.25 <= abs(u_e) <= 1.00
u_e != 0
```

The side is `sign(u_e)`: LONG when the delivery index is above the dominant
wall and SHORT when it is below. This is the **away-from-removed-wall** thesis.
Every eligible expiry is an independent setup; no false-to-true episode filter
is applied.

## Frozen source-support gate

Research splits are:

- train: `[2020-07-01, 2023-01-01)`;
- source-only selection: `[2023-01-01, 2024-01-01)`.

The train clock must have:

- 60–240 accepted entries;
- at least 8 in 2020H2 and at least 18 in each of 2021 and 2022;
- at least 8 in every contained half-year;
- at least 20 LONG and 20 SHORT entries;
- entries in at least 24 of the 30 calendar months;
- no month above 15% of entries; and
- no UTC entry weekday above 30% of entries.

The 2023 selection clock must have:

- 20–100 accepted entries;
- at least 8 in each half and 3 in each quarter;
- at least 6 LONG and 6 SHORT entries;
- entries in at least 8 calendar months;
- no month above 25% of entries; and
- no UTC entry weekday above 35% of entries.

Failure retires DEWH-144 before novelty or economic evaluation. Counts from
ablations cannot replace the primary.

## Frozen source-only novelty gate

Before DEWH incidence is opened, bind a comparator cohort containing the
current AFCS, BAFR, MFIC-fast, MFIC-slow, and three production-sleeve pure
clocks, plus a deterministic reconstruction of the unchanged DEHR-72 singleton
through 2022.

For every unrelated comparator over common coverage, require all of:

- exact-entry Jaccard at most 0.20;
- maximum one-to-one DEWH match coverage within plus/minus six hours at most
  0.35; and
- absolute signed occupied-exposure correlation at most 0.40.

Against DEHR specifically, require exact-entry DEWH coverage at most 0.50 and
absolute signed occupied-exposure correlation at most 0.50. These related-
source limits prevent a renamed DEHR repair while allowing the exchange expiry
clock itself to be shared.

No Binance price, funding, return, or performance artifact may be parsed for
this stage. A novelty failure retires DEWH without economic simulation.

## Frozen controls

Controls are diagnostic and can only falsify the primary:

1. exact direction flip on primary clocks;
2. deterministic timestamp-seeded random side on primary clocks;
3. expiry-time-only deterministic random side on every wall-valid expiry;
4. wall-concentration gate ablation;
5. total-position gate ablation;
6. normalized-distance-band ablation;
7. largest-individual-instrument concentration instead of strike aggregation;
8. frozen DEHR release side at exact DEWH clocks;
9. one additional five-minute execution delay; and
10. a fixed alternating side independent of wall location.

The source-support stage reports their incidence but cannot promote one.

## Conditional strict economic evaluation

Only a support-and-novelty pass may authorize a separately committed strict
evaluator. Train opens first; 2023 opens only if unchanged train passes. Base
exposure is 0.5, base cost is 6 bp/notional/side, and stress cost is 10
bp/notional/side.

Each opened split must have positive absolute return, full-calendar
CAGR/strict-MDD at least 3, strict MDD at most 15%, positive stress-cost
return, positive one-extra-bar-delayed return, mean gross underlying edge at
least 25 bp, and weekly-cluster one-sided sign-flip `p <= 0.10` with a frozen
seed and draw count. Required train years and 2023 halves must be individually
positive with their support minimums.

Strict MDD includes the global/pre-entry high-water mark, entry cost, exact
funding, every held five-minute adverse path, virtual adverse exit fee, and
actual exit. CAGR uses the complete wall-clock split including warm-up and idle
cash. Controls may reject DEWH but never replace it.

## No-repair rule

After the source aggregate or incidence is opened, do not change the 365-day
reference, 180-expiry minimum, 0.50/0.70 ranks, 0.25–1.00 distance band,
away-side map, availability clock, 144-bar hold, support floors, novelty
limits, or comparator membership. A failure requires a genuinely independent
future mechanism and identifier.

## Frozen predecessor anchors

- DEHR rejection document SHA-256:
  `c8b0c18743057e9b217c932e742784f2f810241a4723869cbe9c912db88a25c2`;
- DEHR source manifest SHA-256:
  `b1a2ed3a39b8e71adc0a46a5411d4f568eda3bdaa910cef64d9746fa6f5ea3e5`;
- DEHR aggregate SHA-256:
  `a59953eb0efddbab7a28af9fdd0f61f204fa98d2de330cf1a4090293378b0fda`;
- outcomes opened by this decision: false; and
- DEWH candidate incidence opened by this decision: false.
