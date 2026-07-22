# SMCC-144 preregistration — same-millisecond cascade continuation

## Evidence boundary

This protocol is frozen before rebuilding the same-millisecond source, reading
its real event incidence, or opening any post-signal BTC price, funding,
return, excursion, PnL, equity, CAGR, or MDD.  The only prior evidence used is
the checksum-audited official Binance USD-M `BTCUSDT` aggTrades source contract
and the known failure of dense five-minute microstructure rules whose gross
edge did not clear costs.

SMCC tests a new raw observable that the existing five-minute aggregate does
not retain: the topology of aggregate trades sharing the exact same integer
transaction millisecond.  A same-millisecond group is a matching-engine burst
proxy, not a claimed parent order, liquidation, or participant identity.

## Frozen source transform

For each completed UTC five-minute bar, group official aggregate trades by
their exact `transact_time` millisecond.  Aggressive side is `+1` when
`is_buyer_maker=false` and `-1` when `true`.  For each group `g`:

```text
notional(g) = sum(price * quantity)
signed(g) = sum(side * price * quantity)
coherence(g) = abs(signed(g)) / notional(g)
side(g) = sign(signed(g))
```

Select `g*` by maximum notional, breaking an exact tie with the earlier
millisecond.  The causal predecessor price is the last price of the strictly
preceding millisecond group inside the same five-minute bar.  If `g*` is the
first group in the bar, its sweep and score are zero; no price state is borrowed
across a bar or archive boundary.

### Source-integrity amendment before incidence access

The first blinded source build stopped before support counts were read because
the official `2020-01-15` archive contains one pair of adjacent aggregate rows
that repeats the same underlying trade ID and all economic event fields while
using consecutive aggregate IDs.  A source-only scan confirmed this is the
only underlying-ID overlap in January 2020; the other 47 monthly builds had
already failed closed on any overlap.  No SMCC event incidence, return, or
market outcome was inspected.  The full `2020-01-15` UTC day is therefore
added to the quarantine set, followed by the same 24-bar quarantine.  Positive
underlying-ID holes remain permissible because the frozen aggregate-source
audit defines completeness by aggregate IDs; overlaps remain exact-count
bound and fail closed.

```text
share = notional(g*) / bar_notional
sweep_bp = 10000 * side(g*) * log(last_price(g*) / predecessor_price)
score = share * coherence(g*) * max(sweep_bp, 0)
```

The largest-notional group is chosen before applying coherence or displacement
conditions.  A smaller favorable group may not replace it.  Official daily
archive checksums, monotonic IDs/timestamps, and the existing source-gap day
quarantine are mandatory.  Raw ZIPs are streamed and not persisted.

## Frozen event and execution clock

At completed five-minute bar `t`, SMCC is eligible only when:

- `agg_trade_count >= 64`;
- `g*` contains at least `3` aggregate trades;
- `coherence(g*) >= 0.80`;
- `side(g*) != 0` and `sweep_bp > 0`;
- `score` is at or above its strictly prior 30-day (`8,640` clean bars)
  `q99.5`, excluding current `t` and requiring at least `2,016` observations.

No quantile, side, hold, delay, regime, stop, TP, funding, OI, or ML grid is
allowed.  The action follows `side(g*)`.  One full five-minute calculation bar
is left empty; entry is the open of `t+2`.  Exit is the scheduled open after
exactly `144` held bars (12 hours).  Positions are chronological and
non-overlapping; re-entry is allowed at the preceding scheduled exit open.
Future source completeness after decision `t` never cancels a selected event;
using that information would be look-ahead.  Support-year/month attribution is
by entry timestamp, and entry plus exit must remain before `2024-01-01`.

## Outcome-blind support and novelty gates

Before any post-entry OHLC or funding value may be read, the rebuilt source and
clock must pass all of these gates:

- `150 <=` total non-overlapping events `<= 900` over 2020–2023;
- at least `30` events in each calendar year;
- at least `15` events in each 2023 half;
- long and short shares each between `25%` and `75%`;
- no calendar month contributes more than `15%` of all events.

The selected entry clock must also have:

- exact-entry Jaccard `<= 0.05`;
- one-to-one `+/-12` five-minute-bar tolerant Jaccard `<= 0.15`; and
- primary-clock containment `<= 0.30`

against every available sparse prior aggTrade/microstructure family: MFIC,
AFCS, TAAR, RIFT, and PCP.  Their exact paths, SHA-256 values, member selectors,
and entry-time columns are frozen in the machine-readable preregistration
artifact.  LVRT and Minute Packet Topology have no standalone canonical clock;
NETF has no canonical clock and RIFT is its frozen topology comparator.  These
exclusions are explicit and cannot change after source incidence is read.
Dense BAFR is reported with exact-entry Jaccard and primary containment but
does not receive the tolerant-Jaccard gate because its density makes that
quantity mechanically uninformative.  Missing or malformed registered clocks
fail closed.

Novelty is computed inside each comparator's frozen coverage.  Entries are
sorted and unique.  Primary entries are visited chronologically and matched
one-to-one to the unused comparator entry with the smallest absolute distance,
ties going to the earlier comparator.  Exact matching uses zero distance;
tolerant matching permits at most 12 five-minute bars.  Jaccard is
`matches/(primary+comparator-matches)` and containment is `matches/primary`.
Registered canonical comparator timestamps without an explicit offset are
interpreted as UTC; timestamps already ending in `Z` retain that offset.
Duplicate entries within one comparator member fail closed.  Dense BAFR uses
exact-entry Jaccard and exact primary containment only.

The prior threshold is implemented exactly as a calendar-row rolling window:
mask score to `source_complete`, shift one row, then apply
`rolling(8640,min_periods=2016).quantile(0.995)`.  Invalid rows consume elapsed
clock time but contribute no observation, and current `t` never enters its own
threshold.

## Frozen later economic falsification

Only a passing source/support artifact permits a separately committed strict
evaluator.  Before source incidence is rebuilt, the machine-readable
preregistration freezes the support JSON/clock paths and schema plus exact
control formulas.  In summary, the controls are:

- exact side flip;
- an own-clock five-minute absolute-return q99.5 continuation;
- an own-clock normalized bar-notional times aligned-return q99.5 continuation;
- an own-clock event-HHI times positive burstiness times signed-response q99.5
  continuation;
- one-hour and 24-hour stale SMCC clocks;
- component removals: no coherence, no positive sweep, and no collision-count
  requirement.

All own clocks use the identical strictly prior 8,640/2,016 history, q99.5,
delay, hold, quarantine and non-overlap scheduler.  Component-removal score
formulas and stale offsets are frozen verbatim in
`results/same_millisecond_cascade_preregistration_2026-07-20.json`.  The primary
is rejected if a simpler direct control independently qualifies or if it fails
to exceed the strongest finite control's CAGR/strict-MDD by `0.25`.

Opening is sequential and stops at first failure:

1. train: 2020–2022;
2. selection: calendar 2023 plus H1/H2;
3. test: calendar 2024;
4. eval: calendar 2025;
5. recent 2026 report only.

Each opened annual stage uses `0.5x`, next-open execution, exact realized
funding, `6 bp` notional per side base cost, `10 bp` stress cost, full-calendar
CAGR including idle cash, and strict held-path MDD from the global/pre-entry
high-water mark through entry cost, favorable-before-adverse OHLC, funding,
virtual adverse exit cost, and actual exit.  Promotion requires positive base
absolute return, CAGR/strict-MDD at least `3.0`, strict MDD at most `15%`,
positive stress return with stress ratio at least `2.5`, mean gross underlying
move above `24 bp`, and weekly-cluster sign-flip `p < 0.10`.  Train additionally
requires every calendar year positive; 2023 requires H1/H2 positive.  No failed
stage may be repaired under the SMCC-144 name.

The broader research history has seen many BTC outcomes, so a future pass would
be frozen candidate-level evidence, not a globally pristine discovery.
