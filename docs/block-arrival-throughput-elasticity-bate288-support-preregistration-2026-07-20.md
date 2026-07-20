# BATE-288 block-arrival throughput elasticity support preregistration

Status: **frozen before the complete 2020–2023 block prefix, BATE incidence,
or any BATE market outcome was inspected**.

## Singleton hypothesis

`BATE-288` tests whether Bitcoin blockspace demand is arriving faster or slower
than block production can absorb it. A concordant burst in transaction and
block-weight throughput is tentatively bullish demand continuation; a
concordant drought is tentatively bearish demand continuation.

There is one policy, one packet size, one reference length, one threshold, one
confirmation rule, and one hold. There is no threshold, side, latency, packet,
reference, or hold grid.

Frozen inputs:

- source decision SHA-256:
  `d2643fa3f8f0ead2b705ee48fa83e59b16d73fc2e7df9dabde271f166d709647`;
- source loader commit:
  `addd52cb431d7794a072fa2ed4d4e74fedaeec1b`;
- source loader SHA-256:
  `0628e9e5925087e68b5d0a7a8f74dc91d908a209d140557a8ac690cd0e98bc53`;
- source heights: `610691..823785` inclusive; and
- latest packet end allowed by six-confirmation containment: `823779`.

The mechanism and official source references are frozen in
[`block-arrival-throughput-elasticity-mechanism-decision-2026-07-20.md`](block-arrival-throughput-elasticity-mechanism-decision-2026-07-20.md).

## Exact source-only feature

For every possible packet end height `h`, using blocks `h-6..h`:

1. `elapsed_seconds = timestamp[h] - timestamp[h-6]`;
2. `weight_throughput = sum(weight[h-5:h]) / elapsed_seconds`;
3. `tx_throughput = sum(tx_count[h-5:h]) / elapsed_seconds`;
4. `weight_log = log(weight_throughput)`; and
5. `tx_log = log(tx_throughput)`.

An `elapsed_seconds <= 0` packet is invalid. It is not clipped, absolute-valued,
imputed, or allowed to signal or enter a reference window.

Each channel is standardized independently against the last **2,016 valid
packet endings strictly below `h`**, requiring all 2,016. For channel `x`:

```text
center = median(reference_x)
scale = 1.4826 * median(abs(reference_x - center))
robust_z = (current_x - center) / scale
```

A zero or non-finite scale makes the current packet invalid. The current
packet, confirmation blocks, future blocks, market data, and eventual outcome
never enter its reference statistics.

Frozen states:

- `HIGH` when `weight_z >= +1.25` and `tx_z >= +1.25`;
- `LOW` when `weight_z <= -1.25` and `tx_z <= -1.25`; and
- `NEUTRAL` otherwise.

An onset exists only when the current valid state is `HIGH` or `LOW` and the
immediately preceding **valid** packet state is different. Invalid packets do
not manufacture a transition: the last valid state remains in force across an
invalid run.

The tentative side is fixed:

- `HIGH` onset → long;
- `LOW` onset → short.

## Confirmation, availability, and event selection

Every onset waits through block `h+6`. Its conservative historical synthetic
availability is:

```text
raw_available = max(timestamp[h-6:h+6]) + 2 hours
decision_boundary = ceil raw_available to a 5-minute UTC boundary
entry_time = decision_boundary + 5 minutes
```

Thus one complete latency bar follows both six confirmations and the two-hour
header-time embargo. Header timestamps are miner-reported and are never treated
as historical first-seen timestamps.

Event selection is deterministic in increasing packet height:

1. derive every onset without looking at another onset's outcome;
2. accept the earliest onset whose `entry_time` is not before the prior
   accepted event's `entry_time + 24 hours`;
3. suppress all intervening onsets, including opposite-side onsets; and
4. never replace a suppressed event with a later event selected from price or
   PnL.

The event's support calendar belongs to `entry_time`, not block timestamp.
The eventual fixed hold is **288 five-minute bars (24 hours)** from next-open
entry. There is no early exit, TP, SL, trailing stop, or overlap.

## Source-only integrity gate

Before any BTC market or funding file is opened, the source program must prove:

- exactly 213,095 unique blocks at heights `610691..823785`;
- contiguous heights and `previousblockhash` linkage throughout;
- every header timestamp strictly before `2024-01-01T00:00:00Z`;
- `0 < weight <= 4,000,000`, positive `size` and `tx_count`, and the serialized
  size/weight invariant for every row;
- at least 99.5% of packet spans have `elapsed_seconds > 0`;
- no run of invalid packet spans exceeds 12 consecutive endings;
- every accepted signal has all six confirmation blocks inside the prefix;
  and
- no market, price, funding, premium, OI, liquidation, order-book, FX, return,
  PnL, or post-2023 source field was loaded.

Failure rejects BATE-288 without changing elapsed-time treatment, packet size,
reference, threshold, confirmation count, embargo, or hold.

## Outcome-blind event-support gate

The 2020 source is warm-up only. The non-overlapping source clock must satisfy
all of the following before market outcomes are available:

### 2021–2022 train clock

- at least 80 accepted events total;
- at least 32 events in each calendar year;
- at least 25 `HIGH` and 25 `LOW` events total;
- at least 10 events of each side in each calendar year;
- at least 14 events in each calendar half-year; and
- no calendar month contains more than 15% of train events.

### 2023 selection clock

- at least 35 accepted events total;
- at least 12 `HIGH` and 12 `LOW` events;
- at least 14 events in each calendar half-year;
- at least 5 events of each side in each half-year;
- at least 6 events in every calendar quarter; and
- no calendar month contains more than 20% of 2023 events.

Counts, side balance, invalid-span incidence, and calendar dispersion are the
only values that may be opened at this stage. No price level, future move,
return, funding payment, equity path, MDD, or PnL may be joined or printed.

## Frozen performance gate

Only a source-support pass permits a separately hash-frozen strict evaluator.
That evaluator must use next-open execution, exact funding, 6 bp/notional/side
base costs, 10 bp/notional/side stress costs, full-calendar CAGR including idle
cash, and global strict path MDD with pre-entry high-water mark and
favorable-before-adverse held OHLC ordering.

Both 2021–2022 train and 2023 selection must have:

- positive absolute return;
- CAGR / strict MDD at least 3.0;
- strict MDD at most 15%;
- positive net contribution from both `HIGH` longs and `LOW` shorts;
- weekly-cluster sign-flip `p <= 0.10`;
- mean gross underlying move at least 30 bp;
- positive result at 10 bp/notional/side stress cost; and
- positive result with entry delayed by one additional five-minute bar.

Calendar 2021 and 2022 must each be positive. Both halves of 2023 must be
positive. A split, side, stress, delayed-entry, or statistical failure rejects
the singleton; it does not authorize a one-sided repair.

## Controls and sealed sequence

Frozen controls are exact direction flip, weight-only state, transaction-only
state, denominator-free six-block average, 24-hour stale state, and
year-stratified random non-overlapping clocks with identical side counts. A
component-only or denominator-free control passing every primary gate rejects
the claimed concordant-throughput mechanism even if the primary also passes.

Evaluation order is fixed:

1. complete source integrity and support only;
2. commit one strict evaluator without opening outcomes;
3. open 2021–2022 train; stop on failure;
4. open 2023 selection; stop on failure;
5. test orthogonality and marginal portfolio contribution; and
6. only after all prior gates pass, open 2024, 2025, and 2026 YTD sequentially,
   stopping at the first failed sealed year.

Every opened performance report must include absolute return, CAGR, strict
MDD, CAGR/strict-MDD, and trade count. The branch is globally contaminated by
earlier BTC research, so any pass is candidate-level frozen evidence only.
