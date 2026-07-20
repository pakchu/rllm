# FETD-288 fee–endpoint topology disagreement preregistration

Status: **frozen before any FETD feature value, event incidence, BTC market
value, funding value, return, PnL, CAGR, or MDD was opened**.

## Singleton hypothesis

`FETD-288` tests one symmetric disagreement state in confirmed Bitcoin ledger
composition.  A rise in confirmed fee pressure accompanied by a fall in
input/output endpoint density is tentatively short; the opposite transport is
tentatively long.  Same-direction transport is a control, not a primary
signal.

There is one packet size, transport horizon, rank history, threshold, side
mapping, publication lag, hold, and scheduler.  There is no feature, threshold,
side, horizon, hold, or latency grid.

Frozen source:

- `data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz`;
- SHA-256
  `8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f`;
- heights `610691..823785`, all before the 2024 header-time cutoff; and
- manifest
  `results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json`.

The mechanism and source contract are frozen in
[`fee-endpoint-topology-disagreement-mechanism-decision-2026-07-20.md`](fee-endpoint-topology-disagreement-mechanism-decision-2026-07-20.md).

## Exact source-only feature

Partition blocks by `packet_id = floor(height / 72)`.  Drop an edge packet
unless all 72 heights from `72*packet_id` through `72*packet_id+71` are present
and hash-linked.  This absolute-height alignment does not depend on the frozen
research start.  For complete packet `t`:

```text
total_weight[t]    = sum(weight)
total_fees[t]      = sum(total_fees)
total_endpoints[t] = sum(total_inputs + total_outputs)
fee_pressure[t]    = log(total_fees[t] / total_weight[t])
endpoint_density[t]= log(total_endpoints[t] / total_weight[t])
```

A source packet is valid only when it contains exactly 72 contiguous blocks,
every included row satisfies the frozen source invariants, and all three
aggregate totals are positive and finite.  No block or packet is clipped,
interpolated, forward-filled, reassigned, or imputed.

A feature row requires valid, consecutive packet IDs `t-2`, `t-1`, and `t`:

```text
fee_transport      = fee_pressure[t] - fee_pressure[t-2]
endpoint_transport = endpoint_density[t] - endpoint_density[t-2]
strain_magnitude   = abs(fee_transport * endpoint_transport)
```

All operations use IEEE-754 binary64 and natural logarithms.  There is no
epsilon or rounding.  Exact binary64 equality defines zero and rank ties.

For each base-valid feature row, calculate strict-prior empirical midranks over
the most recent 180 base-valid feature rows, requiring at least 120:

```text
midrank = (count(prior < current) + 0.5*count(prior == current)) / prior_count
```

The current row is excluded and every reference row must have become
available strictly earlier.  Freeze:

- `strain_rank = midrank(strain_magnitude)`;
- `fee_magnitude_rank = midrank(abs(fee_transport))` for the fee-only control;
  and
- `endpoint_magnitude_rank = midrank(abs(endpoint_transport))` for the
  endpoint-only control.

The primary rule is exactly:

```text
fee_transport * endpoint_transport < 0
and strain_rank >= 0.75
```

The side is `-sign(fee_transport)`, equivalently
`sign(endpoint_transport)`:

- fee up / endpoint density down -> short; and
- fee down / endpoint density up -> long.

An exact zero in either transport cannot be primary-eligible.  Every eligible
source clock is considered; there is no onset requirement or score-priority
selection.

## Causal availability and event selection

For every packet ending at height `h`:

1. require hash-linked blocks through `h+6`, all contained in the frozen
   source;
2. set
   `source_available_at = max(timestamp[packet_start:h+6]) + 48 hours`;
3. set `entry_time = ceil_5m(source_available_at) + 5 minutes`; and
4. set `scheduled_exit_time = entry_time + 24 hours`.

The extra five minutes is mandatory even when synthetic availability is
already aligned.  Historical header timestamps are never treated as receipt
times.  Height packets prevent a future backdated header from being inserted
into an apparently closed calendar bucket.

Sort candidates by `(entry_time, packet_id)`.  Accept the earliest candidate
whose entry is at or after the prior accepted exit.  Suppress every
intervening candidate, including opposite-side candidates, without replacement
or outcome-based priority.  Entry equal to the prior exit is allowed.  The
entry and complete half-open hold `[entry_time, scheduled_exit_time)` must be
inside one declared split.

## Frozen calendar

- source warm-up only: calendar 2020;
- train: `[2021-01-01T00:00:00Z, 2023-01-01T00:00:00Z)`;
- selection: `[2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`; and
- sealed: 2024 and later.

No coefficient is fitted.  Selection may not change, rerank, invert, refit, or
repair the policy.  A complete pre-2024 pass is required before any 2024+
source extension or outcome is opened.

## Outcome-blind source/support gate

Before any market or funding value is loaded, the source builder must prove:

- exactly 213,095 unique, contiguous, hash-linked blocks at heights
  `610691..823785`;
- every row is before the frozen 2024 cutoff;
- positive size/weight/transaction counts, valid block weight, nonnegative
  fees/input/output counts, and exact `outputs-inputs == utxo_set_change`;
- exactly 2,959 complete, consecutive 72-block packets from heights
  `610704..823751`, with only incomplete edge packets dropped;
- every feature packet has all six required successor blocks; and
- zero market, funding, premium, OI, liquidation, order-book, return, PnL, or
  post-2023 source value was loaded.

The accepted primary clock must satisfy all gates conjunctively.

### Train, 2021–2022

- at least 80 entries total;
- at least 32 entries in each calendar year;
- at least 25 long and 25 short entries total;
- at least 10 entries of each side in each calendar year;
- at least 14 entries in each calendar half-year; and
- no calendar month contains more than 15% of train entries.

### Selection, 2023

- at least 35 entries total;
- at least 12 long and 12 short entries;
- at least 14 entries in each calendar half-year;
- at least 5 entries of each side in each half-year;
- at least 6 entries in every calendar quarter; and
- no calendar month contains more than 20% of selection entries.

Only counts, side balance, source integrity, and calendar dispersion may be
opened here.  A failure rejects FETD-288 without threshold, side, rank-history,
packet, latency, hold, support-floor, or calendar repair.

## Frozen controls

- direction flip on the exact primary clock, diagnostic only;
- fee-only: `fee_magnitude_rank >= 0.75`, side
  `-sign(fee_transport)`, independent chronological clock;
- endpoint-only: `endpoint_magnitude_rank >= 0.75`, side
  `sign(endpoint_transport)`, independent chronological clock;
- same-direction transport: positive transport product and
  `strain_rank >= 0.75`, side `sign(endpoint_transport)`, independent clock;
- constant long and constant short on the primary clock;
- 14-packet stale primary feature/ranks, applied at the later packet's
  availability and given an independent clock;
- month-and-side-stratified SHA-256 random clocks, seed `20260720`; and
- one complete five-minute bar delayed entry and exit.  Shift both timestamps
  exactly five minutes, deterministically drop a shifted trade if it no longer
  fits its original split, never replace it, and report the dropped count for
  train and selection in the source-only support artifact before outcomes.

If a fee-only, endpoint-only, same-direction, constant-side, stale, or random
control independently satisfies the complete primary performance gate, reject
the specific fee/endpoint-disagreement mechanism.  A passing control may not
replace the primary under FETD-288.

## Frozen performance gate

Only a source/support pass permits a separately committed and hash-frozen
strict evaluator.  Open train first and stop on failure; only an exact train
pass may open 2023 selection.  Each opened window must have:

- positive absolute return;
- full-calendar CAGR / global strict MDD at least 3.0;
- global/pre-entry-HWM strict MDD no greater than 15%;
- weekly-cluster one-sided sign-flip `p <= 0.10` with 100,000 draws and seed
  `20260720`;
- mean gross underlying move at least 30 bp per trade;
- positive absolute return at 10 bp/notional/side stress cost; and
- positive absolute return with entry delayed one additional five-minute bar.

Calendar 2021, 2022, 2023-H1, and 2023-H2 must each be positive.  Long and
short sleeves must each be positive in both train and selection.  Execution is
next-open with 0.5x notional exposure, 6 bp/notional/side base costs, and exact
entry-inclusive/exit-exclusive funding at fixed entry quantity.

CAGR uses the complete declared wall-clock window including idle cash.  Strict
MDD uses a global and pre-entry high-water mark, all costs, exact funding, and
favorable-before-adverse held OHLC/funding path ordering.

## Stopping and contamination rule

Stop permanently at the first support, train, or selection failure.  No failed
policy inversion is allowed.  The source was previously opened for unrelated
UFCP support, and the repository has broad prior BTC outcome exposure.  Any
eventual pass is therefore candidate-level frozen evidence, not a pristine
global holdout claim.
