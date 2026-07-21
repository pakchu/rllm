# BLSR-288 — Blockspace Load-to-Settlement Relay mechanism decision

## Decision and evidence boundary

The next standalone BTC candidate is **BLSR-288**, a sequential confirmed-
ledger relay with a fixed 288 five-minute-bar / 24-hour hold.

BLSR does not trade a fee or endpoint tail immediately. It first observes a
large change in confirmed fee pressure and then waits for the **first later
large endpoint-density response**. A same-signed response confirms broadening
or contracting settlement load; an opposite response cancels the episode.

This document freezes the source axis, exact causal state machine, tentative
direction, latency, hold, support/novelty floors, controls, and stopping rule
before a BLSR feature value or event incidence is opened. It reads no source
row, BTC bar, funding mark, future return, PnL, equity, CAGR, MDD, existing-
alpha outcome, or post-2023 value.

## Why this is a different mechanism

The same confirmed-ledger source has already supported several falsified
hypotheses. BLSR is not an in-place repair of any of them:

- **FETD-288** compared two-packet fee and endpoint transports in the *same*
  feature row and admitted opposite signs. It was rejected before outcomes for
  temporal concentration. BLSR preserves FETD's packet, rank, availability,
  and hold conventions but changes the economic object to an ordered,
  first-response relay with same-signed confirmation. It does not lower a FETD
  threshold or reuse a failed FETD branch.
- **BATE-288** used six-block transaction/weight throughput divided by elapsed
  header time. BLSR uses no elapsed-time denominator and no transaction count.
- **UFCP-1** grouped fees and signed UTXO polarity by UTC day. BLSR never uses
  `utxo_set_change` and never assigns a block to a header-time calendar day.
- **WCTR-288** used witness-discount composition and transaction fullness;
  BLSR uses neither field.

The hypothesis can still be false. Fee changes can reflect inscription bursts,
fee estimation, batching, exchange maintenance, self-transfers, consolidation,
or auction noise. Endpoint counts do not identify owners, value transferred,
or economic purpose. The sequential rule is a falsifiable transport proxy, not
an assertion that confirmed endpoints measure adoption.

## Frozen source binding

- source:
  `data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz`;
- source SHA-256:
  `8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f`;
- source manifest:
  `results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json`;
- manifest SHA-256:
  `ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084`;
- rows / heights: 213,095 contiguous best-chain blocks,
  `610691..823785`; and
- frozen source cutoff: every retained header timestamp is before
  `2024-01-01T00:00:00Z`.

Signal construction may read only `height`, `id`, `previousblockhash`,
`timestamp`, `weight`, `total_fees`, `total_inputs`, and `total_outputs`.
`mediantime`, `size`, `tx_count`, `utxo_set_change`, price, funding, premium,
OI, liquidation, order book, FX, and every post-entry field are forbidden.

Bitcoin Core `getblockstats` is the production definition for fees, inputs,
outputs, and total weight. Historical transport used a hash-audited Mempool
cache; live promotion requires an owned, version-pinned Bitcoin Core node,
local first-seen timestamps, reorg invalidation, and field-by-field parity.

Official references:

- <https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblockstats/>
- <https://mempool.space/docs/api/rest>
- <https://github.com/mempool/mempool>
- <https://developer.bitcoin.org/reference/block_chain.html>

## Frozen packet features

Partition by absolute height, never by the research start or a header-time
calendar:

```text
packet_id = floor(height / 72)
```

A packet is valid only when all 72 heights from `72*packet_id` through
`72*packet_id+71` exist, are unique, and are hash-linked. For packet `t`:

```text
packet_weight[t]    = sum(weight)
packet_fees[t]      = sum(total_fees)
packet_endpoints[t] = sum(total_inputs + total_outputs)

fee_pressure[t]     = log(packet_fees[t] / packet_weight[t])
endpoint_density[t] = log(packet_endpoints[t] / packet_weight[t])

fee_change[t]       = fee_pressure[t] - fee_pressure[t-1]
endpoint_change[t]  = endpoint_density[t] - endpoint_density[t-1]
```

All totals must be positive and finite. There is no clipping, epsilon,
interpolation, reassignment, forward fill, or header-time day bucketing.

For each change, compute an absolute-magnitude empirical midrank against the
latest 180 valid prior packet changes, requiring at least 120. The current
packet is excluded, and every reference packet must have become available
strictly earlier:

```text
midrank(x) = (count(prior < x) + 0.5*count(prior == x)) / prior_count

fee_magnitude_rank[t]      = midrank(abs(fee_change[t]))
endpoint_magnitude_rank[t] = midrank(abs(endpoint_change[t]))
```

An exact binary64 zero has no sign and cannot start or confirm an episode.
The fixed significance boundary is `rank >= 0.75`. There is no threshold,
packet, reference-length, or transform grid.

## Frozen first-response relay

Process valid packets in increasing `packet_id` order.

The synthetic `source_available_at` sequence must be strictly increasing with
`packet_id`; a tie or regression is a source-clock failure rather than a reason
to sort reports into an invented order.

1. When no episode is active, the first packet with
   `fee_magnitude_rank >= 0.75` and nonzero `fee_change` starts a fee-load
   episode with `load_sign = sign(fee_change)`.
2. While active, ignore later fee shocks. Inspect exactly the next three packet
   reports in order.
3. The **first** later packet with
   `endpoint_magnitude_rank >= 0.75` and nonzero `endpoint_change` resolves the
   episode. No stronger or later response may replace it.
4. If `sign(endpoint_change) == load_sign`, emit one candidate. If signs
   disagree, cancel without a candidate.
5. If no significant endpoint response appears by the third later packet,
   expire without a candidate.
6. A resolved or expired episode may not restart on an already-inspected
   packet. The next packet is the earliest possible new onset.

Frozen action:

```text
fee load rises, endpoint density later rises -> LONG
fee load falls, endpoint density later falls -> SHORT
```

The long branch interprets rising fees followed by broad endpoint clearing as
confirmed settlement demand. The short branch interprets falling fees followed
by thinning endpoint clearing as a confirmed demand drought. Opposite first
responses falsify, rather than reverse, the episode.

## Causal availability and execution

Header timestamps are miner-reported event fields, not historical receipt
logs. For every packet report ending at height `h`:

1. require hash-linked successors through `h+6`;
2. set packet availability to
   `max(timestamp[packet_start:h+6]) + 48 hours`;
3. a relay may consume a packet only at that availability;
4. after same-sign confirmation, set entry to
   `ceil_5m(confirming_available_at) + 5 minutes`; and
5. set scheduled exit to exactly 288 five-minute bars / 24 hours after entry.

Sort confirmed candidates by `(entry_time, onset_packet_id,
confirmation_packet_id)`. Accept the earliest candidate whose entry is at or
after the prior accepted exit. Suppress every overlapping candidate without
score priority or replacement. Entry and the complete half-open hold must be
inside one declared split.

Live operation must use the later of synthetic availability and locally
persisted first-seen/confirmation time. A reorg, stale node, missing successor,
or inconsistent Core field cancels the report and cannot be backdated.

## Frozen calendar and source-only gate

- source warm-up: calendar 2020 only;
- train: `[2021-01-01, 2023-01-01)` UTC;
- selection: `[2023-01-01, 2024-01-01)` UTC; and
- sealed: 2024 and later.

Before any BTC market or funding value is opened, primary support must satisfy
every gate:

### Train, 2021–2022

- at least 80 accepted entries total;
- at least 30 in each year;
- at least 12 in each half-year;
- at least 24 LONG and 24 SHORT total;
- at least 8 of each side in each year;
- maximum calendar-month share at most 20%; and
- maximum UTC-entry-weekday share at most 25%.

### Selection, 2023

- at least 35 accepted entries total;
- at least 14 in each half-year;
- at least 6 in every quarter;
- at least 12 LONG and 12 SHORT total;
- at least 4 of each side in each half-year;
- maximum calendar-month share at most 20%; and
- maximum UTC-entry-weekday share at most 25%.

All packet, availability, relay-order, first-response, non-overlap, hold, side,
and split-containment checks must pass exactly.

## Frozen controls

Each source control is formed independently before its own chronological
non-overlap scheduler. A control may not inherit free opportunities released
by a primary abstention.

1. **Fee-only:** every significant fee change, side `sign(fee_change)`.
2. **Endpoint-only:** every significant endpoint change, side
   `sign(endpoint_change)`.
3. **Same-packet agreement:** both changes significant and same-signed in one
   packet, side equal to that sign.
4. **Reverse-order relay:** significant endpoint change starts the episode;
   the first significant fee change within three later packets must agree.
5. **Opposite-response relay:** exact primary onset/deadline contract, but emit
   only when the first significant endpoint response disagrees; side remains
   the original fee-load sign.
6. **One-packet-stale response:** the confirming endpoint state is shifted by
   one complete packet and applied at the later packet's availability.
7. **Exact direction flip:** primary entries with `side=-primary_side`.
8. **Deterministic random side:** exact primary entries; SHA-256 of
   `"BLSR-288-random-side-20260721|" + entry_time` assigns LONG when the first
   digest byte is below 128 and SHORT otherwise.
9. **One-bar latency:** shift primary entry and exit by exactly five minutes;
   drop, never replace, a trade that leaves its original split.

The source-only report publishes only aggregate counts, side/calendar support,
drop reasons, clock hashes, and comparator statistics. Exact source values are
not published. Controls are diagnostics and later mechanism falsifiers; none
may replace a failed primary.

## Frozen novelty gate

Rebuild or load hash-bound pre-2024 clocks for FETD-288, BATE-288, UFCP-1,
WCTR-288, and the frozen prior-microstructure comparator bundle. Comparator
market outcomes and manifest performance values are forbidden.

For every nonempty comparator, require:

- exact five-minute entry-timestamp Jaccard `<= 0.20`;
- one-to-one tolerant matching within plus/minus six hours covers at most
  `35%` of BLSR entries; and
- when direction and exit are available, absolute signed occupied-exposure
  Pearson correlation on the full `[2021-01-01, 2024-01-01)` five-minute UTC
  grid `<= 0.40`.

Timestamp-only comparators omit only the correlation check. Dense calendars do
not excuse the exact or tolerant overlap gates. A comparator may not be removed
after incidence is observed.

## Sequential economic gate

Only a complete support/novelty pass permits a separately implemented, tested,
committed, and hash-frozen strict evaluator. Open 2021–2022 train first. Open
calendar 2023 only after an exact train pass. Open 2024, 2025, and recent 2026
sequentially only after every preceding window passes under an unchanged
policy.

Every opened window must report absolute return, full-calendar CAGR including
idle cash, global/pre-entry-HWM strict MDD over every held five-minute path,
CAGR/strict-MDD, trade count, both sleeves, exact funding, base/stress costs,
extra latency, and weekly-cluster significance.

Primary qualification requires:

- positive absolute return;
- `CAGR / strict MDD >= 3.0`;
- strict MDD `<= 15%`;
- mean gross underlying move at least 30 bp/trade;
- weekly-cluster one-sided sign-flip `p <= 0.10`;
- positive 10 bp/notional/side stress return;
- positive one-bar-delayed return;
- positive contained 2021, 2022, 2023-H1, and 2023-H2 returns; and
- positive LONG and SHORT sleeves in train and selection.

The primary's minimum train/selection ratio must exceed every finite fee-only,
endpoint-only, same-packet, reverse-order, opposite-response, and stale control
by at least 0.25. Direction-flip, random-side, and latency controls are also
mandatory falsifiers. A winning control rejects BLSR-288; it cannot be promoted
under this identity.

## RLLM boundary

BLSR must first establish a deterministic causal edge. Only after a complete
standalone and orthogonality pass may one compact Gemma policy receive symbolic
state such as fee-load sign/rank, response sign/rank, packets-to-deadline,
current position, and time-to-exit. The RLLM may abstain or size within a frozen
risk envelope. It may not create the base event, change direction/hold, consume
raw identifiers/timestamps, or use a sealed failure to redesign BLSR.

## Stop rule

The first source, support, novelty, train, selection, test, eval, or forward
failure retires BLSR-288. No threshold, packet size, rank history, deadline,
sign, availability, latency, hold, support floor, comparator, or control may be
repaired under this identity.

The ledger source has been opened by prior source-only work and the repository
has broad historical BTC exposure. Any pass is candidate-level frozen evidence,
not a pristine global holdout claim.
