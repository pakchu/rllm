# AMTR-48 authorized-minter turnaround relay — outcome-blind mechanism freeze

## Decision and sealed boundary

Freeze one new candidate, **AMTR-48**, before reading USDC address incidence,
forming a real pair, or opening a comparator or BTC outcome.

AMTR-48 tests whether a large USDC authorized-minter activity sequence that
turns from burn to mint or mint to burn precedes a BTC liquidity regime. It is
an event-order and actor-identity hypothesis. It is not an aggregate supply
breadth, stablecoin quote-price, or BTC taker-flow rule.

The complete source event totals and hashes are already known from the prior
source audit. This freeze does **not** inspect minter identities, recipient
identities, per-address counts, amount distributions, tail incidence, pair
incidence, sides, calendar concentration, comparator timestamps, BTC price,
funding, future return, PnL, absolute return, CAGR, or MDD.

Any source, support, actor-concentration, calendar, or novelty failure retires
AMTR-48. Its event types, tail, warmup, matching, direction, gap, ratio,
latency, hold, gates, and comparator set may not be repaired after incidence.

## Contract semantics and interpretation limit

Circle's official FiatTokenV1 contract defines:

```solidity
event Mint(address indexed minter, address indexed to, uint256 amount);
event Burn(address indexed burner, uint256 amount);
```

`mint` may be called only by an authorized minter and may send tokens to a
different recipient. `burn` may be called only by an authorized minter and
burns that caller's own token balance. The exact implementation is documented
in Circle's official repository:

- <https://github.com/circlefin/stablecoin-evm/blob/master/contracts/v1/FiatTokenV1.sol>
- <https://developers.circle.com/stablecoins/usdc-contract-addresses>

AMTR therefore uses `indexed_address_1` only as an **authorized operational
role identity**. It does not call the address a customer, exchange, entity,
treasury owner, or BTC buyer. `indexed_address_2` on mint is a recipient and is
used only for source concentration checks. An address can represent custody or
administrative infrastructure, and one entity can use multiple addresses.

The frozen direction is a hypothesis about activity polarity, not proof of
fiat inflow or redemption:

- a large burn followed by a large mint by the same authorized role is
  tentatively `LONG`, testing renewed issuance activity after contraction;
- a large mint followed by a large burn by the same authorized role is
  tentatively `SHORT`, testing rapid issuance reversal or contraction.

## Frozen source rows and clocks

Use only the hash-bound 2020–2023 Ethereum panel rows satisfying:

```text
asset == "usdc_eth"
event in {"mint", "burn"}
```

For each eligible row:

```text
amount_usd     = integer(amount_raw) / 1_000_000
minter         = indexed_address_1
mint_to        = indexed_address_2 for mint, empty for burn
event_time     = available_at
occurrence_time = block_timestamp
identity       = (block_hash, transaction_hash, log_index)
```

Empty or malformed actor addresses fail closed. Events are ordered by
`(available_at, block_number, transaction_index, log_index, identity)`.
`available_at` controls causality; `block_timestamp` is retained only to prove
that two administrative operations were also separated in occurrence time.

## Frozen strictly-prior tail

Mint and burn have separate global reference histories. For current event `e`,
its reference sample contains only same-event-type rows satisfying:

```text
prior.available_at < e.available_at
e.available_at - 365 calendar days <= prior.available_at
```

At least 256 strictly prior rows are required. Sort prior `amount_usd`
ascending. The frozen nearest-rank 95th percentile is element
`ceil(0.95 * n) - 1` under zero-based indexing. Event `e` is large when its
amount is greater than or equal to that threshold.

Same-`available_at` events are excluded from one another's history. No current
row, full-sample percentile, address-conditioned percentile, interpolation,
future row, or post-2023 row may enter the threshold. A global tail can favor
large-ticket minters, so hard actor-concentration gates below are mandatory and
cannot be replaced by a later per-address normalization.

## Frozen deterministic pair construction

Process large events in causal order. For current event `e`, search unmatched
large events `p` that:

1. have the opposite event type;
2. have `p.minter == e.minter`;
3. satisfy `30 minutes <= e.available_at - p.available_at <= 24 hours`;
4. satisfy
   `30 minutes <= e.block_timestamp - p.block_timestamp <= 24 hours`; and
5. satisfy `min(p.amount_usd, e.amount_usd) / max(...) >= 0.50`.

If several prior legs qualify, choose the one with the latest
`available_at`; break any remaining tie by canonical identity in ascending
order. If none qualifies, emit no pair and consume no event. Once a valid pair
forms, consume both legs permanently. No best-ratio, best-amount, nearest-block,
future-leg, or return-aware matching is permitted.

Pair completion is the later `available_at`. Direction is fixed by order:

```text
burn -> mint = LONG
mint -> burn = SHORT
```

The 30-minute lower bound excludes same-batch and near-immediate bookkeeping.
Requiring it on both causal availability and occurrence time prevents an RPC
confirmation offset from creating artificial separation.

## Frozen execution clock

For pair completion `t`:

```text
latency_bar_start = first UTC five-minute boundary at or after t
entry_time        = latency_bar_start + 5 minutes
scheduled_exit    = entry_time + 48 hours
```

If `t` is exactly on a five-minute boundary, the complete bar beginning at `t`
must elapse. Candidate pairs are considered by `(entry_time, pair identities)`.
Global non-overlap is mandatory: accept only when `entry_time` is at or after
the previous accepted `scheduled_exit`. Pair legs remain consumed even when a
valid pair is skipped by global reservation; the pair clock is formed before
the portfolio scheduler.

## Frozen source-only support gates

All gates are conjunctive:

1. **Source integrity**
   - exact promoted source and manifest hashes;
   - dual canonical-log replay equality;
   - independent event-block header cross-check;
   - block `N+64` causal availability and finalized coverage;
   - no BTC, funding, return, label, PnL, or post-2023 event access.
2. **Incidence**: at least 60 globally accepted pairs over 2020–2023.
3. **Year dispersion**: at least 12 accepted entries in each of 2021, 2022,
   and 2023. The 2020 warmup cannot replace a failed full year.
4. **Side balance**: LONG and SHORT each at least 30%.
5. **Actor breadth**: at least 5 distinct `minter` addresses.
6. **Actor concentration**
   - no minter exceeds 40% of all accepted pairs;
   - no minter exceeds 60% within either side; and
   - no mint recipient exceeds 50% of accepted mint legs.
7. **Calendar concentration**: no UTC entry month exceeds 20% of accepted
   pairs.
8. **Novelty against secondary-market stablecoin clocks**: against each frozen
   SQFD-6 (`primary`, `no_usdt_lag`, `no_participation`), SDDR-12 `primary`,
   and UCBR-12 `primary` clock:
   - exact-entry Jaccard at most 0.05;
   - UTC entry-hour Jaccard at most 0.10; and
   - maximum bidirectional containment within plus/minus six hours at most
     0.25.
9. **Novelty against rejected stablecoin supply breadth**: on the common
   defined interval, UTC entry-date Jaccard at most 0.20 and maximum
   bidirectional containment within plus/minus one calendar day at most 0.40.

Comparator clocks must be checksum-bound before timestamps are opened. A
missing comparator, schema drift, duplicate timestamp, side drift, outcome
field, or grid-only exact-overlap pass fails closed. Novelty is evaluated only
after all earlier source-support gates pass; an earlier failure short-circuits
comparator access.

## Frozen source-only controls

These diagnostics cannot replace the primary after support is opened:

- `cross_minter`: pair the latest qualifying opposite event with a different
  minter, retaining all other rules;
- `no_amount_ratio`: remove only the 0.50 amount-ratio gate;
- `no_minimum_gap`: reduce only the two 30-minute lower bounds to zero while
  retaining the 24-hour upper bounds;
- `stale_6h`: delay completed primary pairs exactly six hours before the same
  latency and reservation scheduler.

Later outcome-only controls, if ever authorized, must include exact direction
flip and a deterministic event-count/side-matched random-side control. Control
incidence cannot authorize a threshold, matching, actor, direction, or hold
repair.

## Later outcome and RLLM boundary

Only a full support and novelty pass may authorize a separately committed,
hash-frozen outcome evaluator. It must use exact next-open BTCUSDT USD-M
perpetual execution, realized funding, full-calendar CAGR, strict intratrade
position-path MDD, 6 bp base and 10 bp stress cost per notional side, and
sequential train/test/eval/holdout opening. Every stage must report absolute
return together with CAGR, strict MDD, CAGR/MDD, trades, sides, and clustered
significance.

Gemma/RLLM may not create, retime, rematch, or reverse AMTR-48. Only after the
deterministic clock demonstrates gross edge above costs may a train-only model
be evaluated as an abstention or risk-routing layer over symbolic event
sequences. A deterministic failure retires AMTR-48 rather than transferring
its choices to an LLM.

## Bound references

- `docs/ethereum-stablecoin-issuance-redemption-source-feasibility-2026-07-21.md`
- `docs/ethereum-stablecoin-issuance-redemption-source-audit-2026-07-21.md`
- `results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json`
- `training/build_ethereum_stablecoin_issuance_redemption.py`
- `docs/stablecoin-supply-breadth-absorption-frozen-oos-2026-07-16.md`
- `docs/stablecoin-quote-flow-diffusion-support-freeze-2026-07-19.md`
- `docs/stablecoin-denominator-dislocation-support-result-2026-07-20.md`
- `docs/usdt-collateral-breadth-relay-support-rejection-2026-07-20.md`

