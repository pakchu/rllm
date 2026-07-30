# ESDI-288 — Ethereum Settlement Demand Impulse preregistration

Date: 2026-07-30

Status: **frozen before the full Ethereum source replay, exact ESDI
incidence, BTC market/funding access, or any outcome**.

## Decision

Freeze one source-new BTC alpha candidate:

```text
ESDI-288 — Ethereum Settlement Demand Impulse, 24-hour hold
```

ESDI-288 treats Ethereum's EIP-1559 base fee as a canonical price of scarce
execution blockspace. A large increase in the median base fee over roughly one
day is interpreted as broad crypto settlement-demand expansion and maps to
`LONG` BTC. A large decrease maps to `SHORT` BTC.

This is a falsifiable cross-asset transmission hypothesis, not an Ethereum
protocol claim and not evidence of profitability. Ethereum-specific activity,
NFT or token launches, liquidations, spam, and application migration can all
move base fees without creating a BTC edge.

## Why this source is admissibly new

Repository-wide search before this freeze found no alpha source, feature, or
clock using:

- `eth_feeHistory`;
- `baseFeePerGas`;
- `gasUsedRatio`; or
- an Ethereum EIP-1559 execution-demand history.

The rejected Coin Metrics stablecoin/network candidate is not continued:
`SplyCur` is a reviewed latest-snapshot history and cannot prove the value
known at each historical timestamp. The rejected ETH+BTC witness alternative
is also not continued: the WCTR source is a rolling present-day aggregate
snapshot and WCTR-288 is already terminal.

ESDI uses no stablecoin value, Bitcoin network value, WCTR value, exchange
feature, BTC price, funding, premium, open interest, order flow, macro value,
Gross9 state, or prior alpha state to construct its source signal.

The source axis is new, but the repository has broad BTC-outcome exposure.
Only the exact ESDI source replay, incidence, and outcome sequence are
candidate-specific unopened evidence. A historical pass cannot be called a
globally pristine discovery.

## Official semantic authority

Only Ethereum mainnet, chain ID `1`, is source authority. Public RPC services
are independent transports, not content authorities.

The source implementation is bound to the following official semantics:

- Ethereum execution API `eth_feeHistory`:
  <https://ethereum.github.io/execution-apis/api/methods/eth_feeHistory/>
- Ethereum execution API `eth_getBlockByNumber`:
  <https://ethereum.github.io/execution-apis/api/methods/eth_getBlockByNumber/>
- EIP-1559 base-fee update rule:
  <https://eips.ethereum.org/EIPS/eip-1559>
- Ethereum proof-of-stake slots, epochs, and finality:
  <https://ethereum.org/developers/docs/consensus-mechanisms/pos/>
- Ethereum archive-node semantics:
  <https://ethereum.org/developers/docs/nodes-and-clients/archive-nodes/>

`eth_feeHistory` returns one `baseFeePerGas` value per returned block plus the
next block's base fee, and one `gasUsedRatio` per returned block. ESDI uses
only the per-block base fees. Gas-used ratios are transport-integrity fields
and a preregistered source-only control; they do not enter the primary rule.

## Excluded transport feasibility probes

Before this freeze, bounded probes established only transport feasibility:

- three transports returned byte-equivalent canonical `eth_feeHistory`
  results for one 1,024-block request ending at block `0x1100000`;
- the canonical result hash was
  `d87bb67a79d84b0f5c691150a69cfde7295cc6fd521d06952473487d1db01ba5`;
- the response shape was 1,025 base-fee values and 1,024 gas-used ratios;
- exact boundary block numbers, timestamps, and hashes were cross-checked;
- no base-fee value, gas-used ratio, epoch aggregate, ESDI feature, rank,
  signal, incidence, BTC row, funding row, return, or PnL was printed or used
  to choose this rule.

These probes are not a source artifact and may not be copied into the source
output. The official builder must refetch the complete frozen range through
both bound transports and fail closed on any disagreement.

## Frozen source envelope

### Canonical calendar boundaries

The following first blocks at or after each UTC boundary were independently
cross-checked:

| UTC boundary | block | hash |
|---|---:|---|
| 2023-01-01 00:00 | 16,308,190 | `0x53dd35d982c984441b3b613919d64dbbf131063d0f85804d77f93f190fa5e106` |
| 2023-06-01 00:00 | 17,382,266 | `0xe0ef11cab4909c80599087b4ffb0bf1e92b1affcc72abc3b802f20a9d5d21096` |
| 2025-01-01 00:00 | 21,525,891 | `0x9512042c5c38145528389a91bd3d63193a1f48fb45d6a3b144ad2d833331fc4c` |
| 2026-01-01 00:00 | 24,136,053 | `0x53e1c0caa885383824d39dc57c0692ea20e971ade409553c4a8031e90f44c516` |
| 2026-06-01 00:00 | 25,218,798 | `0x55f8fdbda40a23cd51a9a2bffba625317ed15d9d1cdc2128c7643bf66e2a906e` |

For each boundary, the immediately preceding block timestamp is strictly
before the boundary and the listed block timestamp is at or after it. The
source builder must reproduce every number, timestamp, hash, and parent-hash
relation through both bound transports before opening the full range.

### Fixed block epochs

The source clock is block-indexed, not timestamp-bucketed:

```text
epoch_size_blocks = 3,600
epoch e blocks     = [3,600*e, 3,600*(e+1)-1]
```

The frozen source contains exactly epoch IDs `4,531` through `7,004`
inclusive:

```text
first source block = 16,311,600
last source block  = 25,217,999
```

The first epoch starts after the 2023-01-01 boundary and supplies more than
the required strictly-prior warm-up before the first permitted ESDI entry on
2023-06-01. The last epoch's confirmation block is still before the
2026-06-01 boundary.

No partial epoch, timestamp-nearest epoch, moving block count, or calendar
resampling is allowed.

## Dual replay and causal availability

The source builder must freeze exactly two HTTPS JSON-RPC transports before
the first full-range request. Both must:

1. report chain ID `1`;
2. reproduce every frozen boundary header;
3. return the complete requested `eth_feeHistory` subsection, never a shorter
   permissible subsection;
4. agree exactly on `oldestBlock`, every `baseFeePerGas`, and every
   `gasUsedRatio`;
5. agree exactly on every epoch-end and confirmation block header; and
6. end at a common finalized head at or after the last required confirmation
   block.

The request chunk size is exactly 1,024 blocks. Adjacent chunks overlap only
through the required extra next-block base fee; that overlap must agree
exactly.

For epoch `e`, define:

```text
end_block          = 3,600*(e+1)-1
confirmation_block = end_block+64
available_at       = canonical timestamp of confirmation_block
```

The 64-produced-block delay is the repository's existing conservative
two-epoch Ethereum confirmation convention. It is not claimed to be a proof
that finality always occurred at exactly that block. The official replay must
also verify that the complete historical range is below the common finalized
head at retrieval. Live promotion requires owned-node or independently
monitored forward parity and explicit finality monitoring.

Any transport error after the one-shot source run begins, shortened range,
provider disagreement, boundary mismatch, reorg/hash mismatch, invalid
quantity, non-monotone block range, or missing block terminally rejects this
exact source build. There is no provider fallback after source values open.

## Exact source statistic

For every epoch, parse each `baseFeePerGas` as a canonical nonnegative integer.
Base fee must be positive for every retained post-London block.

Sort the 3,600 values. Let:

```text
median2[e] = sorted_base_fee[1,799] + sorted_base_fee[1,800]
```

`median2` is exactly twice the epoch median and remains integer. No floating
median, rounding, clipping, winsorization, unit conversion, or gas-price
normalization is allowed.

The normalized source row contains only:

```text
epoch_id
start_block
end_block
end_block_hash
end_block_timestamp_utc
confirmation_block
confirmation_block_hash
available_at_utc
median_base_fee_wei_x2
base_fee_vector_sha256
mean_gas_used_ratio_decimal
```

`mean_gas_used_ratio_decimal` is calculated with exact decimal arithmetic and
is forbidden from the primary feature. It exists only for source diagnostics
and the frozen utilization-only control.

## Exact primary feature and side

For epoch `e >= 4,533`, compare the current median with two epochs earlier:

```text
current = median2[e]
lagged  = median2[e-2]
sign[e] = sign(current-lagged)

magnitude_num[e] = max(current,lagged)
magnitude_den[e] = min(current,lagged)
```

The magnitude ratio is `magnitude_num / magnitude_den`, which is exactly
monotone in the absolute log change. Ratios are compared by integer
cross-multiplication. Floating logarithms never decide a rank or tie.

For each current row, rank its magnitude against exactly the previous 180
finite feature rows, excluding the current row:

```text
L    = count(prior_ratio < current_ratio)
E    = count(prior_ratio == current_ratio)
rank = (L + 0.5*E) / 180
```

The raw primary candidate exists if and only if:

```text
rank >= 0.75
sign != 0
```

Direction is fixed:

- positive sign -> `LONG`;
- negative sign -> `SHORT`.

There is no event-onset filter. Every qualifying epoch may be a raw
candidate; chronological non-overlap decides which candidates execute.

There is no grid over epoch size, lag, warm-up, rank threshold, side, latency,
hold, leverage, cost, or control. Zero change abstains. Exact threshold ties
qualify.

## Execution

For source availability timestamp `T`:

```text
entry_time = ceil_to_5m(T) + 5 elapsed minutes
exit_time  = entry_time + 86,400 elapsed seconds
```

If `T` is already on a five-minute boundary, entry is still five minutes
later. The entry is therefore always after one complete BTC five-minute bar.

- fixed exposure: `0.5x` account notional;
- one global position;
- reservation interval: `[entry_time, exit_time)`;
- raw candidates sort by
  `(entry_time, available_at, epoch_id, side)`;
- accept only when `entry_time >= previous accepted exit_time`;
- suppressed candidates are not queued;
- no pyramiding, stop, take profit, trailing exit, or early close;
- a trade crossing a split boundary is skipped, never truncated.

## Frozen calendars

All calendars are half-open and use both entry and exit containment:

```text
full       [2023-06-01, 2026-06-01)
selection  [2023-06-01, 2025-01-01)
future25   [2025-01-01, 2026-01-01)
future26   [2026-01-01, 2026-06-01)
```

Selection must also report:

```text
2023H2 [2023-06-01, 2024-01-01)
2024H1 [2024-01-01, 2024-07-01)
2024H2 [2024-07-01, 2025-01-01)
```

The complete three-year full-calendar CAGR uses exactly the full wall-clock
interval, including idle time.

## Source-only controls

Controls cannot replace or repair the primary:

1. `base_fee_primary`: exact ESDI primary;
2. `base_fee_one_epoch_stale`: use `median2[e-1]` and
   `median2[e-3]`, but signal no earlier than current epoch `e` availability;
3. `gas_utilization_only`: apply the same two-epoch comparison, exact
   prior-180 midrank, threshold, side, scheduler, and hold to the exact epoch
   mean gas-used ratio;
4. `base_fee_no_tail`: all nonzero primary signs, same scheduler and hold;
5. `exact_direction_flip`: exact accepted primary entries with side flipped;
6. `deterministic_random_side`: exact accepted primary entries with a
   SHA-256-fixed side;
7. `constant_long` and `constant_short`: exact accepted primary entries;
8. `one_bar_delayed_entry`: exact primary parent set shifted five minutes,
   without rerunning non-overlap.

The stale, utilization-only, and no-tail controls build their own
chronological non-overlap clocks. Same-parent side controls never change the
primary parent set.

## Outcome-blind source-support gates

The source stage may load only official Ethereum source rows and construct the
frozen source-only controls above. It must load zero prior-candidate
comparator, BTC market, funding, premium, return, PnL, Gross9, CAGR, or MDD
rows.

The primary accepted clock must pass:

```text
exact source epochs                         = 2,474
missing source epochs                       = 0
dual replay differences                     = 0
boundary/header differences                 = 0
future-append differences in selection      = 0

selection total                             >= 45
2023H2 total                                >= 12
2024H1 total                                >= 12
2024H2 total                                >= 12
selection each side                         >= 14
selection maximum month share               <= 0.20

future25 total                              >= 30
future25 each side                          >= 8
future25 maximum month share                <= 0.25

future26 total                              >= 15
future26 each side                          >= 4
future26 maximum month share                <= 0.30
```

No accepted-entry gap may exceed 90 days and no same-side run may exceed 12.
Every accepted identity, entry, exit, side, rank numerator, tie count, and
source hash must be unique and reproducible.

The primary must not be clock-equivalent to any independent control:

```text
exact-entry Jaccard < 0.90
candidate ±24h containment < 0.95
```

These are anti-degeneracy gates, not claims that a source-only control is
unprofitable. Economic superiority is tested only after the exact evaluator
is separately committed.

Any failed source, incidence, dispersion, append, or control gate retires
ESDI-288 without opening outcomes. No threshold, epoch, lag, side, hold, or
support-floor repair is permitted.

## Downstream novelty gates

Only a complete source-support pass may open comparator reconstruction.

Before ESDI economics, compare its exact accepted entries and occupied
exposure against:

- every available prior Ethereum/stablecoin/WBTC source-family primary clock;
- the chain-activity and network weak-signal clocks;
- WCTR-288 and BFWC-288;
- every positive-weight Gross9 sleeve separately.

For every Gross9 sleeve, require:

```text
exact-entry Jaccard                  <= 0.10
candidate ±6h containment            <= 0.35
occupied-bar Jaccard                 <= 0.25
absolute signed-exposure Pearson     <= 0.35
```

For prior source-family clocks, use exact-entry Jaccard `<=0.20`, candidate
`±24h` containment `<=0.50`, and absolute signed-exposure Pearson `<=0.40`.
Comparators with fewer than ten entries report metrics but do not gate.

Failure is terminal. A comparator cannot be removed after its overlap is
seen.

## Strict standalone economics

The strict evaluator must be implemented, tested, committed, and hash-bound
before any BTC market or funding row is opened.

Common accounting:

- leverage `0.5x`;
- base cost `6 bp` of notional per side;
- stress cost `10 bp` of notional per side;
- exact realized funding where
  `entry_time <= funding_time < exit_time`;
- next-open entry and exact scheduled-open exit;
- no stop or take profit;
- full-calendar CAGR;
- global/pre-entry high-water strict MDD;
- favorable held OHLC and funding credits raise the peak before adverse held
  OHLC, funding debits, liquidation envelope, and exit cost lower the trough.

Every opened period must pass under both base and stress:

```text
absolute return                         > 0
full-calendar CAGR / strict MDD          >= 3.0
strict MDD                              <= 0.15
mean gross underlying move              >= 20 bp
calendar-month clustered sign-flip p    <= 0.10
```

The primary risk-adjusted result must strictly exceed the utilization-only
and one-epoch-stale controls. Direction flip, deterministic random side, or
either constant-side control cannot completely qualify.

Open in order:

1. `2023H2`;
2. `2024`;
3. combined selection;
4. Gross9 selection and frozen weight;
5. `future25`;
6. `future26`;
7. exact stitched three-year report.

Stop permanently at the first failure. Later periods can only veto; they
cannot invert, rerank, repair, select rank two, or alter a parameter.

## Same-gross Gross9 contract

Gross9 remains:

```text
cand_rex_veto_7                 1.6
fresh_kimchi_fx                 2.0
frozen_annual_rank7             3.0
markov_transition_long          2.0
rex_taker_low_range_position    0.4
gross                           9.0
```

Candidate weights are frozen at:

```text
[0.25, 0.50, 0.75, 1.00]
```

At candidate weight `w`, the treatment scales every Gross9 sleeve by
`(9-w)/9` and adds ESDI at weight `w`, so configured gross remains exactly
`9.0`. Compare that treatment directly with the unscaled authoritative
Gross9 baseline, also gross `9.0`, under matching execution, costs, exact
funding, and strict MDD.

Both `2023H2` and calendar 2024 must:

- improve base and stress CAGR/strict-MDD by at least `0.05` versus the
  same-gross comparator;
- retain at least `97%` of unscaled Gross9 absolute return;
- have positive base and stress absolute return; and
- reduce strict MDD versus unscaled Gross9 in at least one selection period.

Rank by the minimum base/stress, 2023H2/2024 improvement; tie-break by lower
weight. Freeze exactly rank one. Future periods evaluate only that frozen
weight and must pass the same no-material-deterioration contract. They cannot
rerank or select another weight.

## Write-once sequence

The only authorized sequence is:

1. commit this mechanism decision;
2. commit the write-once preregistration producer and tests;
3. create and commit the write-once preregistration artifact;
4. commit a dual-replay source builder and synthetic-only tests bound to the
   preregistration hash;
5. execute the full source replay once;
6. commit source artifacts and the source-only support evaluator before
   deriving ESDI incidence;
7. run source support and stop on first failure;
8. only after a pass, commit the strict outcome/Gross9 evaluator;
9. open periods sequentially and stop on first failure;
10. reproduce all hashes/tests from a clean checkout, commit, and push.

At preregistration:

```text
full_source_replay_opened       = false
exact_incidence_opened          = false
candidate_overlap_opened        = false
btc_market_rows_opened          = false
funding_rows_opened             = false
gross9_rows_opened              = false
outcomes_opened                 = false
```

The excluded transport probes above are disclosed separately and cannot
authorize a parameter choice.

## Stop condition

ESDI-288 is complete only if the exact source, support, novelty, standalone,
same-gross Gross9, 2025, 2026, stitched three-year, reproducibility, commit,
and push sequence passes unchanged. Any failure retires this exact policy and
is itself the terminal research result. No ordinary failure permits a repair
under the ESDI-288 identity.
