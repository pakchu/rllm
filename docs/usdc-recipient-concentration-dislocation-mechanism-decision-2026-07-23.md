# URCD-72 USDC recipient-concentration dislocation — mechanism decision

## Decision and sealed boundary

Freeze one new outcome-blind candidate, **URCD-72**.  URCD asks whether a
causal transition into unusually diffuse or unusually concentrated USDC mint
routing marks a BTC liquidity regime for the following 72 elapsed hours.

The mechanism is frozen after the candidate-axis boundary and before parsing
the promoted source CSV, calculating a real recipient concentration, forming
an entry clock, opening comparator timestamps, or reading BTC market outcomes.
Known source-level aggregate counts and the prior role-topology audit are
disclosed in the boundary document.  No URCD window value, incidence, side,
calendar statistic, comparator overlap, price, funding, future return, PnL,
absolute return, CAGR, or strict MDD has been opened.

Any semantic, source-integrity, support, concentration, control-selectivity,
calendar, or novelty failure retires URCD-72.  Its window, statistic,
quantiles, materiality rule, direction, transition rule, latency, hold, and
support floors may not be repaired after source incidence is observed.

## Economic hypothesis and interpretation limit

Circle's USDC `Mint` event records an authorized minter, a recipient address,
and an amount.  URCD aggregates mint amount by `indexed_address_2`, which is
only an on-chain operational endpoint.  It is not labelled as a customer,
exchange, beneficial owner, fiat depositor, or BTC buyer.

The falsifiable routing hypothesis is:

- unusually **diffuse** amount routing across operational endpoints is the
  candidate expansion geometry and maps to `LONG`;
- unusually **concentrated** amount routing is the candidate contraction
  geometry and maps to `SHORT`.

The direction is intentionally independent of USDC burn, USDT, WBTC, BTC
price, funding, order flow, and minter identity.  A result would support only
the routing-state hypothesis, not an ownership interpretation.

## Bound source and causal identity

Use exactly the promoted source artifact:

- path:
  `data/ethereum_stablecoin_issuance_redemption_2020_2023/ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz`;
- source SHA-256:
  `70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901`;
- source manifest:
  `results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json`;
- source-manifest file SHA-256:
  `8ec9ab08c413bf6f5f8170fb800b05105522d4cf1a7932943c214288701e31fe`;
- embedded manifest hash:
  `a0c7740db64f7779fade68d76985c629cabe81983bf594e8258cef16a5725a1b`.

Eligible rows satisfy exactly:

```text
asset == "usdc_eth"
event == "mint"
event_sign == 1
decimals == 6
available_at < 2024-01-01T00:00:00Z
```

For every eligible row:

```text
amount_raw = positive base-10 integer
recipient  = lowercase indexed_address_2
identity   = (block_hash, transaction_hash, log_index)
clock      = available_at
```

Empty, malformed, or zero recipient addresses; nonpositive amounts; duplicate
identities; schema drift; an event timestamp after its `available_at`; or a
source/manifest mismatch fails closed.  `block_timestamp` is retained only for
audit.  Full-period recipient membership is forbidden as a feature.  Each row
becomes usable only at the canonical timestamp of block `N+64` already stored
in `available_at`.

The loader must parse and validate `available_at` before decoding
`amount_raw`, recipient, event, or any other value field.  Rows with
`available_at >= 2024-01-01T00:00:00Z` may contribute only to a timestamp-only
sealed-row count and are otherwise skipped.  Their non-timestamp fields remain
undecoded.

## Frozen decision grid and current routing window

Decision anchors `D` are the exact UTC six-hour grid:

```text
00:00, 06:00, 12:00, 18:00
```

At anchor `D`, the current routing window is the half-open interval:

```text
(D - 24 elapsed hours, D]
```

Only eligible rows with `available_at` in that interval are used.  Aggregate
all current mint amounts by recipient before computing any routing statistic.
The current window is valid only when it contains:

- at least 4 mint events;
- at least 3 distinct nonzero recipients; and
- positive total mint amount.

No burn, USDT, WBTC, market, return, or post-2023 value may enter the current
state.

## Exact amount-weighted concentration

For valid current window `W`, let `a_r` be the sum of integer `amount_raw` for
recipient `r` and `A = sum_r a_r`.  Define the exact Herfindahl concentration
as the rational number:

```text
HHI(W) = sum_r (a_r * a_r) / (A * A)
```

HHI ordering must use integer cross multiplication.  Binary floating point,
decimal rounding, logarithms, full-period normalization, recipient labels,
and minter identities are forbidden in the primary statistic.  Repeated mint
events to the same recipient are consolidated before squaring, so event
fragmentation cannot create artificial breadth.

## Strictly prior seasonal reference panel

For current anchor `D`, form exactly 180 reference endpoints:

```text
D - 1 calendar day, D - 2 calendar days, ..., D - 180 calendar days
```

Each reference endpoint uses its own preceding 24 elapsed-hour window and the
same eligibility and aggregation rules.  The reference windows are daily,
same-UTC-hour, non-overlapping samples; no reference row can overlap the
current window.  Invalid reference windows are retained in an audit count but
excluded from the HHI and amount distributions.  At least 120 valid reference
windows are required.  No search beyond the fixed 180 endpoints is allowed.

Sort valid prior HHI rationals by exact cross multiplication, breaking an
exact rational tie by earlier reference endpoint.  For `n` valid values, use
nearest-rank order statistics under zero-based indexing:

```text
q20_hhi = element ceil(0.20 * n) - 1
q80_hhi = element ceil(0.80 * n) - 1
q50_amt = element ceil(0.50 * n) - 1 of prior total mint amounts
```

The amount sample is sorted **independently** in ascending integer amount,
breaking equal-amount ties by earlier reference endpoint.  `q50_amt` is never
taken from the HHI ordering.

The current window and all same-or-later endpoints are excluded.  The current
window is materially active only when its total mint amount is at least
`q50_amt`.

## Frozen state, transition, and side

At every decision anchor, calculate a raw state:

```text
DIFFUSE      if current is valid and material and HHI <= q20_hhi
CONCENTRATED if current is valid and material and HHI >= q80_hhi
NEUTRAL      otherwise
```

If the two HHI thresholds are equal, the anchor is `NEUTRAL`; it cannot
qualify for both tails.  A candidate is emitted only on entry into a tail:

```text
current DIFFUSE      and immediately prior six-hour anchor was not DIFFUSE
    -> LONG
current CONCENTRATED and immediately prior six-hour anchor was not CONCENTRATED
    -> SHORT
```

The immediately prior anchor uses its independently calculated raw state.
An invalid or unavailable prior anchor is `NEUTRAL`; state is never carried
across a missing anchor.  The tail comparison, transition, and direction are
fixed before source incidence.

## Frozen execution and reservation

For a candidate at decision anchor `D`:

```text
entry_time     = D + 10 elapsed minutes
scheduled_exit = entry_time + 72 elapsed hours
```

The ten-minute delay guarantees two complete BTC five-minute bars after the
decision anchor.  Entry and exit, if later authorized, use the exact BTCUSDT
USD-M perpetual five-minute bar opens at those timestamps.

For every raw candidate, construct `signal_id` as lowercase hexadecimal
SHA-256 of canonical UTF-8 JSON.  The object is:

```json
{
  "candidate": "URCD-72",
  "control": "primary",
  "decision_time": "YYYY-MM-DDTHH:MM:SSZ",
  "row_identities": [["0x<block_hash>", "0x<transaction_hash>", 0]],
  "side": "LONG"
}
```

`row_identities` contains every eligible mint row in the current 24-hour
window, sorted lexicographically by `(block_hash, transaction_hash,
integer(log_index))`.  JSON serialization is Python-compatible
`json.dumps(payload, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)` with no trailing newline.  `side` is the literal `LONG` or
`SHORT`; `control` is the literal clock name.  A duplicate `signal_id` within
a control fails closed.

The frozen source-support splits are:

```text
train     = [2021-01-01T00:00:00Z, 2023-01-01T00:00:00Z)
selection = [2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
```

Reservation is run **independently per split and per control**.  First retain
only candidates satisfying `split_start <= entry_time` and
`scheduled_exit <= split_end`; equality at either retained boundary is
allowed.  A crossing candidate is discarded before reservation and cannot
advance `prior_exit`.  Then process retained candidates by
`(entry_time, signal_id)` and accept only when `entry_time >= prior_exit`.
After acceptance set `prior_exit = scheduled_exit`.  There is no reservation
state before `split_start`, across train/selection, or across controls.

One position is allowed within each independently evaluated clock.  There is
no stop, take-profit, trailing exit, pyramiding, side override, or threshold
search.

## Frozen source-only controls

The source-support evaluator must construct these controls without opening
market outcomes.  None may replace the primary after incidence is observed.

1. `event_count_hhi`: replace recipient amount shares with recipient event
   counts; retain all windows, thresholds, transition, and scheduling rules.
2. `equal_recipient_breadth`: replace HHI ordering with distinct-recipient
   count, mapping high prior-tail breadth to `LONG` and low prior-tail breadth
   to `SHORT`; retain materiality and scheduling.
3. `no_materiality`: remove only the `current amount >= q50_amt` condition.
4. `stale_24h`: evaluate the exact primary state at `D - 24h`, but emit and
   schedule it at `D`.
5. `recipient_year_permutation`: within each `available_at` UTC year,
   deterministically permute recipient labels across eligible rows while
   retaining each destination row's timestamp, amount, and identity.
6. `amount_year_permutation`: within each `available_at` UTC year,
   deterministically permute integer amounts across eligible rows while
   retaining each destination row's recipient, timestamp, and identity.
7. `direction_flip`: exact primary accepted entries with both sides reversed.

For `event_count_hhi`, aggregate integer event counts `c_r` and compare
`sum(c_r*c_r) / total_events**2` with its own strictly-prior q20/q80 panel.
For `equal_recipient_breadth`, sort prior distinct-recipient counts ascending
with earlier-endpoint tie breaks; current count `>= q80` is `LONG`, current
count `<= q20` is `SHORT`, and equal q20/q80 is neutral.  Both controls use the
primary amount materiality gate, tail-entry transition, split containment, and
reservation.  `no_materiality` still computes and audits q50 but ignores only
its comparison.  `stale_24h` takes every raw primary candidate, adds exactly
24 hours to decision/entry/exit, recomputes its `signal_id` with control name
`stale_24h`, and then applies the same split filter and reservation; it does
not recompute a later source state.

The two permutation algorithms are exact.  For each control and UTC year `Y`:

1. the population is every primary-eligible source row whose `available_at`
   year is `Y`;
2. row identity text is lowercase
   `block_hash:transaction_hash:integer(log_index)`;
3. order the source rows by
   `(sha256("URCD-72|<control>|source|<Y>|<identity>"), identity)`;
4. independently order destination rows by
   `(sha256("URCD-72|<control>|destination|<Y>|<identity>"), identity)`;
5. for each zero-based position `k`, copy only the selected field from source
   row `k` to destination row `k`; and
6. preserve all other destination fields exactly.

SHA-256 inputs are literal lowercase ASCII with the shown pipe separators and
no trailing newline.  Hash output is lowercase hexadecimal.  A duplicate row
identity or unequal source/destination population fails closed.  The controls
are offline falsification devices and are not claimed to be live-causal
strategies; only the primary clock can be promoted.

`event_count_hhi`, `equal_recipient_breadth`, `no_materiality`, and
`stale_24h` are audit-only source controls.  Their incidence cannot pass or
fail URCD unless an integrity invariant fails.  The two permutation controls
have the explicit routing-selectivity gates below.  `direction_flip` is an
exact-clock later economic control and has no source-incidence gate.

## Frozen source-support gates

All gates are conjunctive and apply to primary accepted entries in the stated
split after that split's independent reservation:

1. **Integrity and causality**
   - exact source and manifest hashes;
   - independent replay and `N+64` confirmation assertions reproduce;
   - every decision uses only `available_at <= D`;
   - exactly 180 prior endpoints and at least 120 valid reference windows;
   - no duplicate raw or accepted `signal_id`;
   - zero comparator, BTC market, funding, future-return, PnL, or post-2023
     source-value reads before primary support passes.
2. **Train support, 2021–2022**
   - at least 80 accepted entries total;
   - at least 30 entries in each calendar year;
   - at least 12 entries in each half-year;
   - at least 16 entries on each side and each side at least 20%.
3. **Selection support, 2023**
   - at least 30 accepted entries total;
   - at least 10 entries in each half-year;
   - at least 6 entries on each side and each side at least 20%.
4. **Dispersion in each split**
   - no UTC entry month above 20%;
   - no UTC entry quarter above 40%;
   - maximum gap between accepted entries at most 60 calendar days; and
   - no more than 12 consecutive accepted entries on one side.
5. **Routing selectivity**
   - primary exact-entry Jaccard versus `recipient_year_permutation` at most
     0.35 in both train and selection;
   - primary exact-entry Jaccard versus `amount_year_permutation` at most 0.35
     in both train and selection; and
   - for each permutation, at most 60% of primary entries may be reproduced at
     the exact timestamp **and** with the same side in either split.

Any support or selectivity failure short-circuits comparator access and retires
URCD-72 without repair.

## Frozen novelty contract

Only after every primary source-support gate passes may the evaluator open the
following checksum-bound source-only comparator views.  Hashing the whole file
and reading its header before this point are allowed; decoding any data row is
not.

| Views | Exact path | File SHA-256 | Header-line SHA-256 | Overlap `[start,end)` |
|---|---|---|---|---|
| AMTR-48 `primary`, `cross_minter` | `data/authorized_minter_turnaround_relay_clocks_2020_2023.csv.gz` | `30875029daa4d6e2eff9a59f53d45eda57dbced05988df089c38a6c81abfa0f6` | `423287fbc7a50bd00c0ca1de8580c983df1a2d128c1cc497d68e1bc74c224ac8` | `2021-01-01T00:00:00Z`, `2024-01-01T00:00:00Z` |
| UGCI-288 `primary` | `data/usdc_gross_clearing_imbalance_clocks_2021_2023.csv.gz` | `a0f861c69ac171e1efa665dc90a916d0351413ca07e5e46783bb8abd662175fd` | `b79639e44ce1b4488fdf6991e60831221cbc9a48565fa42d053faeb71156ad91` | `2021-01-01T00:00:00Z`, `2024-01-01T00:00:00Z` |
| WCDR-2016 `primary` | `data/wrapped_collateral_dollar_liquidity_rotation_2021_2023/wcdr2016_support_clocks_2021_2023.csv.gz` | `241d96a64a654ba2faeda2d4a8460131269acf21d0bbbf31177d35d1ecd63b3c` | `e67cd52d0cadded15fd49f4ed809707e5d1601260416a93949f452dd7638680e` | `2021-01-01T00:00:00Z`, `2024-01-01T00:00:00Z` |
| WTSL-168 `primary` | `data/wbtc_turnover_stablecoin_liquidity_2021_2023/wtsl168_support_clocks_2021_2023.csv.gz` | `df8cb085d439c9ee9e89334cb891b9e3b04f54c2a8e70bd4f552a90648ea8b6d` | `f206f15f5410c3bb568df4f64c0cffafcf077b5ef08dc8c427ac3af33d873937` | `2021-01-01T00:00:00Z`, `2024-01-01T00:00:00Z` |
| WSCF-72 `primary` | `data/wbtc_stablecoin_finalized_confirmation_relay_2021_2023/wscf72_support_clocks_2021_2023.csv.gz` | `86565774ae97a1024c5a66b4d59a1f5413bf4608398623359dd3ee24572f0ef3` | `adb55cd822efbdcd8469018a51c2b037514758633599a403fae1a1868ef2e9f3` | `2021-01-01T00:00:00Z`, `2024-01-01T00:00:00Z` |
| FCCM-72 `primary` | `data/funding_currency_custody_mobility_consensus_2021_2023/fccm72_support_clocks_2021_2023.csv.gz` | `71180862d9dcc4d76e055c52fd72a2424ee12387a6b8062af8a9382675af3810` | `ffec7a169e71d896d348e875e4753c880050c8011b52eb058eee6932a5d4a6d5` | `2021-01-01T00:00:00Z`, `2024-01-01T00:00:00Z` |
| SQFD-6 `primary`, `no_usdt_lag`, `no_participation` | `data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz` | `a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b` | `2e6d34c734ddc66d15c7718cc0aed3f2c8903fc02370bd9a2446054ff96a2071` | `2023-09-01T00:00:00Z`, `2024-01-01T00:00:00Z` |
| SDDR-12 `primary` | `data/stablecoin_denominator_dislocation_clocks_2023.csv.gz` | `eaf2d6c187af9855e76474d2951fcdc12267174980a72649b73d068982ca8c69` | `91e4b4187dccbba5c9a6407316c4205d17422b1900b319a7ef800a541e1f3550` | `2023-09-01T00:00:00Z`, `2024-01-01T00:00:00Z` |
| UCBR-12 `primary` | `data/usdt_collateral_breadth_relay_clocks_2023.csv.gz` | `20b3ee9f82696222a3adbde0045dfde53e0e240e85162e463166aa8fe90b1a8f` | `a66cd7a33793d7d0b1056171526dd67c9de5cb95b8847435a8ad1c220757ef10` | `2023-09-01T00:00:00Z`, `2024-01-01T00:00:00Z` |

Each header-line hash is over the exact decompressed first CSV line including
its terminal LF byte.  Every comparator header must contain exactly one each
of `candidate`, `control`, `entry_time`, and `side`, and its line hash must
match the table.  The novelty loader may
decode only those four columns.  It must reject forbidden return, price,
funding, PnL, CAGR, or MDD tokens in any header, duplicate `(candidate,
control, entry_time)`, unknown controls, non-UTC timestamps, a side outside
the exact lexical set `{1, -1, LONG, SHORT}`, or a checksum mismatch.

The exact accepted `candidate` literals are `AMTR-48`, `UGCI-288`,
`WCDR-2016`, `WTSL-168-SOURCE-SEEN`, `WSCF-72-SOURCE-FAMILY-SEEN`,
`FCCM-72`, `SQFD-6`, `SDDR-12`, and `UCBR-12`.  Rows for any other candidate or
unlisted control are not decoded beyond the four allowed fields and cannot
enter a view.

For each fixed view and overlap interval, form timestamp sets only; side is an
integrity field and is not part of Jaccard.  Define:

```text
exact_jaccard = |U intersect V| / |U union V|
near_U = count(u in U having any v in V with abs(u-v) <= 6h) / |U|
near_V = count(v in V having any u in U with abs(u-v) <= 6h) / |V|
maximum_bidirectional_containment = max(near_U, near_V)
```

Each exact-entry Jaccard must be at most 0.10 and each maximum bidirectional
containment at most 0.40.  At least ten URCD and five comparator timestamps
must exist on the fixed overlap or the comparison fails closed.  Comparator
values may not alter URCD thresholds, directions, windows, or support floors.

## Later economic sequence, not yet authorized

Only a committed source-support and novelty pass may authorize a separate,
hash-frozen strict outcome evaluator.  The current implementation scope ends
at the 2021–2023 source-support/novelty decision.  It must open economic stages
one way:

1. train `[2021-01-01, 2023-01-01)`;
2. selection `[2023-01-01, 2024-01-01)` only after every train gate passes;
3. immutable test 2024 only after train and selection pass;
4. eval 2025 only after test passes; and
5. recent 2026 only after eval passes.

The evaluator must include exact next-open execution, realized funding,
full-calendar absolute return and CAGR, strict intratrade position-path MDD,
6 bp base and 10 bp stress cost per notional side, side counts, calendar-month
clustered significance, and deterministic controls.  A result cannot be called
an alpha unless untouched OOS stages satisfy their frozen gates; broad prior
human exposure to these market years remains an explicit limitation.

The bound source contains no post-2023 event values.  Before any 2024, 2025,
or 2026 URCD feature is formed, a separate source-extension work unit must be
committed **without opening that stage's BTC outcomes**.  It must reuse the
same Ethereum chain, contract proxy address, event topic, ABI decoding,
`N+64` availability, exact UTC boundary, duplicate/reorg failure rules, two
independent archive-log replays, and independent header materialization as the
promoted 2020–2023 source.  Extensions must be contiguous from
`2024-01-01T00:00:00Z`, deterministic-gzip/hash bound, and staged so the 2024
source freezes before 2024 outcomes, then 2025 source before 2025 outcomes,
and then the 2026 source before 2026 outcomes.  Any source replay or continuity
failure stops the sequence without reading the corresponding market stage.

## RLLM boundary

Gemma/RLLM may not create, retime, reverse, or repair URCD entries.  Only after
the deterministic primary demonstrates gross edge above costs may a train-only
compact model consume bucketed routing-state tokens plus current-position and
time-to-exit state.  Its allowed action is `TRADE_FIXED_SIDE` or `ABSTAIN`, or
a separately preregistered bounded risk size.  It must beat the frozen
deterministic policy on untouched windows after model-selection costs.

## Bound references

- `docs/usdc-recipient-concentration-dislocation-boundary-2026-07-23.md`
- `docs/ethereum-stablecoin-issuance-redemption-source-audit-2026-07-21.md`
- `results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json`
- `training/build_ethereum_stablecoin_issuance_redemption.py`
- `docs/authorized-minter-turnaround-relay-mechanism-freeze-2026-07-21.md`
- `docs/usdc-gross-clearing-imbalance-preregistration-2026-07-22.md`
- `docs/usdc-role-topology-audit-2026-07-21.md`
- `docs/wbtc-stablecoin-finalized-confirmation-relay-support-rejection-2026-07-23.md`
- `docs/wbtc-turnover-stablecoin-liquidity-support-rejection-2026-07-23.md`
