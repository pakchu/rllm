# CLOR-D1 mechanism boundary

Date: 2026-07-25

## Decision

Freeze **CLOR-D1 — Collateral Liquidity Ordering-Relation target policy** as
one source-composition candidate before decoding any CLOR joint state.

CLOR-D1 observes the asynchronous publication sequence of:

1. original-issue nominal Treasury auction bidder allocation;
2. New York Fed SOMA securities-lending award coverage; and
3. OFR preliminary repo venue ordering.

It does not create a source-owned market side or select sparse events. At
every action-independent valid source update, a later single compact RLLM may
choose one target:

```text
TARGET_LONG
TARGET_FLAT
TARGET_SHORT
```

The falsifiable thesis is:

> The same collateral-allocation relation can imply persistence, absorption,
> or irrelevance depending on which Treasury, lending, and repo relation
> changed first and how the following relation states evolved.

The model role is finite relational and temporal deduction. Deterministic code
owns source validation, exact arithmetic, chronology, freshness, safety-flat
decisions, execution, costs, funding, rewards, and strict drawdown.

CLOR-D1 is a **source-incidence-informed successor**. Prior Treasury, SOMA, and
OFR source-support failures influenced the move away from sparse fixed-side
events toward a dense target-position sequence. The candidate is not
independent, pristine, clean-room, or source-value-blind. Only its exact joint
cards, model actions, and outcomes remain unopened.

During preregistration hardening, only the exact gzip header line of each bound
predecessor clock was decoded to freeze its parser. Zero predecessor value or
action rows, zero CLOR joint-state rows, and zero market, funding, reward, or
performance rows were decoded.

## Bound selection authority

```text
docs/post-cefs-d2-alpha-mechanism-audit-2026-07-25.md
commit 006ff4286913c90ef766ce4ba2a563b12b6ec6c0
SHA256 8013f07934a4ef2000e69ba274be06f84142d360e62296b83ca1c2c160930717
```

CEFS-D1/D2 remain terminal. No Cboe source or CEFS formula, token, threshold,
control, split repair, or successor identity may enter CLOR-D1.

## Exact source identities and allowlists

Every CSV loader must decode only its exact allowlist. Loading a wider frame
and dropping columns later is forbidden.

### Treasury auction panel

```text
data/us_treasury_auction_demand_2016_2023/
us_treasury_nominal_original_auctions_2016_2023.csv.gz
SHA256 34a19163630c015a4f9d2671c95ca7cf7cc8a8ada024b3ef985405704fe0e4c1
header SHA256 4f7eab19bebc30f60ded1f6520ee54e2418bc05ef86dde632cad7762e4abf5bf

data/us_treasury_auction_demand_2016_2023/build_manifest.json
SHA256 6da6a3848e89c3418efcbf0d836fda34b537a2da87a8777b74670f3912ad94f2
```

Exact allowlist:

```text
auction_date
result_available_at_utc
original_security_term
competitive_accepted_usd
primary_dealer_accepted_usd
direct_bidder_accepted_usd
indirect_bidder_accepted_usd
source_complete
```

Only complete rows strictly before 2024 are eligible. The three bidder
amounts must be finite nonnegative exact decimals and must sum exactly to
`competitive_accepted_usd`, which must be strictly positive. The term must be
exactly one of:

```text
2-Year
3-Year
5-Year
7-Year
10-Year
20-Year
30-Year
```

The frozen term order is the displayed order. `bid_to_cover_ratio`,
`indirect_competitive_share`, CUSIP, PDF/XML URLs, auction close time, and
updated timestamp are prohibited inputs.

### SOMA securities-lending panels

```text
data/new_york_fed_securities_lending_2019_2023/
new_york_fed_securities_lending_operations_2019_2023.csv.gz
SHA256 99eb8c37c05417789dfad7452c7b2ddc5b6b640078b87451f1c945158af77906
header SHA256 c0d63795e5e53cef816c50472c6941069cb018f30ad1f745f250daa0fa6b9200

data/new_york_fed_securities_lending_2019_2023/
new_york_fed_securities_lending_details_2019_2023.csv.gz
SHA256 27178d8738cb50c4e6c13f1e5940fcfdf4009e6979b006c42fb86fb399d0716d
header SHA256 9f4d54dff4b9c9f0c47c0a85e0bf245276e5a3cb764b3c084017f679586b76dd

data/new_york_fed_securities_lending_2019_2023/build_manifest.json
SHA256 58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019
```

Operation allowlist:

```text
operation_id
operation_date
available_at_utc
total_par_submitted
total_par_accepted
```

Detail allowlist:

```text
operation_id
operation_date
available_at_utc
par_submitted
par_accepted
```

Identifiers must be nonempty and unique. Amounts are finite nonnegative exact
decimals, accepted cannot exceed submitted, and every operation must have at
least one detail. Detail submitted and accepted sums must reconcile exactly
to the operation totals. Security description, CUSIP, fee, holdings,
theoretical/actual availability, loans, settlement/maturity dates, notes, and
raw transport are prohibited.

### OFR preliminary repo panel

```text
data/ofr_repo_preliminary_2019_2023/
ofr_repo_preliminary_observations_2019_2023.csv.gz
SHA256 6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a
header SHA256 d477472ebf9510be24bb4c596c2e795e468a3a3e774f8ba2279ba91fa44e36c4

data/ofr_repo_preliminary_2019_2023/build_manifest.json
SHA256 f937f567e1789ecb39a2b84d6288b2cbab931da4e9f1f4e51addea4b3423b705
```

Exact allowlist:

```text
mnemonic
observation_date
available_at_utc
value
disclosure_edit
```

Only these six preliminary total series are authorized:

```text
REPO-DVP_AR_TOT-P
REPO-GCF_AR_TOT-P
REPO-TRIV1_AR_TOT-P
REPO-DVP_TV_TOT-P
REPO-GCF_TV_TOT-P
REPO-TRIV1_TV_TOT-P
```

Every complete OFR date has exactly one finite, disclosure-edit-zero row for
all six mnemonics. Transaction volumes must be nonnegative with a strictly
positive three-venue sum. Final vintages, `TRI`, collateral subdivisions,
maturity buckets, nulls, interpolation, forward filling, and post-2023 rows
are forbidden.

## Exact source primitives

All source decimals, sums, ratios, comparisons, and ties use exact rational
arithmetic. Binary floating-point source arithmetic is forbidden.

### Treasury bidder-order token

Rows sharing one exact `result_available_at_utc` form one Treasury batch. Sort
them by frozen term order. For each row, compare:

```text
P = primary_dealer_accepted_usd
D = direct_bidder_accepted_usd
I = indirect_bidder_accepted_usd
```

Serialize the exact weak ordering with `>` and `=`. Examples:

```text
P>I>D
I>P=D
P=D=I
```

The Treasury batch token is the ordered list:

```text
2-Year:P>I>D|5-Year:I>P>D
```

No share, change, rank, threshold, scalar score, side, or event is computed.
This is not TADI's bid-to-cover/indirect-share tail or TASCC's settlement
calendar. Because bidder accepted amounts also underlie TADI's indirect-share
field, this token remains explicitly contaminated and is subject to the
mandatory no-Treasury ablation and TADI clock comparison below.

### SOMA award-coverage transition token

Rows sharing one exact `available_at_utc` form one SOMA batch. Sum exact
operation totals after detail reconciliation:

```text
submitted[t] = sum(total_par_submitted)
accepted[t]  = sum(total_par_accepted)
coverage[t]  = accepted[t] / submitted[t]
```

`submitted[t]` must be strictly positive. Compare the current complete batch
with the immediately previous complete, strictly earlier batch:

```text
submitted_step = UP | DOWN | EQUAL
accepted_step  = UP | DOWN | EQUAL
coverage_step  = UP | DOWN | EQUAL
```

The first complete batch and the first complete batch after an invalid batch
establish a baseline but emit no transition token. There is no rolling rank,
tail, vote, divergence, CUSIP distribution, fee component, side, or event.
This is not SLCS or SCAF.

### OFR venue-order token

Rows sharing one exact `available_at_utc` form one causal OFR batch. Every
complete observation date in the batch is validated, but only the greatest
complete observation date becomes the current source state.

For that date, serialize two exact weak orderings in fixed label order
`DVP,GCF,TRIV1`:

```text
rate_order   = weak_order(DVP_AR_TOT, GCF_AR_TOT, TRIV1_AR_TOT)
volume_order = weak_order(DVP_TV_TOT, GCF_TV_TOT, TRIV1_TV_TOT)
```

Examples:

```text
RATE=DVP>GCF=TRIV1
VOLUME=TRIV1>DVP>GCF
```

No dispersion, HHI, collateral mix, rate spread, maturity share, rank,
threshold, product, state side, or handoff is computed. This is not RVFC,
RMSR, RCRE, or DMSH.

## Causal update schedule

Each source batch retains its frozen exact availability timestamp. Convert a
batch to its earliest executable time:

```text
execution_time = ceil_to_5m(available_at_utc) + 5 elapsed minutes
```

An exact five-minute timestamp still receives the extra five-minute latency.
Group all source batches with the same `execution_time`. Apply every grouped
batch in exact `(available_at_utc, source_name)` order, with source order:

```text
TREASURY < SOMA < OFR
```

Then emit at most one action-independent state line for that execution time.
Later archive rows may never move an earlier execution time.

At an execution time, the latest source state is fresh only when:

```text
TREASURY age <= 14 elapsed days
SOMA age     <=  4 elapsed days
OFR age      <=  4 elapsed days
```

Age is measured from the exact source availability timestamp to
`execution_time`. These limits follow publication cadence and are frozen
before joint incidence. Missing, invalid, or stale source state makes the line
invalid.

An invalid line:

- deterministically targets `FLAT`;
- clears the twelve-line model sequence;
- cannot become a model decision; and
- cannot be bridged by a later valid row.

A valid line contains only:

```text
UPDATED=<nonempty TREASURY|SOMA|OFR subset>
TREASURY=<latest bidder-order token>
SOMA=<submitted_step,accepted_step,coverage_step>
OFR=<rate_order,volume_order>
```

No date, timestamp, raw number, source row ID, CUSIP, split, return, price,
funding, or future field enters a line.

## Sequence and action contract

Within each split, start with empty history and `CURRENT_POSITION=FLAT`. Number
consecutive valid lines from one after every split or invalid-line reset. Valid
line `N` is a model decision exactly when `N >= 12`; its prompt uses the
trailing twelve valid lines including line `N`. Therefore the first model
decision is the twelfth consecutive valid line, not the thirteenth. The
sequence length is fixed at twelve because it covers approximately three
independent update cycles per source without introducing a fitted horizon.

Each primary line is canonical UTF-8 ASCII with no leading or trailing
whitespace:

```text
UPDATED=<comma-separated members in TREASURY,SOMA,OFR order>;TREASURY=<token>;SOMA=<submitted_step,accepted_step,coverage_step>;OFR=<rate_order,volume_order>
```

`UPDATED` contains exactly the nonempty set of sources updated at that
execution time. Relation tokens contain no spaces. Prompt bytes use LF
newlines, exactly two LF bytes between the instruction paragraph and
`STATE_01`, and no terminal newline after `TARGET=`.

The exact text envelope is:

```text
You manage one BTC perpetual target from causal collateral-release symbols.
Use only ordering, transition, and update sequence. Do not invent numbers,
dates, rules, or explanations. Return exactly one target token.

STATE_01 <line>
...
STATE_12 <line>
CURRENT_POSITION=<LONG|FLAT|SHORT>
VALID_TARGETS=TARGET_LONG|TARGET_FLAT|TARGET_SHORT
TARGET=
```

The later model output must parse to one exact valid target; any other byte
sequence maps to `TARGET_FLAT`. Hidden reasoning, rationale, free text, raw
numbers, and generated strategy rules are forbidden.

The target transition occurs at the state's already frozen
`execution_time`. Repeating the current target causes no trade. A source
invalidity, sequence reset, split end, process restart without complete
durable state, or stale collector forces flat.

Every valid model decision at time `t` cancels any prior expiry and sets
`target_expiry_time = t + 72 elapsed hours`. At a timestamp shared by a source
execution group and an old expiry, process the source group first. A fresh
valid model decision at that timestamp cancels the old expiry; otherwise the
old expiry then emits a deterministic action-only transition to
`TARGET_FLAT`. An expiry transition:

- does not emit a source line;
- does not enter or clear the twelve-line source history;
- never invokes the model; and
- changes only the durable current target, so the next prompt observes
  `CURRENT_POSITION=FLAT`.

If the target is already flat, expiry is a no-op. Thus a non-flat target can
never persist beyond 72 elapsed hours without a fresh valid decision.

## Research splits

Splits are assigned by `execution_time`:

```text
source warmup only  before 2020-09-10T00:00:00Z
TRAIN               [2020-09-10T00:00:00Z, 2022-01-01T00:00:00Z)
TEST                [2022-01-01T00:00:00Z, 2023-01-01T00:00:00Z)
EVAL                [2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
sealed               2024 onward
```

Sequence and position reset at every split boundary. No source line from an
earlier split enters a later split's twelve-line prompt. TEST may select one
checkpoint under a separately frozen evaluator. EVAL is one-shot and may not
change any source, token, prompt, model, reward, checkpoint rule, or execution
parameter.

## Frozen source-only gates

Before a market, funding, outcome, reward, model, or comparator-action row is
opened, a separately committed source evaluator must pass in this order:

1. exact source, header, manifest, schema, hash, row, chronology, and
   reconciliation validation;
2. causal batching, freshness, split reset, latency, append invariance, and
   no-post-2023 validation;
3. valid model decisions: at least 450 TRAIN, 180 TEST, and 180 EVAL;
4. source-update support on model-decision lines:
   - Treasury at least 40/20/20 in TRAIN/TEST/EVAL;
   - SOMA at least 200/90/90;
   - OFR at least 200/90/90;
5. maximum valid-decision gap at most ten elapsed days in every split;
6. at least 30 decisions in 2020 after the publication floor, at least 50 in
   every 2021 quarter, and at least 40 in every TEST/EVAL quarter;
7. every primitive field has at least two nonzero-support levels in each split
   and no one level exceeds 95% of that field in a split;
8. no complete state-line signature exceeds 25% of a split;
9. at least 150/70/70 unique twelve-line sequence hashes in
   TRAIN/TEST/EVAL;
10. each of the six relation-falsification controls below is deterministic,
    row-preserving where declared, hash-distinct from primary, and changes at
    least 10% of eligible sequence hashes; `future_append` is excluded from
    that changed-hash floor and instead must pass append invariance exactly;
    and
11. every forbidden-access counter remains zero.

Any failed gate retires CLOR-D1 unchanged before outcomes. Counts, freshness,
sequence length, vocabulary, source fields, split, or control limits may not
be repaired from observed incidence.

## Frozen source-language controls

Controls use the same causal rows, freshness, execution grouping, split
resets, and sequence length:

1. `treasury_bidder_label_rotation`: replace every serialized bidder label
   simultaneously by `P -> D`, `D -> I`, and `I -> P`, preserving term order,
   equality groups, batch timing, and every other field;
2. `soma_one_batch_stale`: at each SOMA update after the first emitted
   transition, use the immediately prior emitted valid SOMA transition token;
   the first emitted transition remains unchanged;
3. `ofr_venue_label_rotation`: replace every serialized venue label
   simultaneously by `DVP -> GCF`, `GCF -> TRIV1`, and `TRIV1 -> DVP`,
   preserving equality groups, batch timing, and every other field;
4. `one_merged_update_stale`: the first valid primary line remains unchanged;
   every later valid execution time receives the complete immediately
   preceding valid primary line's four field values while retaining the
   current execution time;
5. `within_year_source_time_reverse`: separately for Treasury, SOMA, and OFR
   and each UTC calendar year, reverse the ordered list of that source's valid
   emitted tokens across its original source-update slots; retain every
   original availability, execution group, update membership, validity
   decision, and non-updating source state;
6. `deterministic_random_relations`: preserve the exact primary schedule,
   validity, term membership, and update membership, but replace each weak
   ordering or `UP|DOWN|EQUAL` field. For a field, form
   `SHA256("CLOR-D1|deterministic_random_relations|<SOURCE>|<execution_time>|<field>")`,
   interpret the first eight digest bytes as one unsigned big-endian integer,
   and take modulo the size of the field's ASCII-lexicographically sorted
   legal vocabulary. A three-label weak-order vocabulary contains every
   serialization produced by contiguous rank vectors over `{0,1,2}` with
   ties joined by `=` and descending groups joined by `>`; Treasury applies
   this independently to every retained term. A step vocabulary is exactly
   `DOWN,EQUAL,UP`; and
7. `future_append`: after building the primary source-token batches, append
   one synthetic valid token batch per source at
   `2024-01-02T00:00:00Z`, `2024-01-02T00:05:00Z`, and
   `2024-01-02T00:10:00Z` in Treasury, SOMA, OFR order. The synthetic tokens
   are respectively `2-Year:P>D>I`, `UP,UP,UP`, and
   `DVP>GCF>TRIV1,TRIV1>GCF>DVP`. Require every pre-2024 primary line,
   validity decision, sequence, sequence hash, and action-only expiry to remain
   byte-identical. This is an append-invariance check only: it is not required
   to be hash-distinct from primary or to change 10% of eligible hashes.

Controls are source-language falsifications only. None may replace a failed
primary or become a trading policy.

## Bound predecessor cohort and non-repackaging gates

The preregistration must hash-bind these exact predecessor clocks before CLOR
joint incidence:

```text
results/treasury_auction_demand_impulse_preregistered_clock_2026-07-17.csv.gz
SHA256 9bb416413a0cfee5a5ebbdb73032e5889735e88098eaa1dc264b6d224fa489f6

data/treasury_auction_settlement_collision_carry_2020_2023/
tascc72_support_clocks_2020_2023.csv.gz
SHA256 0333ba7f523d86a310e76ac51c15e4d273a1f4fb3e98f5e48dad530ac3696de4

results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz
SHA256 b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948

data/soma_collateral_allocation_fracture_clocks_2020_2023.csv.gz
SHA256 64e07005d70442bfa7a110b1e6bea9802ee94be16d95f6e7db9228f4790a28e6

results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz
SHA256 b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e

results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz
SHA256 bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6

results/ofr_repo_collateral_routing_efficiency_clocks_2026-07-23.csv.gz
SHA256 cbe4e5f6fc52b66062abbf931e46ea4aa0d1f3c0157ffd365d0638aa573c2826

results/ofr_dvp_maturity_stock_flow_handoff_clocks_2026-07-23.csv.gz
SHA256 0cfb881b4e3a0123111eeab904eba7bee074767b9c1315f74e7bddf54e3371c3
```

Source support must prove exact hashes and schema-compatible time/side/interval
allowlists without decoding any market outcome or predecessor performance.
The dense action-independent CLOR decision schedule is expected to overlap
public-release clocks, so schedule overlap alone is diagnostic rather than a
pass claim.

Common-window eligibility is additionally bound to:

```text
docs/novelty-comparator-common-window-policy-2026-07-23.md
commit 26c37a88d2286bd6bfe535c00f8d48009ac08dd5
SHA256 928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580
```

The exact predecessor parsers are:

| ID | Header SHA-256 | Required columns | Exact primary filter | Interval columns |
|---|---|---|---|---|
| TADI | `1e798fc8c8cf2dc5c66e40aaa631a2fa20a8f4368b9551290686d92dada55046` | `entry_time,scheduled_exit_time,side,clock_mode` | `clock_mode == "primary"` | `entry_time,scheduled_exit_time,side` |
| TASCC | `c2e97f5b9d7726dc5174ad4bff9e6af962f01e56d8d232236236afd3b82e1f3a` | `candidate,control,entry_time,exit_time,side` | `candidate == "TASCC-72-SOURCE-FAMILY-SEEN"` and `control == "primary"` | `entry_time,exit_time,side` |
| SLCS | `45a24e800b79a30047ffeb5f45c69cf4817262e57b0af1cf5e046332536e5e94` | `control,entry_time,exit_time,side` | `control == "primary"` | `entry_time,exit_time,side` |
| SCAF | `770965eb9e07bbca6f6b3f3c3165fe5c04301ef6573da86dacd161582cfa8c8f` | `control,entry_time,exit_time,side` | `control == "primary"` | `entry_time,exit_time,side` |
| RVFC | `93d6771691150abdb0f571460afa837a2c8e582ec8fafbf2f3203657a9801782` | `control,entry_time,exit_time,side` | `control == "primary"` | `entry_time,exit_time,side` |
| RMSR | `3053df0fdaaf4ab8015d36403f52f927d78e55301fead996ff02ca6cd4bf1660` | `control,entry_time,exit_time,side` | `control == "primary"` | `entry_time,exit_time,side` |
| RCRE | `ae1d29dc71aaf5149a77432f22626736026c10eda680cef47472d7b5a1348638` | `control,entry_time,exit_time,side` | `control == "primary"` | `entry_time,exit_time,side` |
| DMSH | `d8d2cb1cf0ba29c686b7abe9415a9fc9785fe81f24ef0af6516038665d7ec3bb` | `control,entry_time,exit_time,side` | `control == "primary"` | `entry_time,exit_time,side` |

Each parser first verifies the exact full committed gzip header and artifact
hash, then decodes only its required columns. Side must be exactly `LONG` or
`SHORT`; timestamps must be whole-second canonical UTC. Every raw primary
interval must have `exit > entry`, unique entry time, and no chronological
overlap. For the common window, retain only intervals fully contained in
`[W0,W1)`; exclude boundary-crossing rows whole and never clip them. Empty
in-window primary groups, duplicate entries, overlap, invalid side/time, or
zero-variance exposure fail closed.

Before any 2023 EVAL market or funding row opens, the separately frozen
economic/RLLM evaluator must compare the TEST-selected CLOR policy against
every predecessor only over the exact common TRAIN+TEST interval
`[2020-09-10T00:00:00Z, 2023-01-01T00:00:00Z)`:

- exact signless target-transition entry Jaccard at most `0.35`;
- deterministic signless one-to-one entry matching within `+/-6h`, with both
  CLOR-to-predecessor and predecessor-to-CLOR matched fractions at most `0.50`;
- absolute signed occupied-exposure Pearson correlation at most `0.40`; and
- no parser ambiguity, missing required comparator, zero-variance exposure, or
  interval clipping.

For CLOR, a non-flat occupied interval begins only when the durable target
changes to `LONG` or `SHORT`, including a direct reversal, and ends at the next
target change, safety flat, 72-hour expiry, or split end; intervals are
entry-inclusive and exit-exclusive. Repeated targets create no entry. The
signless exact-entry key is the canonical entry timestamp only. Exact Jaccard
is set intersection size
divided by set union size; either empty set fails.

For tolerant matching, give every row the identity
`SHA256("CLOR-D1|comparison|<artifact_sha256>|<group>|<entry>|<exit>|<side>")`.
Sort left rows by `(entry_time,row_identity)`. For each left row, choose among
unmatched right rows within six elapsed hours the row minimizing
`(absolute_time_difference,right_entry_time,right_row_identity)`, match it,
and continue. Run independently in both directions. The gated fractions are
matched rows divided by the corresponding left-row count. Side is ignored for
the gated entry metrics; same-side metrics are report-only and cannot relax a
gate.

Exposure uses every five-minute UTC interval start `t` in the common
TRAIN+TEST window. CLOR is `+1`, `0`, or `-1` from the durable target effective
at `t`; a predecessor is its exact side for `entry_time <= t < exit_time` and
zero otherwise. Compute Pearson correlation over the complete grid including
joint-zero cells and gate its absolute value. No resampling, forward extension,
partial interval, or grid clipping is allowed.

It must also train and select exact no-Treasury, no-SOMA, and no-OFR ablations
under identical seeds, budgets, reward, and checkpoint rules. Ablations retain
the primary execution schedule, full-source validity decisions, split resets,
model-decision timestamps, 72-hour expiry rule, and twelve-line windows. They
remove no source batches and recompute no freshness. At serialization only:

- the omitted source field becomes the exact token `MASKED`;
- the omitted source name is removed from `UPDATED`;
- if it was the only updated source, serialize `UPDATED=NONE`; and
- `CURRENT_POSITION` is the durable position of that separately trained
  ablation.

No ablation identifier enters the prompt. Target-change fraction compares
full and ablation target tokens at their identical model-decision timestamps:
the numerator is unequal tokens and the denominator is all such timestamps.
On TEST, the full three-source policy must:

1. change at least 10% of target decisions relative to each ablation;
2. have positive absolute return and positive CAGR/strict-MDD; and
3. exceed each ablation's CAGR/strict-MDD by at least `0.25`.

These gates are frozen before CLOR incidence and outcomes. A failed predecessor
comparison or source ablation retires CLOR-D1. No component, predecessor, or
ablation may replace the primary.

## Economic/RLLM boundary

Only a complete source-support pass may authorize a new committed evaluator.
That evaluator must freeze before opening market values:

- one exact compact Gemma-family revision and tokenizer;
- one-model architecture only; analyzer/trader dual models are forbidden;
- supervised warm start and constrained offline-RL method;
- TRAIN-only updates, TEST-only checkpoint selection, and one-shot EVAL;
- deterministic seeds, optimizer, quantization/LoRA configuration, maximum
  steps, and checkpoint cleanup;
- current-position handling and malformed-output safety flat;
- exact BTCUSDT bars, funding, costs, turnover, target transitions, full-clock
  absolute return/CAGR, and strict held-path MDD;
- cheap deterministic, constant-position, delayed, permuted, random, and
  no-source baselines; and
- first-failure retirement.

An LLM pass must improve on cheap causal baselines; memorizing source schedule
or calendar is not sufficient. No model may see raw numbers, dates, split
labels, market prices, future returns, rewards, checkpoint scores, or
evaluated outcomes in its prompt.

## Live boundary

Historical source support cannot establish live parity. Real-money admission
requires a separately committed `CLOR-LIVE-D1` protocol with:

- forward raw-response capture for all three official sources;
- retrieval timestamps and immutable revision alarms;
- exact historical/live schema and transform replay;
- stale, missing, late, duplicate, disagreement, and restart fail-flat logic;
- durable source cursor, twelve-line sequence, and current-position state; and
- shadow evidence before orders.

Until that protocol passes, CLOR-D1 is research-only and live target is
unconditionally flat.

## Evidence boundary

This boundary used only committed source audits, headers, manifests, hashes,
and prior terminal aggregate reports. It decoded no Treasury bidder amount,
SOMA operation/detail amount, or OFR rate/volume value. It built no CLOR
primitive, joint line, sequence, action, or incidence and opened no new BTC
bar, funding row, future return, reward, model output, trade, PnL, absolute
return, CAGR, or strict MDD.
