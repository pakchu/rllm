# BCRT-72 source-support implementation contract

Date: 2026-07-24
Status: frozen before BCRT source values, token incidence, or market outcomes

## Purpose

This contract freezes the one permitted source-only implementation for
`BCRT-72`. The implementation may determine whether the hash-bound Bitcoin
block source can produce enough causal twelve-hour relational states. It may
not determine whether any state predicts BTC returns.

The real source-support command is legal only after this document, the builder,
and its synthetic tests are committed and byte-identical to `HEAD`.

## Bound inputs

The builder may open only:

- the canonical BCRT preregistration JSON;
- the BCRT boundary, mechanism, and this implementation contract;
- the frozen UTXO/fee block source CSV and source manifest;
- the frozen basic block-summary reference CSV; and
- its own committed source and tests for hash binding.

It may not open market, funding, premium, OI, liquidation, comparator,
portfolio, label, reward, return, PnL, CAGR, MDD, or post-2023 data.

The source CSV is loaded with the exact preregistered allowlist through
`pandas.read_csv(usecols=allowlist)`. Load-and-drop is forbidden.

## Source validation

The builder fails closed unless:

- the preregistration file SHA and manifest hash match the committed freeze;
- every bound source/document file SHA and CSV header SHA matches;
- rows are the exact contiguous height range `610691..823785`;
- block ids and parent ids are lowercase 64-character hexadecimal strings;
- heights and block ids are unique and every parent link after the first
  matches the previous row id;
- all integer fields are exact integers;
- `tx_count >= 1`, `size > 0`, `weight > 0`, `total_fees >= 0`,
  `total_inputs >= 0`, and `total_outputs >= 0`;
- `size <= weight <= 4*size`;
- `utxo_set_change = total_outputs-total_inputs`;
- timestamp and median time are positive and strictly before 2024;
- median time is nondecreasing by height; and
- all eight basic fields exactly equal the frozen reference CSV.

## Causal bucket formation

For every nominal UTC half-day in `[2020-01-01,2024-01-01)`:

```text
bucket_start = floor(timestamp/43200)*43200
bucket_end   = bucket_start+43200
anchor       = first height with mediantime >= bucket_end
confirmation = anchor+288
members      = rows with height <= confirmation
               and bucket_start <= timestamp < bucket_end
```

A trailing bucket without the exact anchor or confirmation is omitted. Any
later backdated row whose timestamp falls inside a formed bucket is counted as
a late-member diagnostic and excluded permanently.

Anchor and confirmation heights and signal availability must increase
monotonically. Every formed bucket must contain at least one validated member.

For each formed bucket, the builder reselects members from the exact
`height <= confirmation` prefix and from the full frame with the same explicit
height bound. The two byte-level state digests must match. Contiguous heights
then prove that every intermediate later append is also excluded. The report
records the number of buckets replayed and later append rows proved irrelevant.

## Primitive and rank construction

The eight primitives and MAD definition are exactly those in the mechanism.
No clipping, normalization, calendar conditioning, or missing-value fill is
allowed.

Each primitive has its own strictly prior history. The current bucket is
ranked against the last at most 252 source-valid buckets and requires 126.
Current values are appended only after all current ranks are fixed.

The first rank-complete state emits no policy token row but remains the exact
predecessor of the second.

## Token construction

The builder emits only the twelve preregistered relational tokens in canonical
order. Pair boundaries are strict at `+/-1/6`; leader ties have no epsilon;
rank breadth, extreme occupancy, relation breadth, order transition, and
leader transition use the exact preregistered bands.

The stored clock contains no primitive, numeric rank, action, side, market,
funding, return, label, reward, or PnL field.

## Clock, reservation, and split containment

For each token-ready bucket:

```text
raw_available =
    max(bucket_end,prefix_max_timestamp,prefix_max_mediantime)+172800
signal_available = ceil_5m(raw_available)
entry            = signal_available+300
exit             = entry+72*300
```

Reservation is global, half-open, chronological, and action-independent.
Every token-ready interval reserves its slot before split containment.
Abstention cannot release it. A split-crossing state remains reserved but is
not emitted.

An emitted row requires all of the following inside one half-open calendar
split: source bucket start and end, anchor timestamp and median time,
confirmation timestamp and median time, signal availability, latency bar,
entry, held bars, and exit. In particular, scheduled `exit` must be strictly
less than `split_end`; equality would require the next split's execution price.

The source-valid predecessor remains the predecessor even when its token row is
unready, overlap-suppressed, or split-suppressed.

## Development-only support decision

Only `[2020-01-01,2023-01-01)` may enter Boolean support checks.

Development incidence checks:

- at least 2,000 emitted opportunities in 2020-2022;
- at least 1,250 in 2020-2021;
- at least 570 in 2020;
- at least 700 in each of 2021 and 2022;
- at least nine active months in 2020 and all twelve in 2021 and 2022;
- at least 340 per half and 165 per quarter in 2021 and 2022;
- maximum month share at most 13% in 2020 and 10% in 2021 and 2022; and
- maximum entry gap at most three calendar days in 2020-2022.

Token checks run separately on 2020-2021 train and 2022 selection:

- all five pair-token values meet the frozen minimum and maximum shares;
- both leader tokens meet non-tie diversity/share and tie-share limits;
- breadth, occupancy, and transition tokens meet their frozen share limits;
- exact twelve-token signature share is at most 5%;
- every token is valid; and
- every 2022 token value already occurred in train.

Every 2023 incidence, calendar, marginal-token, signature, vocabulary, timing,
nonoverlap, schema, and replay statistic is emitted under
`eval_source_report_only`. It cannot change a Boolean, decision, failure stage,
threshold, token, or implementation. Full-source structural corruption still
fails closed before an artifact can be formed, but no 2023 statistic is copied
into a support-decision Boolean.

Failure retires `BCRT-72` unchanged before any market outcome is opened.
Success authorizes only the separately frozen cheap-baseline/economic
evaluator.

## Deterministic artifacts

Outputs:

- `data/block_clearing_relational_topology_clocks_2020_2023.csv.gz`
- `results/block_clearing_relational_topology_support_2026-07-24.json`

CSV gzip uses an empty filename and `mtime=0`. JSON uses sorted compact keys,
ASCII, finite values only, and a terminal newline. Both outputs are write-once:
an existing byte mismatch fails.

The report binds:

- preregistration, implementation, source, manifest, and reference hashes;
- validation, bucket, rank, token, replay, reservation, and split funnels;
- development Boolean checks and their first failure;
- 2023 report-only statistics;
- clock path, schema, row count, SHA, and deterministic frame hash; and
- explicit zero counters for every forbidden outcome family.

## Mandatory synthetic tests

Before the real source may be decoded, tests must cover:

- preregistration and source/header binding;
- exact integer, chain, UTXO identity, and weight validation;
- anchor and exact 288th-successor lookup;
- later backdated-member exclusion;
- prefix/full-frame replay equivalence;
- all eight primitive equations;
- strict-prior exclusion, ties, cap, minimum, and first-predecessor behavior;
- every token boundary and transition band;
- max-clock plus 48-hour embargo, five-minute ceiling, latency, and six-hour
  hold;
- action-independent half-open reservation;
- split-crossing reservation without emission;
- clock forbidden-column rejection;
- development-only versus 2023 report-only gates;
- deterministic gzip/JSON and write-once drift rejection; and
- refusal to run the real source while protocol files are uncommitted.
