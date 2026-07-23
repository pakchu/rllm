# DCLB-864 source-support implementation contract

## Scope and sequencing

This contract binds the outcome-blind implementation of
**DCLB-864 — Dollar-Collateral Liquidity/Bank Relay**.

The effective preregistration is immutable v2, which binds the immutable v1
base plus the pre-incidence control-only macro-balance amendment. The evaluator
binds both executable preregistration builders as well as the v2 artifact; a
dirty or hash-drifted imported builder blocks before source loading.

The evaluator may decode only the preregistered H.4.1, ON RRP, and H.8 source
allowlists. It may construct their strictly causal source states, deterministic
primary/control clocks, source-support and composition diagnostics, and the
frozen predecessor-clock novelty battery. It may not load BTC OHLC, funding,
returns, labels, rewards, PnL, portfolio results, or any 2024-or-later source
extension.

The evaluator source, tests, and this contract must be committed and unchanged
from `HEAD` before a real source value row is decoded.

Execution order is fail-closed:

1. prove that evaluator source, tests, and this contract are committed and
   clean;
2. validate the preregistration and every bound document/file/header hash;
3. decode only exact source allowlists and build source-only clocks;
4. evaluate source support and relational composition;
5. stop with zero comparator-row reads on any source/support/composition
   failure;
6. only a complete pass may decode exact comparator `usecols` and evaluate
   novelty; and
7. any failure retires DCLB-864 before BTC outcomes.

## Exact source loading

Every source loader calls:

```text
pandas.read_csv(
    usecols=exact_allowlist,
    dtype="string",
    keep_default_na=False,
    na_filter=False,
)
```

Loading a wider frame and dropping columns is forbidden.

### H.4.1

- `release_date` and `observation_date` are exact ISO dates.
- `available_at_utc` is timezone-aware, unique, and strictly increasing.
- release and observation dates are unique and strictly increasing.
- `observation_date < release_date`, and the New York availability date equals
  `release_date`.
- net liquidity is finite and strictly positive.
- every timestamp and date is before 2024.

The first level has no delta. Every later finite log delta is processed against
exactly the prior 104 deltas and is appended afterward even during warm-up.
Delta 105 is the first rankable delta. The integer numerator and centered
numerator are exactly the committed formulas.

For an H.8 anchor, choose the latest H.4.1 feature with availability at or
before the H.8 decision. It is fresh only when its availability is strictly
after the previous archived H.8 decision. No carry or reuse is allowed.

### ON RRP

- operation dates are unique and strictly increasing;
- availability timestamps are timezone-aware, unique, and strictly increasing;
- `source_complete` is exactly `true` or `false`;
- a complete row has an empty quarantine reason and a finite nonnegative
  amount, including zero;
- an incomplete row has a nonempty quarantine reason and a blank amount; and
- every timestamp and date is before 2024.

Each H.8 interval contains every archived ON RRP row satisfying:

```text
previous_h8_decision < result_available_at_utc <= current_h8_decision
```

An interval is complete only with 3–7 rows, all complete, no quarantine text,
and valid amounts. Incomplete intervals emit no level or delta, clear the rank
history, and break adjacency. A complete interval immediately after a reset
has a level but no delta. Every later adjacent complete interval has one delta.
Each finite delta is processed against the prior 13 post-reset deltas and
appended afterward. Delta 14 in a segment is the first rankable delta.

### H.8 SA and NSA

- release dates and release timestamps are unique and strictly increasing;
- `release_weekday` must equal the New York calendar weekday of the release;
- the New York release date must equal `release_date`;
- every retained SA/NSA level is finite and strictly positive; and
- every timestamp and date is before 2024.

For both SA and NSA independently, compute the three committed log-change
components. A robust z-score uses exactly the prior 104 component observations.
The current observation is appended after processing, including warm-up and
rows whose composite state is invalid. Zero MAD or a non-finite z-score makes
that adjustment's current state invalid.

SA owns primary validity. NSA is used only by the exact `nsa_h8` control.
Excluded H.8 releases remain in both robust histories and delimit ON RRP
intervals but cannot originate a normal primary/control state.

## State composition and symbolic tokens

At each archived H.8 release, the decision is 17:00 New York and entry is
17:05 New York. The decision must be strictly later than the archived release
timestamp. Exit is `entry_utc + 4,320` elapsed minutes, and exposure is
`[entry_utc, exit_utc)`.

The primary requires fresh/rank-complete H.4.1, rank-complete ON RRP, valid SA
H.8, a non-excluded release, and nonzero exact macro integer. Direction is the
sign of the exact macro integer.

The emitted clock contains only:

```text
control
signal_id
signal_available_time
decision_time
entry_time
exit_time
side
h41_direction
h41_transition
rrp_direction
rrp_transition
macro_relation
macro_strength
h8_relief
h8_agreement
bank_relation
h41_age_bucket
rrp_count_bucket
prior_side_transition
```

Raw levels, deltas, z-scores, ranks, rank numerators, source dates, release
identities, prices, returns, labels, funding, rewards, PnL, CAGR, and MDD are
not serialized.

Direction tokens are `RELIEF`, `STRESS`, and `NEUTRAL`. H.4.1/RRP transitions
compare the immediately preceding emitted rank on their own causal source
history. Their vocabulary is `NO_PRIOR`, `PERSIST`, `FLIP`, `TO_NEUTRAL`,
`FROM_NEUTRAL`, and `NEUTRAL_PERSIST`.

`prior_side_transition` compares consecutive raw primary-eligible states before
global reservation. It uses `NO_PRIOR`, `PERSIST`, or `FLIP`.

H.4.1 age uses New York calendar-date distance:

- 0: `SAME_DAY`;
- 1: `ONE_DAY`;
- 2–3: `TWO_TO_THREE_DAYS`; and
- 4 or more: `FOUR_PLUS_DAYS`.

ON RRP count buckets are `THREE_TO_FOUR`, `FIVE`, and `SIX_TO_SEVEN`.
Macro relation, strength, H.8 agreement, and bank relation are exactly the
mechanism document.

The primary never has a zero macro integer. Under the committed pre-incidence
amendment, a component-only control may be valid when opposing H.4.1 and ON RRP
terms cancel exactly; its diagnostic clock emits
`MACRO_BALANCED_OPPOSITION`. This token is control-only, is not an eligibility
repair, and is forbidden from any primary RLLM prompt.

`signal_available_time` is the maximum of the selected H.4.1 availability, the
latest ON RRP availability in the current complete interval, and the H.8
release timestamp. It must not exceed the decision.

## Controls, scheduling, and split containment

The fourteen controls and their order are exactly the preregistration.

- component-only controls retain all other causal validity requirements;
- subset controls filter raw primary-eligible states before independent
  reservation;
- stale H.4.1 uses the immediately preceding emitted H.4.1 rank;
- stale ON RRP uses the immediately preceding emitted rank in the same
  post-quarantine segment;
- stale controls recompute exact macro side and bank relation;
- `nsa_h8` replaces SA validity/relation with the exact NSA replay while
  retaining the primary macro side;
- delayed execution keeps the original state and side, uses the next archived
  H.8 17:05 entry, requires entry after that host release, and permits an
  excluded host release; and
- exact flip/random reuse accepted primary timestamps and source tokens while
  changing only side/control identity.

Random side hashes the exact ASCII-compatible UTF-8 string:

```text
DCLB-864|YYYY-MM-DDTHH:MM:SSZ
```

LONG is selected when the first SHA-256 digest byte is below 128.

Every independently scheduled clock is built over the complete pre-2024
source, sorted by `(entry_time, signal_id)`, globally non-overlap-reserved, and
then split-contained. Entry equal to prior accepted exit is allowed. An event
crossing a split boundary belongs to neither split. Warm-up events before 2020
still participate in global reservation.

## Source support and composition

All statistics use globally accepted, split-contained clocks. Empty
denominators and missing required controls fail.

Clock statistics include event/side counts, active months, maximum month and
quarter shares, maximum entry gap, and maximum same-side run. Composition uses
primary accepted events as denominator and applies every preregistered
train/selection threshold exactly.

Maximum gap means New York **calendar-date** difference, not elapsed UTC
hours. Every required control must have at least one split-contained event in
train and selection; these are ordered source-support checks, not composition
checks.

Same-side reproduction is:

```text
candidate primary entries whose exact (entry_time, side) occurs in control
------------------------------------------------------------------------
                    candidate primary entry count
```

The source/support check order and composition check order are deterministic;
the first false check is the frozen first failure.

## Comparator parser and novelty

Comparator hashes and header hashes are validated before source rows are
decoded, but comparator value rows remain unopened until complete source and
composition passes.

For each comparator artifact, use the exact preregistered `usecols` with
string dtype, empty-string preservation, and exact group-filter equality.
Group strings are never stripped, case-folded, coerced, regex-matched, or
substring-matched.

Entry/exit strings must carry an RFC3339 `Z` or `±HH:MM` timezone suffix before
strict UTC parsing. Side is exactly `LONG` or `SHORT`. Timestamp, side, and
`exit > entry` validation applies to **every parsed artifact row before group
filtering**, including rows outside selected groups. Every selected raw group
is then validated in full for nonempty membership, unique entry, and
chronological self-nonoverlap before common-window filtering.

The exact common window is:

```text
[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
```

Only fully contained intervals are used. Each group reports raw selected,
contained, before-window, after-window, and boundary-crossing counts and must
meet its frozen contained-row floor.

Metrics use the complete five-minute grid over the exact four-year window:

- exact-entry Jaccard;
- same-entry same-side reproduction;
- absolute signed occupied-exposure Pearson;
- maximum-cardinality one-to-one ±6-hour Jaccard for asynchronous clocks; and
- maximum-cardinality one-to-one ±7-day Jaccard as report-only.

Undefined correlation fails. Thresholds are selected only by the frozen
`same_h8_anchor` versus `asynchronous` comparator family.

No additional comparator statistic is computed or serialized. In particular,
occupied-time Jaccard is outside the registered metric vocabulary.

Any comparator hash, parser, full-row validation, group, containment-floor,
occupancy, or metric failure is converted into a deterministic
`ComparatorContractFailure`. The failure preserves the cumulative number of
comparator rows already decoded and a stable failure code. It does not abort
artifact construction: the canonical report records the failure as the first
false `comparator_novelty` check and retires DCLB-864 before outcomes.

## Deterministic artifacts

The clock is canonical LF CSV inside deterministic gzip (`mtime=0`, empty
filename). The report is canonical sorted JSON without a wall-clock timestamp.
Both are repository-confined, no-symlink, write-once artifacts. They fsync the
file and containing directory; byte-identical existing files are accepted and
drift or race drift is rejected.

The report records all bindings, decoded source/comparator row counts, feature
funnel, control counts, every gate, first failure, clock hashes, and explicit
zero counts for BTC market, funding, future return, PnL, network, and
external-data subprocess access. The two local Git protocol-integrity checks
are counted separately.
