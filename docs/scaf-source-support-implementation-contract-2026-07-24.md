# SCAF-48 source-support evaluator implementation contract

## Scope

This contract implements the committed SCAF-48 mechanism and immutable
preregistration. It authorizes only:

- exact normalized SOMA operation/detail allowlists;
- causal-batch feature and transition construction;
- deterministic primary/control clocks;
- preregistered source-support and relational-composition gates; and
- only after both pass, the exact SLCS same-source novelty battery.

BTC OHLC, funding, returns, labels, rewards, PnL, CAGR, MDD, model training,
network access, raw API JSON, and 2024-or-later source extensions are forbidden.

## Commit-before-values proof

Before one source value row is decoded, the evaluator must prove that:

1. evaluator source, evaluator tests, and this contract are tracked in `HEAD`
   and byte-clean in both index and worktree;
2. the preregistration builder is tracked and clean;
3. the preregistration artifact hash and manifest hash match;
4. every bound document/source/comparator hash matches; and
5. every CSV header hash and required allowlist matches.

The two local Git checks are protocol-integrity subprocesses, not external-data
access. Any failure stops before source values.

## Exact source loading

Each source loader calls:

```python
pd.read_csv(
    path,
    usecols=exact_allowlist,
    dtype="string",
    keep_default_na=False,
    na_filter=False,
)
```

The resulting frame is reordered to the exact allowlist. No other normalized
column and no raw response is read.

Exact decimal parsing, identity/date/timestamp validation, operation/detail
joining, uniqueness, accepted-not-above-submitted, fee null behavior, and
operation-total reconciliation follow the mechanism. Fractional source
seconds, NUL identifiers, non-UTC timestamps, and noncanonical decimals fail.

## Causal batches and features

All operations sharing an exact `available_at_utc` form one batch. Atoms remain
exact `(operation_id, cusip)` pairs. Equal CUSIPs across operations are not
merged. A zero/invalid required batch total invalidates the whole batch and
breaks continuity.

JSD components use sorted atoms, binary64 normalized shares, natural logs,
`math.fsum`, `0*log(0/m)=0`, division by `ln(2)`, and decimal
`ROUND_HALF_EVEN` quantization to `1e-12`. Unmet-demand mass uses exact Decimal
numerator/denominator arithmetic at precision 80 and the same quantum.

No invalid or simultaneous batch enters another batch's prior state.

## Transitions, controls, and scheduling

Every valid current/previous component pair yields exact `UP/DOWN/FLAT` tokens.
Three or more UP is FRACTURE/SHORT; three or more DOWN is RELIEF/LONG.

Every preregistered source control is built from the same uninterrupted batch
segments. Stale queues reset on invalid batches. The within-batch demand
permutation follows the exact NUL-delimited SHA-256 destination ordering. Side
controls reuse accepted primary entries.

Every source clock independently:

1. constructs canonical signal IDs;
2. enters at `ceil_to_5m(signal) + 5m`;
3. exits after 576 bars / 48 elapsed hours;
4. globally reserves `[entry, exit)` before split assignment; and
5. retains only intervals fully contained in train or selection.

Suppressed opportunities are never queued.

## Source support and composition

Coverage rows are assigned by signal availability. Raw-consensus statistics
use split-contained hypothetical intervals before reservation. Accepted-clock
statistics use globally reserved, split-contained intervals.

All UTC calendar gates, control presence, raw component agreement, exact
three/four agreement, control reproduction, and permutation Jaccard checks
follow the preregistration in insertion order. Empty denominators fail. The
first false check is the frozen first failure.

## Comparator ordering and validation

Comparator values remain unopened until all source-support and composition
checks pass. On pass, the evaluator reads exactly:

```text
control,entry_time,exit_time,side
```

Every raw comparator row is parsed and validated before group filtering.
Groups use exact equality without stripping, case folding, coercion, regex, or
substring matching. Duplicate entries, invalid sides, nonpositive intervals,
self-overlap, or a contained-row floor failure fail closed. Rows wholly before
or after the common window and rows crossing either boundary are excluded from
metrics and reported separately.

Only fully contained rows in:

```text
[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
```

enter novelty metrics.

The evaluator computes only the registered metrics:

- exact-entry Jaccard;
- maximum-cardinality one-to-one New York local-calendar ±1-day Jaccard;
- same-entry same-side reproduction using SCAF as denominator; and
- absolute signed full-grid occupied-exposure Pearson.

The matching denominator, deterministic augmenting-path order, five-minute
grid, `[entry, exit)` convention, side/idle encoding, and undefined-correlation
failure follow the mechanism.

## Comparator failure evidence

Any comparator hash, parser, all-row validation, group, floor, matching, or
occupancy failure becomes a deterministic comparator-contract failure carrying:

- stable failure code;
- cumulative comparator rows decoded; and
- deterministic message.

It does not abort artifact construction. The canonical report records
`first_failing_stage = comparator_novelty`, the exact first false check, decoded
row count, and retirement before outcomes.

## Deterministic artifacts

The clock is canonical LF CSV inside deterministic gzip (`mtime=0`, empty
filename). It contains symbolic directions and scheduling fields only—no raw
amount, rate, JSD, rank, market, return, reward, or PnL value.

The report is canonical sorted JSON without a wall-clock timestamp. Both
artifacts are repository-confined, no-symlink, write-once, fsynced files.
Byte-identical existing files are accepted; drift and races reject.

The report records bindings, source/comparator decoded-row counts, feature
funnel, invalid/reset counts, control clocks, every gate, first failure,
artifact hashes, and explicit zero counts for forbidden market/outcome access.
