# OPRR-288 source-support implementation contract

## Scope and stop rule

The evaluator may decode only the three preregistered CBOE source allowlists.
It must evaluate source integrity, support, and rotation composition first. It
may decode the preregistered comparator columns only after every earlier check
passes. It must never load BTC bars, funding, returns, labels, PnL, rewards,
portfolio records, or 2024-or-later source rows.

A failed source/support/composition check retires OPRR-288 before comparator
rows are read. A failed novelty check retires it before market rows are read.
No control may replace the primary.

## Causal pressure replay

Replay the exact preregistered strict-prior ranks independently in each source:
252 observations maximum, 126 observations minimum, current value appended only
after its rank is fixed. Build term, tail, and option pressure exactly as frozen,
then intersect exact dates. Term and tail VIX must agree exactly.

Keep every rank-complete common date, including pressure ties. The primary
option position requires all three pressures to be pairwise distinct. In the
term- and tail-sponsor permutation controls, a position is unavailable only
when the sponsor itself equals either comparison surface; a tie solely between
the two non-sponsors does not make the sponsor's below/above position
ambiguous. Adjacent transitions use the immediately previous rank-complete
exact common date; they may not skip over a tied primary date.

## Transition replay

For each pair of adjacent states, compute the ordinal position of each possible
sponsor against the other two. The primary uses option as sponsor and requires:

```text
rotation != 0
sign(delta_option) == sign(rotation)
sign(delta_term)   == sign(rotation)
sign(delta_tail)   == sign(rotation)
```

Positive rotation is SHORT and negative rotation is LONG. Zero, missing,
non-finite, tied, or disagreeing values are ineligible. Controls implement the
exact preregistered ablations and sponsor permutations.

## Prospective session schedule

Schedule from the source transition date with the preregistered weekday and
47-date full-day-closure set. The first later eligible date is `S_next`.
Signal time is 09:30 and entry is 09:35 `America/New_York`; DST conversion uses
the IANA timezone. The evaluator may not inspect whether any future CBOE panel
contains `S_next`.

Exposure is `[entry, exit)` for exactly 288 five-minute bars. Global reservation
accepts equality with the prior exit and suppresses, never queues, an overlap.
Reservation is performed over the complete clock before split containment.
Split containment is `entry >= start` and `exit <= end`.

Support months, quarters, and calendar gaps use entry dates converted to
`America/New_York`. Calendar-gap days are differences between local entry
dates, not elapsed UTC hours.

## Outcome-safe clock schema

The emitted clock contains only audit identity, causal times, fixed side, and
symbolic relation tokens:

```text
control
signal_id
source_date
signal_available_time
entry_time
exit_time
side
sponsor_surface
prior_sponsor_position
current_sponsor_position
rotation_direction
rotation_magnitude
option_own_change_agreement
term_confirmation
tail_confirmation
term_tail_order_relation
term_tail_order_changed
calendar_gap_bucket
```

No raw pressure, rank, price, return, funding, label, reward, PnL, CAGR, or MDD
field may be serialized.

## Raw composition metrics

Compute primary retention inside `option_own_confirmed` and
`non_option_pair_only` from raw split-contained transition dates before global
reservation. Accepted-clock Jaccard and same-side reproduction remain separate
metrics. Empty denominators fail.

## Comparator opening and novelty

Only after complete source/support/composition pass, decode exactly the frozen
comparator group, entry, exit, and side columns. The parser and novelty entry
point must require the private authorization capability created by that guarded
pass branch; an absent or invalid capability fails before opening a comparator
file. Validate byte hash, header hash, exact header, selected groups,
timestamps, sides, duplicate entries, and intervals.

Exact metrics use entries in the fixed UTC window. Tolerant matching uses local
calendar dates and the preregistered order-preserving dynamic program. Signed
occupancy uses every five-minute UTC left endpoint and half-open intervals,
clipped to the fixed window. Within-group overlap, zero variance, undefined, or
non-finite metrics fail.

## Commit gate and artifacts

Real source incidence may be opened only when this contract, evaluator source,
and its test file are tracked and byte-identical to `HEAD`. Synthetic tests may
exercise injected states but cannot authorize comparator or market access.

Real artifacts are write-once deterministic files:

```text
data/cboe_option_pressure_rank_rotation_clocks_2020_2023.csv.gz
results/cboe_option_pressure_rank_rotation_support_2026-07-24.json
```

The gzip clock uses an empty filename and `mtime=0`. Existing-byte drift fails.
The report records decoded-row counts and explicitly records zero BTC, funding,
future-return, PnL, CAGR, and MDD values.
