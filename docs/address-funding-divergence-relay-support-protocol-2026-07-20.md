# AFDR-864 source-support evaluator protocol

## Purpose

This evaluator implements the frozen AFDR-864 source-support and novelty stage
without opening BTC prices, settlement-mark values, returns, PnL, CAGR, MDD,
or any 2024+ row. It is not an economic backtest.

The evaluator and this document must be hash-bound in a write-once source
access seal before a numeric address, funding, or comparator row is parsed.
The seal may hash files but records zero inspected feature values and zero
outcome rows.

## Allowed reads

- Coin Metrics exact physical and parsed columns:
  `observation_date,available_at,AdrBalCnt,AdrActCnt`.
- Binance funding physical header: the nine frozen columns in the source
  artifact. The value parser uses only
  `funding_time_ms,funding_time_utc,symbol,funding_rate`; mark and settlement
  values are never parsed.
- Comparator readers use only frozen identity/filter/group, entry, exit, and
  side fields. The prior-microstructure JSON must attest
  `post_entry_outcomes_computed=false` and expose only `signal_date,side` in
  each event object.

Every source and comparator byte is SHA-256 bound by the preregistration.

## Causal feature implementation

The address changes use the exact UTC observation seven calendar days earlier;
there is no nearest match or fill. Complete address-feature availability is
the later of current and lag-row availability. A row with late required input
may become a later reference but cannot emit a backdated signal.

At each current address availability, take the nine most recent funding events
whose `funding_time_utc + 5m` is already available. Their canonical slots must
be consecutive eight-hour slots, reported offsets must be in `[0,60000]` ms,
the newest available event must be at most eight hours old, and timestamp
representations must agree. The pressure is the sum of those nine rates.

Each raw feature receives a tie-midrank against finite observations in the
preceding 365 calendar days whose observation date and complete feature
availability are both strictly prior. The minimum is 180. The current row is
excluded.

The primary and three component states follow the preregistered inclusive
rank tails. An onset requires the immediately previous exact daily row to be
valid and FLAT; an invalid or missing predecessor does not create an onset.

## Scheduling and support

Entry is `ceil(decision_time,5m)+5m`; exit is 864 bars later. Scheduling is
chronological greedy per control, accepts an entry exactly at the prior exit,
and requires full split containment. Direction-flip and random-side controls
retain exact accepted primary intervals. The delay control moves accepted
primary sides to the next fully valid address report and reschedules.

Support and concentration are measured on accepted primary intervals. Rolling
30-day concentration uses `[entry,entry+30d)` anchored at every accepted
entry. All gates are checked independently in train and selection as frozen.

## Novelty

Each comparator group is independent. Timestamp metrics use unique exact UTC
entries, exact Jaccard, and bidirectional elapsed-time containment at the
inclusive six-hour boundary. Low common support fails closed.

Comparator event and coverage timestamps must carry an explicit timezone in
their raw value. Timezone-less strings are malformed and are never localized
to UTC implicitly; valid offset-aware values are converted exactly to UTC.

Every frozen comparator specification must produce at least one evaluated
member. Empty post-filter inputs, empty JSON event lists, missing group keys,
missing timestamps, duplicate member identities, and unknown formats are
serialized as explicit failing registry members; they cannot disappear from
the aggregate novelty decision.

Directional members additionally use complete five-minute entry-inclusive,
exit-exclusive signed exposure over the exact comparison window. Intervals
are clipped only at the fixed comparison boundary. Within-member overlap,
non-alignment, invalid sides, invalid intervals, or zero variance fails closed.
Expected comparator-contract and directional-novelty failures are captured per
member with `comparator_contract_valid=false`; the evaluator continues only far
enough to write the deterministic `REJECT_NO_REPAIR` result and outcome-free
clock artifacts. Such failures never abort before the rejection artifact.

## Output and stopping rule

The clock artifact contains only candidate, control, split, side,
observation/decision/entry/exit timestamps and is deterministically gzipped.
The result binds the source seal and clock hash and reports only source
quality, support counts, novelty statistics, and zeroed outcome counters.

Failure yields `REJECT_NO_REPAIR` and permanently retires AFDR-864. Only a
complete pass authorizes a separately tested, committed, and hash-frozen
strict economic evaluator.
