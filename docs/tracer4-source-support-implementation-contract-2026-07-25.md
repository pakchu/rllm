# TRACER-4H source-support implementation contract

Date: 2026-07-25

## Authority

This contract implements only the source/token support stage authorized by:

```text
docs/tracer4-tri-surface-relational-executor-boundary-2026-07-25.md
SHA256 45c15c2ef2664c35857186363894727010538f878cdda1088eb923b5a653f7d5
latest boundary commit 66218040c8c9aaaaa25fa398a07b823d6b3447ff

results/tracer4_tri_surface_relational_executor_preregistration_2026-07-25.json
file SHA256 5299fc6f803b4faf7cf10655050d71d0f918b0899c6ef8f1a18c5028579a1c8b
manifest hash 67f06047661336ebad4da19f0dd65578520a28f1d1b4b468c6300b4aa4d54318
preregistration commit a0a1a52
```

Implementation files:

```text
training/build_tracer4_tri_surface_relational_executor_support.py
tests/test_build_tracer4_tri_surface_relational_executor_support.py
```

The official runner must refuse to open source payloads unless the boundary,
preregistration source/test/artifact, this contract, runner, and runner test are
tracked, clean, and committed. The runner revision and every protocol-file hash
must be written to the support report.

This contract authorizes no execution kline, funding, future return, reward,
model row, action, target, PnL, CAGR, MDD, prior prediction, comparator row, or
post-2023 numeric source value.

## Physical projection

The runner verifies each full compressed-source SHA-256 and the raw one-line LF
header SHA-256 before converting any row value. It then uses `csv.reader` only
to preserve physical row framing. Unprojected cells may be tokenized as strings
by the CSV parser but may not be converted, retained, hashed by value,
aggregated, or exposed.

For each row:

1. require the exact physical field count;
2. parse only the timestamp field needed to enforce the physical cutoff;
3. stop at the first ordered row with `date >= 2024-01-01T00:00:00Z` before
   converting any other projected field in that row;
4. retain only the exact preregistered allowlist in its frozen order;
5. validate only projected source identities; and
6. stream the projected row to a temporary deterministic gzip file.

Deterministic cuts use UTF-8, LF, `csv.writer`, empty gzip filename, and
`mtime=0`. The temporary file is fsync'd, hashed, then installed write-once. An
existing byte-identical target is reused; drift is terminal.

The premium source is physically longer than 2023 and must hit the cutoff. The
two five-minute sources must end before the cutoff; an unexpected later row is
still stopped, disclosed, and treated as source-contract failure because their
containers are declared pre-2024.

The cut manifest records parent and cut hashes, physical/cut headers, row count,
first/last timestamps, stopped-at timestamp, compressed bytes, and zero
forbidden-field counters. Support calculation reads only the three installed
cuts.

## Source validation

### Leadership

- `date` and `feature_available_time_utc` are unique, monotone UTC timestamps;
- `feature_available_time_utc == date+5m`;
- quote notionals are finite and nonnegative;
- signed notionals are finite and bounded in absolute value by quote notional;
- lagged response and basis fields are finite;
- `source_complete` and `cross_venue_feature_valid` are exact binary values.

### Aggregate trades

- `date` is unique and monotone UTC;
- transaction times are integer milliseconds with
  `date <= first <= last < date+5m`;
- aggregate-trade count is a positive integer;
- quote notional is finite and nonnegative;
- signed notional is finite and bounded by quote notional;
- micro return, HHI, effective count, flip rate, run share, wait, and burstiness
  are finite;
- HHI, flip rate, and run share lie in `[0,1]`;
- effective count is positive and wait is nonnegative.

### Premium

- `date`, close time, and availability are unique/monotone UTC;
- `source_close_time == date+59.999s`;
- `feature_available_time == date+61s`;
- `source_valid` is exact binary;
- a valid row has finite OHLC and a valid high/low envelope;
- an invalid row has no usable OHLC and makes its four-hour boundary invalid.

NaN is never converted to a neutral token.

## Canonical grids and boundary state

Cuts are reindexed onto:

```text
five-minute grid [2020-01-01, 2024-01-01)
one-minute grid  [2020-01-01, 2024-01-01)
```

Missing rows remain missing and make a containing boundary invalid. The first
candidate boundary is `2020-01-01T04:00:00Z`; later boundaries advance exactly
four hours through `2023-12-31T20:00:00Z`.

A boundary is core-valid only with 48 exact valid leadership rows, 48 exact
valid aggregate-trade rows, and 240 exact valid premium rows in `[B-4h,B)`.
Leadership/aggregate rows obey the `B` cutoff. Premium rows obey `B+61s`, which
remains before the `B+5m` decision.

The exact aggregate formulas, rank history, token vocabulary, mapping, safety
line, and transition behavior are those in the boundary. Ranks are computed
before the current valid primitive is appended. Each primitive keeps only the
latest 540 valid prior values and requires 360. Invalid rows never enter rank
history.

`line_ready` means current core-valid plus every required rank ready.
`sequence_ready` means current and two immediately prior canonical boundaries
are consecutive and line-ready. No state may skip an invalid or warm-up line.
The sequence signature is SHA-256 of the three oldest-to-newest canonical lines
joined with one LF and terminated by one LF.

Token output exact columns:

```text
boundary
window_start
window_end
premium_cutoff
decision_time
execution_time
core_source_ready
line_ready
sequence_ready
sponsor
flow_consensus
impact_relation
participation
flow_persistence
auction_tempo
premium_price_relation
basis_premium_relation
sponsor_transition
impact_transition
crowding_transition
canonical_line
sequence_signature
```

It is deterministic gzip, write-once, and contains no raw numeric primitive.

## Controls

Controls rebuild ranks and transitions from their own controlled causal stream.
They may not reuse primary ranks.

### `premium_stale_1440m`

Keep canonical timestamps and Surface A/B unchanged. At every premium timestamp
use the OHLC/validity values from exactly 1,440 rows earlier. The first 1,440
rows become invalid. Availability remains the current canonical timestamp plus
61 seconds, so the control is stale information presented on the current clock,
not a future shift.

### `cash_perpetual_swap`

Keep Surface-A timestamps/validity unchanged. Swap:

```text
spot_quote_notional <-> um_quote_notional
spot_signed_quote_notional <-> um_signed_quote_notional
spot_to_um_lagged_flow_response_bp <-> um_to_spot_lagged_flow_response_bp
```

Negate `open_basis_bp` and `close_basis_bp`. Surface B/C remain unchanged.

### `aggtrade_monthly_rotate_37_rows`

Within each UTC calendar month, keep Surface-B date/transaction timestamps and
validation fields fixed. Circularly move each row's relation values forward by
37 five-minute rows (`numpy.roll(values,+37)`):

```text
quote_notional
signed_quote_notional
micro_log_return
event_notional_hhi
normalized_effective_event_count
sign_flip_rate
max_same_sign_run_share
interarrival_mean_ms
interarrival_burstiness
```

The original final 37 relation rows wrap to the first 37 timestamps of the same
month. Surface A/C remain unchanged.

Each control passes only when its full canonical-line stream hash differs from
the primary. Controls cannot rescue failed primary support.

## Append replay

After the full primary build, rebuild prefixes ending at each of:

```text
2021-01-01T00:00:00Z
2022-01-01T00:00:00Z
2023-01-01T00:00:00Z
2024-01-01T00:00:00Z
```

For each prefix, every already formed row through the last contained boundary
must byte-match the same rows from the full build for readiness flags, eleven
tokens, canonical line, and sequence signature. A mismatch is terminal.

## Gate evaluation

Gate order is exactly boundary gates 1–14. Gates 6–11 use only
sequence-ready/core-valid rows in the named UTC year. `SOURCE_INVALID`, warm-up,
and position are excluded.

- annual join denominator: every canonical five-minute timestamp in that year;
- annual core-valid denominator: every nominal canonical four-hour boundary in
  that year that is constructible from the physically bounded source;
- quarterly readiness excludes 2020Q1 warm-up and requires every later quarter;
- flow buy/sell means exact `CONSENSUS_BUY` and `CONSENSUS_SELL` shares;
- follow/absorb sums the two directional categories in each class;
- signatures use the current eleven-field canonical line, not the three-line
  hash;
- JSD uses complete vocabularies, zero mass for absent levels, base-2 logs, and
  `0*log2(0/M)=0`.

Every check is retained in canonical order. Missing evidence is failure. The
first failure is reported, but all support checks that can be computed without
crossing a failed evidence boundary remain visible.

## Output and decision

Write-once outputs:

```text
data/tracer4_source_cuts/pre2024/{leadership,aggtrade,premium}.csv.gz
data/tracer4_source_cuts/pre2024/token_support.csv.gz
results/tracer4_source_cut_manifest_pre2024_2026-07-25.json
results/tracer4_tri_surface_relational_executor_support_2026-07-25.json
```

Every JSON has canonical sorted serialization, its own `manifest_hash` over the
payload without that field, and a final file SHA-256. Existing drift is
terminal.

Decision:

```text
all 14 gates pass
  authorize_stage_0_5_evaluator_freeze_only
otherwise
  retire_tracer4_unchanged_before_outcomes
```

Even a support pass does not authorize reward or model training directly. It
only authorizes a separately committed Stage 0.5 freeze.
