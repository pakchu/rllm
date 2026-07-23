# OPRR-288 mechanism decision — CBOE option-pressure rank rotation

## Decision

Freeze one candidate:
**OPRR-288 — CBOE Option-Pressure Rank Rotation**.

On each exact common CBOE source date, OPRR places option-flow pressure below,
between, or above term and tail pressure. It emits only when:

1. the option surface changes ordinal position;
2. option pressure itself changes in the same direction as that rotation; and
3. both term and tail pressure also change in that same direction.

The common direction fixes:

```text
all three pressures rising while option rotates upward   -> SHORT BTCUSDT
all three pressures falling while option rotates downward -> LONG BTCUSDT
```

The position begins only on the first later regular CBOE session fixed by the
prospective session calendar below, at 09:35 America/New_York, and holds exactly
288 five-minute bars. A later RLLM may trade the fixed side or abstain.

This document freezes source fields, causal normalization, ordinal and
transition algebra, direction, latency, hold, controls, source-support gates,
novelty cohort, and RLLM boundary before any OPRR pressure row, ordinal state,
transition, timestamp, side, comparator row, or market outcome is computed.

## Falsifiable mechanism

The three surfaces describe distinct observable layers:

- term pressure: short- versus medium-horizon implied-volatility repricing;
- tail pressure: SKEW and volatility-of-volatility repricing; and
- option pressure: changes in index/equity put-call separation, VIX call
  pressure, and index-volume share.

An upward OPRR event requires a broad same-direction stress move while option
pressure overtakes at least one other surface. A downward event requires broad
same-direction relief while option pressure falls behind at least one other
surface. The candidate tests whether that relative acceleration/deceleration
contains next-day BTC directional information.

No option level, option change, term-tail change, or ordinal move is sufficient
alone. OPRR does not identify an investor and does not claim direct causal BTC
demand. Its narrower claim is that synchronized cross-surface movement plus a
change in where option flow sits in that ordering is an observable risk-transfer
transition.

## Immutable source contract

### Term panel

```text
data/cboe_volatility_term_structure_2018_2023/
cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz
SHA256 6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7
```

Manifest SHA-256:
`42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27`.

Allowed columns:

```text
observation_date
VIX9D_close
VIX_close
VIX3M_close
```

### Tail panel

```text
data/cboe_tail_risk_2018_2023/
cboe_tail_risk_2018-01-01_2023-12-31.csv.gz
SHA256 cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a
```

Manifest SHA-256:
`9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd`.

Allowed columns:

```text
observation_date
SKEW_close
VVIX_close
VIX_close
```

### Option-flow panel

```text
data/cboe_option_flow_2020_2023/
cboe_option_flow_2020-01-01_2023-12-31.csv.gz
SHA256 35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78
```

Manifest SHA-256:
`0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e`.

Allowed columns:

```text
observation_date
total_volume
index_call_volume
index_put_volume
index_volume
equity_call_volume
equity_put_volume
vix_call_volume
vix_put_volume
```

The evaluator must pass these exact allowlists to each CSV reader. It may load
no BTC price, return, funding, label, reward, PnL, portfolio, calendar-alpha,
or 2024-or-later field.

Every retained numeric primitive must be finite and strictly positive. Dates
must be unique and strictly increasing. The term and tail `VIX_close` values
must match exactly on common dates. Any mismatch fails closed.

Each surface first computes its own causal history. Only then are exact source
dates intersected. Missing dates are never filled, carried, interpolated,
substituted, or treated as zero.

## Strict-prior pressure construction

Every primitive rank uses:

```text
lookback observations = at most 252 prior observations of that source
minimum prior observations = 126
```

For current value `x` and strictly prior reference `R`, the midrank is:

```text
(count(R < x) + 0.5 * count(R == x)) / len(R)
```

The current primitive is appended only after all current ranks are fixed. A
missing or non-finite primitive makes that source pressure unavailable. No
expanding future normalization, z-score, clipping, winsorization, weekday
conditioning, or later-source substitution is allowed.

### Term pressure

For each term observation:

```text
front_slope = log(VIX9D_close / VIX_close)
broad_slope = log(VIX_close / VIX3M_close)
front_rank  = strict_prior_rank(front_slope)
broad_rank  = strict_prior_rank(broad_slope)
term_pressure = 0.5 * (front_rank + broad_rank)
```

### Tail pressure

For each tail observation:

```text
skew_level    = log(SKEW_close / 100)
vvix_relative = log(VVIX_close / VIX_close)
skew_rank     = strict_prior_rank(skew_level)
vvix_rank     = strict_prior_rank(vvix_relative)
tail_pressure = 0.5 * (skew_rank + vvix_rank)
```

### Option-flow pressure

For each option observation:

```text
institutional_gap =
    log((index_put_volume + 0.5) / (index_call_volume + 0.5))
  - log((equity_put_volume + 0.5) / (equity_call_volume + 0.5))

vix_call_pressure =
    log((vix_call_volume + 0.5) / (vix_put_volume + 0.5))

index_share =
    log((index_volume + 1.0) / (total_volume + 1.0))
```

Take each current level minus the immediately preceding option-source level.
Rank each delta against its own strictly prior delta history:

```text
option_pressure =
    mean(
        delta_institutional_gap_rank,
        delta_vix_call_pressure_rank,
        delta_index_share_rank
    )
```

The half-contract and one-contract pseudocounts are immutable and exist only
to keep log-ratios defined.

## Exact common-date state

Let `D[t]` be an exact common CBOE date after all three pressures have been
computed independently. Let:

```text
P_term[t]   = term_pressure on D[t]
P_tail[t]   = tail_pressure on D[t]
P_option[t] = option_pressure on D[t]
```

All three pairwise comparisons must be strict. If any two pressures are equal,
the option ordinal state on that date is unavailable.

For a pairwise-distinct date:

```text
option_position[t] =
    1{P_term[t] < P_option[t]}
  + 1{P_tail[t] < P_option[t]}
```

Therefore:

```text
0 = option pressure below both other surfaces
1 = option pressure between the other surfaces
2 = option pressure above both other surfaces
```

The prior state is always the immediately preceding exact common date
`D[t-1]`. An unavailable/tied pressure on either `D[t-1]` or `D[t]` makes the
transition unavailable. The evaluator may not skip backward to an older valid
state.

## Rotation, confirmation, and deterministic side

For adjacent exact common dates:

```text
rotation[t] = option_position[t] - option_position[t-1]

delta_option[t] = P_option[t] - P_option[t-1]
delta_term[t]   = P_term[t]   - P_term[t-1]
delta_tail[t]   = P_tail[t]   - P_tail[t-1]
```

The transition is eligible only when all conditions hold:

```text
rotation[t] != 0
sign(delta_option[t]) == sign(rotation[t])
sign(delta_term[t])   == sign(rotation[t])
sign(delta_tail[t])   == sign(rotation[t])
```

Every zero, missing, non-finite, or disagreeing delta makes the transition
ineligible. No epsilon, magnitude threshold, volatility gate, BTC regime gate,
calendar gate, or outcome-selected tie breaker is allowed.

Side is:

```text
rotation[t] > 0 -> SHORT
rotation[t] < 0 -> LONG
```

`abs(rotation)` is exactly `1` or `2` and is diagnostic/token information only;
it does not change side, hold, or leverage.

## Availability, execution, and hold

For transition date `D[t]`:

- source-close values from `D[t]` are unavailable for trading on `D[t]`;
- let `S_next` be the first later regular CBOE session selected by the
  prospective calendar below;
- signal availability: `S_next` at `09:30` America/New_York;
- decision/order/entry: `S_next` at `09:35` America/New_York;
- instrument: Binance USD-M BTCUSDT;
- entry: first five-minute open at the exact UTC-converted decision time;
- exit: exactly 288 five-minute bars after entry;
- leverage: fixed `0.5x`;
- stop, take-profit, trailing exit: none.

For pre-2024 research, a regular CBOE session is any Monday through Friday in
`[2020-01-01, 2025-01-01)` except the following full-day closures:

```text
2020-01-01  2020-01-20  2020-02-17  2020-04-10  2020-05-25
2020-07-03  2020-09-07  2020-11-26  2020-12-25

2021-01-01  2021-01-18  2021-02-15  2021-04-02  2021-05-31
2021-07-05  2021-09-06  2021-11-25  2021-12-24

2022-01-17  2022-02-21  2022-04-15  2022-05-30  2022-06-20
2022-07-04  2022-09-05  2022-11-24  2022-12-26

2023-01-02  2023-01-16  2023-02-20  2023-04-07  2023-05-29
2023-06-19  2023-07-04  2023-09-04  2023-11-23  2023-12-25

2024-01-01  2024-01-15  2024-02-19  2024-03-29  2024-05-27
2024-06-19  2024-07-04  2024-09-02  2024-11-28  2024-12-25
```

`S_next` is found by incrementing one calendar day from `D[t]` until the date
is a weekday not in that immutable closure set. Early closes remain sessions
because entry occurs at 09:35. Historical UTC conversion must use the IANA
`America/New_York` timezone, including the then-applicable DST offset.

The existence or absence of a term, tail, or option row on `S_next` is not read
and cannot create, suppress, or reschedule the entry. A missing future source
row affects only a later transition after that row would have become available.
Consecutive entries exactly equal to the previous exit are allowed. An entry
before the previous accepted exit is suppressed and never queued.

Every raw clock is constructed across the complete pre-2024 source, globally
non-overlap-reserved, and only then split-contained. Split containment requires
`entry >= split_start` and `exit <= split_end`; exposure is the half-open
interval `[entry, exit)`, so `exit == split_end` is allowed.

Live promotion beyond 2024 must hash-freeze the official future session
calendar before opening that year's source or outcome. It must wait for all
three source closes for observation date `D`, validate cross-panel parity, and
fail flat on a missing, duplicated, revised, late, non-finite, nonpositive, or
inconsistent `D` row. A correction cannot retroactively create a trade. Source
rows expected only after the current entry session's close are never required
for that morning's decision.

## Frozen source-only controls

Every independent clock repeats complete-horizon construction, global
reservation, and split containment.

1. `primary` — exact OPRR transition and confirmation.
2. `rank_rotation_only` — option ordinal position changes; direction is the
   rotation sign; ignore all three delta confirmations.
3. `option_own_confirmed` — option ordinal position changes and option delta
   agrees with the rotation; ignore term and tail delta confirmations.
4. `non_option_pair_only` — term and tail deltas agree with one another
   non-neutrally; direction is their common sign; ignore option position and
   option delta.
5. `term_sponsor_rotation` — apply the exact primary algebra with term as the
   ordinal sponsor relative to tail and option.
6. `tail_sponsor_rotation` — apply the exact primary algebra with tail as the
   ordinal sponsor relative to term and option.
7. `one_common_date_stale` — at current transition date `D[t]`, use the exact
   primary decision from `(D[t-2], D[t-1])`; if that prior transition was
   ineligible, emit nothing; schedule on `S_next` after `D[t]`.
8. `exact_direction_flip` — primary timestamps with the opposite side.
9. `deterministic_random_side` — primary timestamps with the frozen SHA-256
   side rule below.
10. `one_day_execution_delay` — primary source state and side with entry and
    exit delayed exactly 288 five-minute bars, followed by fresh global
    reservation and split containment.

For `term_sponsor_rotation` and `tail_sponsor_rotation`, every surface delta
must agree with the sponsor rotation exactly as in the primary. A tie involving
the sponsor makes that control unavailable.

`exact_direction_flip` and `deterministic_random_side` use the exact accepted
primary clock. No source control may replace a failed primary.

For the random control, canonicalize the UTC entry timestamp as exactly:

```text
YYYY-MM-DDTHH:MM:SSZ
```

It is zero-padded, contains no fractional seconds, and always ends in literal
ASCII `Z`. Encode exactly:

```text
b"OPRR-288|" + canonical_entry_time.encode("ascii")
```

Compute `hashlib.sha256(message).digest()` and inspect binary octet
`digest[0]`. Values `0` through `127` are LONG; `128` through `255` are SHORT.
Hexadecimal text is never hashed or indexed.

## Source-support gates

Warm-up uses 2020 only. Primary support is measured on:

```text
train     [2021-01-01, 2023-01-01)
selection [2023-01-01, 2024-01-01)
```

### Train

- at least 100 globally accepted split-contained events;
- at least 40 events in each of 2021 and 2022;
- at least 20 active months;
- LONG and SHORT each at least 20%;
- maximum single-month share at most 15%;
- maximum single-quarter share at most 35%;
- maximum entry gap at most 35 calendar days;
- maximum same-side run at most 10.

### Selection

- at least 45 events;
- at least 18 events in each half-year;
- at least 6 events in every quarter;
- at least 10 active months;
- LONG and SHORT each at least 20%;
- maximum single-month share at most 22%;
- maximum entry gap at most 45 calendar days;
- maximum same-side run at most 8.

An empty required window, missing statistic, zero denominator, or non-finite
metric fails.

## Rotation-composition gates

Across train and selection separately:

- one-step rotations are at least 10% of primary events;
- two-step rotations are at least 10% of primary events;
- each undirected transition family `0<->1`, `1<->2`, and `0<->2` is at least
  5% of primary events;
- each option position `0`, `1`, and `2` appears in at least 8% of prior states
  and at least 8% of current states;
- raw primary-transition retention within `option_own_confirmed` is at most
  75%;
- raw primary-transition retention within `non_option_pair_only` is at most
  75%;
- exact-entry Jaccard against each sponsor-permutation control is at most 65%;
- same-entry same-side reproduction by `one_common_date_stale` is at most 80%;
  and
- same-entry same-side reproduction by deterministic random side is at most
  60%.

Raw retention is computed before global overlap reservation:

```text
count(primary raw transition date in control raw transition dates)
/
count(control raw transition dates)
```

For each split, a raw transition is included in this retention calculation only
when its prospectively scheduled `[entry, exit)` is split-contained. It is not
removed because another transition overlaps. This makes the statistic an
algebraic-incidence comparison rather than a reservation artifact.

Exact-entry Jaccard is computed separately over accepted split-contained entry
timestamp sets:

```text
J(A, B) = |A intersect B| / |A union B|
```

Both sets and their union must be nonempty. Same-entry same-side reproduction
is divided by the accepted split-contained primary count. Every denominator
must be nonzero.

The first two retention gates prove separately that:

- adding non-option confirmation removes at least 25% of the option-own
  transition clock; and
- adding option rank rotation plus own confirmation removes at least 25% of
  the non-option pair clock.

They are not performance gates and cannot be relaxed after counts are opened.

## Comparator novelty

The preregistration must hash-bind and validate exact headers for:

1. CXRT-288 source-only clock:
   `data/cboe_cross_surface_risk_transfer_clocks_2020_2023.csv.gz`,
   SHA-256
   `b3cc6f3d6a19cb39ef63ec0ba9908c983ce03c56a0c7dd8786e51c2ef1c0885f`;
2. CVTR-1 clock:
   `results/cboe_volatility_term_rotation_clocks_2026-07-17.csv.gz`,
   SHA-256
   `47f4ca447daa2b03a0827ad243ed1107eb34a37e5d7bab18ecd3c4331736959d`;
3. CTHD-1 clock:
   `results/cboe_tail_hedge_disagreement_clocks_2026-07-18.csv.gz`,
   SHA-256
   `0e19455e2fb5ab2d36cc996c9adf514adc85c69dd1a325562344a8015464d546`;
4. CIHM-1 clock:
   `results/cboe_institutional_hedge_migration_clocks_2026-07-18.csv.gz`,
   SHA-256
   `5e04cffacb1754c3111fcc32b09d72f06b546a4803b40c77d655a9787b015c0b`.

Selected CXRT groups are:

```text
primary
term_only
tail_only
option_only
term_tail_agreement
one_common_date_stale
exact_direction_flip
deterministic_random_side
one_day_execution_delay
```

Selected CVTR groups are `primary`, `deterministic_random_side`, and
`constant_long`. Selected CTHD and CIHM groups are `primary`.

Every group is compared separately over:

```text
[2021-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
```

For every required group:

- exact entry-time Jaccard is at most `0.35`;
- same-entry same-side reproduction is at most `0.80`;
- absolute signed occupied-exposure Pearson correlation is at most `0.55`;
- one-local-calendar-day tolerant entry Jaccard is report-only.

For exact-entry metrics, define:

```text
A = accepted OPRR primary entries with start <= entry < end
B = comparator-group entries with start <= entry < end
```

Entry timestamps are normalized to UTC before set construction. Duplicate
entry timestamps within either extracted group fail. Exact Jaccard is:

```text
|A intersect B| / |A union B|
```

Both `A` and `B` must be nonempty.

The report-only tolerant metric is a one-local-calendar-day, one-to-one entry
matching:

1. convert every UTC entry to `America/New_York` and retain its local calendar
   date plus UTC timestamp;
2. sort `A` and `B` by UTC timestamp;
3. construct the dynamic-programming table `M[i,j]`, the maximum number of
   order-preserving matches between the first `i` entries of `A` and first `j`
   entries of `B`;
4. initialize `M[0,j] = M[i,0] = 0`;
5. for `i,j > 0`, start with
   `max(M[i-1,j], M[i,j-1])`; if the absolute difference between the two local
   calendar dates is at most one day, also consider `M[i-1,j-1] + 1`;
6. let `m = M[len(A), len(B)]`; and
7. report:

```text
tolerant_jaccard = m / (len(A) + len(B) - m)
```

No side is used in tolerant matching. Empty sets or a zero denominator fail
even though the resulting tolerant value is report-only.

Same-entry same-side reproduction uses accepted OPRR primary events as the
denominator.

Signed occupied-exposure correlation is fixed as follows:

1. use the UTC window
   `[2021-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`;
2. construct every five-minute UTC left endpoint in that window;
3. encode a clock as `+1` for LONG or `-1` for SHORT when the grid point is in
   an event's half-open interval `[entry, exit)`, otherwise `0`;
4. an exit and another entry at the same instant are processed as
   previous-position exit then new-position entry, so the new position owns
   that left endpoint;
5. include any event whose exposure interval intersects the evaluation window
   and clip it to the window;
6. reject overlapping positions within either compared group; and
7. compute ordinary Pearson correlation from the two full equal-length grids:

```text
sum((x - mean(x)) * (y - mean(y)))
/
sqrt(sum((x - mean(x))^2) * sum((y - mean(y))^2))
```

This is a five-minute duration-weighted comparison. A zero variance term,
empty required extraction, duplicate entry timestamp, overlapping position,
hash/header drift, undefined denominator, or non-finite metric fails before
outcomes.

## Economic and RLLM sequence

No BTC execution source may be opened before source support, composition, and
novelty pass. A later stage must separately freeze:

1. exact pre-2024 BTCUSDT OHLC/funding sources and strict MDD accounting;
2. a train-only cheap relational baseline;
3. one small frozen LLM/RLLM model and adapter recipe;
4. reward, optimizer, seed, checkpoint selection, and abstention policy;
5. 2021 fit / 2022 inner-validation / sealed 2023 selection roles;
6. post-2023 source extension and later test/eval sequence.

The LLM may advance only if a non-leaky train/inner-validation baseline shows
recoverable relation-token signal. It may not change the source clock, side,
hold, or leverage.

Final qualification remains:

- positive full-calendar absolute return;
- `CAGR / strict MDD >= 3.0`;
- strict MDD at most 15%;
- positive 10 bp/notional/side stress return;
- positive one-day delayed return;
- statistically meaningful clustered evidence;
- positive LONG and SHORT sleeves; and
- no evaluated-split model, threshold, checkpoint, or token selection.

Exact economic thresholds and inference tests must be frozen in the later
economic/RLLM evaluator after source support passes.

## RLLM relation-token boundary

Allowed tokens:

- fixed side;
- prior and current option position `BELOW`, `MIDDLE`, or `ABOVE`;
- rotation magnitude `ONE_STEP` or `TWO_STEP`;
- rotation direction `UP` or `DOWN`;
- option-own-change agreement;
- term and tail directional confirmation;
- term-versus-tail ordering and whether it changed;
- common-calendar gap bucket `1`, `2-3`, or `4+` days;
- source validity; and
- current position state.

Forbidden inputs:

- raw numeric values or ranks;
- observation date, year, month, weekday, timestamp, row number, hash, or
  source identifier;
- BTC price, return, funding, future path, label, PnL, reward, CAGR, MDD, or
  portfolio state beyond current position;
- candidate creation, side reversal, hold change, leverage change, or
  timestamp choice.

Action space:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

No prompt may reveal split identity or historical outcome summaries.

## Failure action

Any source, support, rotation-composition, comparator-novelty, economic, or
RLLM gate failure retires OPRR-288 unchanged at that stage. A changed pressure,
rank, transition, confirmation, side, clock, hold, control, gate, token, or
model recipe is a new candidate and may not reuse an evaluated split as a
holdout.
