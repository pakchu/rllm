# SMAF-72 preregistration — 2026-07-30

## Decision

**Freeze exactly one source-seen, outcome-unseen candidate before incidence.**
`SMAF-72` is the SOMA Maturity Allocation Fracture clock. It projects the
already frozen New York Fed SOMA securities-lending detail panel onto security
maturity, a field not used by the retired SLCS or SCAF candidates.

This is not a new source and not a causal claim. It is a same-source,
source-seen maturity projection with a falsifiable fixed polarity:

- unusually low fracture: `LONG`;
- unusually high fracture: `SHORT`;
- entry after one complete five-minute bar;
- 72-hour hold;
- 0.5x gross exposure.

No SMAF feature, rank, event count, event time, overlap, BTC return, funding
cash flow, PnL, CAGR, or MDD has been calculated. The first failed gate retires
this exact singleton without changing its parser, statistic, sign, rank window,
tail, onset, latency, hold, support floor, comparator, cost, or threshold.

## Frozen official source

The source is the official New York Fed securities-lending history audited in:

- `docs/new-york-fed-securities-lending-source-audit-2026-07-23.md`;
- <https://www.newyorkfed.org/markets/desk-operations/securities-lending>;
- <https://www.newyorkfed.org/markets/sec_faq>;
- <https://www.newyorkfed.org/markets/domestic-market-operations/monetary-policy-implementation/securities-lending>.

The New York Fed describes a daily competitive multiple-price auction in which
primary dealers borrow eligible SOMA securities. The candidate uses only the
frozen normalized operation and CUSIP-detail files. It does not infer an
original publication timestamp beyond the conservative `available_at_utc`
already produced by the audited builder.

Frozen identities:

| Artifact | SHA-256 |
|---|---|
| Source audit | `c812998be0bd44efc09b8120d9bc0b2a96f4e1e95f9414a4c8458d97319307bc` |
| Source builder | `2f0b5b3daca253ca015c7f691faf0ab75d11c200c11f5bc1c47b34ed1b85ef45` |
| Operations panel | `99eb8c37c05417789dfad7452c7b2ddc5b6b640078b87451f1c945158af77906` |
| Operations header | `c0d63795e5e53cef816c50472c6941069cb018f30ad1f745f250daa0fa6b9200` |
| CUSIP-detail panel | `27178d8738cb50c4e6c13f1e5940fcfdf4009e6979b006c42fb86fb399d0716d` |
| Detail header | `9f4d54dff4b9c9f0c47c0a85e0bf245276e5a3cb764b3c084017f679586b76dd` |
| Source manifest file | `58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019` |
| Source manifest canonical hash | `748db33b3ea40eb48d126d0e9882b05e1994741bf851a8f3c7b89d5166db969c` |

Only these operation columns are authorized:

1. `operation_id`;
2. `operation_date`;
3. `available_at_utc`;
4. `total_par_submitted`;
5. `total_par_accepted`.

Only these detail columns are authorized:

1. `operation_id`;
2. `operation_date`;
3. `available_at_utc`;
4. `cusip`;
5. `security_description`;
6. `par_submitted`;
7. `par_accepted`;
8. `actual_available_to_borrow`.

The operation-panel `maturity_date` is the lending operation's maturity, not
the underlying security maturity, and is explicitly forbidden.

## Disclosed bounded grammar probe

Before this preregistration, one bounded source-only probe inspected exactly
the first eight normalized detail rows under the file's existing order. The
selection rule was: open the detail gzip, retain only
`operation_id,operation_date,cusip,security_description`, identify the first
`operation_id`, and stop after eight rows from that operation. It did not read
amounts, rates, availability, counts beyond eight, any candidate statistic, or
any market/funding value.

The exact disclosed rows are:

| operation_id | operation_date | cusip | security_description |
|---|---|---|---|
| `SL 010219 1` | `2019-01-02` | `912810EC8` | `T 08.875 02/15/19` |
| `SL 010219 1` | `2019-01-02` | `912810ED6` | `T 08.125 08/15/19` |
| `SL 010219 1` | `2019-01-02` | `912810EE4` | `T 08.500 02/15/20` |
| `SL 010219 1` | `2019-01-02` | `912810EF1` | `T 08.750 05/15/20` |
| `SL 010219 1` | `2019-01-02` | `912810EG9` | `T 08.750 08/15/20` |
| `SL 010219 1` | `2019-01-02` | `912810EH7` | `T 07.875 02/15/21` |
| `SL 010219 1` | `2019-01-02` | `912810EJ3` | `T 08.125 05/15/21` |
| `SL 010219 1` | `2019-01-02` | `912810EK0` | `T 08.125 08/15/21` |

The probe disclosed only the trailing `MM/DD/YY` grammar. Any later parser
failure is terminal; the grammar may not be repaired from newly seen rows.

## Exact parser and operation integrity

All input is read as strings with `keep_default_na=False` and
`na_filter=False`. No trimming, Unicode normalization, case folding, exponent
notation, float parsing, or row dropping is allowed.

### Security description

Every description must match this ASCII regular expression exactly:

```text
\A(?P<label>[A-Z][A-Z0-9/-]{0,15}) (?P<coupon>[0-9]{1,2}(?:\.[0-9]{1,6})?) (?P<maturity>(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])/[0-9]{2})\Z
```

Consequences:

- exactly one ASCII space separates the three tokens;
- `label` is 1–16 uppercase ASCII letters, digits, slash, or hyphen and starts
  with a letter;
- `coupon` is a one- or two-digit, optionally zero-padded, nonnegative plain
  decimal from zero through less than 100, with at most six fractional digits;
- no sign, comma, percent symbol, exponent, tab, leading space, trailing space,
  or extra token is accepted;
- the date receives Gregorian calendar validation after the regex;
- year `YY` means exactly `2000 + YY`;
- `tau = maturity_date - operation_date` is an integer number of calendar days;
- `1 <= tau <= 18,263` is required.

Coupon and label are validation tokens only. They never enter the feature.

### Numeric strings

`par_submitted`, `par_accepted`, `actual_available_to_borrow`, and operation
totals must match `\A(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z`. Each is converted to an
exact rational from its decimal digits. Values must be finite and nonnegative.
For every detail, accepted may not exceed submitted.

### Join and complete-operation contract

The evaluator must fail closed unless:

1. operation IDs are unique;
2. `(operation_id,cusip)` detail keys are unique;
3. every detail joins exactly one operation;
4. joined operation date and `available_at_utc` agree exactly;
5. every operation has at least one detail;
6. detail submitted and accepted sums reconcile exactly to operation totals;
7. every description parses and every `tau` is valid;
8. each of submitted, accepted, and available-to-borrow total weight is
   strictly positive.

One failed detail invalidates its whole operation. No partial centroid,
available-only subset, row deletion, or imputation is allowed. Failure reason
counts are reported before retirement.

## Causal batch and uninterrupted history

`available_at_utc` is consumed exactly from the hash-bound operation panel.
The builder already interprets `lastUpdated` in `America/New_York`, rejects DST
ambiguity/nonexistence, and takes the later of that value and the next UTC
midnight.

All operations with exactly equal `available_at_utc` form one causal batch.
SMAF requires exactly one complete operation per causal batch. A missing,
invalid, or multi-operation batch invalidates the entire batch and resets:

- all rolling rank history;
- the prior lower-tail state;
- the prior upper-tail state.

No current-batch value may enter another current-batch value's history.
Weekend, holiday, and ordinary no-operation calendar gaps do not interrupt a
segment. Only an invalid causal batch resets it.

## Exact feature

For operation `o`, detail `i`, integer maturity distance `tau_i`, and exact
nonnegative weight `W_i`, define:

```text
C_o(W) = sum_i(W_i * tau_i) / sum_i(W_i)
```

The three centroids are:

- `C(S)`: `par_submitted` weighted maturity centroid;
- `C(A)`: `par_accepted` weighted maturity centroid;
- `C(V)`: `actual_available_to_borrow` weighted maturity centroid.

They are calendar-day centroids, not duration, DV01, risk, or causal estimates.
The singleton fracture is:

```text
F = 2*C(S) - C(V) - C(A)
  = (C(S)-C(V)) + (C(S)-C(A))
```

Thus positive `F` means submitted maturity is long relative to the **average**
of the available-to-borrow and accepted-award centroids. It does not require
submitted maturity to exceed each centroid separately.

The fixed economic hypothesis is that an unusually high `F` represents
long-maturity collateral demand not mirrored by the average of available
inventory and awards, a Treasury-collateral scarcity state that transmits
`SHORT` to BTC over 72 hours. Unusually low `F` is the fixed inverse `LONG`
hypothesis. This polarity is conjectural and may not be reversed after seeing
incidence or outcomes.

## Exact rank, tails, and onset

Within an uninterrupted segment, each complete operation is ranked against
exactly the latest 126 strictly prior complete operations. Fractions are
compared by integer cross multiplication; floats are forbidden.

For current `F`, let:

```text
L = count(prior F < current F)
E = count(prior F = current F)
R = (2*L + E) / 252
```

Tail tests are also integer comparisons:

```text
LOW  iff 10*(2*L + E) <= 252
HIGH iff 10*(2*L + E) >= 2268
otherwise NEUTRAL
```

The first rank-ready operation in a segment only establishes the prior tail
baseline and cannot trigger. Thereafter:

- current `LOW` and prior state not `LOW` emits one raw `LONG`;
- current `HIGH` and prior state not `HIGH` emits one raw `SHORT`;
- persistence inside the same tail emits nothing;
- neutral or the opposite tail updates state without emitting;
- invalid batch resets state and history.

The canonical primary signal ID is lowercase hex:

```text
sha256(
  UTF8("SMAF-72|primary|<operation_id>|<available_at_utc>|<LOW_OR_HIGH>")
)
```

Source strings are inserted byte-for-byte and the tail token is uppercase.

## Availability, scheduling, and split assignment

For each raw signal:

```text
decision_time = available_at_utc
entry_time = ceil_to_5m(decision_time) + 5 elapsed minutes
exit_time = entry_time + 72 elapsed hours
```

`ceil_to_5m(t)` is the smallest UTC timestamp on a Unix-epoch multiple of 300
seconds that is not earlier than `t`. Even an already aligned decision waits
five elapsed minutes. The hold is exactly 864 five-minute bars.

Raw signals are sorted by `(entry_time, signal_id)`. For each clock
independently, accept a signal only when `entry_time >= previous accepted
exit_time`; equality is allowed. The reserved interval is `[entry_time,
exit_time)`. An overlapping signal is suppressed without queue, replacement,
or later release. Source tail state evolves independently of suppression.

Global non-overlap is applied before split filtering. A retained interval must
be fully contained:

- train: `[2020-01-01T00:00:00Z, 2023-01-01T00:00:00Z)`;
- selection: `[2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`.

Operation date, decision, entry, and exit must all belong to the same declared
split; exit may equal the exclusive split end. Crossing intervals are reported
and excluded whole, never clipped.

## Source-only controls

Every scalar clock uses the same 126-operation exact rank, tails, onset,
availability, ordering, non-overlap, and containment rules:

1. `primary`: `2*C(S)-C(V)-C(A)`;
2. `submitted_inventory_tilt`: `C(S)-C(V)`;
3. `submitted_award_tilt`: `C(S)-C(A)`;
4. `award_inventory_tilt`: `C(A)-C(V)`;
5. `aggregate_demand_intensity`: `sum(S)/sum(V)`.

The pairwise tilts are algebraically related; they are attribution controls,
not independent features. Each control receives its own signal ID by replacing
`primary` in the canonical ID input with the exact control name.

Outcome controls reuse accepted primary intervals:

- `exact_direction_flip`: opposite side;
- `deterministic_random_side`: `LONG` iff the first byte of
  `SHA256("SMAF-72|<primary_signal_id>|RANDOM_SIDE")` is below 128;
- `constant_long`;
- `constant_short`;
- `one_extra_bar_delay`: entry and exit both shifted by five elapsed minutes,
  then exact bars and full containment are required;
- `one_operation_delay`: entry is the scheduled entry of the next complete
  causal operation in the same uninterrupted segment after the primary
  decision, exit is 72 hours later, then chronological global non-overlap and
  split containment are reapplied.

The raw parent signal IDs and sides are unchanged. Independent delayed
reservation or split containment may shrink the delayed accepted set; every
such suppression, crossing interval, and missing same-segment successor is
reported. No delayed control may create a new parent event, change a side or
hold, or queue a suppressed event.

## Source-support gates

All gates use accepted, globally non-overlapping primary entries unless the
metric explicitly says raw or rank-ready. Operation and batch attribution uses
`available_at_utc`, not `operation_date`. Split boundaries do not reset rank
history; only an invalid causal batch does. Undefined or empty denominators
fail. Every metric and failure reason is written even when an earlier metric in
the same already-open source stage fails.

Coverage is measured over the complete source interval
`[2019-01-01T00:00:00Z, 2024-01-01T00:00:00Z)` and separately over warmup
`[2019-01-01T00:00:00Z, 2020-01-01T00:00:00Z)`, train, and selection:

- parser coverage =
  valid parsed joined detail rows / all joined detail rows;
- complete-operation share =
  complete operations / all operation rows;
- singleton-batch share =
  valid one-complete-operation availability batches /
  all distinct `available_at_utc` batches.

Each of those three ratios must equal exactly `1.0` in the full interval and
in every one of warmup, train, and selection. Thus 2019 rows that can enter
rank history receive the same fail-closed parser and integrity requirements as
2020–2023 rows.

### Completeness and selectivity

| Coverage gate | Full | Warmup | Train | Selection |
|---|---:|---:|---:|---:|
| Description parser coverage | `1.0` | `1.0` | `1.0` | `1.0` |
| Complete-operation share | `1.0` | `1.0` | `1.0` | `1.0` |
| Single-operation causal-batch share | `1.0` | `1.0` | `1.0` | `1.0` |

| Selectivity gate | Train | Selection |
|---|---:|---:|
| Rank-ready complete operations | at least `740` | at least `240` |
| Raw LOW share of rank-ready operations | `[0.05, 0.20]` | `[0.05, 0.20]` |
| Raw HIGH share of rank-ready operations | `[0.05, 0.20]` | `[0.05, 0.20]` |

### Accepted-event support

| Gate | Train | Selection |
|---|---:|---:|
| Total entries | `[60, 180]` | `[18, 70]` |
| Each calendar year | at least `15` | N/A |
| Each half-year | at least `6` | at least `7` |
| Each quarter | at least `2` | at least `3` |
| Each side count | at least `15` | at least `5` |
| Each side share | at least `0.20` | at least `0.20` |
| Active UTC months | at least `20` | at least `8` |
| Maximum UTC-month share | at most `0.15` | at most `0.25` |
| Maximum UTC-quarter share | at most `0.30` | at most `0.45` |
| Maximum elapsed entry gap | at most `90` days | at most `90` days |
| Maximum same-side run | at most `8` | at most `6` |

Month/quarter shares use accepted split entries as denominator. Gaps and runs
use chronological `(entry_time,signal_id)` order. Each year/half/quarter is a
fixed UTC calendar subperiod and requires full interval containment.

### Internal component distinctness

Each of the four non-primary source clocks must have at least 30 contained train
entries and 10 contained selection entries, with both sides at least 20% in
each split. Against primary in each split, each must satisfy:

- exact distinct-entry-set Jaccard `<= 0.70`;
- exact-entry same-side reproduction divided by primary entries `<= 0.70`;
- absolute signed five-minute occupied-exposure Pearson correlation `<= 0.80`.

Zero-variance, undefined, or nonfinite correlation fails. A failure means the
primary is not sufficiently distinct from its own amount-only projection and
retires SMAF before external comparators or outcomes.

The source gate order is:

1. frozen identity and exact header;
2. schema, join, uniqueness, and exact reconciliation;
3. parser coverage and complete operations;
4. singleton causal batches;
5. rank coverage and tail selectivity;
6. primary event support;
7. internal component distinctness.

The first failed numbered gate retires SMAF unchanged. No external comparator
or market row is opened after failure.

## External novelty contract

Novelty opens only after every source-support gate passes. It follows
`docs/novelty-comparator-common-window-policy-2026-07-23.md`, SHA-256
`928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580`.

The exact common window is
`[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`. Raw comparator schema,
side vocabulary, group vocabulary, timestamps, ordering, duplicate identity,
and interval validity are checked before filtering. Only intervals with
`entry >= window_start` and `exit <= window_end` are retained; crossing rows
are reported and excluded whole.

### Immutable comparator A: SLCS

- path:
  `results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz`;
- file SHA-256:
  `b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948`;
- header SHA-256:
  `45a24e800b79a30047ffeb5f45c69cf4817262e57b0af1cf5e046332536e5e94`;
- columns: `control,entry_time,exit_time,side`;
- side map: string `1 -> LONG`, string `-1 -> SHORT`;
- exact full group vocabulary:
  `primary`, `carry_intensity_only`, `constant_long`, `constant_short`,
  `demand_breadth_only`, `demand_intensity_only`,
  `deterministic_random_side`, `exact_direction_flip`,
  `five_operation_stale`, `mean_without_consensus`,
  `one_operation_stale`, `same_sign_without_magnitude`,
  `weighted_fee_only`, `year_component_permutation`;
- comparison subset, with groups compared separately:
  `primary`, `demand_intensity_only`, `weighted_fee_only`,
  `carry_intensity_only`, `demand_breadth_only`.

### Immutable comparator B: SCAF

- path:
  `data/soma_collateral_allocation_fracture_clocks_2020_2023.csv.gz`;
- file SHA-256:
  `64e07005d70442bfa7a110b1e6bea9802ee94be16d95f6e7db9228f4790a28e6`;
- header SHA-256:
  `770965eb9e07bbca6f6b3f3c3165fe5c04301ef6573da86dacd161582cfa8c8f`;
- columns: `control,entry_time,exit_time,side`;
- side map: `LONG -> LONG`, `SHORT -> SHORT`;
- exact full group vocabulary:
  `primary`, `inventory_mismatch_only`, `award_distortion_only`,
  `unmet_demand_mass_only`, `fee_distortion_only`,
  `mean_change_without_consensus`, `two_of_four_without_opposition`,
  `one_batch_stale`, `five_batch_stale`,
  `within_batch_demand_permutation`, `exact_direction_flip`,
  `deterministic_random_side`, `constant_long`, `constant_short`;
- comparison subset, with groups compared separately:
  `primary`, `inventory_mismatch_only`, `award_distortion_only`,
  `unmet_demand_mass_only`, `fee_distortion_only`.

SCAF was retired source-only, but its immutable outcome-blind clock remains a
valid family comparator. Every selected comparator group must contain at least
20 fully contained rows.

### Exact metrics and thresholds

For each selected group separately:

1. **Exact-entry Jaccard**: intersection over union of distinct UTC entry-time
   sets, `<= 0.20`.
2. **Exact-entry same-side reproduction**: exact timestamp and normalized side
   matches divided by the number of candidate entries, `<= 0.30`.
3. **24-hour one-to-one containment**: sort distinct candidate and comparator
   entries. Apply a two-pointer maximum-cardinality match: discard the earlier
   timestamp only when it lies more than 24 elapsed hours before the other;
   otherwise match the pair and advance both. Let `m` be matched pairs. Both
   `m/N_candidate` and `m/N_comparator` must be `<= 0.40`.
4. **Signed occupied-exposure correlation**: create the complete five-minute
   UTC grid over the common window; encode `[entry,exit)` as `+1` for LONG,
   `-1` for SHORT, and `0` otherwise. Absolute Pearson correlation must be
   `<= 0.35`.

Duplicate entry times inside one group, overlapping intervals, a non-five-
minute boundary, zero variance, undefined correlation, empty denominator, or
nonfinite result fails. All ten selected groups must pass. Comparator removal,
threshold relaxation, side remapping, clipping, or a wider tolerance after
incidence is forbidden.

## Sealed economic contract

The following artifacts are identity-frozen now but their data rows remain
unread until source support and novelty both pass and a separate strict
evaluator is committed:

| Artifact | SHA-256 | Header SHA-256 |
|---|---|---|
| `data/binance_um_kline_reference_btc_2020_2023/build_manifest.json` | `c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e` | N/A |
| `data/binance_um_kline_reference_btc_2020_2023/BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz` | `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d` | `5e8d51e7e1218929db6a54ca59280eb4306171b81d5d0880467a85cf9d23eff2` |
| `results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json` | `a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b` | N/A |
| `data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz` | `3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6` | `71b2b1395313f631969674c43e569c8f1619a9fb23c8316e2e0478c32f01d61f` |

At freeze time, the two market paths under
`data/binance_um_kline_reference_btc_2020_2023/` are not materialized in this
worktree. Matching bytes were hash-checked in a local artifact store without
decoding market rows. The logical repository-relative paths and hashes above,
not the local store location, define identity.

Before the economic evaluator commit, a separate hydration step must:

1. copy bytes from any read-only source into the exact logical paths above;
2. refuse symlinks, devices, FIFOs, sockets, and non-regular files;
3. set the hydrated files read-only;
4. validate all four full-file SHA-256 identities;
5. validate both gzip header SHA-256 identities;
6. write exactly
   `results/smaf_72_economic_artifact_hydration_2026-07-30.json`;
7. open no market or funding data row until all checks pass.

The hydration manifest uses UTF-8, one trailing LF, sorted keys, two-space
indentation, `ensure_ascii=true`, and `allow_nan=false`. It has exactly:

```text
{
  "protocol_version": "smaf_72_economic_artifact_hydration_v1",
  "artifacts": [
    {
      "logical_path": "<repository-relative path>",
      "portable_source_locator": "local-cache:sha256:<64 lowercase hex>",
      "copied_at_utc": "YYYY-MM-DDTHH:MM:SS.ffffffZ",
      "size_bytes": <nonnegative integer>,
      "sha256": "<64 lowercase hex>",
      "header_sha256": null or "<64 lowercase hex>",
      "regular_file": true,
      "symlink": false,
      "mode": "0444",
      "rows_opened_before_validation": 0
    }
  ],
  "market_rows_opened": 0,
  "funding_rows_opened": 0,
  "manifest_hash": "<canonical SHA-256>"
}
```

The four artifact objects are sorted by `logical_path` and no other key is
allowed. `copied_at_utc` is UTC RFC3339 with exactly six fractional digits and
literal `Z`. Each portable locator suffix equals that object's `sha256`.
`header_sha256` is null for JSON manifests and the frozen header hash for gzip
data. `manifest_hash` is SHA-256 of the compact sorted-key JSON encoding of the
object without `manifest_hash`, using separators `(",",":")`,
`ensure_ascii=true`, and `allow_nan=false`.

The host absolute source paths may appear only in the uncommitted, gitignored
local log `.omx/local/smaf-72-hydration-source-path.json`; they are forbidden
from the committed hydration manifest, evaluator output, and research result.
The later economic evaluator must freeze both the committed hydration-manifest
file SHA-256 and its internal `manifest_hash` as code constants, validate both
before opening any row, and fail on absence or drift.

The evaluator has no fallback artifact root and refuses an absent logical path.
No absolute local-store path enters this preregistration or later results. Any
identity, exact-header, or hydration-manifest drift fails before loading rows.

### Accounting

- instrument: Binance USD-M `BTCUSDT` perpetual;
- initial equity: `1.0`;
- side sign: LONG `+1`, SHORT `-1`;
- entry: exact five-minute `open` at `entry_time`;
- scheduled exit: exact five-minute `open` at `exit_time`;
- quantity: `0.5 * pre_entry_equity / entry_open`, fixed through exit;
- base cost: `6 bp/notional/side`;
- stress cost: `10 bp/notional/side`;
- cost cash: `abs(quantity) * execution_price * bp / 10,000`;
- funding inclusion: `entry_time <= funding_time < exit_time`;
- funding cash:
  `-side_sign * quantity * funding_rate * settlement_mark_price`;
- stops, take-profit, liquidation, early exit, compounding inside a trade, and
  overlapping positions: forbidden.

Missing or duplicate market bars, nonpositive OHLC, malformed funding rows, or
a missing exact entry/exit bar fails the stage. Full-calendar CAGR includes all
idle cash and uses elapsed seconds divided by `365.25*86400`.

Strict MDD uses a global high-water mark established before each entry. It
deducts the entry fee, then for every held bar applies funding credits before
the favorable side-specific extreme and funding debits before the adverse
extreme. The adverse mark includes a hypothetical exit fee. It finally marks
and charges the scheduled exit. This favorable-then-adverse ordering, global
pre-entry high-water mark, intratrade extremes, funding, and virtual exit cost
are required for primary and every control.

`mean_gross_underlying_bp` is:

```text
mean(side_sign * (exit_open / entry_open - 1) * 10,000)
```

The weekly cluster test groups base-cost net trade PnL by UTC ISO entry week.
The exact stage tokens are `TRAIN_2020_2022` and `SELECTION_2023`. Draw indices
are integers `0..19999` formatted as five decimal digits. ISO year is four
digits and ISO week is two digits. For each draw/cluster, hash this exact UTF-8
byte string with no newline:

```text
SMAF-72|<STAGE_TOKEN>|<DRAW_00000>|<ISO_YEAR_4>-W<ISO_WEEK_2>
```

The bit is the most significant bit of digest byte zero
(`digest[0] & 0x80`). Bit one multiplies cluster PnL by `-1`; bit zero
multiplies it by `+1`. The one-sided add-one p-value is:

```text
(1 + count(flipped_total >= observed_total)) / 20,001
```

### Sequential outcome gates

No 2023 row is loaded during the train stage. If train fails, selection remains
sealed and SMAF retires. If train passes, the unchanged evaluator opens 2023
once. Parser, side, threshold, controls, and accounting cannot change between
stages.

Train is exactly 2020–2022 and must satisfy:

- executed trades `>= 60`, each year `>= 15`, each side `>= 15`;
- base and stress absolute return each strictly positive;
- base absolute return strictly positive in each calendar year;
- base full-calendar CAGR/strict-MDD ratio `>= 3.0`;
- stress full-calendar CAGR/strict-MDD ratio `>= 2.5`;
- base and stress strict MDD each `<= 0.15`;
- mean gross underlying move `>= 35 bp`;
- weekly cluster sign-flip p-value `<= 0.10`;
- one-extra-bar-delay and one-operation-delay absolute return each positive;
- primary base absolute return greater than exact flip, deterministic random,
  constant-long, and constant-short;
- primary base CAGR/MDD at least `0.25` above the strongest of the four
  mechanism controls, and primary base absolute return greater than each.

Selection is exactly 2023 and must satisfy:

- executed trades `>= 18`, each half `>= 7`, each side `>= 5`;
- base and stress absolute return each strictly positive;
- base absolute return strictly positive in each half;
- base full-calendar CAGR/strict-MDD ratio `>= 3.0`;
- stress full-calendar CAGR/strict-MDD ratio `>= 2.5`;
- base and stress strict MDD each `<= 0.15`;
- mean gross underlying move `>= 35 bp`;
- weekly cluster sign-flip p-value `<= 0.20`;
- one-extra-bar-delay and one-operation-delay absolute return each positive;
- the same direction/random/constant and `+0.25` mechanism-control margins.

Absolute return, CAGR, strict MDD, ratio, trade and side counts, subperiod
returns, gross basis points, funding, costs, cluster counts and p-value are
reported together. A positive return without its CAGR/MDD pair is never a pass.
Nonfinite ratios or zero-MDD controls fail the margin comparison rather than
being treated as favorable infinity.

## RLLM boundary and post-2023 seal

RLLM is forbidden until deterministic train and selection both pass. A later
separately committed policy may only choose `TRADE_FIXED_SIDE` or `ABSTAIN`
from symbolic source-state buckets. It may not see raw maturity, amount, rank
numerators, timestamps, split identity, price, return, funding, reward, PnL,
CAGR, or MDD, and may not change direction, entry, exit, hold, leverage, or
accounting. Fit is limited to 2020–2022; 2023 feedback may not alter it.

All 2024+ source and outcomes remain sealed. Promotion requires:

1. a separately committed official-source extension manifest;
2. append-only raw response capture and retrieval timestamps;
3. revision alarms;
4. exact schema and parser parity with this preregistration;
5. a separately committed evaluator and forward split.

The historical API is a current snapshot rather than a vintage archive, so no
live claim is permitted without that forward capture.

## Evidence boundary at freeze

The following is an operator-reported, conservative contamination envelope,
not a claim that historical access can be independently reconstructed from
pre-existing `HEAD`. No pre-SMAF audit artifact recorded the interactive probe
or comparator inventory scan at the time they occurred. Repeating those reads
now could confirm the disclosed rows and artifact group totals but could not
prove the prior access history. SMAF therefore treats every item below as
exposed and makes no pristine-source claim.

Disclosed before this document:

- security-description grammar probe rows: `8`;
- source identity fields in that probe: `8`;
- source amount/rate/availability value rows read for SMAF: `0`;
- SMAF centroids, fracture values, ranks, tails, or events derived: `0`;
- SLCS comparator rows scanned only to inventory immutable group counts:
  `1,685`;
- SCAF comparator rows scanned only to inventory immutable group counts:
  `5,809`;
- SMAF-to-comparator overlap or correlation computed: `false`;
- BTC market rows loaded: `0`;
- funding data rows loaded: `0`;
- forward returns, PnL, CAGR, or MDD opened: `false`.

Market and funding artifact hashes and manifest metadata were inspected without
decoding market or funding data rows. The next authorized step is only to
commit a write-once preregistration artifact and its tests. Source incidence
may open only under a later committed source-support evaluator.
