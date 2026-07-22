# QLCD-288 strict economic evaluator freeze — 2026-07-20

## Outcome boundary

The QLCD-288 phase-one evaluator is frozen before parsing any BTC execution
OHLC row, funding row, return, PnL, CAGR, or MDD. Freeze may hash and parse only
the outcome-blind QLCD feature source, primary clock, derived control clocks, and
their manifests. It records previously declared execution-container identities
but does not hash, decompress, or numerically parse either execution container.

## Frozen phase scope

- `train`: `[2020-01-01, 2023-01-01)`;
- `selection`: `[2023-01-01, 2024-01-01)`, opened only after train passes;
- `test=2024`, `eval=2025`, and `recent_report=2026+` remain inaccessible;
- phase two requires a separately committed evaluator after both phase-one
  stages pass and before any 2024+ signal or execution outcome is opened.

The source-only primary clock has 489 events. Full stage containment admits 377
train events and 111 selection events. One position is excluded rather than
reassigned or shortened because it crosses the train/selection boundary:

- decision: `2022-12-31T04:20:00Z`;
- entry: `2022-12-31T04:25:00Z`;
- exit: `2023-01-01T04:25:00Z`;
- side: `-1`;
- score: `0x1.14a088d9370aep-4`;
- threshold: `0x1.78612625d4192p-5`;
- exclusion identity hash:
  `7b7824b732032476f7409ea93cf942fd9731bcf949de8a42faf7ec0fa4b5eff3`.

`decision_time` is the completed signal bar's close timestamp. Entry is the
next five-minute open (`decision_time + 5m`), exactly matching the source-only
clock's preregistered `t+2` bucket. Exit is the open exactly 288 five-minute
bars later (24 hours). Every schedule remains non-overlapping.

## Frozen source-only falsification controls

All preregistered controls are derived and clocked before execution outcomes:

| Clock | Events | Frozen construction |
|---|---:|---|
| primary | 489 | preregistered QLCD-288 clock |
| exact_side_flip | 489 | identical times and scores, side multiplied by `-1` |
| medium_vs_fine | 547 | medium replaces coarse, own prior-only q99.75 clock |
| remove_opposition | 504 | coarse share × coherence, no opposition term, own clock |
| all_quantity_imbalance | 486 | absolute all-size imbalance, own prior-only clock |
| stale_one_hour | 489 | primary decision, entry, and exit shifted exactly `+12` bars |
| stale_twenty_four_hours | 489 | primary decision, entry, and exit shifted exactly `+288` bars |

The six controls are mandatory report-only diagnostics. The preregistration did
not set a control-margin promotion threshold, so no control can pass, repair,
reverse, tune, or otherwise rescue a failed primary clock.

## Frozen accounting

- exposure: `0.5x` current pre-entry equity;
- base cost: `6 bp` per notional side;
- stress cost: `10 bp` per notional side;
- entry and exit: frozen BTCUSDT USD-M five-minute opens;
- funding: exact funding time/rate and frozen settlement mark;
- funding boundary: exact entry/exit credits are dropped, while debits are
  retained; interior events are symmetric;
- strict MDD: one global pre-entry high-water mark across the declared stage,
  entry cost, funding marks, every held five-minute favorable-then-adverse OHLC
  path, virtual adverse exit cost, and actual exit cost;
- CAGR: full declared stage calendar, including warm-up and idle cash;
- gross move: unlevered side-signed entry-open to exit-open return in basis
  points;
- weekly diagnostic: nominal two-sided ISO-entry-week clustered sign flips,
  exact through 20 clusters and otherwise 20,000 Monte Carlo draws with seed
  `20260720`.

The weekly value is one frozen randomization diagnostic and one preregistered
gate. It is not claimed as a standalone discovery p-value, a multiple-search
adjusted result, or sufficient evidence by itself.

For train-year and selection-half positivity, only trades fully contained in
the corresponding frozen subperiod are used. The aggregate stage likewise uses
only positions fully contained in its stage window.

## Frozen stopping gates

Every opened phase-one stage must pass all of the following without repair:

1. base absolute return `> 0`;
2. base CAGR / strict MDD `>= 3.0`;
3. strict MDD `<= 15%`;
4. stress absolute return `> 0`;
5. stress CAGR / strict MDD `>= 2.5`;
6. mean gross underlying move `>= 24 bp`;
7. nominal weekly-cluster two-sided sign-flip diagnostic `p < 0.10`.

Train additionally requires positive absolute return in each fully contained
calendar year 2020, 2021, and 2022. Selection additionally requires positive
absolute return in each fully contained 2023 half. The first failed stage
retires QLCD-288; no threshold, side, delay, hold, cost, split, subperiod,
control, or gate may change.

## Physical source isolation

Freeze verifies the parent manifests and records their previously declared
compressed-container hashes as metadata. It deliberately does **not** recompute
either full 2020-2023 parent hash, because that would touch selection bytes
before train is admitted.

After freeze, stage preparation verifies every prior-stage pass first. It then
copies only the opened stage prefix/slice, computes the new slice hash, and
stops immediately after the exact expected row count without reading the first
future row. Exact five-minute and funding grids are numerically validated only
inside that opened slice. Stage reload hashes only the physically isolated
slice, reconstructs that same prefix/slice without reading beyond its bound,
and rejects changed paths, bytes, row counts, prior/future rows, or a stage
artifact that no longer equals the frozen parent slice.

The train preparation and reload regression tests make any direct call to
`_sha256` on either unsliced parent container fail. Selection source access is
ordered after verification of a passing train artifact.

## Phase-two contract

The current evaluator approves only train and selection. A new committed
phase-two evaluator must, before opening 2024 outcomes:

1. obtain official Binance USD-M BTCUSDT daily aggTrade archives and bind every
   archive to its official `CHECKSUM`;
2. abort on missing/mismatched checksums or parse/quantity-precision failures;
3. apply the frozen whole-day quarantine rules for ID anomalies, cross-day
   discontinuity, duplicates, or unexplained missing five-minute buckets;
4. concatenate the hash-bound 2020-2023 source with future source, continue the
   strictly-prior 8,640-bar q99.75 baseline without reset, and continue the
   global non-overlap scheduler without reset;
5. reproduce the complete pre-2024 primary clock byte-for-byte before admitting
   a future event;
6. freeze every future stage's full-containment exclusions before its outcomes;
7. use checksum-bound official 5m klines and exact funding events/marks,
   physically isolate each stage, and reject symlinks or stale caches;
8. freeze the latest completed UTC day used by `recent_report`; and
9. keep all rules immutable after phase one.

No current code path supports post-2023 access.

## Frozen identities

- evaluator source SHA-256:
  `1ed4ffa7aca2bbe3c84dc7dca05c537ab020a11ee5bde2d4219770721d755f2d`;
- evaluation-clock SHA-256:
  `c699c2d8c462b465579eb4035c76dda96923a4f39663395b371a04e9ad6de4a9`;
- evaluator manifest hash:
  `9ea2049f8c4ae02350b241a716284cc01463a5658c2469d0cab2adef2e12f992`;
- evaluator JSON SHA-256:
  `417ef91ed16f35c99093d6809fbd515a6981aca9db539b0a3419e7b0b4702ed7`;
- source clock SHA-256:
  `ed882ac8a28f1f0b2b7ad7bf3d2de1f37b175cde63b20d4d1c7a290f3eb89bec`;
- support result SHA-256:
  `d5b5f2e59fe2f8d8df775a9ee7a05da0bab2898af210d6e724669d9781efe640`.

At freeze time, opened windows must be `[]`, sealed windows must be
`[train, selection, test, eval, recent]`, parsed execution/funding rows must be
`0/0`, outcome bytes hashed must be `false`, and simulation must be `false`.
